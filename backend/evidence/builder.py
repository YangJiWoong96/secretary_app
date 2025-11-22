import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from backend.config import get_settings
from backend.utils.tracing import traceable
from backend.rag.retrieval import retrieve_from_rag
from backend.search_engine.service import build_web_context
from backend.utils.logger import log_event


def _now() -> float:
    return time.time()


def _hash_q(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# 내부 헬퍼 (동작 불변, 가독성/중복 제거 목적)
# ─────────────────────────────────────────────────────────────
def _blocks_to_web_docs(blocks: list[str]) -> list["WebDoc"]:
    """
    3줄 블록 문자열 목록을 WebDoc 리스트로 변환한다.
    - 각 블록: 제목/설명/URL (부족한 경우 안전 폴백 유지)
    - 기존 구현과 동일한 필드 매핑, id 포맷(web:{idx})을 보장한다.
    """
    docs: list[WebDoc] = []
    for idx, block in enumerate(blocks):
        lines = (block or "").split("\n")
        title = lines[0] if len(lines) > 0 else None
        desc = lines[1] if len(lines) > 1 else None
        url = lines[2] if len(lines) > 2 else ""
        docs.append(
            WebDoc(
                id=f"web:{idx}",
                url=url,
                title=title,
                published_at=None,
                snippet=desc,
            )
        )
    return docs


def _score_trust_consistency(
    docs: list["WebDoc"],
) -> tuple[float, float, dict[str, float]]:
    """
    교차검증 스코어 계산 (trust, consistency, domain_score).
    기존 내부 중첩 함수와 동일한 로직을 모듈 헬퍼로 승격하였다.
    """
    try:
        from backend.proactive.consistency_checker import ConsistencyChecker
        from backend.proactive.trust_scorer import calculate_trust_score
        from urllib.parse import urlparse

        sources = []
        for d in docs:
            try:
                dom = (urlparse(d.url or "").netloc or "").lower()
                if dom.startswith("www."):
                    dom = dom[4:]
            except Exception:
                dom = ""
            content = f"{(d.title or '').strip()}\n{(d.snippet or '').strip()}".strip()
            sources.append({"domain": dom, "content": content, "published_at": None})

        trusts: list[float] = []
        domain_score: dict[str, float] = {}
        for it in sources:
            sc = float(calculate_trust_score(it, sources))
            trusts.append(sc)
            dom = it.get("domain") or ""
            if dom:
                domain_score[dom] = max(sc, domain_score.get(dom, 0.0))
        trust_val = max(trusts) if trusts else 0.0
        cons_val = ConsistencyChecker().calculate_consistency(
            [{"content": s.get("content", "")} for s in sources]
        )
        return trust_val, float(cons_val), domain_score
    except Exception:
        # 실패 시, 기존 로직과 동일하게 0.0 기반 값들을 반환
        return 0.0, 0.0, {}


@dataclass
class WebDoc:
    id: str
    url: str
    title: str | None
    published_at: str | None
    snippet: str | None


@dataclass
class RagDoc:
    id: str
    path: str | None
    chunk_id: str | None
    snippet: str | None


@dataclass
class EvidenceBundle:
    web_docs: List[WebDoc]
    rag_docs: List[RagDoc]
    citations_policy: str
    # 공통 blocks 포맷(3줄 블록 텍스트). web/rag 동일 필드 유지
    web_blocks: str = ""
    rag_blocks: str = ""


_CACHE: Dict[str, Tuple[float, EvidenceBundle, str, str]] = {}
_TTL_SEC = 180.0  # 3분


@traceable(name="Evidence: build_evidence", run_type="chain", tags=["evidence"])
async def build_evidence(
    mcp_url: str,
    session_id: str,
    q2: str,
    web_on: bool,
    rag_on: bool,
    timeout_s: float,
    rag_date_filter: Optional[Tuple[int, int]] = None,
) -> Tuple[EvidenceBundle, str, str]:
    """
    q2(재작성 쿼리)에 대해 증거를 수집한다. 3분 캐시.
    반환: (EvidenceBundle, web_ctx_blocks, rag_ctx_blocks)
    """
    # 캐시 키: (session_id, web_on, rag_on, q_hash) — 사용자/옵션 격리
    key = f"{session_id}:{int(bool(web_on))}:{int(bool(rag_on))}:{_hash_q(q2)}"
    now = _now()
    if key in _CACHE and (now - _CACHE[key][0]) <= _TTL_SEC:
        ts, bundle, web_ctx, rag_ctx = _CACHE[key]
        return bundle, web_ctx, rag_ctx

    web_docs: List[WebDoc] = []
    rag_docs: List[RagDoc] = []
    web_ctx = ""
    rag_ctx = ""
    web_blocks = ""
    rag_blocks = ""

    if web_on:
        # freshness_days: 태거 freshness → 기간 결정(high:30, med:180, low:365)
        try:
            # 태깅 결과가 세션에 있다면 참조
            from backend.policy import SESSION_STATE

            st = SESSION_STATE.get(session_id, {}) or {}
            tag = ((st.get("last_tagging") or {}).get("tags") or {}).get(
                "freshness", "low"
            )
            fd = (
                30
                if str(tag).lower() == "high"
                else 180 if str(tag).lower() == "medium" else 365
            )
        except Exception:
            fd = 365
        kind, web_ctx = await build_web_context(
            mcp_url, q2, display=3, timeout_s=timeout_s
        )
        # MCP freshness는 서버에서 처리하므로, 필요시 client로도 전달 가능
        # 현재 build_web_context는 블록 문자열 제공
        web_blocks = web_ctx or ""
        try:
            if bool(get_settings().TRACE_TEXT):
                logging.getLogger("evidence").info(
                    "[evidence] web_blocks_full=%s", web_blocks
                )
        except Exception:
            pass
        blocks = (web_blocks or "").split("\n\n")
        # Rerank: STWM 앵커 기반 가중 제거(오염 방지). 순수 결과 유지.
        web_docs.extend(_blocks_to_web_docs(blocks))

        # (신규) 증거 cross-check 루프: 신뢰/일관성 임계 미달 시 제외 도메인 재시도
        if get_settings().FEATURE_EVIDENCE_CROSSCHECK:
            try:
                _s = get_settings()
                min_trust = float(_s.EVIDENCE_TRUST_MIN)
                min_cons = float(_s.EVIDENCE_CONSIST_MIN)
                loop = 0
                trust_val, cons_val, dom_score = _score_trust_consistency(web_docs)
                while (trust_val < min_trust or cons_val < min_cons) and loop < 2:
                    loop += 1
                    # 낮은 신뢰 도메인을 제외 대상으로 선정(최대 3개)
                    low_domains = [
                        d
                        for d, s in sorted(dom_score.items(), key=lambda kv: kv[1])
                        if s < min_trust
                    ][:3]
                    if not low_domains:
                        break
                    kind_retry, web_ctx_retry = await build_web_context(
                        mcp_url,
                        q2,
                        display=3,
                        timeout_s=timeout_s,
                        endpoints=None,
                        exclude_sites=low_domains,
                    )
                    web_blocks = web_ctx_retry or ""
                    # 웹 문서 재구성
                    blocks2 = (web_blocks or "").split("\n\n")
                    web_docs = _blocks_to_web_docs(blocks2)
                    trust_val, cons_val, dom_score = _score_trust_consistency(web_docs)
                    try:
                        log_event(
                            "evidence.crosscheck.loop",
                            {
                                "loop": loop,
                                "trust": round(trust_val, 4),
                                "cons": round(cons_val, 4),
                                "exclude": low_domains,
                            },
                        )
                    except Exception:
                        pass
            except Exception:
                # 모델/환경 미구성 등으로 실패 시 원 결과 유지
                pass

    if rag_on:
        # 기존 RAG는 문자열 컨텍스트를 반환하므로 동일 방식으로 문서화
        rag_ctx = retrieve_from_rag(
            session_id, q2, top_k=3, date_filter=rag_date_filter
        )
        # rag_ctx를 web과 동일한 3줄 블록 문자열로 정규화
        # 규칙: 각 문단의 첫 줄을 제목, 둘째 줄을 스니펫, 셋째 줄은 비워두는 대신 가상의 식별자 링크
        # 링크는 추후 소스 경로가 생기면 대체 가능
        parts = (rag_ctx or "").split("\n\n")
        norm_blocks: List[str] = []
        for i, block in enumerate(parts):
            lines = [ln for ln in (block or "").split("\n") if ln.strip()]
            if not lines:
                continue
            title = lines[0][:80]
            desc = (" ".join(lines[1:])[:140]) if len(lines) > 1 else "(설명 없음)"
            link = f"rag://session/{session_id}/hit/{i}"
            norm_blocks.append(
                "\n".join([title or "(제목 없음)", desc or "(설명 없음)", link])
            )
        rag_blocks = "\n\n".join(norm_blocks)
        try:
            _LOGGER = logging.getLogger("evidence")
            _LOGGER.info(
                "[evidence] q='%s' rag_date_filter=%s web_on=%s rag_on=%s rag_blocks_len=%d",
                (q2[:80] if isinstance(q2, str) else ""),
                str(rag_date_filter),
                str(web_on),
                str(rag_on),
                len(rag_blocks or ""),
            )
            if bool(get_settings().TRACE_TEXT):
                _LOGGER.info("[evidence] rag_blocks_full=%s", rag_blocks)
        except Exception:
            pass
        for idx, block in enumerate((rag_blocks or "").split("\n\n")):
            # 각 블록은 라벨 라인 포함 가능 → 스니펫으로 저장
            rag_docs.append(
                RagDoc(id=f"rag:{idx}", path=None, chunk_id=None, snippet=block)
            )

    bundle = EvidenceBundle(
        web_docs=web_docs[:5],
        rag_docs=rag_docs[:5],
        citations_policy="strict",
        web_blocks=web_blocks,
        rag_blocks=rag_blocks,
    )
    # 상세 로깅(미리보기)
    try:
        import json as _json

        _LOGGER = logging.getLogger("evidence")
        _LOGGER.info(
            "[evidence:bundle] sid=%s web_on=%s rag_on=%s web_len=%d rag_len=%d",
            session_id,
            str(web_on),
            str(rag_on),
            len(web_blocks or ""),
            len(rag_blocks or ""),
        )
        if bool(get_settings().TRACE_TEXT):
            _LOGGER.info("[evidence:web:preview]\n%s", (web_blocks or "")[:500])
            _LOGGER.info("[evidence:rag:preview]\n%s", (rag_blocks or "")[:500])
    except Exception:
        pass
    _CACHE[key] = (now, bundle, web_blocks, rag_blocks)
    return bundle, web_blocks, rag_blocks
