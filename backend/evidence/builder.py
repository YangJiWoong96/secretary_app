import time
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from backend.search_engine.service import build_web_context
from backend.rag.retrieval import retrieve_from_rag


def _now() -> float:
    return time.time()


def _hash_q(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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


_CACHE: Dict[str, Tuple[float, EvidenceBundle, str, str]] = {}
_TTL_SEC = 180.0  # 3분


async def build_evidence(
    mcp_url: str,
    session_id: str,
    q2: str,
    web_on: bool,
    rag_on: bool,
    timeout_s: float,
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

    if web_on:
        kind, web_ctx = await build_web_context(
            mcp_url, q2, display=3, timeout_s=timeout_s
        )
        # 현재 build_web_context는 블록 문자열만 제공 → Evidence는 최소 구조 유지
        blocks = (web_ctx or "").split("\n\n")
        # Rerank: STWM anchors(장소/행위/대상/주제)를 포함하는 블록 우대
        try:
            from backend.memory.stwm import get_stwm_snapshot

            stwm = get_stwm_snapshot(session_id)
            anchors = (
                [
                    str(stwm.get(k) or "")
                    for k in ("last_loc", "last_act", "last_target", "last_topic")
                ]
                if stwm
                else []
            )

            def _score(block: str) -> int:
                s = 0
                for a in anchors:
                    if a and a in block:
                        s += 1
                return s

            blocks = sorted(blocks, key=_score, reverse=True)
        except Exception:
            pass
        for idx, block in enumerate(blocks):
            lines = block.split("\n")
            title = lines[0] if len(lines) > 0 else None
            desc = lines[1] if len(lines) > 1 else None
            url = lines[2] if len(lines) > 2 else ""
            web_docs.append(
                WebDoc(
                    id=f"web:{idx}",
                    url=url,
                    title=title,
                    published_at=None,
                    snippet=desc,
                )
            )

    if rag_on:
        # 기존 RAG는 문자열 컨텍스트를 반환하므로 동일 방식으로 문서화
        rag_ctx = retrieve_from_rag(session_id, q2, top_k=3)
        for idx, block in enumerate((rag_ctx or "").split("\n\n")):
            # 각 블록은 라벨 라인 포함 가능 → 스니펫으로 저장
            rag_docs.append(
                RagDoc(id=f"rag:{idx}", path=None, chunk_id=None, snippet=block)
            )

    bundle = EvidenceBundle(
        web_docs=web_docs[:5], rag_docs=rag_docs[:5], citations_policy="strict"
    )
    _CACHE[key] = (now, bundle, web_ctx, rag_ctx)
    return bundle, web_ctx, rag_ctx
