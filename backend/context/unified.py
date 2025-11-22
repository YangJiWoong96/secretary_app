import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory

from backend.memory.stwm import get_stwm_snapshot
from backend.memory.turns import get_summaries as tb_get_summaries

# 내부 모듈 의존성은 지연 임포트 대신 명시 임포트로 고정
from backend.rag.embeddings import embed_documents, embed_query_openai

# ---------------------------------------------
# 통합 파이프라인 구성 요소 (최소 침습 주입용)
# ---------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    v = float(np.dot(a, b) / denom)
    if v < 0:
        return 0.0
    return v


def _time_weight(ts: float, now: float, half_life_sec: float = 3600.0) -> float:
    # 최근 항목에 더 높은 가중치 부여(지수 감쇠)
    try:
        age = max(0.0, now - float(ts or 0.0))
    except Exception:
        age = 0.0
    lam = np.log(2.0) / max(1.0, half_life_sec)
    return float(np.exp(-lam * age))


@dataclass
class FocusedItem:
    content: str
    score: float
    type: str
    timestamp: float


class QueryFocusedProcessor:
    """
    현재 쿼리가 장기 히스토리에 묻히지 않도록, 모든 메모리 레이어에서
    쿼리 중심 관련성을 계산해 상위 항목만 추출한다.
    출력 컨텍스트 키 순서는 고정(stwm, redis_recent, redis_all, summaries).
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")

    async def extract_query_focused_context(
        self, session_id: str, user_input: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        now = time.time()

        # 임베딩: 통일된 엔트리 포인트 사용 (캐시는 embedding 레이어에서 처리)
        qv = embed_query_openai(user_input)

        # STWM
        stwm = get_stwm_snapshot(session_id) or {}
        stwm_items: List[FocusedItem] = []
        if stwm:
            base_ts = float(stwm.get("ts", now))
            for k, t in [
                ("위치", stwm.get("last_loc", "")),
                ("주제", stwm.get("last_topic", "")),
                ("감정", stwm.get("last_emotion", "")),
            ]:
                s = str(t or "").strip()
                if not s:
                    continue
                iv = _emb_q(s)
                sc = (
                    _cosine(qv, iv)
                    * 3.0
                    * _time_weight(base_ts, now, half_life_sec=900.0)
                )
                stwm_items.append(
                    FocusedItem(
                        content=f"{k}: {s}", score=sc, type=k, timestamp=base_ts
                    )
                )

        # Redis recent/all — 배치 임베딩 최적화
        redis_recent: List[FocusedItem] = []
        redis_all: List[FocusedItem] = []
        try:
            hist = RedisChatMessageHistory(session_id=session_id, url=self.redis_url)
            msgs = [m for m in getattr(hist, "messages", []) if hasattr(m, "content")]
            # recent(최신 20개)
            tail = msgs[-20:]
            tail_texts = [(getattr(m, "content", "") or "").strip() for m in tail]
            sel_idx = [i for i, t in enumerate(tail_texts) if t]
            if sel_idx:
                vecs = [embed_documents([tail_texts[i]])[0] for i in sel_idx]
                k = 0
                for i, m in enumerate(tail):
                    txt = tail_texts[i]
                    if not txt:
                        continue
                    iv = vecs[k]
                    k += 1
                    role = getattr(m, "type", "")
                    r_w = 1.5 if role == "human" else 0.8
                    sc = _cosine(qv, iv) * r_w * _time_weight(now, now, 1800.0)
                    redis_recent.append(
                        FocusedItem(
                            content=txt, score=sc, type="conversation", timestamp=now
                        )
                    )
            # all(전체)
            all_texts = [(getattr(m, "content", "") or "").strip() for m in msgs]
            all_idx = [i for i, t in enumerate(all_texts) if t]
            if all_idx:
                vecs_all = [embed_documents([all_texts[i]])[0] for i in all_idx]
                k = 0
                for i, m in enumerate(msgs):
                    txt = all_texts[i]
                    if not txt:
                        continue
                    iv = vecs_all[k]
                    k += 1
                    role = getattr(m, "type", "")
                    r_w = 1.5 if role == "human" else 0.8
                    sc = _cosine(qv, iv) * r_w * _time_weight(now, now, 7200.0)
                    redis_all.append(
                        FocusedItem(
                            content=txt, score=sc, type="conversation", timestamp=now
                        )
                    )
        except Exception:
            pass

        # Summaries(최근 10개) — 배치 임베딩
        summaries_items: List[FocusedItem] = []
        try:
            sums = tb_get_summaries(session_id)
            last10 = sums[-10:] if sums else []
            texts = [(getattr(s, "answer_summary", "") or "").strip() for s in last10]
            idx = [i for i, t in enumerate(texts) if t]
            if idx:
                vecs = [embed_documents([texts[i]])[0] for i in idx]
                k = 0
                for i, s in enumerate(last10):
                    txt = texts[i]
                    if not txt:
                        continue
                    iv = vecs[k]
                    k += 1
                    ts = float(
                        getattr(getattr(s, "ts_range", {}), "get", lambda *_: 0)(
                            "end", 0
                        )
                    )
                    sc = _cosine(qv, iv) * 1.2 * _time_weight(ts, now, 14400.0)
                    summaries_items.append(
                        FocusedItem(content=txt, score=sc, type="summary", timestamp=ts)
                    )
        except Exception:
            pass

        def _topk(items: List[FocusedItem], k: int = 3) -> List[Dict[str, Any]]:
            if not items:
                return []
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            return [
                {
                    "content": it.content,
                    "score": float(it.score),
                    "type": it.type,
                    "timestamp": it.timestamp,
                }
                for it in items_sorted[:k]
            ]

        # 키 순서 고정
        return {
            "stwm": _topk(stwm_items),
            "redis_recent": _topk(redis_recent),
            "redis_all": _topk(redis_all),
            "summaries": _topk(summaries_items),
        }


# (B9) Dead code 제거: HierarchicalContextManager는 참조되지 않으며 기능 혼선 방지를 위해 삭제.


class EvidenceContextIntegrator:
    """
    증거와 맥락을 충돌 없이 융합하기 위한 쿼리 강화만 책임진다.
    실제 검색 실행/응답 합성은 기존 파이프라인을 그대로 사용한다.
    """

    def enhance_queries(
        self, user_input: str, integrated_info: Dict[str, Any]
    ) -> Dict[str, str]:
        cf = integrated_info.get("confirmed_facts", {}) or {}

        web_parts: List[str] = [user_input]
        loc = cf.get("location")
        if isinstance(loc, dict) and float(loc.get("confidence", 0.0)) >= 0.8:
            val = (loc.get("value") or "").strip()
            if val and val not in user_input:
                web_parts.append(val)
        pref = cf.get("food_preference")
        if isinstance(pref, dict) and float(pref.get("confidence", 0.0)) >= 0.8:
            val = (pref.get("value") or "").strip()
            if val and val not in user_input:
                web_parts.append(val)

        rag_parts: List[str] = [user_input]
        for k, v in cf.items():
            try:
                if float(v.get("confidence", 0.0)) >= 0.6:
                    val = (v.get("value") or "").strip()
                    if val and val not in rag_parts:
                        rag_parts.append(val)
            except Exception:
                pass

        return {
            "web_query": " ".join(web_parts).strip(),
            "rag_query": " ".join(rag_parts).strip(),
        }


class AdaptiveResponseSynthesizer:
    """
    응답 합성은 기존 main_response를 사용하므로, 여기서는 포맷팅 유틸만 제공한다.
    """

    def format_confirmed_facts(self, facts: Dict[str, Any]) -> str:
        if not facts:
            return "확인된 사실 없음"
        out = []
        for k, info in facts.items():
            v = (info.get("value") or "").strip()
            c = float(info.get("confidence", 0.0))
            mark = "✅" if c >= 0.9 else "⚡"
            out.append(f"{k}: {v} {mark}")
        return "\n".join(out)


class UnifiedContextualChatbot:
    """4단계 파이프라인의 경량 래퍼(아이디어 차용, 구현은 현 코드 스타일)"""

    def __init__(self, redis_url: Optional[str] = None):
        self.query_processor = QueryFocusedProcessor(redis_url)
        # 미사용 경로 제거: HierarchicalContextManager 주입 중단
        self.evidence_integrator = EvidenceContextIntegrator()
        self.response_synthesizer = AdaptiveResponseSynthesizer()


# ---------------------------------------------
# 앱 통합을 위한 보조 유틸(최소 침습 주입)
# ---------------------------------------------


def apply_query_focused_overrides(
    session_state: dict,
    session_id: str,
    user_input: str,
    focused_context: Dict[str, List[Dict[str, Any]]],
    integrated_info: Dict[str, Any],
) -> Dict[str, str]:
    """
    RAG/WEB 쿼리 오버라이드 제안값을 세션 상태에 저장하고 반환.
    - 사실채널(rag/web)과 개인화(aux/conv) 분리를 유지하기 위해, 여기서는 쿼리 텍스트만 제안한다.
    """
    integrator = EvidenceContextIntegrator()
    qs = integrator.enhance_queries(user_input, integrated_info)
    st = session_state.setdefault(session_id, {})
    st["qf_ctx"] = focused_context  # 키 순서: stwm, redis_recent, redis_all, summaries
    st["qf_overrides"] = {
        "rag_query": qs.get("rag_query", ""),
        "web_query": qs.get("web_query", ""),
    }
    st["qf_facts"] = integrated_info.get("confirmed_facts", {})
    return st["qf_overrides"]


def update_route_ema(
    session_state: dict, session_id: str, p_rag: float, p_web: float, alpha: float = 0.6
) -> Tuple[float, float]:
    """
    라우팅 확률의 EMA를 업데이트하여 동적 임계값(게이트)을 제공.
    반환: (ema_rag, ema_web)
    """
    st = session_state.setdefault(session_id, {})
    ema_rag = float(st.get("ema_p_rag", 0.0))
    ema_web = float(st.get("ema_p_web", 0.0))
    ema_rag = alpha * p_rag + (1.0 - alpha) * ema_rag
    ema_web = alpha * p_web + (1.0 - alpha) * ema_web
    st["ema_p_rag"], st["ema_p_web"] = float(ema_rag), float(ema_web)
    return float(ema_rag), float(ema_web)


def cap_context_tokens(
    enc,
    rag_ctx: str,
    web_ctx: str,
    conv_ctx: str,
    aux_ctx: str,
    total_cap: int = 2800,
    rag_cap: int = 1200,
    web_cap: int = 1200,
    conv_cap: int = 1100,
    aux_cap: int = 500,
) -> Tuple[str, str, str, str]:
    """
    토큰 버짓 하드 캡을 적용하여 LLM 입력 오염/과적을 방지한다.
    - 증거 채널(rag/web)을 우선 보존, conv/aux는 톤 shaping용으로만 소폭 허용
    """

    def _trim_by_tokens(txt: str, cap: int) -> str:
        if not txt:
            return ""
        ids = enc.encode(txt)
        if len(ids) <= cap:
            return txt
        return enc.decode(ids[:cap])

    rag_trim = _trim_by_tokens(rag_ctx or "", rag_cap)
    web_trim = _trim_by_tokens(web_ctx or "", web_cap)
    conv_trim = _trim_by_tokens(conv_ctx or "", conv_cap)
    aux_trim = _trim_by_tokens(aux_ctx or "", aux_cap)

    tot = sum(len(enc.encode(x)) for x in [rag_trim, web_trim, conv_trim, aux_trim])
    if tot <= total_cap:
        return rag_trim, web_trim, conv_trim, aux_trim

    # 총량 초과 시 conv/aux를 먼저 축소
    remain = total_cap - sum(len(enc.encode(x)) for x in [rag_trim, web_trim])
    remain = max(0, remain)
    conv_trim = _trim_by_tokens(conv_trim, remain // 2)
    aux_trim = _trim_by_tokens(aux_trim, remain - len(enc.encode(conv_trim)))
    return rag_trim, web_trim, conv_trim, aux_trim
