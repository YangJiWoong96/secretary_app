from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, TypedDict


class ContextAnalyzerInput(TypedDict):
    """
    Context Analyzer 입력 스키마
    - 세션2에서는 Strategic Planner의 info_needs 없이도 동작 가능하도록 최소 필수 필드만 명시한다.
    """

    user_id: str
    session_id: str
    timestamp: str  # ISO8601


class ContextBundle(TypedDict):
    """다층 내부 컨텍스트 묶음"""

    rag_ctx: str
    mobile_ctx: str
    conversation_ctx: str
    memory_ctx: str


class InsightSignal(TypedDict):
    """탐지된 인사이트 시그널 구조"""

    type: str  # "health_alert" | "schedule_conflict" | "sentiment_pattern"
    pattern: str
    severity: str  # "low" | "medium" | "high"
    confidence: float
    evidence: List[str]
    timestamp: str


class CausalLink(TypedDict):
    """인과관계 링크 구조"""

    cause: str
    effect: str
    implication: str
    confidence: float


class ContextAnalyzerOutput(TypedDict):
    """Context Analyzer 출력 스키마"""

    internal_contexts: ContextBundle
    context_insights: List[InsightSignal]
    causal_links: List[CausalLink]
    collection_latencies: Dict[str, int]


def hash_user_id(user_id: str) -> str:
    """
    사용자 식별자 최소화: 로그에서 원문 대신 해시(앞 16자)만 사용.
    """

    try:
        return hashlib.sha256((user_id or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def get_latest_mtm_summary(user_id: str, session_id: str) -> str:
    """
    최신 MTM 요약 텍스트 반환. 실패 시 빈 문자열.
    - 읽기 전용 접근만 수행한다.
    """

    try:
        from backend.memory.coordinator import get_memory_coordinator

        coord = get_memory_coordinator()
        latest = coord.mtm.get_latest(user_id, session_id)
        if latest:
            # 우선순위: mtm_summary > summary > routing_summary
            return (
                latest.get("mtm_summary")
                or latest.get("summary")
                or latest.get("routing_summary")
                or ""
            ).strip()
        return ""
    except Exception:
        return ""


def get_conversation_texts(session_id: str, max_messages: int = 200) -> List[str]:
    """
    최근 대화 원문 텍스트 목록을 반환한다(최대 max_messages).
    - Redis 기반 단기 메모리에서 읽기 전용으로 조회.
    - 메시지 순서는 오래된 → 최신 순서로 정규화한다.
    """

    try:
        from backend.memory.redis_memory import get_short_term_memory

        stm = get_short_term_memory(session_id)
        msgs = getattr(stm, "chat_memory", {}).messages or []
        # 오래된 → 최신 순
        texts: List[str] = [
            str(getattr(m, "content", "") or "") for m in msgs[-max_messages:]
        ]
        return texts
    except Exception:
        return []
