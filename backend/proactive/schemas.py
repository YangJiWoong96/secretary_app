"""
backend.proactive.schemas - 프로액티브 파이프라인 스키마 정의

- TypedDict: LangGraph 상태 전파용 (가벼움/선호)
- Pydantic 모델: 외부 I/O 검증 및 직렬화 일관성 보장

세션 1 범위:
- ProactiveState (입/출력 상태 정의)
- FinalNotification (최종 알림 페이로드)
- ProactiveError (에러 스키마)
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ProactiveState(TypedDict, total=False):
    # ===== 입력 =====
    user_id: str
    session_id: str
    trigger_type: str  # "scheduled" | "event_driven" | "manual"
    timestamp: str  # ISO8601
    trace_id: str  # 파이프라인 추적 ID (세션4)

    # ===== Strategic Planner 출력 =====
    planner_analysis: Dict
    info_needs: List[str]
    urgency_scores: Dict[str, float]  # {"event_id": urgency_score}

    # ===== Context Analyzer 출력 =====
    internal_contexts: Dict  # {"rag": "...", "mobile": "...", "calendar": "...", "conversation": "..."}
    context_insights: List[Dict]  # [{"type": "health_alert", ...}]

    # ===== Web Research 출력 =====
    web_queries: List[str]
    web_results: List[
        Dict
    ]  # [{"url": "...", "title": "...", "trust_score": 0.85, ...}]
    source_credibility: Dict  # {"domain": trust_score}

    # ===== Synthesizer 출력 =====
    synthesized_insights: List[Dict]
    verified_facts: List[Dict]
    actionable_items: List[Dict]  # SMART 액션
    # 도메인별 구조화 팩트
    domain_facts: Dict[str, List[Dict]]

    # ===== Notification Generator 출력 =====
    notification_candidates: List[Dict]
    final_notification: Optional[Dict]

    # ===== 메타 =====
    confidence_score: float
    agent_latencies: Dict[str, int]  # ms


class FinalNotification(BaseModel):
    """최종 푸시 알림 페이로드.

    세션 4에서 실제 전송 직전까지 유지되는 구조체. 세션 1에서는 모델만 정의한다.
    """

    push_id: str = Field(..., description="UUID")
    user_id: str
    title: str = Field(..., max_length=40, description="≤ 40자 한글")
    body: str = Field(..., max_length=150, description="≤ 150자 한글, 2~4문장")
    urgency: int = Field(..., ge=0, le=10)
    category: str = Field(..., description='"건강" | "관계" | "재무" | "여가" | "업무"')
    tone: str = Field(
        default="formal", description='"formal" | "friendly" | "empathetic"'
    )
    emoji_level: str = Field(
        default="medium", description='"none" | "low" | "medium" | "high"'
    )
    contexts_meta: Dict = Field(default_factory=dict)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: str = Field(..., description="ISO8601")
    variant: str = Field(default="baseline_v1")
    # 설명가능 푸시 Why-Tag (옵션)
    why_tag: Optional[Dict[str, str]] = Field(
        default=None,
        description='{ "text": "근거 요약", "sensitivity": "low|medium|high" }',
    )


class ProactiveError(BaseModel):
    error_type: (
        str  # "timeout" | "confidence_low" | "context_empty" | "guard_violation"
    )
    agent: str
    message: str
    timestamp: str
    recoverable: bool = True


# =============================
# 세션 3: Web Research & Verification 스키마 추가
# =============================


class WebSearchResult(TypedDict):
    """웹 검색 결과 항목(신뢰도 포함).

    - trust_score: 0.0 ~ 1.0 (최소 0.7 이상만 채택 권장)
    - source_type: "news" | "community" | "official" | "blog"
    - cross_validated: 교차 검증 여부
    - evidence_ids: 교차 검증 시 참조한 다른 소스 식별자들
    """

    url: str
    title: str
    excerpt: str
    content: str
    trust_score: float
    domain: str
    published_at: Optional[str]
    source_type: str
    cross_validated: bool
    evidence_ids: List[str]


class ConsistencyReport(TypedDict):
    """내용 일관성 리포트(DBSCAN 기반)."""

    overall_score: float
    cluster_count: int
    noise_count: int
    largest_cluster_size: int


class VerifiedFact(TypedDict):
    """검증된 사실(교차 검증된 요약 단위)."""

    fact: str
    sources: List[str]
    trust_score: float
    evidence_count: int


# ===== 도메인별 팩트 스키마 =====


class FinanceFact(TypedDict):
    """재무 도메인 구조화 팩트."""

    symbol: str
    price: float
    change: float
    change_percent: str
    timestamp: str
    sources: List[str]
    trust_score: float


class WeatherFact(TypedDict):
    """날씨 도메인 구조화 팩트."""

    location: str
    lat: float
    lon: float
    temperature_c: float
    apparent_c: float
    precipitation_mm: float
    humidity_pct: float
    wind_ms: float
    time: str
    sources: List[str]
    trust_score: float


class TrafficFact(TypedDict):
    """교통 도메인 구조화 팩트."""

    origin: str
    destination: str
    distance_km: float
    duration_min: float
    sources: List[str]
    trust_score: float


class SMARTAction(TypedDict):
    """SMART 원칙 기반 실행 항목."""

    action: str
    rationale: str
    urgency: int  # 0~10
    time_bound: str
    measurable: str


class SynthesizerOutput(TypedDict):
    """합성/검증 에이전트 출력."""

    verified_facts: List[VerifiedFact]
    actionable_items: List[SMARTAction]
    confidence_score: float
    contradictions_found: int


# =============================
# 세션 4: Notification Generator 보조 스키마
# =============================


class NotificationCandidate(TypedDict):
    """노티 후보(검증 전 중간 결과)."""

    title: str
    body: str
    urgency: int
    category: str
    tone: str
    emoji_level: str
    confidence: float


__all__ = [
    "ProactiveState",
    "FinalNotification",
    "ProactiveError",
    # S3 추가 내보내기
    "WebSearchResult",
    "ConsistencyReport",
    "VerifiedFact",
    "FinanceFact",
    "WeatherFact",
    "TrafficFact",
    "SMARTAction",
    "SynthesizerOutput",
    # S4 추가 내보내기
    "NotificationCandidate",
]
