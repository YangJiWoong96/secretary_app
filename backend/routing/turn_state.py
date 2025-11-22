"""
backend.routing.turn_state - 턴 단일 출처 상태(SSOT)

라우팅 의사결정과 정책·제약을 불변 구조로 캡슐화하여
이후 단계(rewrite/retrieve/generate/validate)가 읽기 전용으로 일관되게 따르도록 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RouteLabel = Literal["conv", "web", "rag"]
CtxKind = Literal["conv_only", "web_ctx", "rag_ctx"]


def _allowed_ctx_by_route(route: RouteLabel) -> List[CtxKind]:
    if route == "web":
        return ["conv_only", "web_ctx"]
    if route == "rag":
        return ["conv_only", "rag_ctx"]
    return ["conv_only"]


@dataclass(frozen=True)
class TurnState:
    """
    턴 상태(불변).

    - route: 최종 라우트(conv/web/rag)
    - ambiguous: 모호성 플래그(스코어 마진/τ 기준)
    - allow_web: 정책적으로 웹 사용 허용 여부(리라이터 결과에 따라 보수적으로 False 가능)
    - protected_domain: 보호 도메인(금융/개인정보 등) 강제 비활성화 여부
    - penalties: 라우팅 보정 페널티 기록(진단용)
    - lambda_ema: m_ctx 혼합 가중치(採用 시 기록)
    - reason: 강제 전환/폴백 사유
    - max_sim/margin: 라우팅 스코어 진단값
    - need_rag/need_web: 하위 파이프라인 활성 플래그(편의)
    - session_id/turn_id: 키 스코프
    - allowed_ctx_mask: 서버가 허용하는 컨텍스트 종류 마스크
    """

    session_id: str
    turn_id: str
    route: RouteLabel
    ambiguous: bool = False
    allow_web: bool = True
    protected_domain: bool = False
    penalties: Dict[str, float] = field(default_factory=dict)
    lambda_ema: float = 0.0
    reason: str = ""
    max_sim: float = 0.0
    margin: float = 0.0
    need_rag: bool = False
    need_web: bool = False
    allowed_ctx_mask: List[CtxKind] = field(default_factory=list)

    @staticmethod
    def build(
        *,
        session_id: str,
        turn_id: str,
        route: RouteLabel,
        ambiguous: bool,
        max_sim: float,
        margin: float,
        need_rag: bool,
        need_web: bool,
        allow_web: bool = True,
        protected_domain: bool = False,
        penalties: Optional[Dict[str, float]] = None,
        lambda_ema: float = 0.0,
        reason: str = "",
    ) -> "TurnState":
        mask = _allowed_ctx_by_route(route)
        return TurnState(
            session_id=session_id,
            turn_id=turn_id,
            route=route,
            ambiguous=ambiguous,
            allow_web=allow_web,
            protected_domain=protected_domain,
            penalties=dict(penalties or {}),
            lambda_ema=lambda_ema,
            reason=reason,
            max_sim=max_sim,
            margin=margin,
            need_rag=need_rag,
            need_web=need_web,
            allowed_ctx_mask=mask,
        )


# ─────────────────────────────────────────────────────────────
# Redis 기반 세션별 턴 상태 저장/조회 (Evidence 재사용용)
# ─────────────────────────────────────────────────────────────


def _get_redis():
    """Redis 연결 인스턴스 반환"""
    try:
        import redis
        from backend.config import get_settings

        settings = get_settings()
        return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception:
        return None


def get_turn_state(session_id: str) -> Optional[Dict[str, Any]]:
    """
    세션별 턴 상태 조회 (Redis 기반)

    Args:
        session_id: 세션 ID

    Returns:
        Dict: 턴 상태 딕셔너리 (없으면 None)

    스키마:
        {
            "session_id": str,
            "active_eids": List[str],  # 이전 턴에서 생성된 증거 ID 리스트
            "last_route": str,
            "timestamp": int,
            ...
        }
    """
    try:
        import json

        r = _get_redis()
        if not r:
            return None

        data = r.get(f"turn_state:{session_id}")
        if not data:
            return None

        return json.loads(data)
    except Exception:
        return None


def set_turn_state(session_id: str, state: Dict[str, Any], ttl_sec: int = 3600) -> None:
    """
    세션별 턴 상태 저장 (Redis 기반)

    Args:
        session_id: 세션 ID
        state: 턴 상태 딕셔너리
        ttl_sec: TTL (기본 1시간)

    동작:
        - Redis에 JSON으로 직렬화하여 저장
        - 실패 시 조용히 실패 (로깅 없음)
    """
    try:
        import json

        r = _get_redis()
        if not r:
            return

        data = json.dumps(state, ensure_ascii=False)
        r.setex(f"turn_state:{session_id}", ttl_sec, data)
    except Exception:
        pass
