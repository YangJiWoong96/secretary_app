"""
backend.routing.router_context - 세션→사용자 매핑 컨텍스트

목적:
- 레거시/신규 WebSocket 경로를 모두 지원하면서, RAG/프로필 스코프는 사용자 단위로 고정.
- 멀티프로세스 환경에서도 견고하게 동작하도록 Redis를 1차 저장소로 사용(있으면),
  프로세스 내 단기 캐시도 병행.
"""

from __future__ import annotations

import os
from typing import Optional

_LOCAL_MAP: dict[str, str] = {}


def _redis_client():
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
        )
    except Exception:  # pragma: no cover
        return None


def map_session_to_user(session_id: str, user_id: str, ttl_sec: int = 86400) -> None:
    """세션→사용자 매핑 저장(로컬 캐시 + Redis).

    Args:
        session_id: 세션 ID
        user_id: 사용자 ID
        ttl_sec: Redis TTL(초)
    """
    if not session_id or not user_id:
        return
    _LOCAL_MAP[session_id] = user_id
    r = _redis_client()
    if r is not None:
        try:
            r.setex(f"router:session_user:{session_id}", int(ttl_sec), user_id)
        except Exception:
            pass


def user_for_session(session_id: str) -> Optional[str]:
    """세션에 대응하는 사용자 ID 조회. 로컬→Redis 순으로 조회한다."""
    if not session_id:
        return None
    # L1: local
    uid = _LOCAL_MAP.get(session_id)
    if uid:
        return uid
    # L2: redis
    r = _redis_client()
    if r is not None:
        try:
            v = r.get(f"router:session_user:{session_id}")
            if v:
                _LOCAL_MAP[session_id] = v
                return v
        except Exception:
            return None
    return None


def clear_session(session_id: str) -> None:
    if not session_id:
        return
    try:
        _LOCAL_MAP.pop(session_id, None)
    except Exception:
        pass
    r = _redis_client()
    if r is not None:
        try:
            r.delete(f"router:session_user:{session_id}")
        except Exception:
            pass
