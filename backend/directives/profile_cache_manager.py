"""
backend.directives.profile_cache_manager - 프로필 캐시 관리자

단일 턴 내 프로필 조회를 메모이제이션하여 중복 호출을 제거한다.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

_turn_cache = threading.local()


def _get_turn_cache() -> Dict[str, List[Dict]]:
    if not hasattr(_turn_cache, "data"):
        _turn_cache.data = {}
    return _turn_cache.data  # type: ignore[return-value]


def clear_turn_cache() -> None:
    if hasattr(_turn_cache, "data"):
        _turn_cache.data.clear()


async def get_profile_items_cached(
    user_id: str,
    tier: str,
    user_input: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict]:
    import hashlib

    from backend.rag.profile_rag import get_profile_rag
    from backend.rag.profile_schema import ProfileTier

    cache = _get_turn_cache()
    key_parts = [user_id, tier]
    if user_input and tier == "dynamic":
        key_parts.append(hashlib.md5(user_input.encode()).hexdigest()[:8])
    cache_key = ":".join(key_parts)

    if cache_key in cache:
        return cache[cache_key]

    rag = get_profile_rag()
    tier_enum = ProfileTier[tier.upper()]
    if tier == "dynamic":
        items = await rag.query_by_tier(
            user_id=user_id, tier=tier_enum, user_input=user_input, top_k=top_k
        )
    else:
        items = await rag.query_by_tier(user_id=user_id, tier=tier_enum)

    cache[cache_key] = items or []
    return cache[cache_key]
