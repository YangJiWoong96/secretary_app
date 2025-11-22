from typing import Any, Dict, List, Tuple


class ProfileGuard:
    """
    Guard 계층 검증기

    역할:
    - Guard 위반 여부 검증 (금지 주제 등)
    - 위반 시 사유 반환(차단/대체문 로직에서 사용)
    """

    def __init__(self):
        from backend.rag.profile_rag import get_profile_rag

        self.profile_rag = get_profile_rag()
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    async def load_guard(self, user_id: str) -> List[Dict[str, Any]]:
        """사용자 Guard 항목을 로드(프로세스 메모리 캐시 우선)."""
        if user_id in self._cache:
            return self._cache[user_id]
        from backend.rag.profile_schema import ProfileTier

        items = await self.profile_rag.query_by_tier(
            user_id=user_id, tier=ProfileTier.GUARD
        )
        self._cache[user_id] = items
        return items

    async def validate(
        self, user_id: str, user_input: str, ai_output: str = ""
    ) -> Tuple[bool, str]:
        """
        Guard 위반 검증

        Returns:
            (is_valid, reason)
        """
        items = await self.load_guard(user_id)

        taboo_topics: list[str] = []
        for it in items or []:
            kp = str(it.get("key_path") or "")
            val = it.get("value")
            if "taboo_topics" in kp:
                try:
                    if isinstance(val, str):
                        import json as _json

                        parsed = _json.loads(val)
                    else:
                        parsed = val
                    if isinstance(parsed, list):
                        taboo_topics.extend([str(x) for x in parsed])
                except Exception:
                    continue

        combined = f"{user_input}\n{ai_output}".lower()
        for topic in taboo_topics:
            try:
                if str(topic).lower() in combined:
                    return False, f"Guard violation: taboo topic '{topic}' detected"
            except Exception:
                continue

        return True, "OK"


_GUARD_INSTANCE: ProfileGuard | None = None


def get_profile_guard() -> ProfileGuard:
    global _GUARD_INSTANCE
    if _GUARD_INSTANCE is None:
        _GUARD_INSTANCE = ProfileGuard()
    return _GUARD_INSTANCE
