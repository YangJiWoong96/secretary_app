"""
backend.policy.profile_utils - 프로필 유틸리티

사용자 프로필 관련 헬퍼 함수들을 제공합니다.
"""

from typing import Dict, List


def pinned_facts_of(session_id: str, profile_db: Dict[str, Dict]) -> List[str]:
    """
    세션의 고정된 사실(pinned facts) 추출

    프로필에서 interests, preferences, name, location, occupation, constraints 등
    핵심 정보를 추출하여 요약 시 보호할 사실 목록을 생성합니다.

    Args:
        session_id: 세션 ID
        profile_db: 프로필 데이터베이스 딕셔너리

    Returns:
        List[str]: 고정 사실 리스트 (최대 50개)

    Example:
        >>> profile = {
        ...     "name": "홍길동",
        ...     "location": "서울",
        ...     "interests": ["개발", "음악"],
        ...     "preferences": {"음식": "한식"},
        ... }
        >>> facts = pinned_facts_of("user123", {"user123": profile})
        >>> print(facts)
        >>> # ['개발', '음악', '음식:한식', 'name:홍길동', 'location:서울']
    """
    pf = profile_db.get(session_id, {})
    out = []

    # interests: 리스트
    try:
        ints = pf.get("interests") or []
        if isinstance(ints, list):
            out.extend([str(x) for x in ints if x])
    except Exception:
        pass

    # preferences: dict
    try:
        prefs = pf.get("preferences") or {}
        if isinstance(prefs, dict):
            out.extend([f"{k}:{v}" for k, v in prefs.items() if v])
    except Exception:
        pass

    # name/location/occupation
    for k in ("name", "location", "occupation"):
        v = str(pf.get(k) or "").strip()
        if v:
            out.append(f"{k}:{v}")

    # constraints는 키만 반영
    try:
        cons = pf.get("constraints") or {}
        if isinstance(cons, dict):
            out.extend([f"constraint:{k}" for k in cons.keys()])
    except Exception:
        pass

    return out[:50]


# ===== 호환성을 위한 함수 =====


def get_pinned_facts(session_id: str) -> List[str]:
    """
    고정 사실 가져오기 (호환성 래퍼)

    GlobalState의 profile_db를 사용하여 고정 사실을 추출합니다.

    Args:
        session_id: 세션 ID

    Returns:
        List[str]: 고정 사실 리스트
    """
    from backend.policy.state import get_global_state

    state = get_global_state()
    return pinned_facts_of(session_id, state.profile_db)
