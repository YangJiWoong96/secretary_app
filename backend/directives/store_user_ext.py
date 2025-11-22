from __future__ import annotations

import json
from typing import Dict

from backend.directives.store import _r, TTL_SEC  # 재사용


def _kp_user(user_id: str) -> str:
    return f"dirpersona:user:{user_id}"


def load_persona_user(user_id: str) -> Dict:
    """
    사용자 범위(persona) 로드.
    - 세션 기반이 아닌 사용자 단위 Big Five/MBTI 누적값 저장소
    """
    if not user_id:
        return {}
    try:
        j = _r.hget(_kp_user(user_id), "json")
        return json.loads(j) if j else {}
    except Exception:
        return {}


def save_persona_user(user_id: str, persona: Dict) -> None:
    """
    사용자 범위(persona) 저장.
    """
    if not user_id:
        return
    try:
        key = _kp_user(user_id)
        _r.hset(key, mapping={"json": json.dumps(persona, ensure_ascii=False)})
        _r.expire(key, TTL_SEC)
    except Exception:
        return


# ----- Directives (User-scope) -----


def _kd_user(user_id: str) -> str:
    return f"directives:user:{user_id}"


def _kdm_user(user_id: str) -> str:
    return f"directives_meta:user:{user_id}"


def load_directives_user(user_id: str) -> tuple[Dict, Dict]:
    """
    사용자 범위(user_id) 지시문과 메타 로드.
    - 새 세션에서도 동일 사용자 룰북을 이어서 사용할 수 있게 한다.
    """
    if not user_id:
        return {}, {}
    try:
        j = _r.hget(_kd_user(user_id), "json")
        m = _r.hget(_kdm_user(user_id), "json")
        dirs = (json.loads(j) if j else {}) or {}
        meta = (json.loads(m) if m else {}) or {}
        return dirs, meta
    except Exception:
        return {}, {}


def save_directives_user(user_id: str, directives: Dict, meta: Dict) -> None:
    """사용자 범위(user_id) 지시문과 메타 저장."""
    if not user_id:
        return
    try:
        _r.hset(
            _kd_user(user_id),
            mapping={"json": json.dumps(directives, ensure_ascii=False)},
        )
        _r.expire(_kd_user(user_id), TTL_SEC)
        _r.hset(
            _kdm_user(user_id), mapping={"json": json.dumps(meta, ensure_ascii=False)}
        )
        _r.expire(_kdm_user(user_id), TTL_SEC)
    except Exception:
        return
