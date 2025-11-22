"""
backend.directives.invalidator - Directives 캐시 무효화 및 버전 관리 유틸

기능:
- Dirty Bit 설정/해제: `dir:dirty:{user_id}`
- 티어별 버전 카운터 증분(원자적): `dir:ver:{user_id}:{tier}`

설계 원칙:
- Dirty Bit은 통합 컴파일 캐시(베이스/오버레이/압축)의 미스를 강제하여 재컴파일을 유도한다.
- 티어(Core/Guard) 버전은 감사/관측을 위한 보조 수단이며, 캐시 키 직접 구성에는 사용하지 않는다.
  (실제 base_version은 내용 기반 해시(version_of)로 계산되며, 이 유틸은 변경 전파를 보장하기 위해 Dirty Bit을 함께 세팅한다.)
"""

from __future__ import annotations

import os
from typing import Dict


def _get_client():
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
        )
    except Exception:  # pragma: no cover - 환경별 선택적 의존성
        return None


def mark_dirty(
    user_id: str, reason: str | None = None, ttl_sec: int | None = None
) -> None:
    """
    Dirty Bit 설정.

    - 키: dir:dirty:{user_id}
    - 값: "1"
    - TTL: 선택(미지정 시 영속) — 재컴파일 시 store 측에서 Dirty Bit을 삭제한다.
    - reason은 최근 사유를 메타 키에 기록(디버깅/관측용, 비필수)
    """
    if not user_id:
        return
    r = _get_client()
    if r is None:
        return
    try:
        key = f"dir:dirty:{user_id}"
        r.set(key, "1")
        if ttl_sec and ttl_sec > 0:
            r.expire(key, int(ttl_sec))
        if reason:
            r.hset(f"dir:dirty:meta:{user_id}", mapping={"reason": str(reason)})
    except Exception:
        # 조용히 폴백(무효화 실패 시에도 기능은 계속 진행)
        pass


def clear_dirty(user_id: str) -> None:
    """Dirty Bit 제거."""
    if not user_id:
        return
    r = _get_client()
    if r is None:
        return
    try:
        r.delete(f"dir:dirty:{user_id}")
    except Exception:
        pass


def update_tier_version_atomic(user_id: str, tier: str) -> int:
    """
    티어별 버전 카운터를 원자적으로 증가시키고 Dirty Bit을 설정한다.

    Args:
        user_id: 사용자 식별자(현 구조에서는 session_id와 동일하거나 매핑됨)
        tier: "guard" | "core" | "dynamic" (동적은 버전 의미가 옅으나 사유 기록용 허용)

    Returns:
        int: 증가된 현재 버전(실패 시 0)
    """
    if not user_id or not tier:
        return 0
    r = _get_client()
    if r is None:
        return 0
    try:
        key = f"dir:ver:{user_id}:{tier.strip().lower()}"
        ver = int(r.incr(key))
        # 관측 편의를 위해 30일 TTL 부여(선택)
        try:
            r.expire(key, 30 * 24 * 3600)
        except Exception:
            pass
        # 변경 전파를 위해 Dirty Bit 함께 설정
        mark_dirty(user_id, reason=f"tier_updated:{tier}")
        return ver
    except Exception:
        return 0


def get_tier_versions(user_id: str) -> Dict[str, int]:
    """사용자 티어별 버전 조회(디버깅/관측용). 실패 시 빈 딕셔너리 반환."""
    if not user_id:
        return {}
    r = _get_client()
    if r is None:
        return {}
    out: Dict[str, int] = {}
    try:
        for tier in ("guard", "core", "dynamic"):
            try:
                v = int(r.get(f"dir:ver:{user_id}:{tier}") or 0)
                if v:
                    out[tier] = v
            except Exception:
                continue
    except Exception:
        return {}
    return out


import hashlib
import json
import logging
import os
import time

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_r = (
    redis.Redis.from_url(REDIS_URL, decode_responses=True)
    if redis is not None
    else None
)


def _safe_redis_call(fn, default=None):
    try:
        if _r is None:
            return default
        return fn()
    except Exception:
        return default


def mark_dirty(user_id: str, reason: str = "unknown") -> None:
    """
    Directives 캐시 무효화 플래그 설정 (Dirty Bit)

    Args:
        user_id: 사용자 ID (현 구조에서는 session_id와 동일하게 사용)
        reason: 무효화 이유 (로깅용)
    """
    if not user_id:
        return
    dirty_key = f"dir:dirty:{user_id}"
    _safe_redis_call(lambda: _r.set(dirty_key, "1", ex=3600))
    # Pub/Sub 알림
    try:
        _r.publish(
            "dir.invalidate",
            json.dumps(
                {"user": user_id, "reason": reason, "tier": "", "ts": int(time.time())},
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass

    # 로깅 (비차단)
    try:
        logger = logging.getLogger("invalidator")
        logger.info(f"[invalidator] Marked dirty: user_id={user_id} reason={reason}")
    except Exception:
        pass


def clear_dirty(user_id: str) -> None:
    """Dirty Bit 초기화"""
    if not user_id:
        return
    dirty_key = f"dir:dirty:{user_id}"
    _safe_redis_call(lambda: _r.delete(dirty_key))


def update_tier_version(user_id: str, tier: str) -> str:
    """
    계층별 버전 해시 갱신 + Dirty Bit 설정

    Args:
        user_id: 사용자 ID
        tier: "guard" | "core"

    Returns:
        str: 새 버전 해시
    """
    if not user_id or not tier:
        return ""
    new_version = hashlib.sha256(
        f"{user_id}:{tier}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    version_key = f"prof:{tier}_ver:{user_id}"
    _safe_redis_call(lambda: _r.set(version_key, new_version, ex=7 * 24 * 3600))
    mark_dirty(user_id, reason=f"{tier}_version_updated")
    # Pub/Sub 알림(버전 변경)
    try:
        _r.publish(
            "dir.invalidate",
            json.dumps(
                {
                    "user": user_id,
                    "reason": "tier_version",
                    "tier": tier,
                    "ts": int(time.time()),
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass
    return new_version


# 원자적 업데이트(Lua)
LUA_UPDATE = """
local ver_key = KEYS[1]; local dirty_key = KEYS[2]
redis.call('SET', ver_key, ARGV[1])
redis.call('SET', dirty_key, '1', 'EX', ARGV[2])
return 1
"""


def update_tier_version_atomic(user_id: str, tier: str, dirty_ttl: int = 3600) -> str:
    """
    Lua 스크립트를 통해 버전 갱신과 Dirty Bit 설정을 원자적으로 수행한다.
    실패 시 비원자 `update_tier_version`로 폴백한다.
    """
    if not user_id or not tier:
        return ""
    new_ver = hashlib.sha256(
        f"{user_id}:{tier}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    ver_key = f"prof:{tier}_ver:{user_id}"
    dirty_key = f"dir:dirty:{user_id}"

    def _eval():
        return _r.eval(LUA_UPDATE, 2, ver_key, dirty_key, new_ver, int(dirty_ttl))

    ok = _safe_redis_call(_eval, default=None)
    if ok is None:
        # 폴백
        return update_tier_version(user_id, tier)
    # Pub/Sub 알림(버전 변경)
    try:
        _r.publish(
            "dir.invalidate",
            json.dumps(
                {
                    "user": user_id,
                    "reason": "tier_version",
                    "tier": tier,
                    "ts": int(time.time()),
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass
    return new_ver
