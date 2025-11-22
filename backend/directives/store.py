# # Redis(캐시) + (옵션) Milvus 스냅샷
import hashlib
import json
import os
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple

import redis

from backend.config import get_settings
from backend.utils.logger import log_event

from .schema import Directives

_s = get_settings()
REDIS_URL = _s.REDIS_URL
TTL_SEC = int(getattr(_s, "DIR_TTL_SEC", 604800))  # 7d


# lazy-load Redis client to avoid heavy init at import time
class _RedisLazy:
    """Redis 클라이언트 지연 초기화 래퍼.

    - 첫 속성 접근 시점에만 실제 Redis 연결을 생성한다.
    - 생성 실패 시 AttributeError를 발생시켜 호출부의 기존 try/except에 포착되도록 한다.
    """

    def __init__(self, url: str):
        self._url = url
        self._client = None  # type: ignore[var-annotated]

    def _ensure(self):
        if self._client is None:
            try:
                import redis as _redis  # type: ignore

                self._client = _redis.Redis.from_url(self._url, decode_responses=True)
            except Exception:
                self._client = None
        return self._client

    def __getattr__(self, name: str):
        client = self._ensure()
        if client is None:
            raise AttributeError("Redis client not available")
        return getattr(client, name)


_r = _RedisLazy(REDIS_URL)


def _bg_submit(op: str, fn, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    백그라운드 작업 등록 헬퍼.
    - 이벤트 루프가 있으면 asyncio.create_task + asyncio.to_thread 사용
    - 이벤트 루프가 없으면 daemon Thread로 비차단 실행
    - 시작/성공/실패를 log_event로 기록
    """
    meta = meta or {}

    def _runner():
        try:
            fn()
            try:
                log_event("dir.bg.ok", {"op": "store." + op, **meta})
            except Exception:
                pass
        except Exception as e:
            try:
                log_event("dir.bg.err", {"op": "store." + op, "error": str(e), **meta})
            except Exception:
                pass

    try:
        try:
            log_event("dir.bg.schedule", {"op": "store." + op, **meta})
        except Exception:
            pass
        loop = asyncio.get_running_loop()

        async def _async_runner():
            await asyncio.to_thread(_runner)

        loop.create_task(_async_runner())
    except RuntimeError:
        t = threading.Thread(target=_runner, daemon=True)
        t.start()


def _k(session_id: str) -> str:
    return f"dir:{session_id}"


def _km(session_id: str) -> str:
    return f"dirmeta:{session_id}"


def _kc(session_id: str) -> str:
    return f"dircompiled:{session_id}"  # system용 캐시


def _ks(session_id: str) -> str:
    return f"dirsig:{session_id}"  # signals


def _kp(session_id: str) -> str:
    return f"dirpersona:{session_id}"  # persona


ACTIVE_SET = "dir:active_users"


def _is_dirty(user_id: str) -> bool:
    """Dirty Bit 확인: 1이면 재생성 필요 → 캐시 미스 처리"""
    try:
        return bool(user_id) and (_r.get(f"dir:dirty:{user_id}") == "1")
    except Exception:
        return False


def load_directives(session_id: str) -> Tuple[Directives, Dict]:
    j = _r.hget(_k(session_id), "json")
    m = _r.hget(_km(session_id), "json")
    dirs = json.loads(j) if j else {}
    meta = json.loads(m) if m else {"last_changed": {}}
    return dirs, meta


def save_directives(session_id: str, directives: Directives, meta: Dict, reasons=None):
    _r.hset(
        _k(session_id), mapping={"json": json.dumps(directives, ensure_ascii=False)}
    )
    _r.expire(_k(session_id), TTL_SEC)
    _r.hset(_km(session_id), mapping={"json": json.dumps(meta, ensure_ascii=False)})
    _r.expire(_km(session_id), TTL_SEC)
    # 감사로그(선택): 최근 이유 보관
    if reasons:

        def _bg_reasons() -> None:
            _r.hset(
                _k(session_id),
                mapping={"reasons": json.dumps(reasons, ensure_ascii=False)},
            )

        _bg_submit("save_directives.reasons", _bg_reasons, {"session_id": session_id})

    def _bg_active() -> None:
        _r.sadd(ACTIVE_SET, session_id)

    _bg_submit("active_set.sadd", _bg_active, {"session_id": session_id})
    # Directives 변경 시 base 캐시 무효화(세션 ID를 user_id로 사용)
    _bg_submit(
        "invalidate_unified_base",
        lambda: invalidate_unified_base(session_id),
        {"user_id": session_id},
    )


def set_compiled(session_id: str, prompt: str, version: str):
    def _bg() -> None:
        _r.hset(
            _kc(session_id),
            mapping={
                "prompt": (prompt or ""),
                "version": (version or ""),
                "ts": str(int(time.time())),
            },
        )
        _r.expire(_kc(session_id), TTL_SEC)
        # Dirty Bit 초기화
        try:
            _r.delete(f"dir:dirty:{session_id}")
        except Exception:
            pass

    _bg_submit(
        "set_compiled",
        _bg,
        {"session_id": session_id, "prompt_len": len(prompt or ""), "version": version},
    )


def get_compiled(session_id: str) -> Tuple[str, str]:
    # Dirty Bit가 설정되면 미스 처리
    if _is_dirty(session_id):
        return "", ""
    h = _r.hgetall(_kc(session_id)) or {}
    return h.get("prompt", ""), h.get("version", "")


def _canon(obj: Any) -> str:
    """결정론 직렬화: 키 정렬 + 압축 구분자 + NaN 불가.

    float는 소수점 6자리로 고정하여 근소 차이로 인한 해시 변동을 방지한다.
    """

    def _norm(x):
        if isinstance(x, float):
            return float(f"{x:.6f}")
        if isinstance(x, dict):
            return {k: _norm(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_norm(v) for v in x]
        return x

    body = json.dumps(
        _norm(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return body


def version_of(obj: Any) -> str:
    """결정론 버전 해시(sha256).

    동일 입력 → 동일 해시, 미세 변경도 즉시 키 분기.
    """
    body = _canon(obj)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ----- signals/persona 저장/불러오기 -----


def load_signals(session_id: str) -> Dict:
    j = _r.hget(_ks(session_id), "json")
    return json.loads(j) if j else {}


def save_signals(session_id: str, signals: Dict):
    def _bg() -> None:
        key = _ks(session_id)
        _r.hset(key, mapping={"json": json.dumps(signals, ensure_ascii=False)})
        _r.expire(key, TTL_SEC)
        try:
            _r.sadd(ACTIVE_SET, session_id)
        except Exception:
            pass

    _bg_submit("save_signals", _bg, {"session_id": session_id})


def load_persona(session_id: str) -> Dict:
    j = _r.hget(_kp(session_id), "json")
    return json.loads(j) if j else {}


def save_persona(session_id: str, persona: Dict):
    def _bg() -> None:
        key = _kp(session_id)
        _r.hset(key, mapping={"json": json.dumps(persona, ensure_ascii=False)})
        _r.expire(key, TTL_SEC)
        try:
            _r.sadd(ACTIVE_SET, session_id)
        except Exception:
            pass

    _bg_submit("save_persona", _bg, {"session_id": session_id})


def load_all(session_id: str) -> Tuple[Dict, Dict, Dict, Dict]:
    dirs, meta = load_directives(session_id)
    sig = load_signals(session_id)
    per = load_persona(session_id)
    return dirs, sig, per, meta


def get_active_users() -> List[str]:
    try:
        return list(_r.smembers(ACTIVE_SET))
    except Exception:
        return []


# ----- 통합 컴파일 캐시 (쿼리 해시 기반, TTL 24h) -----


def _kc_unified(session_id: str, query_hash: str) -> str:
    """
    통합 컴파일 캐시 키 생성

    Args:
        session_id: 세션 ID
        query_hash: 쿼리 해시(앞 8자 등 짧은 버전)
    """
    return f"dir:unified:{session_id}:{query_hash}"


def get_compiled_unified(session_id: str, query: str) -> Tuple[str, str]:
    """
    통합 컴파일 결과 로드(캐시)

    Returns:
        (prompt, version)
    """
    # Dirty Bit가 설정되면 미스 처리
    if _is_dirty(session_id):
        return "", ""
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    ck = _kc_unified(session_id, qh)
    h = _r.hgetall(ck) or {}
    return h.get("prompt", ""), h.get("version", "")


def set_compiled_unified(
    session_id: str,
    query: str,
    prompt: str,
    version: str,
    ttl_sec: int = 86400,
) -> None:
    """
    통합 컴파일 결과 저장(캐시)

    Args:
        session_id: 세션 ID
        query: 현재 쿼리(해시 계산용)
        prompt: 컴파일된 시스템 프롬프트
        version: 버전 해시
        ttl_sec: TTL(기본 24시간)
    """
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    ck = _kc_unified(session_id, qh)

    def _bg() -> None:
        _r.hset(
            ck,
            mapping={
                "prompt": prompt or "",
                "version": version or "",
                "ts": str(int(time.time())),
            },
        )
        _r.expire(ck, int(ttl_sec))
        # Dirty Bit 초기화 (현 구조에서 session_id == user_id 가정)
        try:
            _r.delete(f"dir:dirty:{session_id}")
        except Exception:
            pass

    _bg_submit(
        "set_compiled_unified",
        _bg,
        {
            "session_id": session_id,
            "qh": qh,
            "prompt_len": len(prompt or ""),
            "version": version,
        },
    )


# ----- 2계층 캐시 (base: 버전 키, overlay: 쿼리 해시 키) -----


def _kc_unified_base(user_id: str, version: str) -> str:
    return f"dir:unified:base:{user_id}:{version}"


def _kc_unified_overlay(session_id: str, query_hash8: str) -> str:
    return f"dir:unified:overlay:{session_id}:{query_hash8}"


def get_unified_base(user_id: str, version: str) -> str:
    """base 프롬프트 로드 (없으면 빈 문자열)."""
    if not user_id or not version:
        return ""
    h = _r.hgetall(_kc_unified_base(user_id, version)) or {}
    return h.get("prompt", "")


def set_unified_base(
    user_id: str, version: str, prompt: str, ttl_sec: int = 86400
) -> None:
    if not user_id or not version:
        return
    ck = _kc_unified_base(user_id, version)

    def _bg() -> None:
        _r.hset(
            ck,
            mapping={
                "prompt": prompt or "",
                "ts": str(int(time.time())),
            },
        )
        _r.expire(ck, int(ttl_sec))

    _bg_submit(
        "set_unified_base",
        _bg,
        {"user_id": user_id, "version": version, "prompt_len": len(prompt or "")},
    )


def get_unified_overlay(session_id: str, query: str) -> tuple[str, str]:
    """
    overlay 프롬프트와 base_version 로드.
    Returns: (overlay, base_version)
    """
    if not session_id:
        return "", ""
    # Dirty Bit가 설정되면 미스 처리
    if _is_dirty(session_id):
        return "", ""
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    ck = _kc_unified_overlay(session_id, qh)
    h = _r.hgetall(ck) or {}
    return h.get("overlay", ""), h.get("base_version", "")


def set_unified_overlay(
    session_id: str,
    query: str,
    overlay: str,
    base_version: str,
    ttl_sec: int = 7200,
) -> None:
    if not session_id:
        return
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    ck = _kc_unified_overlay(session_id, qh)

    def _bg() -> None:
        _r.hset(
            ck,
            mapping={
                "overlay": overlay or "",
                "base_version": base_version or "",
                "ts": str(int(time.time())),
            },
        )
        _r.expire(ck, int(ttl_sec))
        # Dirty Bit 초기화 (현 구조에서 session_id == user_id 가정)
        try:
            _r.delete(f"dir:dirty:{session_id}")
        except Exception:
            pass

    _bg_submit(
        "set_unified_overlay",
        _bg,
        {
            "session_id": session_id,
            "qh": qh,
            "overlay_len": len(overlay or ""),
            "base_version": base_version,
        },
    )


def invalidate_unified_base(user_id: str) -> None:
    """해당 사용자 base 캐시 전체 무효화."""

    def _bg() -> None:
        try:
            cursor = 0
            pattern = f"dir:unified:base:{user_id}:*"
            while True:
                cursor, keys = _r.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    _r.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass

    _bg_submit("invalidate_unified_base", _bg, {"user_id": user_id})


def acquire_compile_lock(
    user_id: str, query: str, ex_sec: int = 10
) -> tuple[str, bool]:
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
    lock_key = f"lock:compile:{user_id}:{qh}"
    try:
        ok = _r.set(lock_key, "1", nx=True, ex=int(ex_sec))
        return lock_key, bool(ok)
    except Exception:
        return lock_key, False


def release_compile_lock(lock_key: str) -> None:
    try:
        if lock_key:
            _r.delete(lock_key)
    except Exception:
        pass


# ----- 세션6: 압축 캐시(Base/Overlay) + Single-Flight 락 -----


def _kc_comp_base(user_id: str, version: str, model: str, budget: int) -> str:
    return f"dir:prompt:comp:base:{user_id}:{version}:{model}:{int(budget)}"


def _kc_comp_overlay(session_id: str, query_hash8: str, model: str, budget: int) -> str:
    return f"dir:prompt:comp:overlay:{session_id}:{query_hash8}:{model}:{int(budget)}"


def get_compressed_base(user_id: str, version: str, model: str, budget: int) -> str:
    if not user_id or not version:
        return ""
    try:
        h = _r.hgetall(_kc_comp_base(user_id, version, model, budget)) or {}
        return h.get("text", "")
    except Exception:
        return ""


def set_compressed_base(
    user_id: str, version: str, model: str, budget: int, text: str, ttl_sec: int = 86400
) -> None:
    if not user_id or not version:
        return

    def _bg() -> None:
        try:
            ck = _kc_comp_base(user_id, version, model, budget)
            _r.hset(ck, mapping={"text": text or "", "ts": str(int(time.time()))})
            _r.expire(ck, int(ttl_sec))
        except Exception:
            pass

    _bg_submit(
        "set_compressed_base",
        _bg,
        {
            "user_id": user_id,
            "version": version,
            "model": model,
            "budget": int(budget),
            "text_len": len(text or ""),
        },
    )


def get_compressed_overlay(session_id: str, query: str, model: str, budget: int) -> str:
    if not session_id:
        return ""
    try:
        qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
        h = _r.hgetall(_kc_comp_overlay(session_id, qh, model, budget)) or {}
        return h.get("text", "")
    except Exception:
        return ""


def set_compressed_overlay(
    session_id: str, query: str, model: str, budget: int, text: str, ttl_sec: int = 7200
) -> None:
    if not session_id:
        return
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]

    def _bg() -> None:
        try:
            ck = _kc_comp_overlay(session_id, qh, model, budget)
            _r.hset(ck, mapping={"text": text or "", "ts": str(int(time.time()))})
            _r.expire(ck, int(ttl_sec))
        except Exception:
            pass

    _bg_submit(
        "set_compressed_overlay",
        _bg,
        {
            "session_id": session_id,
            "qh": qh,
            "model": model,
            "budget": int(budget),
            "text_len": len(text or ""),
        },
    )


def acquire_compress_lock(
    session_id: str, query: str, model: str, budget: int, ex_sec: int = 5
) -> tuple[str, bool]:
    try:
        qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]
        lock_key = f"lock:comp:{session_id}:{qh}:{model}:{int(budget)}"
        ok = _r.set(lock_key, "1", nx=True, ex=int(ex_sec))
        return lock_key, bool(ok)
    except Exception:
        return "", False


def release_compress_lock(lock_key: str) -> None:
    try:
        if lock_key:
            _r.delete(lock_key)
    except Exception:
        pass
