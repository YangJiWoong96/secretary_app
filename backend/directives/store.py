# # Redis(캐시) + (옵션) Milvus 스냅샷
import os, json, time, hashlib
import redis
from typing import Tuple, Optional, Dict, List, Any
from .schema import Directives

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TTL_SEC = int(os.getenv("DIR_TTL_SEC", "604800"))  # 7d
_r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


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
        _r.hset(
            _k(session_id), mapping={"reasons": json.dumps(reasons, ensure_ascii=False)}
        )
    try:
        _r.sadd(ACTIVE_SET, session_id)
    except Exception:
        pass


def set_compiled(session_id: str, prompt: str, version: str):
    _r.hset(
        _kc(session_id),
        mapping={"prompt": prompt, "version": version, "ts": str(int(time.time()))},
    )
    _r.expire(_kc(session_id), TTL_SEC)


def get_compiled(session_id: str) -> Tuple[str, str]:
    h = _r.hgetall(_kc(session_id)) or {}
    return h.get("prompt", ""), h.get("version", "")


def version_of(obj: Any) -> str:
    """임의의 JSON 직렬화 가능 객체의 버전 해시.
    directives, signals, persona를 함께 넣어 버전 관리 가능.
    """
    blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----- signals/persona 저장/불러오기 -----


def load_signals(session_id: str) -> Dict:
    j = _r.hget(_ks(session_id), "json")
    return json.loads(j) if j else {}


def save_signals(session_id: str, signals: Dict):
    _r.hset(_ks(session_id), mapping={"json": json.dumps(signals, ensure_ascii=False)})
    _r.expire(_ks(session_id), TTL_SEC)
    try:
        _r.sadd(ACTIVE_SET, session_id)
    except Exception:
        pass


def load_persona(session_id: str) -> Dict:
    j = _r.hget(_kp(session_id), "json")
    return json.loads(j) if j else {}


def save_persona(session_id: str, persona: Dict):
    _r.hset(_kp(session_id), mapping={"json": json.dumps(persona, ensure_ascii=False)})
    _r.expire(_kp(session_id), TTL_SEC)
    try:
        _r.sadd(ACTIVE_SET, session_id)
    except Exception:
        pass


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
