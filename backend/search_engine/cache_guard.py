from __future__ import annotations

"""
Search Cache Guard

역할:
- (정규화된 쿼리, 도메인) → 결과 블록(요약/키팩트) 캐시를 TTL로 보존
- 차감 검색/증분 탐색 전에 캐시 적중 시 즉시 반환하여 외부 호출을 줄임

설계:
- 프로세스 메모리 캐시(간단): { key -> (ts, data) }
- 키: sha1("{domain}:{normalized_query}")
"""

import hashlib
import os
import time
from typing import Dict, Optional, Tuple

from backend.config import get_settings

_CACHE: Dict[str, Tuple[float, Dict]] = {}


def _norm(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def _key(q: str, domain: str = "*") -> str:
    return hashlib.sha1(f"{domain}:{_norm(q)}".encode()).hexdigest()


def get(q: str, domain: str = "*") -> Optional[Dict]:
    try:
        ttl = float(get_settings().CACHE_TTL_SEC)
    except Exception:
        ttl = 1800.0
    k = _key(q, domain)
    v = _CACHE.get(k)
    if not v:
        return None
    ts, data = v
    if time.time() - ts > ttl:
        _CACHE.pop(k, None)
        return None
    return data


def put(q: str, data: Dict, domain: str = "*") -> None:
    _CACHE[_key(q, domain)] = (time.time(), data)


__all__ = ["get", "put"]
