import os
import time
import json
import hashlib
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import redis

from .milvus import ensure_collections
from .embeddings import embed_query_cached


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _simhash64(text: str) -> int:
    # 간단 whitespace 토큰 기반 simhash(64-bit)
    tokens = [t for t in _norm_text(text).split() if t]
    if not tokens:
        return 0
    bits = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)  # 64-bit
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if bits[i] >= 0:
            out |= 1 << i
    return out


def _canon_url(u: str) -> str:
    try:
        p = urlparse(u)
        # UTM/트래킹 파라미터 제거
        q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")]
        clean = p._replace(query=urlencode(q, doseq=True), fragment="")
        # 호스트 소문자, 기본 포트 제거
        netloc = clean.netloc.lower()
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        if netloc.endswith(":443"):
            netloc = netloc[:-4]
        clean = clean._replace(netloc=netloc)
        return urlunparse(clean)
    except Exception:
        return (u or "").strip()


def _url_hash(u: str) -> str:
    return hashlib.sha256(_canon_url(u).encode("utf-8")).hexdigest()


def _content_hash(t: str) -> str:
    return hashlib.sha256(_norm_text(t).encode("utf-8")).hexdigest()


def extract_web_items(blocks: str) -> List[Tuple[str, str]]:
    """
    3줄 블록(web)에서 (title, url)만 추출.
    """
    items: List[Tuple[str, str]] = []
    for block in (blocks or "").split("\n\n"):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) >= 3:
            title = lines[0]
            url = lines[2]
            if url:
                items.append((title, url))
    return items[:8]


def _rpush_refs(session_id: str, refs: List[str]):
    try:
        key = f"evrefs:{session_id}"
        if refs:
            _r.rpush(key, *refs)
            _r.ltrim(key, -200, -1)  # 최근 200개 유지
            _r.expire(key, int(os.getenv("EVIDENCE_REFS_TTL_SEC", "15552000")))  # 180d
    except Exception:
        pass


def upsert_web_ref(session_id: str, title: str, url: str) -> str:
    """
    log_coll에 type="web_ref"로 업서트. text는 "title | url" 형식.
    반환: pointer token (e.g., "web:<url_hash>")
    """
    _, log_coll = ensure_collections()
    url_h = _url_hash(url)
    text = f"{title or ''} | {url}"
    emb = embed_query_cached(title or url)
    try:
        log_coll.upsert(
            [
                {
                    "id": f"{session_id}:web:{url_h}",
                    "embedding": emb,
                    "text": text,
                    "user_id": session_id,
                    "type": "web_ref",
                    "created_at": int(time.time_ns()),
                    "date_start": 0,
                    "date_end": 99999999,
                    "date_ym": 0,
                }
            ]
        )
    except Exception:
        pass
    return f"web:{url_h}"


def upsert_rag_ref(session_id: str, block_text: str) -> str:
    """
    log_coll에 type="rag_ref"로 업서트. text는 간단 preview 문자열.
    반환: pointer token (e.g., "rag:<content_hash>")
    """
    _, log_coll = ensure_collections()
    h = _content_hash(block_text or "")
    # preview: 첫 2문장/250자
    t = (block_text or "").strip()
    preview = t[:250]
    emb = embed_query_cached(preview)
    try:
        log_coll.upsert(
            [
                {
                    "id": f"{session_id}:rag:{h}",
                    "embedding": emb,
                    "text": preview,
                    "user_id": session_id,
                    "type": "rag_ref",
                    "created_at": int(time.time_ns()),
                    "date_start": 0,
                    "date_end": 99999999,
                    "date_ym": 0,
                }
            ]
        )
    except Exception:
        pass
    return f"rag:{h}"


def store_refs_from_contexts(session_id: str, web_ctx: str, rag_ctx: str) -> List[str]:
    refs: List[str] = []
    # 웹 refs
    for title, url in extract_web_items(web_ctx or ""):
        refs.append(upsert_web_ref(session_id, title, url))
    # RAG refs: 블록 단위
    for block in (rag_ctx or "").split("\n\n"):
        bt = block.strip()
        if bt:
            refs.append(upsert_rag_ref(session_id, bt))
    # Redis 포인터 저장(경량)
    if refs:
        _rpush_refs(session_id, refs)
    return refs
