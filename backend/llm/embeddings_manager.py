"""
backend.llm.embeddings_manager - 임베딩 배치·캐시·단일화 매니저

기능:
- 입력 텍스트를 정규화 후 SHA1으로 중복 제거
- LRU+TTL 인메모리 캐시 조회
- 남은 항목은 OpenAI Embeddings 배치 호출(AsyncOpenAI 재사용)
- 부분 실패 시 소배치/단건 재시도로 복구
- 원 순서 복원

메트릭 이벤트:
- embeddings.cache_hit
- embeddings.batch_size
- embeddings.calls_per_turn
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Dict, List, Tuple, Optional

from backend.config import get_settings
from backend.config.clients import get_openai_client
from backend.utils.logger import log_event
from backend.utils.retry import get_retry_manager

# ===== 인메모리 LRU+TTL 캐시 =====
_CACHE: Dict[str, Tuple[float, List[float]]] = {}
_CACHE_ACCESS: Dict[str, float] = {}
_CACHE_MAXSIZE = 4096
_CACHE_TTL_S = 300  # 5분

# 동시성 세마포어 (전역)
_SEM: Optional[asyncio.Semaphore] = None


def _get_sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(max(1, int(get_settings().EMBED_CONCURRENCY)))
    return _SEM


def _norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text or ""):
        return "ko"
    if re.search(r"[a-zA-Z]", text or ""):
        return "en"
    return "mixed"


def _make_key(provider: str, model: str, dim: int, text_norm: str) -> str:
    sha = hashlib.sha1(text_norm.encode("utf-8")).hexdigest()
    lang = _detect_language(text_norm)
    # norm은 L2로 고정
    return f"prov={provider}|model={model}|dim={dim}|norm=l2|lang={lang}|sha1={sha}"


def _cache_get(key: str) -> Optional[List[float]]:
    now = time.time()
    v = _CACHE.get(key)
    if not v:
        return None
    ts, vec = v
    if now - ts > _CACHE_TTL_S:
        # TTL 만료
        _CACHE.pop(key, None)
        _CACHE_ACCESS.pop(key, None)
        return None
    _CACHE_ACCESS[key] = now
    return vec


def _cache_set(key: str, vec: List[float]) -> None:
    now = time.time()
    _CACHE[key] = (now, vec)
    _CACHE_ACCESS[key] = now
    # LRU 제거
    if len(_CACHE) > _CACHE_MAXSIZE:
        oldest_key = min(_CACHE_ACCESS, key=_CACHE_ACCESS.get)
        _CACHE.pop(oldest_key, None)
        _CACHE_ACCESS.pop(oldest_key, None)


async def _embed_batch(client, model: str, inputs: List[str]) -> List[List[float]]:
    """
    OpenAI Embeddings 배치 호출. 오류 시 재시도(backoff).
    """
    retry = get_retry_manager()
    attempts = int(get_settings().MAX_RETRIES_OPENAI) + 1
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            resp = await client.embeddings.create(model=model, input=inputs)
            # OpenAI 응답에서 data는 입력 순서에 대응
            return [list(map(float, d.embedding)) for d in resp.data]  # type: ignore[attr-defined]
        except Exception as e:
            last_err = e
            if i + 1 >= attempts:
                break
            await retry.backoff_sleep(i)
    # 최종 실패
    if last_err is not None:
        raise last_err
    raise RuntimeError("embeddings batch failed with unknown error")


async def embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    """
    텍스트 리스트 임베딩(배치/캐시/순서복원).

    내부 처리:
    1) 정규화+SHA1로 중복 제거
    2) LRU 캐시 조회(hit는 즉시 복원)
    3) 미스 항목만 배치 호출(BATCH_MAX 단위)
    4) 결과 캐시에 저장 후 원래 순서로 복원
    """
    settings = get_settings()
    mdl = (model or settings.EMBEDDING_MODEL).strip()
    provider = "openai"  # 현재는 OpenAI만
    if not texts:
        return []

    # 정규화 & 인덱스 매핑
    normed = [_norm_text(t) for t in texts]
    # SHA1 기반 de-dup
    uniq_map: Dict[str, List[int]] = {}
    for idx, t in enumerate(normed):
        sha = hashlib.sha1(t.encode("utf-8")).hexdigest()
        uniq_map.setdefault(sha, []).append(idx)

    # 캐시 조회
    cached_hits = 0
    to_fetch: List[Tuple[str, str]] = []  # (sha, text)
    key_of_sha: Dict[str, str] = {}
    # dim은 아직 모르므로 -1로 플레이스홀더
    for sha, idxs in uniq_map.items():
        key = _make_key(provider, mdl, -1, normed[idxs[0]])
        key_of_sha[sha] = key
        vec = _cache_get(key)
        if vec is None:
            to_fetch.append((sha, normed[idxs[0]]))
        else:
            cached_hits += 1

    # 임베딩 호출
    client = get_openai_client()
    results_by_sha: Dict[str, List[float]] = {}
    BATCH_MAX = 128
    if to_fetch:
        # 메트릭: batch_size
        try:
            log_event("embeddings.batch_size", {"size": len(to_fetch), "model": mdl})
        except Exception:
            pass

        # 동시에 embed_texts가 여러 번 호출될 수 있으므로 세마포어로 제한
        async with _get_sem():
            # 소배치 반복
            for i in range(0, len(to_fetch), BATCH_MAX):
                chunk = to_fetch[i : i + BATCH_MAX]
                inputs = [txt for _, txt in chunk]
                try:
                    vecs = await _embed_batch(client, mdl, inputs)
                except Exception:
                    # 부분 실패 복구: 단건 재시도
                    vecs = []
                    for _, single in chunk:
                        try:
                            one = await _embed_batch(client, mdl, [single])
                            vecs.append(one[0])
                        except Exception as e:
                            # 마지막 방어: 에러 로깅 후 빈 벡터로 대체하지 않고 예외 전파
                            try:
                                log_event(
                                    "embeddings.error",
                                    {"model": mdl, "error": repr(e)},
                                    level=40,
                                )
                            except Exception:
                                pass
                            raise

                # 캐시 저장 및 매핑
                for (sha, txt), vec in zip(chunk, vecs):
                    # 이제 dim을 알 수 있지만, 초기 키는 dim=-1로 생성됨 → 동일 키로 저장
                    key = key_of_sha.get(sha) or _make_key(provider, mdl, -1, txt)
                    _cache_set(key, vec)
                    results_by_sha[sha] = vec

    # 결과 복원
    out: List[Optional[List[float]]] = [None] * len(texts)
    for sha, idxs in uniq_map.items():
        # 캐시에서 꺼내거나 새로 계산된 벡터
        key = key_of_sha.get(sha) or _make_key(provider, mdl, -1, normed[idxs[0]])
        vec = _cache_get(key) or results_by_sha.get(sha)
        if vec is None:
            # 이 경우는 거의 없지만, 방어적으로 단건 호출 시도
            single_vecs = await _embed_batch(client, mdl, [normed[idxs[0]]])
            vec = single_vecs[0]
            _cache_set(key, vec)
        for i in idxs:
            out[i] = list(vec)

    # 메트릭 로깅
    try:
        log_event(
            "embeddings.cache_hit",
            {"hits": cached_hits, "total_unique": len(uniq_map), "model": mdl},
        )
        log_event("embeddings.calls_per_turn", {"calls": 1})
    except Exception:
        pass

    # 타입 안정성 보장
    return [v or [] for v in out]


__all__ = ["embed_texts"]
