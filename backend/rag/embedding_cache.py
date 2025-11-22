# C:\My_Business\backend\rag\embedding_cache.py
from __future__ import annotations

"""
backend.rag.embedding_cache - 다계층 임베딩 캐시

L1 인메모리(LRU+TTL) → L2 Redis(TTL) → API 호출 순서로 임베딩을 제공한다.
캐시 키에는 모델/버전/언어를 포함하여 오염을 방지한다.
"""

import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import redis

from backend.config import get_settings


class MultiLevelCache:
    """L1 (인메모리) → L2 (Redis) → API 캐싱"""

    def __init__(self) -> None:
        self.settings = get_settings()

        # L1: 인메모리 LRU (기본 60분 TTL)
        self._l1_cache: Dict[str, List[float]] = {}
        self._l1_access: Dict[str, float] = {}
        self._l1_maxsize: int = 4096
        # L1 TTL: 5분 (중복 임베딩 호출 방지 목적)
        self._l1_ttl_s: int = 300

        # L2: Redis (기본 24시간 TTL)
        self._redis = redis.Redis.from_url(
            self.settings.REDIS_URL, decode_responses=True
        )

    def _detect_language(self, text: str) -> str:
        """간단한 언어 감지 (한글/영문/혼합)"""
        if re.search(r"[가-힣]", text or ""):
            return "ko"
        if re.search(r"[a-zA-Z]", text or ""):
            return "en"
        return "mixed"

    def _make_keys(
        self, text: str, model_id: Optional[str]
    ) -> Tuple[str, str, str, str]:
        model = (model_id or self.settings.EMBEDDING_MODEL) or "text-embedding-3-small"
        version = getattr(self.settings, "EMBEDDING_MODEL_VERSION", "v1") or "v1"
        lang = self._detect_language(text)
        cache_key_data = f"{model}:{version}:{lang}:{text}"
        short_md5 = hashlib.md5(cache_key_data.encode("utf-8")).hexdigest()
        l1_key = short_md5
        redis_key = f"emb:{model}:{version}:{lang}:{short_md5}"
        return l1_key, redis_key, model, version

    def _l1_evict_if_needed(self) -> None:
        if len(self._l1_cache) <= self._l1_maxsize:
            return
        oldest_key = min(self._l1_access, key=self._l1_access.get)
        self._l1_cache.pop(oldest_key, None)
        self._l1_access.pop(oldest_key, None)

    def get_embedding(
        self, text: str, model_id: Optional[str] = None, normalize: bool = True
    ) -> List[float]:
        """
        임베딩 벡터를 반환한다. (L1→L2→API 순)

        - L1: 인메모리 LRU+TTL
        - L2: Redis JSON(+TTL)
        - Miss 시 API 호출 후 상위 계층에 채움
        """
        import numpy as np

        text = (text or "").strip()
        l1_key, redis_key, model, version = self._make_keys(text, model_id)

        # 1) L1 조회
        now = time.time()
        if l1_key in self._l1_cache:
            last = self._l1_access.get(l1_key, 0.0)
            if (now - last) < self._l1_ttl_s:
                self._l1_access[l1_key] = now
                return self._l1_cache[l1_key]
            # TTL 만료
            self._l1_cache.pop(l1_key, None)
            self._l1_access.pop(l1_key, None)

        # 2) L2 조회
        try:
            cached = self._redis.get(redis_key)
            if cached:
                data = json.loads(cached)
                emb = data.get("embedding")
                if isinstance(emb, list) and emb:
                    self._l1_cache[l1_key] = emb
                    self._l1_access[l1_key] = now
                    self._l1_evict_if_needed()
                    return emb
        except Exception:
            # Redis 장애는 캐시 미스로 취급하고 진행
            pass

        # 3) API 호출
        emb = self._call_api(text, model)

        # 4) 정규화 (cosine 유사도 일관성 보장)
        if normalize:
            try:
                arr = np.array(emb, dtype=np.float32)
                denom = float(np.linalg.norm(arr)) or 1.0
                emb = (arr / denom).astype(np.float32).tolist()
            except Exception:
                # 정규화 실패 시 원본 유지
                emb = [float(x) for x in emb]

        # 5) L1/L2 저장
        self._l1_cache[l1_key] = emb
        self._l1_access[l1_key] = now
        self._l1_evict_if_needed()

        try:
            ttl = int(getattr(self.settings, "EMBEDDING_CACHE_TTL", 86400))
            payload = {
                "embedding": emb,
                "model_id": model,
                "version": version,
                "lang": self._detect_language(text),
                "cached_at": time.time(),
            }
            self._redis.setex(redis_key, ttl, json.dumps(payload))
        except Exception:
            # Redis 저장 실패는 무시 (L1로도 충분히 혜택 제공)
            pass

        return emb

    def _call_api(self, text: str, model_id: str) -> List[float]:
        """실제 임베딩 API 호출 (OpenAIEmbeddings 사용)."""
        from langchain_openai import OpenAIEmbeddings

        api_key = (
            getattr(self.settings, "OPENAI_API_KEY", None)
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in env before using OpenAI embeddings."
            )

        emb = OpenAIEmbeddings(openai_api_key=api_key, model=model_id)
        vec = emb.embed_query(text)
        return [float(x) for x in vec]


# ===== 싱글톤 =====
_cache_instance: Optional[MultiLevelCache] = None


def get_embedding_cache() -> MultiLevelCache:
    """전역 임베딩 캐시 인스턴스를 반환한다."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MultiLevelCache()
    return _cache_instance
