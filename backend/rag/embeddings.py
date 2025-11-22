# backend/rag/embeddings.py
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings

from backend.rag.embedding_cache import get_embedding_cache

from .config import (
    EMBEDDING_BACKEND,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GEMMA_MODEL_PATH,
    OPENAI_API_KEY,
)

_EMBeddings_INSTANCE: Optional[object] = None  # OpenAIEmbeddings | SentenceTransformer
_BACKEND_READY = None

# 백엔드 전용 인스턴스(동시 보유 허용: 기능별 선택 적용용)
_EMB_OPENAI: Optional[OpenAIEmbeddings] = None
_EMB_GEMMA: Optional[object] = None  # SentenceTransformer


def _init_openai() -> OpenAIEmbeddings:
    key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in env before using OpenAI embeddings."
        )
    return OpenAIEmbeddings(openai_api_key=key, model=EMBEDDING_MODEL)


def _init_gemma():
    # sentence-transformers 로컬 모델 로드
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers가 설치되어 있지 않습니다. requirements.txt에 추가하세요."
        ) from e
    model = SentenceTransformer(GEMMA_MODEL_PATH, device="cpu")
    return model


def _get_openai() -> OpenAIEmbeddings:
    global _EMB_OPENAI
    if _EMB_OPENAI is None:
        _EMB_OPENAI = _init_openai()
    return _EMB_OPENAI


def _get_gemma():
    global _EMB_GEMMA
    if _EMB_GEMMA is None:
        _EMB_GEMMA = _init_gemma()
    return _EMB_GEMMA


def get_embeddings():
    """
    임베딩 백엔드(OpenAI/Gemma)를 지연 초기화하고 단일 인스턴스를 반환한다.
    """
    global _EMBeddings_INSTANCE, _BACKEND_READY
    if _EMBeddings_INSTANCE is not None:
        return _EMBeddings_INSTANCE
    if EMBEDDING_BACKEND == "gemma":
        _EMBeddings_INSTANCE = _init_gemma()
        _BACKEND_READY = "gemma"
    else:
        _EMBeddings_INSTANCE = _init_openai()
        _BACKEND_READY = "openai"
    return _EMBeddings_INSTANCE


def _get_backend_name() -> str:
    """Ensure lazy-init and return resolved backend name."""
    global _BACKEND_READY
    if _BACKEND_READY is None:
        get_embeddings()
    return _BACKEND_READY or "openai"


def _norm_for_cache(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


@lru_cache(maxsize=10000)
def _embed_query_cached_key(model: str, dim: int, text_norm: str) -> tuple[float, ...]:
    backend_name = _get_backend_name()
    if backend_name == "gemma":
        mdl = _get_gemma()
        vec = mdl.encode([text_norm], normalize_embeddings=True)[0]
    else:
        emb = _get_openai()
        vec = emb.embed_query(text_norm)
    return tuple(float(x) for x in vec)


def embed_query_cached(text: str) -> np.ndarray:
    """
    단일 쿼리 임베딩 엔트리포인트(백엔드 추상화).
    - OpenAI: L2 캐시 포함 경로(embed_query_openai) 사용
    - Gemma: ST encode + LRU 캐시
    """
    if _get_backend_name() == "openai":
        return embed_query_openai(text)
    tn = _norm_for_cache(text)
    v = _embed_query_cached_key(EMBEDDING_MODEL, EMBEDDING_DIM, tn)
    return np.array(v, dtype=np.float32)


def embed_documents(texts: list[str]) -> list[np.ndarray]:
    """
    문서 배치 임베딩을 백엔드 추상화로 제공(OpenAI/Gemma 호환).
    반환: np.ndarray 리스트
    """
    if not texts:
        return []
    backend = get_embeddings()
    if _BACKEND_READY == "gemma":
        vecs = backend.encode(texts, normalize_embeddings=True)
        return [np.array(v, dtype=np.float32) for v in vecs]
    else:
        # Embeddings Manager 경유(배치·캐시·중복 제거·순서 복원)
        try:
            from backend.llm.embeddings_manager import embed_texts as _em
            import asyncio as _a

            # 실행 중인 루프가 있으면 안전하게 폴백(동기 컨텍스트에서만 매니저 사용)
            try:
                _a.get_running_loop()
                vecs = backend.embed_documents(texts)
            except RuntimeError:
                vecs = _a.get_event_loop().run_until_complete(
                    _em([(t or "").strip() for t in texts])
                )
        except Exception:
            # 폴백: 기존 경로(OpenAIEmbeddings) 사용
            vecs = backend.embed_documents(texts)
        return [np.array(v, dtype=np.float32) for v in vecs]  # type: ignore[arg-type]


# ------------------------------
# 백엔드 고정 임베딩 (직접 호출이 꼭 필요한 특수 경로용)
# ------------------------------


@lru_cache(maxsize=10000)
def _openai_embed_query_cached_key(text_norm: str) -> tuple[float, ...]:
    emb = _get_openai()
    vec = emb.embed_query(text_norm)
    return tuple(float(x) for x in vec)


@lru_cache(maxsize=10000)
def _gemma_embed_query_cached_key(text_norm: str) -> tuple[float, ...]:
    mdl = _get_gemma()
    vec = mdl.encode([text_norm], normalize_embeddings=True)[0]
    return tuple(float(x) for x in vec)


def embed_query_openai(text: str) -> np.ndarray:
    """OpenAI 전용 쿼리 임베딩(다계층 캐시 통합).

    우선순위: L1(In-Mem) → L2(Redis) → OpenAI API
    """
    try:
        cache = get_embedding_cache()
        vec = cache.get_embedding((text or "").strip(), model_id=EMBEDDING_MODEL)
        return np.array(vec, dtype=np.float32)
    except Exception:
        # 캐시 계층 장애 시 기존 lru_cache 경로로 폴백
        tn = _norm_for_cache(text)
        v = _openai_embed_query_cached_key(tn)
        return np.array(v, dtype=np.float32)


def embed_query_gemma(text: str) -> np.ndarray:
    """Gemma(Sentence-Transformers) 전용 쿼리 임베딩(캐시 포함)."""
    tn = _norm_for_cache(text)
    v = _gemma_embed_query_cached_key(tn)
    return np.array(v, dtype=np.float32)
