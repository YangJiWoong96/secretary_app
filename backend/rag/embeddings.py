from functools import lru_cache
import os
import numpy as np
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from .config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BACKEND,
    GEMMA_MODEL_PATH,
)


_EMBeddings_INSTANCE: Optional[object] = None  # OpenAIEmbeddings | SentenceTransformer
_BACKEND_READY = None


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


def _norm_for_cache(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


@lru_cache(maxsize=10000)
def _embed_query_cached_key(model: str, dim: int, text_norm: str) -> tuple[float, ...]:
    backend = get_embeddings()
    # 백엔드별 API 통일
    if _BACKEND_READY == "gemma":
        # sentence-transformers: encode로 쿼리/문서 모두 처리
        vec = backend.encode([text_norm], normalize_embeddings=True)[0]
    else:
        vec = backend.embed_query(text_norm)
    return tuple(float(x) for x in vec)


def embed_query_cached(text: str) -> np.ndarray:
    text_norm = _norm_for_cache(text)
    vec = _embed_query_cached_key(EMBEDDING_MODEL, EMBEDDING_DIM, text_norm)
    return np.array(vec, dtype=np.float32)


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
        # OpenAIEmbeddings.embed_documents는 리스트 반환
        vecs = backend.embed_documents(texts)
        return [np.array(v, dtype=np.float32) for v in vecs]
