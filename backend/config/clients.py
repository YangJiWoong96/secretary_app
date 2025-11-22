"""
backend.config.clients - 외부 클라이언트 초기화 및 관리
OpenAI, LangChain, Firestore 등 외부 서비스 클라이언트의
싱글톤 인스턴스를 제공하는 팩토리 함수들.
"""

from __future__ import annotations

import logging
from functools import lru_cache
import os
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from backend.config.settings import get_settings

# Firestore는 선택적 의존성
try:
    from google.cloud import firestore  # type: ignore

    FIRESTORE_AVAILABLE = True
except ImportError:
    firestore = None  # type: ignore
    FIRESTORE_AVAILABLE = False

logger = logging.getLogger("clients")
"""
backend.config.clients - 외부 클라이언트 초기화 및 관리
공용 싱글톤 팩토리 (Firestore / OpenAI / LLM / Embeddings)
"""

# ===== 전역 클라이언트 인스턴스 캐시 =====
_openai_client: Optional[AsyncOpenAI] = None
_llm_instance: Optional[ChatOpenAI] = None
_llm_cold_instance: Optional[ChatOpenAI] = None
_embeddings_instance = None
_firestore_client = None


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """
    AsyncOpenAI 클라이언트 싱글톤 반환

    OpenAI API 키를 사용하여 비동기 클라이언트를 초기화합니다.
    전역에서 동일한 인스턴스를 재사용합니다.

    Returns:
        AsyncOpenAI: OpenAI 비동기 클라이언트

    Example:
        >>> from backend.config import get_openai_client
        >>> client = get_openai_client()
        >>> response = await client.chat.completions.create(...)
    """
    global _openai_client

    if _openai_client is None:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # LangSmith 트레이싱 활성 시 OpenAI 클라이언트 래핑
        try:
            if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
                from langsmith.wrappers import wrap_openai  # type: ignore

                client = wrap_openai(client)  # type: ignore
                logger.info("[clients] AsyncOpenAI client wrapped by LangSmith")
        except Exception:
            # 래핑 실패 시 기본 클라이언트 사용
            pass
        _openai_client = client
        logger.info("[clients] AsyncOpenAI client initialized")

    return _openai_client


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    LangChain ChatOpenAI 인스턴스 반환 (일반 용도)

    LangChain 체인 및 일반 LLM 호출에 사용되는 인스턴스입니다.

    Returns:
        ChatOpenAI: LangChain LLM 인스턴스

    Example:
        >>> from backend.config import get_llm
        >>> llm = get_llm()
        >>> result = llm.invoke("Hello")
    """
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
        logger.info(f"[clients] LangChain LLM initialized (model={settings.LLM_MODEL})")

    return _llm_instance


@lru_cache(maxsize=1)
def get_llm_cold() -> ChatOpenAI:
    """
    LangChain ChatOpenAI 인스턴스 반환 (메모리/요약 전용)

    메모리 관리 및 대화 요약 전용으로 사용되는 별도 인스턴스입니다.
    일반 LLM 인스턴스와 독립적으로 관리됩니다.

    Returns:
        ChatOpenAI: 메모리/요약 전용 LLM 인스턴스

    Example:
        >>> from backend.config import get_llm_cold
        >>> llm_cold = get_llm_cold()
        >>> memory = ConversationSummaryBufferMemory(llm=llm_cold, ...)
    """
    global _llm_cold_instance

    if _llm_cold_instance is None:
        settings = get_settings()
        _llm_cold_instance = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
        logger.info(
            f"[clients] LangChain LLM (cold) initialized (model={settings.LLM_MODEL})"
        )

    return _llm_cold_instance


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """
    임베딩 백엔드 인스턴스 반환

    backend.rag.embeddings 모듈의 추상화된 임베딩 백엔드를 반환합니다.
    OpenAI 또는 로컬 모델(Gemma 등)을 사용할 수 있습니다.

    Returns:
        임베딩 인스턴스 (OpenAIEmbeddings 또는 로컬 모델)

    Example:
        >>> from backend.config import get_embeddings
        >>> embeddings = get_embeddings()
        >>> vectors = embeddings.embed_documents(["text1", "text2"])
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        # 순환 의존성 방지를 위해 지연 임포트
        from backend.rag.embeddings import get_embeddings as _get_embeddings_impl

        _embeddings_instance = _get_embeddings_impl()
        logger.info("[clients] Embeddings backend initialized")

    return _embeddings_instance


def get_firestore_client():
    """
    Firestore 클라이언트 반환 (지연 초기화)

    Firestore가 활성화된 경우에만 클라이언트를 초기화합니다.
    환경변수 FIRESTORE_ENABLE=0인 경우 또는 google-cloud-firestore가
    설치되지 않은 경우 None을 반환합니다.

    Returns:
        firestore.Client | None: Firestore 클라이언트 또는 None

    Example:
        >>> from backend.config import get_firestore_client
        >>> db = get_firestore_client()
        >>> if db:
        ...     users = db.collection('users').stream()
    """
    global _firestore_client

    # 이미 초기화 시도를 했으면 캐시된 결과 반환 (None 포함)
    if _firestore_client is not None:
        return _firestore_client

    settings = get_settings()

    # Firestore 비활성화 또는 라이브러리 미설치
    if not settings.FIRESTORE_ENABLE or not FIRESTORE_AVAILABLE:
        logger.info("[clients] Firestore client disabled or not available")
        return None

    try:
        _firestore_client = firestore.Client()
        logger.info("[clients] Firestore client initialized")
        return _firestore_client
    except Exception as e:
        logger.warning(f"[clients] Firestore init failed: {e}")
        return None


def reset_clients() -> None:
    """
    모든 클라이언트 인스턴스 초기화 (테스트용)

    싱글톤 캐시를 초기화하여 다음 호출 시 새로운 인스턴스를 생성하도록 합니다.
    주로 테스트 환경에서 설정을 변경한 후 클라이언트를 재생성할 때 사용합니다.

    Warning:
        프로덕션 환경에서는 사용하지 마세요.
    """
    global _openai_client, _llm_instance, _llm_cold_instance, _embeddings_instance, _firestore_client

    _openai_client = None
    _llm_instance = None
    _llm_cold_instance = None
    _embeddings_instance = None
    _firestore_client = None

    # lru_cache 초기화
    get_openai_client.cache_clear()
    get_llm.cache_clear()
    get_llm_cold.cache_clear()
    get_embeddings.cache_clear()

    logger.info("[clients] All client instances reset")
