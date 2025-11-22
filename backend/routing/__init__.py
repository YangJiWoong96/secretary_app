"""
backend.routing - 라우팅 모듈 (3단계: 임베딩 → ML → LLM)

사용자 입력을 분석하여 적절한 처리 경로(conv/rag/web)를 결정하는 라우팅 로직을 제공합니다.

구조:
1. 임베딩 라우터: 의미 유사도 기반 초기 라우팅
2. ML 분류기: Scikit-learn/HuggingFace 기반 저신뢰 케이스 폴백
3. LLM 라우터: 최종 폴백 (max_sim < 0.4)
"""

from .intent_router import IntentRouter, get_intent_router
from .llm_router import (
    LLMRouter,
    get_llm_router,
    route_guard,
    router_one_call,
)
from .ml_classifier import classify_intent

__all__ = [
    "IntentRouter",
    "get_intent_router",
    "LLMRouter",
    "get_llm_router",
    "router_one_call",
    "route_guard",
    "classify_intent",
]
