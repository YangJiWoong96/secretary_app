"""
backend.routing.ml_classifier - 머신러닝 기반 의도 분류기

Scikit-learn, HuggingFace Transformers, 또는 LLM을 사용하여
사용자 입력의 need_rag/need_web 확률을 예측합니다.

임베딩 라우터의 저신뢰 케이스 폴백으로 사용 가능합니다.

우선순위:
1. Scikit-learn (PROACTIVE_INTENT_SKLEARN_PATH 설정 시)
2. HuggingFace Transformers (PROACTIVE_INTENT_HF_PATH 설정 시)
3. LLM 폴백 (THINKING_MODEL 사용)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

# 전역 모델 인스턴스 캐시
_SKLEARN_MODEL = None
_SKLEARN_VECTORIZER = None
_HF_TOKENIZER = None
_HF_MODEL = None


def _try_load_sklearn() -> bool:
    """
    환경변수로 지정된 경로에서 scikit-learn 분류기와 벡터라이저를 로드한다.

    필수 환경변수:
    - PROACTIVE_INTENT_SKLEARN_PATH (*.joblib)
    - PROACTIVE_INTENT_VECTORIZER_PATH (*.joblib)

    Returns:
        bool: 모델 로딩 성공 여부
    """
    global _SKLEARN_MODEL, _SKLEARN_VECTORIZER
    if _SKLEARN_MODEL is not None and _SKLEARN_VECTORIZER is not None:
        return True
    path_m = os.getenv("PROACTIVE_INTENT_SKLEARN_PATH")
    path_v = os.getenv("PROACTIVE_INTENT_VECTORIZER_PATH")
    if not path_m or not path_v:
        return False
    try:
        from joblib import load as _joblib_load

        _SKLEARN_MODEL = _joblib_load(path_m)
        _SKLEARN_VECTORIZER = _joblib_load(path_v)
        return True
    except Exception:
        _SKLEARN_MODEL = None
        _SKLEARN_VECTORIZER = None
        return False


def _try_load_hf() -> bool:
    """
    HuggingFace 분류기(한국어 Electra 등)를 로드한다.

    환경변수 또는 기본 경로(models/router_kor_electra_small)를 사용한다.
    필수 파일: config.json, tokenizer.*, pytorch_model.bin 등이 경로에 있어야 한다.

    Returns:
        bool: 모델 로딩 성공 여부
    """
    global _HF_TOKENIZER, _HF_MODEL
    if _HF_TOKENIZER is not None and _HF_MODEL is not None:
        return True
    path = os.getenv(
        "PROACTIVE_INTENT_HF_PATH", "c:/My_Business/models/router_kor_electra_small"
    )
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        _HF_TOKENIZER = AutoTokenizer.from_pretrained(path)
        _HF_MODEL = AutoModelForSequenceClassification.from_pretrained(path)
        return True
    except Exception:
        _HF_TOKENIZER = None
        _HF_MODEL = None
        return False


def _clf_labels() -> Tuple[str, str]:
    """
    레이블 정의: need_rag, need_web 두 축을 독립 이진으로 본다.

    스몰 분류기는 멀티라벨(두 개의 시그모이드) 또는
    다중 클래스(4클래스)를 지원한다.

    Returns:
        Tuple[str, str]: ("need_rag", "need_web")
    """
    return ("need_rag", "need_web")


def _predict_sklearn(text: str) -> Dict[str, float]:
    """
    Scikit-learn 모델을 사용한 의도 분류

    벡터화 후 이진 확률 추정. 양방향 멀티라벨 또는 4-클래스를 가정한다.

    Args:
        text: 입력 텍스트

    Returns:
        Dict[str, float]: {"need_rag": prob, "need_web": prob}
    """
    X = _SKLEARN_VECTORIZER.transform([text])
    probs: Dict[str, float] = {}
    try:
        if hasattr(_SKLEARN_MODEL, "predict_proba"):
            pp = _SKLEARN_MODEL.predict_proba(X)
            # 멀티라벨: list of arrays
            if isinstance(pp, list) and len(pp) >= 2:
                probs = {
                    "need_rag": float(pp[0][:, 1][0]),
                    "need_web": float(pp[1][:, 1][0]),
                }
            else:
                # 다중 클래스: 00, 01, 10, 11
                pc = pp[0]
                p00 = float(pc[0]) if len(pc) > 0 else 0.0
                p01 = float(pc[1]) if len(pc) > 1 else 0.0
                p10 = float(pc[2]) if len(pc) > 2 else 0.0
                p11 = float(pc[3]) if len(pc) > 3 else 0.0
                probs = {
                    "need_rag": p10 + p11,
                    "need_web": p01 + p11,
                }
        else:
            probs = {"need_rag": 0.5, "need_web": 0.5}
    except Exception:
        probs = {"need_rag": 0.5, "need_web": 0.5}
    return probs


def _predict_hf(text: str) -> Dict[str, float]:
    """
    HuggingFace Transformers 모델을 사용한 의도 분류

    시그모이드 2-출력(멀티라벨) 또는 소프트맥스 4-클래스를 지원한다.

    Args:
        text: 입력 텍스트

    Returns:
        Dict[str, float]: {"need_rag": prob, "need_web": prob}
    """
    import torch
    from torch.nn.functional import sigmoid, softmax

    tok = _HF_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = _HF_MODEL(**tok)
        logits = out.logits
        if logits.shape[-1] == 2:  # 멀티라벨 2-출력
            p = sigmoid(logits).squeeze(0).tolist()
            return {"need_rag": float(p[0]), "need_web": float(p[1])}
        else:  # 4-클래스
            pc = softmax(logits, dim=-1).squeeze(0).tolist()
            p00 = float(pc[0]) if len(pc) > 0 else 0.0
            p01 = float(pc[1]) if len(pc) > 1 else 0.0
            p10 = float(pc[2]) if len(pc) > 2 else 0.0
            p11 = float(pc[3]) if len(pc) > 3 else 0.0
            return {"need_rag": p10 + p11, "need_web": p01 + p11}


def _predict_llm(text: str) -> Dict[str, float]:
    """
    LLM 폴백을 사용한 의도 분류

    JSON만 반환하도록 강제하고, [0,1] 확률을 추정한다.

    Args:
        text: 입력 텍스트

    Returns:
        Dict[str, float]: {"need_rag": prob, "need_web": prob}
    """
    from langchain_openai import ChatOpenAI

    model = os.getenv("THINKING_MODEL", "gpt-5-thinking")
    llm = ChatOpenAI(model=model)
    sys = (
        "너는 입력 문장을 보고 다음 두 확률을 0~1 실수로 예측한다. "
        "1) need_rag: 개인 장기기억/대화 로그/프로필 검색이 유효한가, "
        "2) need_web: 웹/뉴스/현황 검색이 유효한가. "
        'JSON만 반환하라. 예: {"need_rag":0.7,"need_web":0.2}'
    )
    usr = f"문장: {text}".strip()
    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ]
        )
        content = (getattr(resp, "content", "") or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        need_rag = float(data.get("need_rag", 0.5))
        need_web = float(data.get("need_web", 0.5))
        need_rag = max(0.0, min(1.0, need_rag))
        need_web = max(0.0, min(1.0, need_web))
        return {"need_rag": need_rag, "need_web": need_web}
    except Exception:
        return {"need_rag": 0.5, "need_web": 0.5}


def classify_intent(text: str, threshold: float = 0.5) -> Dict[str, float | int]:
    """
    텍스트에 대한 의도 분류 결과를 반환한다.

    우선순위: scikit-learn -> HuggingFace -> LLM

    Args:
        text: 입력 텍스트
        threshold: 이진 분류 임계값 (기본 0.5)

    Returns:
        Dict: {
            "prob_rag": 0.73,   # RAG 확률
            "prob_web": 0.22,   # WEB 확률
            "need_rag": 1,      # prob >= threshold
            "need_web": 0
        }

    Example:
        >>> from backend.routing.ml_classifier import classify_intent
        >>> result = classify_intent("내가 저번에 설정한 목표 알려줘")
        >>> print(result)
        {'prob_rag': 0.85, 'prob_web': 0.15, 'need_rag': 1, 'need_web': 0}
    """
    text = (text or "").strip()
    if not text:
        return {"prob_rag": 0.0, "prob_web": 0.0, "need_rag": 0, "need_web": 0}

    probs: Optional[Dict[str, float]] = None
    if _try_load_sklearn():
        probs = _predict_sklearn(text)
    elif _try_load_hf():
        probs = _predict_hf(text)
    else:
        probs = _predict_llm(text)

    pr = float(probs.get("need_rag", 0.5))
    pw = float(probs.get("need_web", 0.5))
    return {
        "prob_rag": pr,
        "prob_web": pw,
        "need_rag": 1 if pr >= threshold else 0,
        "need_web": 1 if pw >= threshold else 0,
    }
