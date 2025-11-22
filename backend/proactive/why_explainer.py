from __future__ import annotations

"""
Why-Tag 생성기

목표:
- 내부 컨텍스트(RAG/모바일/웹) 반영 여부를 짧고 안전한 문장으로 요약한다.
- 민감도는 초기 버전에서 보수적으로 "low"로 제공한다(프론트/정책에서 상향 가능).

설계 원칙:
- 길이 제한: WHY_MAX_LEN(기본 120자) 이내로 자르고 종결 처리를 보수적으로 유지
- 프라이버시: 세부 내용/PII 미노출, "반영" 여부만 진술
"""

import os
from typing import Dict, Optional

from backend.config import get_settings


def _clip(text: str, max_len: int) -> str:
    try:
        t = (text or "").strip()
        return t[: max(0, max_len)].rstrip()
    except Exception:
        return text or ""


def build_why_tag(internal: Dict[str, str]) -> Dict[str, str]:
    """
    입력: {"rag_ctx": str, "mobile_ctx": str, "web_ctx": str}
    출력: {"text": str, "sensitivity": "low|medium|high"}

    규칙:
    - 우선순위: 모바일 → RAG → 웹 (사용자 근접성 기준)
    - 기본 문장 집합만 사용(민감 요약/PII 제거)
    - 길이 제한 적용
    """
    rag = (internal or {}).get("rag_ctx", "") or ""
    mob = (internal or {}).get("mobile_ctx", "") or ""
    web = (internal or {}).get("web_ctx", "") or ""

    # 근접 신호 우선순위
    if mob:
        reason = "최근 기기 컨텍스트를 반영했어요."
    elif rag:
        reason = "이전 대화·메모 맥락을 반영했어요."
    elif web:
        reason = "최근 확인한 근거 자료를 반영했어요."
    else:
        reason = "최근 맥락을 반영했어요."

    try:
        max_len = int(get_settings().WHY_MAX_LEN)
    except Exception:
        max_len = 120
    text = _clip(reason, max_len)
    try:
        sens = str(get_settings().WHY_SENS_FILTER)
    except Exception:
        sens = "low"
    return {"text": text, "sensitivity": sens}


__all__ = ["build_why_tag"]
