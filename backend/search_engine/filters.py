from __future__ import annotations

"""
관심사 기반 3계층 필터

역할:
- +필수(required), 일반(normal), !거부(denied) 3계층 규칙을 간단하고 확장 가능한 방식으로 적용한다.
- 제목/본문과 같은 텍스트에 대해 매칭 여부를 반환한다.
"""

import re
from typing import Iterable, List


_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_text(text: str) -> str:
    """
    간단 정규화:
    - HTML 태그 제거
    - 소문자화
    - 다중 공백 축약
    """
    t = text or ""
    t = _TAG_RE.sub(" ", t)
    t = " ".join(t.lower().split())
    return t


def match_interest(
    title_or_text: str,
    required: Iterable[str] | None,
    normal: Iterable[str] | None,
    denied: Iterable[str] | None,
) -> bool:
    """
    3계층 필터 매칭.

    Args:
        title_or_text: 검사 대상 텍스트(제목/본문 등).
        required: 모두 포함되어야 하는 키워드(없으면 검사 생략).
        normal: 최소 하나 포함되어야 하는 키워드(없으면 검사 생략).
        denied: 포함되면 즉시 거부되는 키워드(없으면 검사 생략).

    Returns:
        bool: 매칭 여부(True=통과, False=제외)
    """
    t = _normalize_text(title_or_text)
    req = [str(w).lower().strip() for w in (required or []) if str(w).strip()]
    nor = [str(w).lower().strip() for w in (normal or []) if str(w).strip()]
    den = [str(w).lower().strip() for w in (denied or []) if str(w).strip()]

    if den and any(w in t for w in den):
        return False
    if req and not all(w in t for w in req):
        return False
    if nor and not any(w in t for w in nor):
        return False
    return True


__all__ = ["match_interest"]
