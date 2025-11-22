"""
backend.generation.formatters - 응답 포맷팅 유틸리티

최종 답변 문자열 정리 및 [참고] 섹션 생성을 담당합니다.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from backend.config import get_settings


def cleanup_final_answer(text: str, show_citation: Optional[bool] = None) -> str:
    """
    최종 답변 문자열 정리.

    - Top-N 메타 표식 제거
    - [출처: ...] 표식은 SHOW_CITATION=true가 아니면 제거
    - 과다 개행 정리

    Args:
        text: 정리할 답변 문자열
        show_citation: 출처 표시 여부 (None이면 환경변수 참조)

    Returns:
        str: 정리된 답변 문자열
    """
    if not text:
        return ""

    try:
        # Top-N 메타 표식 제거
        text = re.sub(r"Top-\d+|\[Top \d+\]", "", text, flags=re.I)

        # 출처 표식 제거 (설정 기반)
        if show_citation is None:
            try:
                show_citation = bool(get_settings().SHOW_CITATION)
            except Exception:
                show_citation = False

        if not show_citation:
            text = re.sub(r"\[출처:.*?\]", "", text, flags=re.I)

        # 과다 개행 정리
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
    except Exception:
        return text


def format_references(web_ctx: str, max_refs: int = 5) -> str:
    """
    웹 컨텍스트에서 [참고] 섹션을 생성한다.

    Args:
        web_ctx: 웹 검색 컨텍스트 (3줄 블록 형식)
        max_refs: 최대 참고 항목 수

    Returns:
        str: [참고] 섹션 문자열(없으면 빈 문자열)
    """
    try:
        from backend.search_engine.formatter import blocks_to_items

        items = blocks_to_items(web_ctx or "")
        if not items:
            return ""

        lines = ["[참고]"]
        for item in items[:max_refs]:
            title = item.get("title", "").strip() or "(제목 없음)"
            desc = item.get("desc", "").strip() or "-"
            url = item.get("url", "").strip()
            if url:
                lines.append(f"- {title}: {desc}")
                lines.append(f"  {url}")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


def ensure_references_in_answer(final_answer: str, web_ctx: str) -> str:
    """웹 컨텍스트가 존재하면, 답변 끝에 [참고] 섹션으로 링크를 반드시 포함한다.

    - search_engine.formatter.blocks_to_items 형태(제목/설명/URL)를 사용
    - 이미 참고 섹션이 있다면 중복 추가하지 않음
    """
    try:
        if not (web_ctx or "").strip():
            return final_answer
        low = (final_answer or "").lower()
        if "[참고]" in low or "[references]" in low:
            return final_answer
        from backend.search_engine.formatter import blocks_to_items

        items = blocks_to_items(web_ctx)
        if not items:
            return final_answer
        lines = ["[참고]"]
        for it in items[:5]:
            title = (it.get("title") or "").strip()
            desc = (it.get("desc") or "").strip() or "-"
            url = (it.get("url") or "").strip()
            if title and url:
                lines.append(f"- {title}: {desc}")
                lines.append(f"  {url}")
        if len(lines) == 1:
            return final_answer
        sep = "\n\n" if final_answer and not final_answer.endswith("\n") else "\n"
        return (final_answer or "").rstrip() + sep + "\n".join(lines)
    except Exception:
        return final_answer


def enforce_web_results_in_answer(
    final_answer: str, web_ctx: str, top_k: int = 3
) -> str:
    """웹 컨텍스트에 있는 항목명이 최종 답변에 하나도 포함되지 않으면,
    안전하게 상위 결과를 이름/간단설명/URL 형식으로 제시한다.

    - LLM의 환각으로 존재하지 않는 장소명/잘못된 주소가 나오는 문제를 방지
    - web_ctx는 3줄 블록(title/desc/url) 규격을 가정
    """
    try:
        from backend.search_engine.formatter import blocks_to_items

        items = blocks_to_items(web_ctx or "")
        if not items:
            return final_answer or ""

        fa = (final_answer or "").strip()
        titles = [str(it.get("title") or "").strip() for it in items]
        if any(t and t in fa for t in titles):
            return fa

        lines: list[str] = []
        for it in items[: max(1, int(top_k))]:
            t = (it.get("title") or "").strip()
            d = (it.get("desc") or "").strip() or "-"
            u = (it.get("url") or "").strip()
            if not (t and u):
                continue
            lines.append(f"- {t}: {d}")
            lines.append(f"  {u}")

        if not lines:
            return fa

        return "다음은 추천 결과입니다:\n" + "\n".join(lines)
    except Exception:
        return final_answer or ""
