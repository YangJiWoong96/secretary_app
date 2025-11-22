from __future__ import annotations

"""
도메인 라우터 — 상태의 태그를 바탕으로 도메인을 결정한다.

반환:
- "finance" | "health" | "general"
"""

from typing import Dict


def route_domain(state: Dict) -> str:
    """
    상태의 태그를 바탕으로 도메인을 결정한다.

    Returns:
        "finance" | "health" | "general"
    """
    tags = set([str(t).lower() for t in (state.get("tags") or [])])
    if {"finance", "stock", "fx"} & tags:
        return "finance"
    if {"health", "sleep", "diet"} & tags:
        return "health"
    return "general"


__all__ = ["route_domain"]
