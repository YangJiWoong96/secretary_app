from __future__ import annotations

"""
backend.proactive.source_selector - 도메인/소스 선택 에이전트

역할:
- 플래너 출력(info_needs)과 컨텍스트 요약(internal_contexts)을 바탕으로
  웹/날씨/유튜브 등 소스를 선택하고, 필요한 보조 정보(예: youtube_urls)를 상태에 추가한다.
"""

import re
from typing import Any, Dict, List, Tuple

from backend.proactive.schemas import ProactiveState

try:
    # 도메인 카탈로그: 정보 필요에 따른 시드 도메인
    from backend.proactive.domain_catalog import get_seed_domains  # type: ignore
except Exception:  # pragma: no cover
    get_seed_domains = None  # type: ignore


def _extract_youtube_urls(text: str) -> List[str]:
    urls: List[str] = []
    try:
        # 간단한 URL 정규식 (YouTube 전용)
        for m in re.finditer(
            r"(https?://(?:www\.)?(?:youtube\.com|youtu\.be)[^\s]+)",
            text or "",
            flags=re.I,
        ):
            u = m.group(1).strip().rstrip(").,]}")
            urls.append(u)
    except Exception:
        pass
    # 중복 제거
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _extract_tickers(text: str) -> List[str]:
    """
    보수적 티커 추출: $AAPL, $TSLA 같은 패턴만 인식.
    """
    out: List[str] = []
    try:
        for m in re.finditer(r"\$([A-Za-z]{1,5})\b", text or ""):
            t = m.group(1).upper()
            if t not in out:
                out.append(t)
    except Exception:
        pass
    return out


def _extract_two_coords(text: str) -> List[str]:
    """
    (lat,lng) 패턴이 2개 이상 존재하면, 경로 질의 문자열로 사용하기 위해 원문을 반환.
    멀티에이전트에서는 원문 전체를 교통 서비스에 전달해 좌표 파싱.
    """
    try:
        matches = re.findall(r"\(([^,]+),\s*([^)]+)\)", text or "")
        if len(matches) >= 2:
            return [text]
    except Exception:
        pass
    return []


def _should_include_weather(state: ProactiveState) -> bool:
    """
    매우 보수적인 규칙:
    - info_needs에 '여가' 또는 '건강'이 포함되거나
    - 모바일 컨텍스트에 '현재 위치' 또는 '비/우산/날씨' 키워드가 있을 때
    """
    needs = [str(x) for x in (state.get("info_needs") or [])]
    internal = state.get("internal_contexts", {}) or {}
    mobile = (internal.get("mobile_ctx") or "").lower()
    convo = (internal.get("conversation_ctx") or "").lower()
    if any(x in needs for x in ["여가", "건강"]):
        return True
    if any(k in mobile for k in ["현재 위치", "날씨", "비", "우산"]):
        return True
    if any(k in convo for k in ["날씨", "비", "우산"]):
        return True
    return False


def select_sources(state: ProactiveState) -> ProactiveState:
    """
    입력: ProactiveState (planner/context 이후)
    출력: selected_sources: List[str], youtube_urls: List[str]
    """
    internal = state.get("internal_contexts", {}) or {}
    text_blocks = [
        internal.get("rag_ctx") or "",
        internal.get("mobile_ctx") or "",
        internal.get("conversation_ctx") or "",
        internal.get("memory_ctx") or "",
    ]
    joined = "\n".join(text_blocks)
    youtube_urls = _extract_youtube_urls(joined)
    tickers = _extract_tickers(joined)
    traffic_queries = _extract_two_coords(joined)

    sources: List[str] = ["web"]  # 웹은 기본 선택
    if _should_include_weather(state):
        sources.append("weather")
    if youtube_urls:
        sources.append("youtube")
    # 재무: 티커가 감지된 경우
    if tickers:
        sources.append("finance")
    # 교통: 좌표쌍이 감지된 경우
    if traffic_queries:
        sources.append("traffic")

    # 정보 필요 기반 시드 도메인(있으면 주입)
    seeds: List[str] = []
    try:
        if get_seed_domains is not None:
            needs = [str(x) for x in (state.get("info_needs") or [])]
            # 내부 컨텍스트에서 최근 소비 도메인 힌트(있으면) 사용
            recent = []
            try:
                recent = list((internal.get("recent_domains") or []) or [])
            except Exception:
                recent = []
            seeds = get_seed_domains(needs, topk=6, recent_domains=recent)
    except Exception:
        seeds = []

    out: ProactiveState = {
        **state,
        "selected_sources": sources,
        "youtube_urls": youtube_urls,
        "finance_symbols": tickers,
        "traffic_queries": traffic_queries,
        "seed_domains": seeds,
    }
    return out


__all__ = ["select_sources"]
