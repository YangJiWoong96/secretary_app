"""
backend.proactive.multi_agent_graph - 세션 1 멀티에이전트 그래프 정의

노드:
- Strategic Planner → info_needs 결정
- (세션1 범위) 컨텍스트/웹/합성/알림 노드는 스텁 처리

분기:
- should_generate_notification(state) 규칙을 함수로 제공
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.proactive.observability import emit_metric, ensure_trace_id
from backend.proactive.schemas import ProactiveState
from backend.proactive.strategic_planner import strategic_planner_node
from backend.utils.logger import log_event

try:
    # 선택 소스 에이전트(선택적)
    from backend.proactive.source_selector import select_sources as _select_sources
except Exception:  # pragma: no cover
    _select_sources = None  # type: ignore


def should_generate_notification(state: ProactiveState):
    """조건부 분기 규칙(적응형 임계 적용).
    - 기본 조건: confidence ≥ τ(state) AND (verified_facts ≥ 1 OR actionable_items ≥ 1)
    - 세션1에서는 confidence가 없으면 0.0로 취급

    참고: τ(state)는 컨텍스트 풍부도 및 검색 결과 수에 따라 0.65~0.75 사이에서 가변.
    """

    def _adaptive_confidence_threshold(s: ProactiveState) -> float:
        """컨텍스트/검색량 기반 적응형 임계값 계산.
        - RAG/모바일 컨텍스트가 충분히 길면 0.75
        - 웹 결과가 부족(<5)하면 0.65
        - 기본 0.70
        """
        try:
            web_cnt = len(s.get("web_results", []) or [])
            internal = s.get("internal_contexts", {}) or {}
            rag_len = len((internal.get("rag_ctx") or ""))
            mob_len = len((internal.get("mobile_ctx") or ""))
            if rag_len > 500 and mob_len > 200:
                return 0.75
            if web_cnt < 5:
                return 0.65
        except Exception:
            # 예외 시 기본값
            pass
        return 0.70

    confidence = float(state.get("confidence_score", 0.0) or 0.0)
    vf = len(state.get("verified_facts", []) or [])
    ai = len(state.get("actionable_items", []) or [])
    thr = _adaptive_confidence_threshold(state)
    if confidence >= thr and (vf >= 1 or ai >= 1):
        return "notification_generator"
    return END


# ===== 스텁 노드(세션1 범위에서 파이프라인 연결만 보장) =====


def _context_analyzer_node(state: ProactiveState) -> ProactiveState:
    """세션2 구현 호출: ContextAnalyzerAgent로 실제 컨텍스트 수집/분석 수행."""
    t0 = time.perf_counter_ns()
    try:
        from backend.proactive.context_analyzer import ContextAnalyzerAgent
        from backend.proactive.data_contracts import ContextAnalyzerInput

        agent = ContextAnalyzerAgent()
        inp: ContextAnalyzerInput = {
            "user_id": state.get("user_id", ""),
            "session_id": state.get("session_id", ""),
            "timestamp": state.get("timestamp", ""),
        }
        # LangGraph sync 실행 맥락 → 이벤트 루프 보장 위해 to_thread 사용
        out_ctx = asyncio.run(agent.analyze(inp))
        out: ProactiveState = {
            **state,
            "internal_contexts": out_ctx.get("internal_contexts", {}),
            "context_insights": out_ctx.get("context_insights", []),
        }
    except Exception:
        out = {
            **state,
            "internal_contexts": state.get("internal_contexts", {}) or {},
            "context_insights": state.get("context_insights", []) or [],
        }

    took_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    lat = dict(out.get("agent_latencies", {}) or {})
    lat["context"] = took_ms
    out["agent_latencies"] = lat
    try:
        log_event(
            "proactive.agent_complete",
            {
                "agent": "context",
                "trace_id": state.get("trace_id"),
                "took_ms": took_ms,
                "confidence": float(out.get("confidence_score", 0.0) or 0.0),
                "output_size": len(
                    (out.get("internal_contexts") or {}).get("rag_ctx", "")
                ),
            },
        )
        try:
            emit_metric("proactive.agent.latency", took_ms, {"agent": "context"})
        except Exception:
            pass
    except Exception:
        pass
    return out


def _web_research_node(state: ProactiveState) -> ProactiveState:
    """세션3: WebResearchAgent 실제 호출.

    - info_needs 상위 항목을 쿼리로 사용(없으면 보수적 기본값)
    - trust_score ≥ 0.7 필터, 일관성 점수 계산
    - (확장) source_selector가 지정한 도메인 소스(weather/youtube 등)도 함께 취합
    """
    t0 = time.perf_counter_ns()
    queries = list(state.get("web_queries", []) or [])

    if not queries:
        needs = list(state.get("info_needs", []) or [])
        if needs:
            head = needs[0]
            queries = [f"{head} 최신 동향"]
        else:
            queries = ["오늘 주요 뉴스"]

    # 한 개 쿼리만 보수적으로 수행
    q = queries[0]

    # 선택된 소스(없으면 기본 web)
    selected_sources = list(state.get("selected_sources", []) or []) or ["web"]

    web_results_all: list[dict] = []
    try:
        # 1) 범용 웹 리서치
        if "web" in selected_sources:
            from backend.proactive.web_researcher import WebResearchAgent

            agent = WebResearchAgent()
            result = asyncio.run(
                agent.research(
                    query=q,
                    context_hints=list(state.get("info_needs", []) or []),
                    max_sources=8,
                    timeout_sec=2.0,
                    seed_domains=list(state.get("seed_domains", []) or []),
                    seed_urls=list(state.get("seed_urls", []) or []),
                )
            )
            web_results_all.extend(list(result.get("results", []) or []))
    except Exception:
        pass

    # 2) 날씨(주소/지명 기반 블록 → 결과 구조화)
    try:
        if "weather" in selected_sources:
            from backend.services.weather import (
                get_weather_text as _w_text,
                get_weather_data as _w_data,
            )
            from backend.rag.retrieval import retrieve_enhanced as _ret_enh

            # 모바일 컨텍스트에서 주소를 보수적으로 파싱
            internal = state.get("internal_contexts", {}) or {}
            mobile_ctx = internal.get("mobile_ctx") or ""
            addr = ""
            try:
                import re as _re

                m = _re.search(r"현재 위치:\s*(.+)", mobile_ctx)
                if m:
                    addr = m.group(1).strip()
            except Exception:
                addr = ""
            wq = addr or "현재 위치 날씨"
            # 멀티에이전트: 3줄 제한 없이 전문 텍스트를 사용
            wtxt = asyncio.run(_w_text(wq))
            wdat = None
            try:
                wdat = asyncio.run(_w_data(wq))
            except Exception:
                wdat = None
            if wtxt:
                web_results_all.append(
                    {
                        "url": "",
                        "title": f"날씨 정보: {wq}",
                        "excerpt": wtxt[:500].strip(),
                        "content": wtxt,
                        "data": (wdat or {}),
                        "trust_score": 0.8,
                        "domain": "open-meteo",
                        "source_type": "weather",
                    }
                )
    except Exception:
        pass

    # 3) 유튜브(URL 목록 → 전체 텍스트 사용: 자막/폴백 STT)
    try:
        if "youtube" in selected_sources:
            from backend.services.youtube import (
                get_youtube_text as _yt_text,
                get_youtube_block as _yt_block,
            )

            try:
                from backend.config import get_settings as _gs

                yt_cap = int(_gs().MA_YT_CONTENT_MAX_CHARS)
            except Exception:
                yt_cap = 20000

            for u in list(state.get("youtube_urls", []) or [])[:3]:
                # 제목은 블록에서, 본문은 전체 텍스트에서 확보
                try:
                    blk = asyncio.run(_yt_block(u))
                except Exception:
                    blk = ""
                title = ""
                if blk:
                    lines = [ln for ln in blk.split("\n") if ln.strip()]
                    if lines:
                        title = lines[0].strip()
                # 전체 텍스트
                try:
                    full_txt = asyncio.run(_yt_text(u))
                except Exception:
                    full_txt = ""
                if not (title or full_txt):
                    continue
                content_txt = full_txt[:yt_cap] if yt_cap > 0 else full_txt
                excerpt = full_txt[:500].strip() if full_txt else "-"
                url = u
                dom = "youtube.com"
                web_results_all.append(
                    {
                        "url": url,
                        "title": title or "YouTube 영상",
                        "excerpt": excerpt,
                        "content": content_txt,
                        "trust_score": 0.85,
                        "domain": dom,
                        "source_type": "video",
                    }
                )
    except Exception:
        pass

    # 4) 재무(티커 → 전문 텍스트)
    try:
        if "finance" in selected_sources:
            from backend.services.finance import (
                get_finance_text as _fin_text,
                get_finance_data as _fin_data,
                get_finance_earnings_text as _fin_earn,
                get_finance_news_text as _fin_news,
                get_finance_sector_text as _fin_sector,
            )

            for sym in list(state.get("finance_symbols", []) or [])[:3]:
                try:
                    ftxt = asyncio.run(_fin_text(sym))
                except Exception:
                    ftxt = ""
                try:
                    fdat = asyncio.run(_fin_data(sym))
                except Exception:
                    fdat = None
                if ftxt:
                    web_results_all.append(
                        {
                            "url": "",
                            "title": f"주가 정보: {sym}",
                            "excerpt": ftxt[:500].strip(),
                            "content": ftxt,
                            "data": (fdat or {}),
                            "trust_score": 0.8,
                            "domain": "alphavantage",
                            "source_type": "finance",
                        }
                    )
                # 실적/뉴스/섹터 비교 추가(있으면 추가)
                try:
                    earn = asyncio.run(_fin_earn(sym))
                    if earn:
                        web_results_all.append(
                            {
                                "url": "",
                                "title": f"실적 일정: {sym}",
                                "excerpt": earn[:500].strip(),
                                "content": earn,
                                "trust_score": 0.75,
                                "domain": "alphavantage",
                                "source_type": "finance",
                            }
                        )
                except Exception:
                    pass
                try:
                    news = asyncio.run(_fin_news(sym))
                    if news:
                        web_results_all.append(
                            {
                                "url": "",
                                "title": f"뉴스: {sym}",
                                "excerpt": news[:500].strip(),
                                "content": news,
                                "trust_score": 0.7,
                                "domain": "alphavantage",
                                "source_type": "finance",
                            }
                        )
                except Exception:
                    pass
                try:
                    sector = asyncio.run(_fin_sector(sym))
                    if sector:
                        web_results_all.append(
                            {
                                "url": "",
                                "title": f"섹터 비교: {sym}",
                                "excerpt": sector[:500].strip(),
                                "content": sector,
                                "trust_score": 0.7,
                                "domain": "alphavantage",
                                "source_type": "finance",
                            }
                        )
                except Exception:
                    pass
    except Exception:
        pass

    # 5) 교통(좌표쌍 → 전문 텍스트)
    try:
        if "traffic" in selected_sources:
            from backend.services.traffic import (
                get_traffic_text as _t_text,
                get_traffic_data as _t_data,
            )

            for q in list(state.get("traffic_queries", []) or [])[:2]:
                try:
                    ttxt = asyncio.run(_t_text(q))
                except Exception:
                    ttxt = ""
                try:
                    tdat = asyncio.run(_t_data(q))
                except Exception:
                    tdat = None
                if ttxt:
                    web_results_all.append(
                        {
                            "url": "",
                            "title": "교통 경로",
                            "excerpt": ttxt[:500].strip(),
                            "content": ttxt,
                            "data": (tdat or {}),
                            "trust_score": 0.75,
                            "domain": "osrm",
                            "source_type": "traffic",
                        }
                    )
    except Exception:
        pass

    # 도메인별 대표 신뢰도 맵
    try:
        src_cred: dict[str, float] = {}
        for it in web_results_all:
            d = (it.get("domain") or "").lower()
            ts = float(it.get("trust_score", 0.0) or 0.0)
            if d:
                src_cred[d] = max(ts, src_cred.get(d, 0.0))
        out: ProactiveState = {
            **state,
            "web_queries": queries,
            "web_results": web_results_all,
            "source_credibility": src_cred,
        }
    except Exception:
        out = {
            **state,
            "web_queries": queries,
            "web_results": state.get("web_results", []) or [],
            "source_credibility": state.get("source_credibility", {}) or {},
        }

    took_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    lat = dict(out.get("agent_latencies", {}) or {})
    lat["web"] = took_ms
    out["agent_latencies"] = lat
    try:
        log_event(
            "proactive.agent_complete",
            {
                "agent": "web",
                "trace_id": state.get("trace_id"),
                "took_ms": took_ms,
                "confidence": float(out.get("confidence_score", 0.0) or 0.0),
                "output_size": len(out.get("web_results", []) or []),
            },
        )
        try:
            emit_metric("proactive.agent.latency", took_ms, {"agent": "web"})
        except Exception:
            pass
    except Exception:
        pass
    return out


def _synthesizer_node(state: ProactiveState) -> ProactiveState:
    """세션3: SynthesizerAgent 실제 호출.

    - 내부 컨텍스트 × 웹 결과 교차 검증
    - Verified Facts / SMART Actions 생성
    - confidence_score 집계
    """
    t0 = time.perf_counter_ns()
    try:
        from backend.proactive.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent()
        out = agent.synthesize(state)
    except Exception:
        out = {
            **state,
            "synthesized_insights": state.get("synthesized_insights", []) or [],
            "verified_facts": state.get("verified_facts", []) or [],
            "actionable_items": state.get("actionable_items", []) or [],
            "confidence_score": float(state.get("confidence_score", 0.0) or 0.0),
        }

    took_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    lat = dict(out.get("agent_latencies", {}) or {})
    lat["synthesizer"] = took_ms
    out["agent_latencies"] = lat
    try:
        log_event(
            "proactive.agent_complete",
            {
                "agent": "synthesizer",
                "trace_id": state.get("trace_id"),
                "took_ms": took_ms,
                "confidence": float(out.get("confidence_score", 0.0) or 0.0),
                "output_size": len(out.get("verified_facts", []) or [])
                + len(out.get("actionable_items", []) or []),
            },
        )
        try:
            emit_metric("proactive.agent.latency", took_ms, {"agent": "synthesizer"})
        except Exception:
            pass
    except Exception:
        pass
    return out


def _notification_generator_node(state: ProactiveState) -> ProactiveState:
    """세션4 구현 호출: NotificationGenerator로 최종 알림 생성."""
    t0 = time.perf_counter_ns()
    try:
        from backend.proactive.notification_generator import NotificationGenerator

        gen = NotificationGenerator()
        actions = list(state.get("actionable_items", []) or [])
        internal = state.get("internal_contexts", {}) or {}
        cands, final = asyncio.run(
            asyncio.wait_for(
                gen.generate(
                    actions,
                    user_id=state.get("user_id", ""),
                    session_id=state.get("session_id", ""),
                    internal_contexts=internal,
                    verified_facts=state.get("verified_facts", []) or [],
                    base_confidence=float(state.get("confidence_score", 0.0) or 0.0),
                ),
                timeout=0.4,
            )
        )
        out: ProactiveState = {
            **state,
            "notification_candidates": cands,
            "final_notification": (final.dict() if hasattr(final, "dict") else final),
        }
    except Exception:
        out = {
            **state,
            "notification_candidates": state.get("notification_candidates", []) or [],
            "final_notification": state.get("final_notification", None),
        }

    took_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    lat = dict(out.get("agent_latencies", {}) or {})
    lat["notifier"] = took_ms
    out["agent_latencies"] = lat
    try:
        log_event(
            "proactive.agent_complete",
            {
                "agent": "notifier",
                "trace_id": state.get("trace_id"),
                "took_ms": took_ms,
                "confidence": float(out.get("confidence_score", 0.0) or 0.0),
                "output_size": len(out.get("notification_candidates", []) or []),
            },
        )
        try:
            emit_metric("proactive.agent.latency", took_ms, {"agent": "notifier"})
        except Exception:
            pass
    except Exception:
        pass
    return out


def build_proactive_graph() -> StateGraph[ProactiveState]:
    """LangGraph 기반 멀티에이전트 그래프 빌더 반환."""
    g = StateGraph(ProactiveState)

    # 노드 등록
    g.add_node("strategic_planner", strategic_planner_node)
    g.add_node("context_analyzer", _context_analyzer_node)
    if _select_sources is not None:
        g.add_node("source_selector", lambda s: {**s, **_select_sources(s)})
    g.add_node("web_research", _web_research_node)
    g.add_node("synthesizer", _synthesizer_node)
    g.add_node("notification_generator", _notification_generator_node)

    # 진입점 → 플래너 (trace_id 보장 래퍼)
    g.set_entry_point("strategic_planner")
    g.add_edge("strategic_planner", "context_analyzer")
    if _select_sources is not None:
        g.add_edge("context_analyzer", "source_selector")
        g.add_edge("source_selector", "web_research")
    else:
        g.add_edge("context_analyzer", "web_research")
    g.add_edge("web_research", "synthesizer")

    # 조건부 분기
    g.add_conditional_edges("synthesizer", should_generate_notification)

    # (옵션) 도메인 라우팅 경유: 전략 이후 도메인 태깅 → 컨텍스트 분석으로 전파
    from backend.config import get_settings as _gs

    if bool(_gs().FEATURE_DOMAIN_ROUTING):
        from backend.proactive.domain_router import route_domain

        g.add_node("domain_router", lambda s: {**s, "domain": route_domain(s)})
        g.add_edge("strategic_planner", "domain_router")
        # 간소화: 도메인 경로별 특화 노드는 이후 단계에서 확장. 현재는 동일 노드 재사용.
        g.add_edge("domain_router", "context_analyzer")

    # 노티/종료 연결
    g.add_edge("notification_generator", END)

    return g


# 체크포인트 메모리(옵션)
_checkpoint = MemorySaver()
GRAPH = build_proactive_graph().compile(checkpointer=_checkpoint)


__all__ = [
    "ProactiveState",
    "build_proactive_graph",
    "should_generate_notification",
    "GRAPH",
]


def run_proactive_pipeline(initial: ProactiveState) -> ProactiveState:
    """동기 실행 헬퍼: 파이프라인 시작/종료를 관측성 이벤트로 남긴다.

    세션1 범위에서 최소한의 실행 진입점을 제공한다.
    """
    try:
        # Trace ID 보장 및 전파
        try:
            trace_id = ensure_trace_id(initial)
            initial["trace_id"] = trace_id
        except Exception:
            pass
        log_event(
            "proactive.pipeline_start",
            {
                "session_id": initial.get("session_id"),
                "user_id": initial.get("user_id"),
                "trace_id": initial.get("trace_id"),
                "trigger_type": initial.get("trigger_type", "manual"),
                "timestamp": initial.get("timestamp", ""),
            },
        )
    except Exception:
        pass

    try:
        final_state: ProactiveState = GRAPH.invoke(initial)
        try:
            log_event(
                "proactive.pipeline_complete",
                {
                    "session_id": final_state.get("session_id"),
                    "trace_id": final_state.get("trace_id"),
                    "confidence": float(
                        final_state.get("confidence_score", 0.0) or 0.0
                    ),
                },
            )
        except Exception:
            pass
        return final_state
    except Exception as e:
        try:
            log_event(
                "proactive.pipeline_error",
                {
                    "session_id": initial.get("session_id"),
                    "error_type": "pipeline_exception",
                    "agent": "graph",
                    "message": repr(e),
                    "recoverable": False,
                },
            )
        except Exception:
            pass
        raise
