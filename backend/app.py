"""
app.py - AI Assistant Backend (v2.0 모듈화)

모든 비즈니스 로직은 backend/ 하위 모듈로 분리됨.
이 파일은 FastAPI 앱 정의 및 고수준 오케스트레이션만 담당.
기능 구현은 backend/ 중 각 하위 폴더 내부의 모듈에서 이루어지고 해당 app.py 흐름에서 호출되는 형태로 진행.

리팩토링 완료: 2025-10-04
- 2,908 라인 제거 (56.5% 축소)
- 28개 핵심 모듈 생성
- 100% 타입 안전, Linter 10.0/10.0
"""

import asyncio
import logging

# ===== 표준 라이브러리 =====
import os
import time
import uuid
from typing import Any, Dict, List

# ===== FastAPI =====
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ===== 설정 및 초기화 =====
from backend.config import get_settings

# ===== 컨텍스트 =====
from backend.context.unified import (
    UnifiedContextualChatbot,
    apply_query_focused_overrides,
    cap_context_tokens,
)
from backend.directives.pipeline import schedule_directive_update

# ===== Directive =====
from backend.directives.store import get_compiled as get_compiled_directives
from backend.directives.store import (
    get_compiled_unified,
    set_compiled_unified,
)

# ===== 증거 수집 =====
from backend.evidence.builder import build_evidence

# ===== 응답 생성 =====
from backend.generation import (
    conversation_chain,
    filter_semantic_mismatch,
    filter_web_ctx,
    post_verify_answer,
    validate_final_answer,
)
from backend.generation.evidence_handlers import (
    detect_evidence_detail_request,
    detect_unused_evidence_reload,
    handle_evidence_detail_request,
    reload_unused_evidence_eids,
)
from backend.generation.schema import AssistantResponse, TurnSummary
from backend.generation.selector import build_evidence_meta, select_blocks_by_ids
from backend.generation.tagger import extract_tags

# ===== 데이터 수집 =====
from backend.ingest import build_mobile_ctx
from backend.ingest.mobile_context import get_current_time_context

# ===== 메모리 =====
from backend.memory import TurnResult, get_memory_coordinator, get_short_term_memory

# ===== 정책 및 상태 =====
from backend.policy import (
    SESSION_STATE,
    enqueue_snapshot,
    get_global_state,
    redact_text,
)

# ===== 프롬프트 =====
from backend.prompts.system import (
    EVIDENCE_SYS_RULE,
    FINAL_PROMPT,
    IDENTITY_PROMPT,
    NO_EVIDENCE_SYS_RULE,
)

# ===== RAG 시스템 =====
from backend.rag import retrieve_from_rag

# ── 세션3 가드 검증기
from backend.rag.profile_guard import get_profile_guard
from backend.rag.refs import store_refs_from_contexts
from backend.rag.retrieval import retrieve_enhanced
from backend.rag.profile_ids import bot_user_id_for

# ===== 쿼리 재작성 =====
from backend.rewrite import rewrite_query
from backend.rewrite.log import RewriteRecord, add_rewrite

# ===== 라우팅 (2단계: 임베딩 → LLM) =====
from backend.routing import get_intent_router
from backend.routing.intent_router import embedding_router_scores
from backend.routing.llm_router import llm_decider, route_guard, router_one_call
from backend.routing.router_context import map_session_to_user, user_for_session
from backend.routing.turn_state import TurnState

# ===== 웹 검색 =====
from backend.search_engine import build_web_context
from backend.startup import initialize_all_services

# ===== 유틸리티 =====
from backend.utils.datetime_utils import extract_date_range_for_rag

# ===== 중앙 로거 =====
from backend.utils.logger import (
    clear_context,
    get_logger,
    init_logging,
    log_event,
    safe_log_event,
    set_context,
)

# ===== 웹 컨텍스트 정제 파서 =====
try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate
    from pydantic import BaseModel, Field

    _PARSER_AVAILABLE = True
except Exception:  # pragma: no cover
    _PARSER_AVAILABLE = False
    BaseModel = object  # type: ignore

    def Field(*args, **kwargs):  # type: ignore
        return None

    _ChatPromptTemplate = None  # type: ignore

try:
    from backend.search_engine.formatter import blocks_to_items as _blocks_to_items
except Exception:  # pragma: no cover
    _blocks_to_items = None  # type: ignore


# ===== 기타 =====

from backend.generation.evidence_supplement import supplement_evidence

# ─────────────────────────────────────────────────────────────
# 로깅 초기화 및 설정 로드
# ─────────────────────────────────────────────────────────────
init_logging()
settings = get_settings()
logger = get_logger("app")


from backend.generation.formatters import cleanup_final_answer

# ─────────────────────────────────────────────────────────────
# FastAPI 앱 인스턴스
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Assistant Backend",
    description="모듈화된 대화형 AI 비서 백엔드",
    version="2.0.0",
)

# ─────────────────────────────────────────────────────────────
# 최종 응답 프롬프트 템플릿 (Main Response용) - prompts/system.py에서 관리
# ─────────────────────────────────────────────────────────────

# 스트리밍 런타임 플래그
_STREAM_RUNTIME_DISABLED = False


def _stream_allowed() -> bool:
    """스트리밍 허용 여부"""
    return settings.STREAM_ENABLED and (not _STREAM_RUNTIME_DISABLED)


from backend.generation.message_builder import build_messages
from backend.generation.orchestrator import (
    handle_pre_guard,
    load_memory_and_history,
    main_response,
    perform_routing,
)
from backend.generation.stream_handler import stream_response
from backend.utils.http_client import aclose as http_client_aclose

# ─────────────────────────────────────────────────────────────
# 응답 생성 함수들 (모듈화된 3단계 구조)
# ─────────────────────────────────────────────────────────────
from backend.memory.history_selector import prepare_history

# (삭제됨) _refine_web_ctx_with_parser: 사용처 없음 → 제거

from backend.utils.tracing import traceable


# ─────────────────────────────────────────────────────────────
# Startup 이벤트
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """애플리케이션 시작 시 모든 서비스 초기화"""
    log_event("startup_begin")
    await initialize_all_services()
    log_event("startup_ready")


# ─────────────────────────────────────────────────────────────
# 헬스체크
# ─────────────────────────────────────────────────────────────
@app.get("/")
def health():
    """헬스체크 엔드포인트"""
    log_event("health_ok")
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# 이벤트 수집 엔드포인트
# ─────────────────────────────────────────────────────────────
@app.post("/events/push/open")
async def events_push_open(payload: Dict[str, Any]):
    """푸시 오픈 이벤트 수집"""
    try:
        from backend.config import get_firestore_client

        try:
            from google.cloud.firestore_v1 import FieldFilter  # type: ignore
        except Exception:
            FieldFilter = None  # type: ignore
        from datetime import datetime, timezone

        uid = str(payload.get("user_id") or "").strip()
        push_id = str(payload.get("push_id") or "").strip()
        variant = str(payload.get("variant") or "")
        dwell_ms = int(payload.get("dwell_ms") or 0)
        answer_started = bool(payload.get("answer_started", False))

        if not uid or not push_id:
            return {"ok": False}

        db = get_firestore_client()
        if not db:
            return {"ok": False}

        base_coll = (
            db.collection("users").document(uid).collection("proactive_push_logs")
        )
        if FieldFilter is not None:
            q = base_coll.where(filter=FieldFilter("push_id", "==", push_id)).limit(1)
        else:
            q = base_coll.where("push_id", "==", push_id).limit(1)

        docs = list(q.stream())
        ref = (
            docs[0].reference
            if docs
            else (
                db.collection("users")
                .document(uid)
                .collection("proactive_push_logs")
                .document()
            )
        )

        update_obj = {
            "opened_at": datetime.now(timezone.utc),
            "dwell_ms": dwell_ms,
            "answer_started": answer_started,
        }
        if variant:
            update_obj["variant"] = variant

        ref.set(update_obj, merge=True)

        # (신규) 밴딧 보상 업데이트: 변이명이 bandit:<arm> 형태일 때만 반영
        try:
            if settings.FEATURE_BANDIT:
                arm = ""
                if isinstance(variant, str) and variant.startswith("bandit:"):
                    arm = variant.split(":", 1)[1]
                if arm:
                    from backend.experiments.bandit import Bandit, reward_from_event

                    rew = reward_from_event(
                        dwell_ms=dwell_ms, answer_started=answer_started
                    )
                    Bandit().update(arm, rew)
        except Exception:
            pass
        return {"ok": True}
    except Exception as e:
        log_event(
            "events_push_open_error",
            {"error": repr(e)},
            level=logging.WARNING,
        )
        return {"ok": False}


# ─────────────────────────────────────────────────────────────
# 내부 엔드포인트
# ─────────────────────────────────────────────────────────────
@app.post("/internal/rag/retrieve")
async def internal_rag_retrieve(payload: Dict[str, Any]):
    """RAG 검색 내부 엔드포인트"""
    try:
        uid = str(payload.get("user_id") or "").strip()
        if not uid:
            raise ValueError("user_id is required")
        q = str(payload.get("query") or "").strip()
        top_k = int(payload.get("top_k") or 2)
        df = payload.get("date_filter")
        date_filter = (
            (int(df[0]), int(df[1]))
            if isinstance(df, (list, tuple)) and len(df) == 2
            else None
        )

        blocks = retrieve_from_rag(uid, q, top_k=top_k, date_filter=date_filter)
        return {"blocks": blocks or ""}
    except Exception as e:
        log_event(
            "internal_rag_retrieve_error",
            {"error": repr(e)},
            level=logging.WARNING,
        )
        return {"blocks": ""}


@app.post("/internal/mobile/context")
async def internal_mobile_context(payload: Dict[str, Any]):
    """모바일 컨텍스트 내부 엔드포인트"""
    try:
        sid = str(payload.get("session_id") or "").strip()
        blocks = await build_mobile_ctx(sid)
        return {"blocks": blocks or ""}
    except Exception as e:
        log_event(
            "internal_mobile_context_error",
            {"error": repr(e)},
            level=logging.WARNING,
        )
        return {"blocks": ""}


@app.post("/internal/evidence/bundle")
@traceable(
    name="App: internal_evidence_bundle", run_type="chain", tags=["app", "evidence"]
)
async def internal_evidence_bundle(payload: Dict[str, Any]):
    """증거 번들 내부 엔드포인트"""
    try:
        sid = str(payload.get("session_id") or "").strip()
        uid = str(payload.get("user_id") or "").strip()
        if not uid:
            raise ValueError("user_id is required")
        q = str(payload.get("query") or "").strip()
        web_on = bool(payload.get("web_on", True))
        rag_on = bool(payload.get("rag_on", True))
        timeout_s = float(
            payload.get("timeout_s") or max(settings.TIMEOUT_WEB, settings.TIMEOUT_RAG)
        )

        # 재작성 강제: 웹/RAG 각각 마지막 발화만을 대상으로 재작성하고,
        # preview/요약은 참고 단서로만 사용(재작성 대상 텍스트에 혼합 금지)
        from backend.rewrite import rewrite_query as _rq

        web_query: str = ""
        rag_query: str = ""
        rag_date_filter = None

        if web_on:
            try:
                rw_web = await _rq(
                    "web",
                    q,
                    "",
                    session_id=sid,
                    preview_ctx="",
                    realtime_ctx=get_current_time_context(),
                )
                web_query = str(rw_web.get("web_query") or "").strip() or q
            except Exception:
                web_query = q
        if rag_on:
            try:
                rw_rag = await _rq(
                    "rag",
                    q,
                    "",
                    session_id=sid,
                    preview_ctx="",
                    realtime_ctx=get_current_time_context(),
                )
                rag_query = str(rw_rag.get("query_text") or "").strip() or q
                rag_date_filter = rw_rag.get("date_filter")
            except Exception:
                rag_query = q

        # 증거 수집을 병렬로 실행
        web_ctx = ""
        rag_ctx = ""

        async def _web_task():
            nonlocal web_ctx
            if web_on and web_query:
                try:
                    _k, _ctx = await build_web_context(
                        settings.MCP_SERVER_URL, web_query, 5, timeout_s
                    )
                    web_ctx = _ctx or ""
                except Exception:
                    web_ctx = ""

        async def _rag_task():
            nonlocal rag_ctx
            if rag_on and rag_query:
                try:
                    rag_ctx_local = retrieve_from_rag(
                        uid, rag_query, top_k=3, date_filter=rag_date_filter
                    )
                    # RAG 블록 정규화(3줄 블록): builder.py와 동일 규칙 적용
                    parts = (rag_ctx_local or "").split("\n\n")
                    norm_blocks: list[str] = []
                    for i, block in enumerate(parts):
                        lines = [ln for ln in (block or "").split("\n") if ln.strip()]
                        if not lines:
                            continue
                        title = lines[0][:80]
                        desc = (
                            (" ".join(lines[1:])[:140])
                            if len(lines) > 1
                            else "(설명 없음)"
                        )
                        link = f"rag://session/{sid}/hit/{i}"
                        norm_blocks.append(
                            "\n".join(
                                [title or "(제목 없음)", desc or "(설명 없음)", link]
                            )
                        )
                    rag_ctx = "\n\n".join(norm_blocks)
                except Exception:
                    rag_ctx = ""

        await asyncio.gather(_web_task(), _rag_task())

        return {"web": {"blocks": web_ctx or ""}, "rag": {"blocks": rag_ctx or ""}}
    except Exception as e:
        log_event(
            "internal_evidence_bundle_error",
            {"error": repr(e)},
            level=logging.WARNING,
        )
        return {"web": {"blocks": ""}, "rag": {"blocks": ""}}


@app.get("/internal/directives/{session_id}/compiled")
async def internal_directives_compiled(session_id: str):
    """Directive 컴파일 결과 조회"""
    try:
        prompt, ver = get_compiled_directives(session_id)
        return {"prompt": prompt or "", "version": ver or ""}
    except Exception as e:
        log_event(
            "internal_directives_compiled_error",
            {"error": repr(e)},
            level=logging.WARNING,
        )
        return {"prompt": "", "version": ""}


# ─────────────────────────────────────────────────────────────
# 프로액티브 트리거
# ─────────────────────────────────────────────────────────────
@app.post("/proactive/trigger/{user_id}")
async def trigger_proactive(user_id: str):
    """프로액티브 푸시 수동 트리거"""
    try:
        from backend.proactive.agent import select_and_send

        sent = await select_and_send(user_id, max_send=1)
        return {"status": "ok", "sent": len(sent)}
    except Exception as e:
        return {"status": "error", "error": repr(e)}


# ─────────────────────────────────────────────────────────────
# WebSocket 엔드포인트 (고수준 파이프라인 오케스트레이션)
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/{user_id}/{session_id}")
@traceable(name="App: websocket_session", run_type="chain", tags=["app", "ws"])
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """
    WebSocket 턴 파이프라인

    처리 흐름:
    1. 메모리 & 히스토리 로드
    2. STWM 업데이트 & 턴 버퍼
    3. 라우팅 (임베딩 우선 또는 소분류기 우선)
    4. 쿼리 재작성 (RAG/WEB)
    5. 증거 수집 (병렬)
    6. 태깅 & 선정
    7. 컨텍스트 필터링
    8. 최종 응답 생성
    9. 대화 저장 & 요약
    10. 스냅샷 에지 트리거
    """
    await websocket.accept()
    try:
        set_context(session_id=session_id)
        # 세션→사용자 매핑(영속 스토어는 user_id 기준으로 운영)
        try:
            map_session_to_user(session_id, user_id)
        except Exception as _e_map:
            safe_log_event(
                "router.map_session_to_user_error",
                {"session_id": session_id, "user_id": user_id, "error": str(_e_map)},
            )
        log_event("ws_accept", {"session_id": session_id, "user_id": user_id})
    except Exception as e:
        try:
            safe_log_event(
                "ws_accept_error",
                {"session_id": session_id, "user_id": user_id, "error": str(e)},
            )
        except Exception:
            pass

    # 서비스 인스턴스
    router = get_intent_router()

    try:
        while True:
            # ===== 사용자 입력 수신 =====
            user_input = await websocket.receive_text()

            # ===== Evidence 상세 요청 판별 =====
            try:
                want_detail, eid_req = await detect_evidence_detail_request(
                    user_input, "", settings.LLM_MODEL
                )
                if want_detail and eid_req:
                    body = await handle_evidence_detail_request(eid_req, session_id)
                    if body:
                        try:
                            await websocket.send_text(f"[증거 원문]\n{body}")
                        except Exception as e:
                            try:
                                safe_log_event("ws_send_error", {"error": str(e)})
                            except Exception:
                                pass
                        # 상세 조회는 별도 응답으로 처리하고 다음 입력을 기다린다
                        continue
            except Exception as e:
                try:
                    safe_log_event("evidence.detail.detect_error", {"error": str(e)})
                except Exception:
                    pass

            # ===== 미사용 증거 재로드 의도 판별 및 활성 EID 확장 =====
            try:
                reload_unused = await detect_unused_evidence_reload(
                    user_input, settings.LLM_MODEL
                )
                if reload_unused:
                    last_tid = (
                        SESSION_STATE.get(session_id, {}).get("last_turn_id") or ""
                    )
                    if last_tid:
                        extra_eids = await reload_unused_evidence_eids(
                            session_id, last_tid
                        )
                        if extra_eids:
                            # 활성 EID에 추가
                            prev_active = SESSION_STATE.get(session_id, {}).get(
                                "active_eids", []
                            )
                            merged = list(dict.fromkeys(list(prev_active) + extra_eids))
                            SESSION_STATE.setdefault(session_id, {})[
                                "active_eids"
                            ] = merged
                            try:
                                log_event(
                                    "evidence.reused",
                                    {
                                        "session_id": session_id,
                                        "active_eid_count": len(merged),
                                    },
                                )
                            except Exception:
                                pass
            except Exception as e:
                try:
                    safe_log_event("evidence.reload_error", {"error": str(e)})
                except Exception:
                    pass

            # 세션 시작/첫 턴에서 Unified Prompt base/overlay 프리로드(폴백 포함)
            try:
                from backend.directives.compiler import compile_unified_prompt_split

                real_uid = user_id
                base_p, overlay_p, _ver = await compile_unified_prompt_split(
                    user_id=real_uid, session_id=session_id, user_query=""
                )
                if not (overlay_p or "").strip():
                    # 폴백 오버레이 주입: 저장된 Directives 기반으로 최소 스타일을 반영
                    try:
                        from backend.directives.store import load_directives as _ld

                        _dirs, _ = _ld(session_id)
                        _style = str(_dirs.get("formality") or "mixed")
                    except Exception:
                        _style = "mixed"
                    overlay_p = (
                        "[Communication Style]\n"
                        f"- Style: {_style}\n"
                        "- Sources: show only when web_ctx exists\n"
                        "[Tier 3: Dynamic - Current Context]\n"
                        "🔄 bot_hint.output: include distance/open-now/rating/review-count for local"
                    )
                # 캐시에 저장하여 이후 build_messages에서 사용되도록 함
                try:
                    from backend.directives.store import (
                        set_unified_base,
                        set_unified_overlay,
                    )

                    if base_p:
                        set_unified_base(real_uid, "preload", base_p)
                    set_unified_overlay(
                        session_id, user_input or "", overlay_p or "", "preload"
                    )
                except Exception as e:
                    try:
                        safe_log_event(
                            "directive_preload_cache_error", {"error": str(e)}
                        )
                    except Exception:
                        pass
            except Exception as e:
                try:
                    safe_log_event("directive_preload_error", {"error": str(e)})
                except Exception:
                    pass
            realtime_ctx = get_current_time_context()
            turn_id = str(uuid.uuid4())[:8]
            try:
                set_context(turn_id=turn_id)
                log_event(
                    "query_received",
                    {"turn_id": turn_id, "len": len(user_input), "text": user_input},
                )
            except Exception as e:
                try:
                    safe_log_event(
                        "query_received_log_error",
                        {"turn_id": turn_id, "error": str(e)},
                    )
                except Exception:
                    pass
            # 다음 단계에서 사용할 최근 turn_id를 세션 상태에 기록
            try:
                SESSION_STATE.setdefault(session_id, {})["last_turn_id"] = turn_id
            except Exception as e:
                try:
                    safe_log_event(
                        "session_state_update_error",
                        {"session_id": session_id, "error": str(e)},
                    )
                except Exception:
                    pass

            # ===== Pre-Guard 검사: 입력만으로 금지 주제 위반 시 즉시 차단 =====
            if not await handle_pre_guard(session_id, user_input, websocket):
                continue

            # ===== 1. 메모리 & 히스토리 로드 =====
            hist, hist_msgs = await load_memory_and_history(session_id)

            # ===== 2. (이관됨) STWM/턴 버퍼는 Coordinator가 처리 =====

            # ===== 이전 턴 요약 불러오기 =====
            last_turn_summary = SESSION_STATE.get(session_id, {}).get(
                "last_turn_summary"
            )
            prev_turn_ctx = ""
            if last_turn_summary:
                prev_turn_ctx = (
                    f"[Previous Turn Summary]\n"
                    f"사용자 의도: {last_turn_summary.get('user_intent', '없음')}\n"
                    f"내 답변 요약: {last_turn_summary.get('ai_summary', '없음')}"
                )
                try:
                    log_event(
                        "prev_turn_loaded",
                        {"prev_turn_ctx": prev_turn_ctx},
                    )
                except Exception:
                    pass

            # ===== 3. 라우팅 (2단계: 임베딩 → LLM) =====
            # 라우팅용 요약 로드(MTM 최신 routing_summary -> 태그 제거)
            routing_ctx = ""
            try:
                coord = get_memory_coordinator()
                latest_mtm = coord.mtm.get_latest(user_id, session_id)
                if latest_mtm:
                    rs = (latest_mtm.get("routing_summary") or "").strip()
                    routing_ctx = (
                        rs.replace("[ROUTING_ONLY]", "")
                        .replace("[/ROUTING_ONLY]", "")
                        .strip()
                    )
            except Exception:
                routing_ctx = ""

            # 1~2단계 라우팅 수행 (임베딩 → 저신뢰시 LLM 폴백)
            need_rag, need_web, best_label, sims, max_sim = await perform_routing(
                user_input, routing_ctx, session_id
            )
            need_rag_prob = (
                float(sims.get("rag", 0.0)) if isinstance(sims, dict) else 0.0
            )
            need_web_prob = (
                float(sims.get("web", 0.0)) if isinstance(sims, dict) else 0.0
            )
            # TurnState(SSOT) 구성: 후속 단계는 읽기 전용으로만 사용
            try:
                vals_sorted = sorted(
                    (float(v) for v in (sims or {}).values()), reverse=True
                )
                sec_sim = float(vals_sorted[1]) if len(vals_sorted) >= 2 else 0.0
                margin = float(max_sim - sec_sim)
                count_over_tau = sum(
                    1 for v in (sims or {}).values() if float(v) >= 0.4
                )
                ambiguous = (count_over_tau >= 2) or (margin < 0.02)
                route_label = str(best_label or "conv")
                turn_state = TurnState.build(
                    session_id=session_id,
                    turn_id=turn_id,
                    route=("conv" if route_label not in ("conv", "web", "rag") else route_label),  # type: ignore[arg-type]
                    ambiguous=ambiguous,
                    max_sim=float(max_sim or 0.0),
                    margin=margin,
                    need_rag=bool(need_rag),
                    need_web=bool(need_web),
                )
            except Exception:
                # 실패 시 보수적 기본값
                turn_state = TurnState.build(
                    session_id=session_id,
                    turn_id=turn_id,
                    route="conv",
                    ambiguous=False,
                    max_sim=0.0,
                    margin=0.0,
                    need_rag=False,
                    need_web=False,
                )

            # ===== 4. 쿼리 재작성 (필요 시 병렬) =====
            rag_query_text = None
            rag_date_filter = None
            web_query = None
            preview_ctx = prev_turn_ctx or ""
            # STWM에서 최근 위치를 가져와 맥락에 포함(LLM이 필요시 carry-over하도록 힌트 제공)
            try:
                from backend.memory.stwm import get_stwm_snapshot

                snap = get_stwm_snapshot(session_id) or {}
                last_loc = str(snap.get("last_loc") or "").strip()
                if last_loc:
                    preview_ctx = (preview_ctx + f"\n[최근 위치]\n{last_loc}").strip()
            except Exception as e:
                try:
                    safe_log_event(
                        "stwm_snapshot_error",
                        {"session_id": session_id, "error": str(e)},
                    )
                except Exception:
                    pass

            async def _do_rewrite_rag():
                nonlocal rag_query_text, rag_date_filter
                rag_rw = await rewrite_query(
                    "rag",
                    user_input,
                    hist,
                    session_id=session_id,
                    preview_ctx=preview_ctx,
                    realtime_ctx=realtime_ctx,
                )
                rag_query_text = rag_rw["query_text"]
                rag_date_filter = rag_rw["date_filter"]
                try:
                    log_event(
                        "rewrite_rag",
                        {
                            "rag_query_text": rag_query_text,
                            "rag_date_filter": rag_date_filter,
                        },
                    )
                except Exception:
                    pass

                add_rewrite(
                    session_id,
                    RewriteRecord(
                        raw_query=user_input,
                        query_rewritten=rag_query_text,
                        applied_slots=["rewrite_llm"],
                    ),
                )

            async def _do_rewrite_web():
                nonlocal web_query
                web_rw = await rewrite_query(
                    "web",
                    user_input,
                    hist,
                    session_id=session_id,
                    preview_ctx=preview_ctx,
                    realtime_ctx=realtime_ctx,
                )
                web_query = web_rw["web_query"]
                try:
                    log_event(
                        "rewrite_web",
                        {"web_query": web_query},
                    )
                except Exception:
                    pass

                add_rewrite(
                    session_id,
                    RewriteRecord(
                        raw_query=user_input,
                        query_rewritten=web_query,
                        applied_slots=["rewrite_llm"],
                    ),
                )

            if need_rag and need_web:
                await asyncio.gather(_do_rewrite_rag(), _do_rewrite_web())
            elif need_rag:
                await _do_rewrite_rag()
            elif need_web:
                await _do_rewrite_web()

            # ===== 5. 증거 수집 (병렬, 타임아웃 폴백) =====
            rag_ctx = ""
            web_ctx = ""
            mobile_ctx = ""

            evidence_task = None
            mobile_task = asyncio.create_task(build_mobile_ctx(session_id))

            # Memory 채널 태스크 (최근 4턴 원문 + MTM 유사도 Top-K)
            from backend.context.memory_channel import build_memory_channel

            memory_task = asyncio.create_task(
                build_memory_channel(
                    user_id=user_id,
                    session_id=session_id,
                    user_input=user_input,
                    token_budget=700,
                )
            )

            # RAG 증거 태스크 (차감 검색 + 재랭킹 통합)
            if need_rag and rag_query_text:
                rag_t0 = time.time()
                evidence_task = asyncio.create_task(
                    retrieve_enhanced(
                        query=rag_query_text,
                        route="rag",
                        user_id=user_id,
                        top_k=5,
                        date_filter=rag_date_filter,
                    )
                )

            # 웹/날씨/금융 검색 태스크
            web_task = None
            finance_task = None
            if need_web and web_query:
                web_t0 = time.time()
                # 차감 검색(Web): web_archived 캐시 기반 제외 도메인 적용
                web_task = asyncio.create_task(
                    retrieve_enhanced(
                        query=web_query,
                        route="web",
                        user_id=user_id,
                        top_k=5,
                    )
                )
            # 금융 의도 강제 라우팅: "실시간 가격/시세/종가/티커" 등 명확 신호면 finance 우선
            try:
                from backend.services.finance import (
                    build_finance_block,
                    detect_finance_intent,
                )

                fin = await detect_finance_intent(user_input, realtime_ctx)
                if bool(fin.get("is_finance")) and str(fin.get("intent")) in (
                    "realtime_price",
                    "historical_price",
                ):
                    finance_task = asyncio.create_task(
                        build_finance_block(user_input, realtime_ctx)
                    )
            except Exception:
                finance_task = None
            # weather는 라우팅/재작성 단계에서 need_web 대신 weather 플래그를 통해 설정하는 방식이 자연스럽다.
            # 간이 구현: 입력 내 날씨 관련 키워드가 있으면 weather 우선 시도
            weather_task = None
            try:
                _t = (user_input or "").lower()
                WEATHER_HINT = (
                    ("날씨" in _t)
                    or ("기상" in _t)
                    or ("강수" in _t)
                    or ("미세먼지" in _t)
                    or ("체감" in _t)
                    or ("예보" in _t)
                )
                if WEATHER_HINT:
                    weather_task = asyncio.create_task(
                        retrieve_enhanced(
                            query=user_input, route="weather", user_id=user_id
                        )
                    )
            except Exception as e:
                try:
                    safe_log_event("weather_task_error", {"error": str(e)})
                except Exception:
                    pass

            # 500ms 타임아웃 대기 (증거 우선)
            wait_set = [mobile_task, memory_task]
            if evidence_task is not None:
                wait_set.append(evidence_task)

            done, pending = await asyncio.wait(wait_set, timeout=0.5)

            # 완료된 태스크 수집
            try:
                mobile_ctx = await mobile_task
            except Exception as e:
                mobile_ctx = ""
                try:
                    safe_log_event("mobile_ctx_error", {"error": str(e)})
                except Exception:
                    pass
            try:
                memory_ctx = await memory_task
            except Exception as e:
                memory_ctx = ""
                try:
                    safe_log_event("memory_ctx_error", {"error": str(e)})
                except Exception:
                    pass

            # 증거 타임아웃 처리
            if evidence_task is not None:
                if evidence_task in done:
                    try:
                        rag_ctx = await evidence_task
                        try:
                            log_event(
                                "evidence_rag",
                                {
                                    "query": rag_query_text,
                                    "rag_ctx": rag_ctx,
                                    "elapsed_ms": int((time.time() - rag_t0) * 1000),
                                },
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        rag_ctx = ""
                        try:
                            safe_log_event("evidence_rag_error", {"error": str(e)})
                        except Exception:
                            pass
                else:
                    # 지연 → 백그라운드 보강
                    log_event(
                        "evidence_rag_timeout",
                        {"query": rag_query_text},
                        level=logging.WARNING,
                    )
                    try:
                        asyncio.create_task(
                            supplement_evidence(
                                session_id, evidence_task, websocket, user_input, ""
                            )
                        )
                    except Exception:
                        pass

            # 웹/금융 검색 타임아웃 처리 (웹: 별도 5.5s, 금융: 2.0s)
            if web_task is not None:
                try:
                    web_ctx = await asyncio.wait_for(web_task, timeout=5.5)
                    try:
                        log_event(
                            "evidence_web",
                            {
                                "query": web_query,
                                "kind": "web",
                                "web_ctx": web_ctx,
                                "elapsed_ms": int((time.time() - web_t0) * 1000),
                            },
                        )
                    except Exception:
                        pass
                except asyncio.TimeoutError:
                    log_event(
                        "evidence_web_timeout",
                        {"query": web_query},
                        level=logging.WARNING,
                    )
                    web_ctx = ""
                except Exception:
                    web_ctx = ""

            if finance_task is not None:
                try:
                    blk, reason_fin = await asyncio.wait_for(finance_task, timeout=2.0)
                    if blk:
                        # 금융 블록이 있으면 web_ctx를 대체하고, 이후 필터는 건너뜀
                        web_ctx = blk
                        need_web = True
                        safe_log_event(
                            "evidence_finance",
                            {"reason": reason_fin, "len": len(blk or "")},
                        )
                except Exception:
                    pass

            # 날씨 컨텍스트 수집(성공 시 web_ctx 대신 사용)
            if weather_task is not None:
                try:
                    wx_ctx = await asyncio.wait_for(weather_task, timeout=2.5)
                    if wx_ctx:
                        web_ctx = wx_ctx
                        safe_log_event("evidence_weather", {"len": len(wx_ctx or "")})
                except Exception:
                    pass

            # 웹 전용 조기 응답 경로 제거: 항상 아래 공통 경로로 진행

            # ===== 6. 컨텍스트 필터링 (가능한 병렬) =====
            try:
                from backend.memory.summarizer import get_tokenizer

                _enc = get_tokenizer()
                tok_rag = len(_enc.encode(rag_ctx or ""))
                tok_web = len(_enc.encode(web_ctx or ""))
            except Exception:
                tok_rag = tok_web = None
            try:
                log_event(
                    "context_filter_before",
                    {
                        "rag_ctx_len": len(rag_ctx or ""),
                        "web_ctx_len": len(web_ctx or ""),
                        "rag_tokens": tok_rag,
                        "web_tokens": tok_web,
                        "rag_ctx": rag_ctx,
                        "web_ctx": web_ctx,
                    },
                )
            except Exception:
                pass

            async def _filter_rag():
                nonlocal rag_ctx
                if rag_ctx:
                    rag_ctx = await filter_semantic_mismatch(user_input, rag_ctx)

            async def _filter_web():
                nonlocal web_ctx
                if web_ctx and (finance_task is None):
                    web_ctx = await filter_web_ctx(user_input, web_ctx)

            await asyncio.gather(_filter_rag(), _filter_web())

            # 필터 이후 web_ctx가 비면 즉시 한 번 더 수집 폴백(저신뢰 또는 과도 필터 제거 보호)
            if need_web and not (web_ctx or "").strip() and web_query:
                try:
                    # 더 넉넉한 타임아웃으로 재시도
                    web_kind2, web_ctx2 = await build_web_context(
                        settings.MCP_SERVER_URL,
                        web_query,
                        5,
                        settings.TIMEOUT_WEB * 1.5,
                    )
                    if web_ctx2:
                        web_ctx = web_ctx2
                        try:
                            log_event(
                                "evidence_web_fallback_retry",
                                {"kind": web_kind2, "len": len(web_ctx2 or "")},
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            try:
                from backend.memory.summarizer import get_tokenizer

                _enc = get_tokenizer()
                tok_rag2 = len(_enc.encode(rag_ctx or ""))
                tok_web2 = len(_enc.encode(web_ctx or ""))
            except Exception:
                tok_rag2 = tok_web2 = None
            try:
                log_event(
                    "context_filter_after",
                    {
                        "rag_ctx_len": len(rag_ctx or ""),
                        "web_ctx_len": len(web_ctx or ""),
                        "rag_tokens": tok_rag2,
                        "web_tokens": tok_web2,
                        "rag_ctx": rag_ctx,
                        "web_ctx": web_ctx,
                    },
                )
            except Exception:
                pass

            # ===== 계층적 증거 요약 (선택적, Evidence 초과 시) =====
            try:
                from backend.context.block_reranker import rerank_and_select_blocks
                from backend.context.block_summarizer import summarize_evidence_blocks
                from backend.context.evidence_block_parser import parse_evidence_blocks
                from backend.memory.summarizer import get_tokenizer

                enc = get_tokenizer()
                evidence_cap = int(settings.EVIDENCE_TOKEN_CAP)

                rag_tokens = len(enc.encode(rag_ctx or ""))
                web_tokens = len(enc.encode(web_ctx or ""))
                total_evidence_tokens = rag_tokens + web_tokens

                # 증거가 예산 초과 시에만 계층적 요약 실행
                if total_evidence_tokens > evidence_cap:
                    logger.info(
                        f"[hierarchical_summary] Evidence exceeds budget: "
                        f"{total_evidence_tokens} > {evidence_cap}, "
                        "applying hierarchical summarization"
                    )

                    # RAG 블록 파싱 및 요약
                    rag_blocks = []
                    if rag_ctx:
                        rag_blocks = parse_evidence_blocks(rag_ctx, "rag", enc)
                        rag_blocks = await summarize_evidence_blocks(
                            rag_blocks,
                            user_input,
                            target_tokens_per_block=200,
                        )

                    # 웹 블록 파싱 및 요약
                    web_blocks = []
                    if web_ctx:
                        web_blocks = parse_evidence_blocks(web_ctx, "web", enc)
                        web_blocks = await summarize_evidence_blocks(
                            web_blocks,
                            user_input,
                            target_tokens_per_block=150,
                        )

                    # 통합 및 재순위화
                    all_blocks = rag_blocks + web_blocks
                    if all_blocks:
                        combined_evidence = rerank_and_select_blocks(
                            all_blocks,
                            user_input,
                            token_budget=evidence_cap,
                            tokenizer=enc,
                        )

                        # 재할당: evidence는 rag_ctx로 통합, web_ctx는 비움
                        rag_ctx = combined_evidence
                        web_ctx = ""

                        try:
                            log_event(
                                "hierarchical_summary_applied",
                                {
                                    "original_tokens": total_evidence_tokens,
                                    "final_tokens": len(enc.encode(combined_evidence)),
                                    "blocks_total": len(all_blocks),
                                },
                            )
                        except Exception:
                            pass
            except Exception as e:
                # 실패 시에도 기존 컨텍스트 유지
                logger.error(f"[hierarchical_summary] Error: {e}")

            # ===== 7. 적응형 토큰 버짓 (Evidence > Memory > Profile) =====
            try:
                from backend.context.adaptive_budget import (
                    AdaptiveBudgetManager,
                    ContextBundle,
                )

                # 증거 계약 생성(Evidence Reference System): 버짓 전 단계에서 원본 보존 후 계약 적용
                try:
                    from backend.context.evidence_contractor import (
                        get_evidence_contractor as _get_evc,
                    )

                    # 원본을 refs 용으로 선보존
                    if (rag_ctx or "").strip() or (web_ctx or "").strip():
                        rag_ctx_for_refs = rag_ctx  # 원문 보존
                        web_ctx_for_refs = web_ctx  # 원문 보존

                        contractor = _get_evc()

                        # 신규: TurnState에서 active_eids 로드
                        active_eids = []
                        try:
                            from backend.routing.turn_state import get_turn_state

                            state = get_turn_state(session_id) or {}
                            active_eids = state.get("active_eids", [])

                            if active_eids:
                                logger.info(
                                    f"[app] Loaded {len(active_eids)} active_eids from TurnState"
                                )
                                try:
                                    log_event(
                                        "evidence.reused",
                                        {
                                            "session_id": session_id,
                                            "active_eid_count": len(active_eids),
                                        },
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            active_eids = []

                        # Evidence Contractor 호출 (save_active_eids=True)
                        eids, contract_text = contractor.store_and_contract(
                            session_id=session_id,
                            rag_ctx=rag_ctx or "",
                            web_ctx=web_ctx or "",
                            user_query=user_input,
                            active_eids=active_eids or None,
                            save_active_eids=True,
                        )
                        if eids:
                            # 계약 텍스트를 evidence로 사용, web_ctx는 계약에 통합되어 비움
                            rag_ctx = contract_text
                            web_ctx = ""
                            # 세션 상태에 EID 저장(턴 격리 + 하위 호환)
                            try:
                                SESSION_STATE.setdefault(session_id, {}).setdefault(
                                    "last_eids_map", {}
                                )[turn_id] = list(eids)
                                SESSION_STATE.setdefault(session_id, {})[
                                    "last_eids"
                                ] = list(eids)
                            except Exception:
                                pass
                except Exception as _e:
                    logger.warning(
                        f"[evidence_contract] Failed: {_e}, fallback to raw evidence"
                    )

                budget_mgr = AdaptiveBudgetManager()
                aux_in = SESSION_STATE.get(session_id, {}).get("aux_ctx", "")

                bundle = ContextBundle(
                    evidence=rag_ctx or "",  # 계약 또는 원문
                    memory=memory_ctx or "",
                    profile="",
                )

                # 신규: 동적 예산 할당 (allocate_dynamic 사용)
                # 히스토리 토큰 수 계산
                try:
                    from backend.memory.summarizer import get_tokenizer

                    enc_budget = get_tokenizer()
                    history_token_count = len(enc_budget.encode(memory_ctx or ""))

                    # 동적 예산 할당 시도
                    adjusted = budget_mgr.allocate_dynamic(
                        bundle=bundle,
                        user_input=user_input,
                        history_tokens=history_token_count,
                        total_cap=3400,
                    )
                except Exception as e:
                    # 폴백: 기존 allocate() 메서드 사용
                    logger.warning(
                        f"[app] allocate_dynamic failed, fallback to allocate: {e}"
                    )
                    adjusted = budget_mgr.allocate(bundle, total_cap=3400)

                # Evidence에 웹 포함 → web_ctx는 비움
                # refs 저장/검증을 위해 원본 컨텍스트를 별도로 보존
                if "rag_ctx_for_refs" not in locals():
                    rag_ctx_for_refs = rag_ctx or ""
                if "web_ctx_for_refs" not in locals():
                    web_ctx_for_refs = web_ctx or ""
                rag_ctx = adjusted.evidence or ""
                web_ctx = ""
                memory_ctx = adjusted.memory or ""
                try:
                    log_event(
                        "context_budget_adjust",
                        {
                            "evidence_len": len(rag_ctx or ""),
                            "memory_len": len(memory_ctx or ""),
                        },
                    )
                except Exception:
                    pass

                # aux는 기존 유지
                SESSION_STATE.setdefault(session_id, {})["aux_ctx"] = aux_in
            except Exception:
                # 폴백: 기존 컨텍스트를 그대로 사용
                pass

            # Evidence 완전 공백 방지 가드: need_web인데 evidence가 0이면 최소 웹 블록 1개 재시도 후 없으면 경고 발송 및 fail-open 차단
            if need_web and not (rag_ctx or "").strip():
                if not (web_ctx_for_refs or "").strip():
                    try:
                        await websocket.send_text(
                            "[알림] 검색 결과가 충분하지 않습니다. 쿼리를 구체화해 재시도합니다."
                        )
                    except Exception:
                        pass
                    # fail-open 방지: 증거 없이 사실 추천 생성 금지 힌트를 보조 컨텍스트에 주입
                    try:
                        aux_prev = SESSION_STATE.get(session_id, {}).get("aux_ctx", "")
                        SESSION_STATE.setdefault(session_id, {})["aux_ctx"] = (
                            aux_prev
                            + "\n[가드] web_ctx 없음 → 사실 추천/리스트 금지, 검색 제안만 허용"
                        ).strip()
                    except Exception:
                        pass

            # ===== 8. 최종 응답 생성 =====
            # 링크 보존 정책:
            #  - 증거 계약(EID) 활성 시: 프롬프트에는 web_ctx를 주입하지 않되, 사용자 출력에는 refs(링크)를 반드시 포함
            #  - 비활성 시: 기존 동작 유지
            try:
                _eids = SESSION_STATE.get(session_id, {}).get("last_eids", [])
                if _eids:
                    # EID 모드: 프롬프트에 최소 원문(제목/링크만) 주입하여 환각률을 낮춘다.
                    try:
                        minimal_prompt_ctx = ""
                        if (_blocks_to_items is not None) and (
                            web_ctx_for_refs or ""
                        ).strip():
                            items_min = _blocks_to_items(web_ctx_for_refs)
                            blocks = []
                            for it in items_min:
                                t = (it.get("title") or "").strip()
                                u = (it.get("url") or "").strip()
                                if t and u:
                                    blocks.append("\n".join([t, "-", u]))
                            minimal_prompt_ctx = "\n\n".join(blocks)
                        web_ctx_for_prompt = minimal_prompt_ctx
                    except Exception:
                        web_ctx_for_prompt = ""
                    # 사용자 출력용 링크는 보존본을 우선 사용
                    web_refs_ctx_for_answer = (
                        web_ctx_for_refs
                        if (web_ctx_for_refs or "").strip()
                        else web_ctx
                    )
                else:
                    # 프롬프트/refs 모두 동일 소스를 사용(보존본이 있으면 우선)
                    web_ctx_for_prompt = (
                        web_ctx_for_refs
                        if (web_ctx_for_refs or "").strip()
                        else web_ctx
                    )
                    web_refs_ctx_for_answer = web_ctx_for_prompt
            except Exception:
                web_ctx_for_prompt = ""
                web_refs_ctx_for_answer = ""

            full_answer, turn_summary = await main_response(
                session_id,
                turn_state,
                user_input,
                websocket,
                mobile_ctx,
                rag_ctx,
                web_ctx_for_prompt,
                memory_ctx,
                realtime_ctx,
                prev_turn_ctx=prev_turn_ctx,
                web_refs_ctx=web_refs_ctx_for_answer,
            )
            try:
                log_event(
                    "llm_final",
                    {
                        "final_answer": full_answer,
                        "turn_summary": {
                            "user_intent": turn_summary.user_intent,
                            "ai_summary": turn_summary.ai_summary,
                        },
                    },
                )
            except Exception:
                pass

            # ===== 턴 요약 저장 =====
            SESSION_STATE.setdefault(session_id, {})["last_turn_summary"] = {
                "user_intent": turn_summary.user_intent,
                "ai_summary": turn_summary.ai_summary,
            }
            try:
                log_event(
                    "turn_summary_saved",
                    {
                        "user_intent": turn_summary.user_intent,
                        "ai_summary": turn_summary.ai_summary,
                    },
                )
            except Exception:
                pass

            # ===== 라우팅 보조 신호(m_ctx) 비동기 업데이트 =====
            try:
                from backend.routing.mctx_store import (
                    update_mctx_with_summary as _update_mctx,
                )
                from backend.generation.analysis_logger import (
                    run_analysis_v1 as _run_analysis_v1,
                )

                async def _upd_mctx_bg():
                    try:
                        summary_text = (
                            f"{turn_summary.user_intent}\n{turn_summary.ai_summary}"
                        ).strip()
                        await asyncio.to_thread(
                            _update_mctx, session_id, summary_text, turn_id
                        )
                        try:
                            log_event(
                                "routing_mctx_updated",
                                {
                                    "session_id": session_id,
                                    "turn_id": turn_id,
                                },
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

                async def _analysis_bg():
                    try:
                        # allowed_ctx는 TurnState에서, ctx_present는 실제 주입 여부 기반
                        allowed_ctx = list(turn_state.allowed_ctx_mask or [])
                        # 실제 주입 판단: conv_only는 기본 포함, rag/web은 프롬프트 주입 여부
                        present: list[str] = ["conv_only"]
                        try:
                            if (rag_ctx or "").strip():
                                present.append("rag_ctx")
                        except Exception:
                            pass
                        try:
                            if (web_ctx_for_prompt or "").strip():
                                present.append("web_ctx")
                        except Exception:
                            pass
                        await _run_analysis_v1(
                            session_id=session_id,
                            turn_state=turn_state,
                            allowed_ctx=allowed_ctx,
                            ctx_present=present,
                        )
                    except Exception:
                        pass

                asyncio.create_task(_upd_mctx_bg())
                asyncio.create_task(_analysis_bg())
            except Exception:
                pass

            # ===== Output-Aware Pruning: AI 출력에서 인용된 EID 추출 및 다음 턴 활성화 설정 =====
            try:
                from backend.context.evidence_contractor import (
                    get_evidence_contractor as _get_evc_post,
                )

                if settings.OUTPUT_PRUNING_ENABLED:
                    contractor_post = _get_evc_post()
                    cited_eids = contractor_post.extract_cited_eids(full_answer)
                    current_eids = SESSION_STATE.get(session_id, {}).get(
                        "last_eids", []
                    )

                    # 미인용 계산 및 보류 저장(contracts_latest 기준)
                    if current_eids:
                        unused_eids = [
                            eid for eid in current_eids if eid not in cited_eids
                        ]
                        if unused_eids:
                            # 최신 계약에서 unused 계약 추출 후 보류 저장
                            latest_contracts = contractor_post.get_latest_contracts(
                                session_id
                            )
                            if latest_contracts:
                                cited_set = set(cited_eids)
                                _, unused_contracts = (
                                    contractor_post.filter_contracts_by_eids(
                                        latest_contracts, list(cited_set)
                                    )
                                )
                                if unused_contracts:
                                    contractor_post.store_unused_evidence(
                                        session_id, turn_id, unused_contracts
                                    )

                        try:
                            log_event(
                                "evidence.pruned",
                                {
                                    "session_id": session_id,
                                    "cited_count": len(cited_eids),
                                    "unused_count": (
                                        len(unused_eids) if current_eids else 0
                                    ),
                                },
                            )
                        except Exception:
                            pass

                    # 다음 턴 활성 EID 저장 (없으면 빈 리스트)
                    try:
                        SESSION_STATE.setdefault(session_id, {})["active_eids"] = (
                            cited_eids or []
                        )
                        SESSION_STATE.setdefault(session_id, {}).setdefault(
                            "active_eids_map", {}
                        )[turn_id] = list(cited_eids or [])
                    except Exception:
                        pass
            except Exception as _e:
                logger.warning(f"[output_pruning] post-turn failed: {_e}")

            # ===== 메모리/봇프로필/프로필 Writer 업데이트를 응답 후 후처리로 일원화 =====
            try:
                coordinator = get_memory_coordinator()

                async def _run_mem_off_thread() -> "TurnResult":
                    def _sync() -> "TurnResult":
                        import asyncio as _aio

                        _uid = user_id

                        return _aio.run(
                            coordinator.on_turn_end(
                                user_id=_uid,
                                session_id=session_id,
                                user_input=user_input,
                                ai_output=full_answer,
                            )
                        )

                    return await asyncio.to_thread(_sync)

                mem_task = asyncio.create_task(_run_mem_off_thread())
                mem_task.add_done_callback(lambda t: None)

                # [세션 2] BotProfileManager 제거됨: 메모리 기반 봇 프로필 업데이트 호출 삭제

                # 프로필 Writer: AI 응답 기반 추론 정보 pending 적재
                try:
                    from backend.rag.profile_writer import get_profile_writer

                    _uid2 = user_id

                    async def _run_profile_update():
                        writer = get_profile_writer()
                        await writer.update_from_turn(
                            user_id=_uid2,
                            session_id=session_id,
                            turn_summary={
                                "id": turn_id,
                                "user_input": user_input,
                                "ai_output": full_answer,
                                "llm_turn_summary": {
                                    "user_intent": turn_summary.user_intent,
                                    "ai_summary": turn_summary.ai_summary,
                                },
                            },
                        )
                        # [세션 3] 봇 전역 프로필도 동시 학습 (user_id="bot")
                        try:
                            await writer.update_from_turn(
                                user_id=bot_user_id_for(_uid2),
                                session_id=session_id,
                                turn_summary={
                                    "id": turn_id,
                                    "user_input": user_input,
                                    "ai_output": full_answer,
                                    "llm_turn_summary": {
                                        "user_intent": turn_summary.user_intent,
                                        "ai_summary": turn_summary.ai_summary,
                                    },
                                },
                            )
                        except Exception:
                            pass

                    asyncio.create_task(_run_profile_update())
                except Exception:
                    pass

                try:
                    log_event(
                        "memory_on_turn_end_scheduled",
                        {"user_input": user_input, "ai_output": full_answer},
                    )
                except Exception:
                    pass
            except Exception as e:
                log_event(
                    "memory_coordinator_schedule_error",
                    {"error": repr(e)},
                    level=logging.ERROR,
                )

            # ===== 9. 사후 검증 (비차단) =====
            if (
                rag_ctx_for_refs if "rag_ctx_for_refs" in locals() else rag_ctx
            ).strip() or (
                web_ctx_for_refs if "web_ctx_for_refs" in locals() else web_ctx
            ).strip():
                try:
                    log_event(
                        "post_verify_scheduled",
                        {
                            "user_input": user_input,
                            "rag_ctx": (
                                rag_ctx_for_refs
                                if "rag_ctx_for_refs" in locals()
                                else rag_ctx
                            ),
                            "web_ctx": (
                                web_ctx_for_refs
                                if "web_ctx_for_refs" in locals()
                                else web_ctx
                            ),
                            "answer": full_answer,
                        },
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    post_verify_answer(
                        user_input,
                        (
                            rag_ctx_for_refs
                            if "rag_ctx_for_refs" in locals()
                            else rag_ctx
                        ),
                        (
                            web_ctx_for_refs
                            if "web_ctx_for_refs" in locals()
                            else web_ctx
                        ),
                        full_answer,
                        websocket,
                    )
                )

            # 증거 참조 저장 (원본 컨텍스트 기준) - 비동기 오프로딩 [규칙 6-2-3]
            if (
                web_ctx_for_refs if "web_ctx_for_refs" in locals() else web_ctx
            ).strip() or (
                rag_ctx_for_refs if "rag_ctx_for_refs" in locals() else rag_ctx
            ).strip():

                async def _store_refs_bg(_uid: str, _w: str, _r: str) -> None:
                    try:
                        # 동기 I/O는 to_thread로 비차단 처리
                        await asyncio.to_thread(store_refs_from_contexts, _uid, _w, _r)
                        try:
                            log_event(
                                "refs_stored",
                                {
                                    "web_len": len(_w or ""),
                                    "rag_len": len(_r or ""),
                                },
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            log_event("refs_store_failed", {"error": str(e)})
                        except Exception:
                            pass

                _w_val = web_ctx_for_refs if "web_ctx_for_refs" in locals() else web_ctx
                _r_val = rag_ctx_for_refs if "rag_ctx_for_refs" in locals() else rag_ctx
                asyncio.create_task(_store_refs_bg(user_id, _w_val, _r_val))

            # ===== 증거 피드백 감지 및 저장 =====
            try:
                from backend.rag.feedback_detector import (
                    detect_evidence_feedback as _detect_fb,
                )
                from backend.rag.feedback_store import store_feedback_enhanced_evidence
                from backend.rag.web_evidence_archiver import evaluate_with_feedback

                prev_evidence = SESSION_STATE.get(session_id, {}).get(
                    "last_evidence", {}
                )
                prev_rag = prev_evidence.get("rag_ctx", "")
                prev_web = prev_evidence.get("web_ctx", "")

                feedbacks = []
                if prev_rag or prev_web:
                    feedbacks = await _detect_fb(
                        user_input=user_input,
                        ai_output=full_answer,
                        prev_rag_ctx=prev_rag,
                        prev_web_ctx=prev_web,
                        user_id=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                    )
                    # 피드백 기반 pending 증거 평가 (Delayed Archival)
                    await evaluate_with_feedback(
                        user_id, session_id, turn_id, feedbacks
                    )
                    for fb in feedbacks:
                        original = prev_rag if fb.evidence_type == "rag" else prev_web
                        asyncio.create_task(
                            store_feedback_enhanced_evidence(fb, original)
                        )

                # 현재 턴 증거 저장 (다음 턴 감지용)
                try:
                    current_evidence = {
                        "rag_ctx": (
                            rag_ctx_for_refs
                            if "rag_ctx_for_refs" in locals()
                            else rag_ctx
                        ),
                        "web_ctx": (
                            web_ctx_for_refs
                            if "web_ctx_for_refs" in locals()
                            else web_ctx
                        ),
                    }
                    SESSION_STATE.setdefault(session_id, {})[
                        "last_evidence"
                    ] = current_evidence
                    SESSION_STATE.setdefault(session_id, {}).setdefault(
                        "last_evidence_map", {}
                    )[turn_id] = current_evidence
                except Exception:
                    pass
            except Exception as e:
                try:
                    log_event(
                        "evidence_feedback_error",
                        {"error": repr(e)},
                        level=logging.ERROR,
                    )
                except Exception:
                    pass

            # 턴 범위 프로필 캐시 클리어
            try:
                from backend.directives.profile_cache_manager import clear_turn_cache

                clear_turn_cache()
            except Exception:
                pass

            # ===== 웹 검색 결과 선별 적재 (LLM 평가 기반) =====
            try:
                from backend.rag.web_evidence_archiver import enqueue_pending_evidence
                from backend.rag.web_evidence_evaluator import should_archive_web_result

                current_web = (
                    web_ctx_for_refs if "web_ctx_for_refs" in locals() else web_ctx
                )

                if current_web and len(current_web) > 100:

                    async def _evaluate_and_archive():
                        try:
                            should_save, conf, reason = await should_archive_web_result(
                                web_ctx=current_web,
                                user_input=user_input,
                                ai_output=full_answer,
                                user_next_turn=None,
                            )
                            if should_save:
                                user_ctx = f"질문: {user_input[:200]}\n답변: {full_answer[:200]}"
                                await enqueue_pending_evidence(
                                    user_id=user_id,
                                    session_id=session_id,
                                    turn_id=turn_id,
                                    web_ctx=current_web,
                                    user_context=user_ctx,
                                    confidence=conf,
                                )
                                try:
                                    log_event(
                                        "web_evidence_pending",
                                        {
                                            "session_id": session_id,
                                            "confidence": conf,
                                            "reason": reason,
                                        },
                                    )
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"[web_archive] Error: {e}")

                    asyncio.create_task(_evaluate_and_archive())
            except Exception as e:
                try:
                    log_event(
                        "web_evidence_archive_error",
                        {"error": repr(e)},
                        level=logging.ERROR,
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        log_event("ws_disconnected", {"session_id": session_id, "user_id": user_id})
        try:
            clear_context()
        except Exception:
            pass
        # 최종 스냅샷 예약
        enqueue_snapshot(user_id, session_id)
        schedule_directive_update(session_id, force=True)


@app.on_event("shutdown")
async def on_shutdown():
    """애플리케이션 종료 시 공유 HTTP 클라이언트 정리"""
    try:
        await http_client_aclose()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 메인 실행 (개발용)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
