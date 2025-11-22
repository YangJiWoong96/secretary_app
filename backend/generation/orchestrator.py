"""
backend.generation.orchestrator - 응답 생성 오케스트레이터

3단계 응답 생성 파이프라인을 조율합니다.
"""

from __future__ import annotations

from typing import Tuple

from fastapi import WebSocket

from backend.generation.message_builder import build_messages
from backend.generation.schema import TurnSummary
from backend.generation.stream_handler import stream_response
from backend.memory.history_selector import prepare_history
from backend.routing.turn_state import TurnState

from backend.utils.tracing import traceable


@traceable(name="Orchestrator: main_response", run_type="chain", tags=["orchestrator"])
async def main_response(
    session_id: str,
    turn_state: TurnState | None,
    user_input: str,
    websocket: WebSocket,
    mobile_ctx: str,
    rag_ctx: str,
    web_ctx: str,
    memory_ctx: str,
    realtime_ctx: str,
    prev_turn_ctx: str = "",
    web_refs_ctx: str = "",
) -> Tuple[str, TurnSummary]:
    """
    최종 응답 생성 (3단계 모듈화 구조).

    1. prepare_history(): 히스토리 선별
    2. build_messages(): 프롬프트 조립
    3. stream_response(): JSON 응답 및 파싱
    """
    # 1단계: 히스토리 준비
    hist_msgs = await prepare_history(session_id, user_input)

    # 2단계: 메시지 조립
    messages = await build_messages(
        hist_msgs,
        session_id,
        user_input,
        rag_ctx,
        web_ctx,
        mobile_ctx,
        memory_ctx,
        realtime_ctx,
        prev_turn_ctx,
    )

    # 증거 모드 판별
    evidence_mode = bool(rag_ctx.strip() or web_ctx.strip())

    # 3단계: 스트리밍 응답
    full_json, parsed_response = await stream_response(
        messages,
        session_id,
        evidence_mode,
        websocket,
        turn_state=turn_state,
        user_input=user_input,
        rag_ctx=rag_ctx,
        web_ctx=web_ctx,
        web_refs_ctx=web_refs_ctx,
    )

    return parsed_response.final_answer, parsed_response.turn_summary


import asyncio  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

# ─────────────────────────────────────────────────────────────
# WebSocket 핸들러 보조 함수 (리팩토링 분리)
# ─────────────────────────────────────────────────────────────
from typing import Any, Dict, List, Tuple  # noqa: E402


@traceable(name="Orchestrator: handle_pre_guard", run_type="tool", tags=["guard"])
async def handle_pre_guard(
    session_id: str, user_input: str, websocket: WebSocket
) -> bool:
    """
    Pre-Guard 검사를 수행한다.
    - 사용자 입력만으로 금지 주제 위반 시 즉시 차단 메시지를 전송하고 False를 반환한다.
    - 오류 발생 시 fail-open으로 True를 반환한다.
    """
    try:
        from backend.rag.profile_guard import get_profile_guard  # 지연 임포트
        from backend.utils.logger import log_event  # 지연 임포트

        guard = get_profile_guard()
        ok, reason = await guard.validate(
            user_id=session_id, user_input=user_input, ai_output=""
        )
        if not ok:
            try:
                log_event(
                    "guard_violation_pre_block",
                    {"session_id": session_id, "reason": reason},
                )
            except Exception:
                pass
            try:
                await websocket.send_text(
                    "죄송합니다. 해당 주제는 답변드릴 수 없습니다."
                )
            except Exception:
                pass
            return False
        return True
    except Exception:
        # 보수적 처리: 예외 시 통과
        return True


@traceable(
    name="Orchestrator: load_memory_and_history", run_type="tool", tags=["memory"]
)
async def load_memory_and_history(session_id: str) -> Tuple[str, List[Any]]:
    """
    세션의 단기 메모리에서 대화 히스토리를 로드하고, 문자열 히스토리와 메시지 목록을 반환한다.
    반환:
        hist: "role: content" 형식으로 합쳐진 문자열
        hist_msgs: 원본 메시지 객체 리스트
    """
    try:
        from backend.memory import get_short_term_memory  # 지연 임포트
        from backend.utils.logger import safe_log_event  # 지연 임포트

        stm = get_short_term_memory(session_id)
        hist_msgs = list(stm.chat_memory.messages)
        hist = "\n".join(f"{m.type}: {m.content}" for m in hist_msgs)
        safe_log_event(
            "history_loaded", {"messages": len(hist_msgs), "hist_text": hist}
        )
        return hist, hist_msgs
    except Exception:
        # 실패 시 빈 히스토리 반환
        return "", []


@traceable(name="Orchestrator: perform_routing", run_type="chain", tags=["routing"])
async def perform_routing(
    user_input: str,
    routing_ctx: str,
    session_id: str,
) -> Tuple[bool, bool, str, Dict[str, float], float]:
    """
    2단계 라우팅(임베딩 → 저신뢰시 LLM 폴백)을 수행하여
    최종 need_rag / need_web 플래그와 진단 정보를 반환한다.
    반환:
        (need_rag, need_web, best_label, sims, max_sim)
    """
    from backend.routing.intent_router import (  # 지연 임포트
        embedding_router_scores,
        get_intent_router,
    )
    from backend.routing.llm_router import llm_decider  # 지연 임포트
    from backend.utils.logger import safe_log_event  # 지연 임포트
    from backend.rag.embeddings import embed_query_cached  # 지연 임포트
    from backend.routing.mctx_store import load_mctx  # 지연 임포트
    import numpy as _np  # 지연 임포트

    # 1단계: 임베딩 라우터
    # 원칙: query-only 임베딩으로 결정 (prev_ctx 결합 금지)
    best_label, sims = await asyncio.to_thread(
        embedding_router_scores, user_input, 0.4, "", session_id
    )
    safe_log_event(
        "routing_scores",
        {
            "embedding_best": best_label,
            "scores": sims,
            "routing_ctx": routing_ctx,
            "routing_ctx_len": len(routing_ctx or ""),
            "routing_ctx_sha1": (
                hashlib.sha1((routing_ctx or "").encode("utf-8")).hexdigest()
                if routing_ctx
                else ""
            ),
        },
    )

    # 초기 플래그 (query-only 기준)
    need_rag = False
    need_web = False
    if best_label == "rag":
        need_rag = True
    elif best_label == "web":
        need_web = True
    elif best_label == "conv":
        need_rag, need_web = False, False

    # 모호 판정: count_over_τ(0.4)≥2 또는 margin < 0.02
    max_sim = max(sims.values()) if sims else 0.0
    sorted_vals = sorted((float(v) for v in sims.values()), reverse=True)
    sec_sim = float(sorted_vals[1]) if len(sorted_vals) >= 2 else 0.0
    margin = float(max_sim - sec_sim)
    count_over_tau = sum(1 for v in sims.values() if float(v) >= 0.4)
    ambiguous = (count_over_tau >= 2) or (margin < 0.02)

    # 2-a) 모호하고, m_ctx가 유효하면 보조 신호 적용 시도
    if ambiguous:
        try:
            # 병렬: emb(q), load(m_ctx)
            q_vec_fut = asyncio.to_thread(
                embed_query_cached, (user_input or "").strip()
            )
            mctx_fut = asyncio.to_thread(load_mctx, session_id)
            q_vec, mctx = await asyncio.gather(q_vec_fut, mctx_fut)

            accepted = False
            align = 0.0
            age_ok = False
            max_ema = 0.0
            margin_ema = 0.0

            if (
                isinstance(q_vec, _np.ndarray)
                and mctx
                and isinstance(mctx.get("vec"), _np.ndarray)
            ):
                # 정규화
                qv = _np.array(q_vec, dtype=_np.float32)
                qv = qv / (float(_np.linalg.norm(qv)) or 1.0)
                mv = _np.array(mctx["vec"], dtype=_np.float32)
                mv = mv / (float(_np.linalg.norm(mv)) or 1.0)
                align = float(float(_np.dot(qv, mv)))

                # age ≤ 600s (직전 턴/시간 조건의 보수적 근사)
                try:
                    age_ok = (time.time() - float(mctx.get("updated_at") or 0.0)) <= 600.0  # type: ignore[name-defined]
                except Exception:
                    age_ok = False

                # 보호 도메인은 현 버전에서 미사용(보수적 False)
                protected = False

                if (align >= 0.35) and age_ok and (not protected):
                    lam = 0.08  # λ=0.05~0.10 권장
                    e_star = (1.0 - lam) * qv + lam * mv
                    e_star = e_star / (float(_np.linalg.norm(e_star)) or 1.0)

                    # 보조 벡터로 재스코어
                    router = get_intent_router()
                    best2, sims2 = await asyncio.to_thread(
                        router.route_with_scores_by_vec, e_star, 0.4
                    )

                    if sims2:
                        max_ema = max(sims2.values())
                        vals2 = sorted((float(v) for v in sims2.values()), reverse=True)
                        sec2 = float(vals2[1]) if len(vals2) >= 2 else 0.0
                        margin_ema = float(max_ema - sec2)

                    # 채택 기준: margin_ema ≥ ε(0.01) AND max_ema ≥ τ(0.40)
                    if (margin_ema >= 0.01) and (max_ema >= 0.40) and best2:
                        accepted = True
                        best_label = best2
                        sims = sims2
                        need_rag = best2 == "rag"
                        need_web = best2 == "web"

            # 진단 로깅
            try:
                safe_log_event(
                    "routing_mctx_mixed",
                    {
                        "ambiguous": ambiguous,
                        "align": align,
                        "age_ok": age_ok,
                        "accepted": accepted,
                        "margin": margin,
                        "margin_ema": margin_ema,
                        "max_sim": max_sim,
                        "max_ema": max_ema,
                    },
                )
            except Exception:
                pass
        except Exception:
            # 보조 실패는 무시
            pass

    # 2-b) 저신뢰 케이스 LLM 폴백 (임계값 0.4)
    max_sim = max(sims.values()) if sims else 0.0
    if max_sim < 0.4:
        try:
            decided = await llm_decider(user_input)
            safe_log_event(
                "llm_decider_result",
                {"decided": decided, "max_sim": max_sim, "scores": sims},
            )
            if decided == "rag":
                need_rag, need_web = True, False
            elif decided == "web":
                need_rag, need_web = False, True
            else:
                need_rag, need_web = False, False
        except Exception:
            # 실패 시 임베딩 결과 유지
            pass

    safe_log_event(
        "routing_decision_final",
        {
            "best_label": best_label,
            "sims": sims,
            "max_sim": max_sim,
            "need_rag": need_rag,
            "need_web": need_web,
            "routing_ctx_len": len(routing_ctx or ""),
            "routing_ctx_sha1": (
                hashlib.sha1((routing_ctx or "").encode("utf-8")).hexdigest()
                if routing_ctx
                else ""
            ),
        },
    )

    return need_rag, need_web, best_label, sims, max_sim
