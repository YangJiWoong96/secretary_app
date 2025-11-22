"""
backend.generation.stream_handler - 스트리밍 응답 처리

LLM 스트리밍 응답 및 JSON 구조화 출력을 처리합니다.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Tuple

from fastapi import WebSocket

from backend.generation.schema import AssistantResponse, TurnSummary
from backend.routing.turn_state import TurnState

logger = logging.getLogger("stream_handler")

from backend.utils.tracing import traceable


@traceable(name="Gen: stream_response", run_type="chain", tags=["gen", "llm"])
async def stream_response(
    messages: list[dict[str, str]],
    session_id: str,
    evidence_mode: bool,
    websocket: WebSocket,
    *,
    turn_state: TurnState | None = None,
    user_input: str | None = None,
    rag_ctx: str | None = None,
    web_ctx: str | None = None,
    web_refs_ctx: str | None = None,
) -> tuple[str, AssistantResponse]:
    """
    LLM 스트리밍 응답 (JSON 구조화 출력).

    Args:
        messages: LLM에 전달할 메시지 배열
        session_id: 세션 ID
        evidence_mode: 증거 모드 여부
        websocket: WebSocket 연결
        user_input: 사용자 입력 (검증용)
        rag_ctx: RAG 컨텍스트 (검증용)
        web_ctx: WEB 컨텍스트 (검증용)

    Returns:
        tuple[str, AssistantResponse]: (전체 XML 문자열, 파싱된 응답 객체)
    """
    from backend.config import get_settings
    from backend.generation.formatters import cleanup_final_answer, format_references
    from backend.generation.validators import validate_final_answer
    from backend.memory import model_supports_response_format
    from backend.rag.profile_guard import get_profile_guard
    from backend.utils.logger import log_event
    from backend.utils.retry import openai_chat_with_retry
    from backend.utils.schema_builder import build_json_schema
    from backend.policy import SESSION_STATE

    settings = get_settings()
    full_response = ""
    parsed_response = None

    try:
        # LLM 호출
        create_kwargs = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.3 if evidence_mode else 0.7,
        }

        # 모델이 지원하면 JSON 강제 응답
        try:
            if model_supports_response_format(settings.LLM_MODEL):
                schema = {
                    "type": "object",
                    "properties": {
                        "internal_analysis": {"type": "string"},
                        "final_answer": {"type": "string"},
                        "turn_summary": {
                            "type": "object",
                            "properties": {
                                "user_intent": {"type": "string"},
                                "ai_summary": {"type": "string"},
                            },
                            "required": ["user_intent", "ai_summary"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["internal_analysis", "final_answer", "turn_summary"],
                    "additionalProperties": False,
                }
                create_kwargs["response_format"] = build_json_schema(
                    "AssistantResponse", schema, strict=True
                )
        except Exception:
            pass

        log_event(
            "llm_call",
            {
                "model": create_kwargs.get("model"),
                "temperature": create_kwargs.get("temperature"),
                "messages_count": len(messages),
            },
        )

        t0 = time.perf_counter()
        resp = await openai_chat_with_retry(**create_kwargs)
        took_ms = (time.perf_counter() - t0) * 1000.0
        full_response = (resp.choices[0].message.content or "").strip()

        log_event("llm_raw", {"raw": full_response, "took_ms": round(took_ms, 1)})

        # JSON 파싱 (코드블록 래핑도 처리)
        try:
            raw = full_response.strip()
            if raw.startswith("```"):
                # ```json\n{...}\n```
                try:
                    start = raw.find("\n")
                    end = raw.rfind("```")
                    candidate = (
                        raw[start + 1 : end].strip()
                        if start != -1 and end != -1
                        else raw
                    )
                    data = (
                        json.loads(candidate)
                        if candidate.strip().startswith("{")
                        else {}
                    )
                except Exception:
                    data = {}
            else:
                data = json.loads(raw) if raw.startswith("{") else {}
            internal_analysis = (
                data.get("internal_analysis") or ""
            ).strip() or "분석 없음"
            final_answer = (data.get("final_answer") or "").strip() or full_response
            ts = data.get("turn_summary") or {}
            user_intent = (ts.get("user_intent") or "").strip() or "불명"
            ai_summary = (ts.get("ai_summary") or "").strip() or "불명"

            parsed_response = AssistantResponse(
                internal_analysis=internal_analysis,
                final_answer=final_answer,
                turn_summary=TurnSummary(
                    user_intent=user_intent, ai_summary=ai_summary
                ),
            )

            # Post-Guard 재검증
            try:
                guard = get_profile_guard()
                ok, reason = await guard.validate(
                    user_id=session_id,
                    user_input=(user_input or ""),
                    ai_output=final_answer,
                )
                if not ok:
                    final_answer = "죄송합니다. 해당 주제는 답변드릴 수 없습니다."
                    parsed_response = AssistantResponse(
                        internal_analysis=internal_analysis,
                        final_answer=final_answer,
                        turn_summary=TurnSummary(
                            user_intent=user_intent, ai_summary=ai_summary
                        ),
                    )
                    log_event("guard_violation_blocked", {"reason": reason})
            except Exception:
                pass

            # 사전 검증 및 보완 재생성
            if user_input is not None:
                ok = await validate_final_answer(
                    user_input, rag_ctx or "", web_ctx or "", final_answer
                )
                if (
                    (not ok)
                    and (turn_state is not None)
                    and (turn_state.route == "web")
                ):
                    # 결정적 렌더(증거 기반 강제): web_refs_ctx/web_ctx의 블록만으로 정형 출력
                    try:
                        from backend.search_engine.formatter import (
                            blocks_to_items as _bti,
                        )

                        refs_ctx = (web_refs_ctx or "").strip() or (
                            web_ctx or ""
                        ).strip()
                        items = _bti(refs_ctx) if refs_ctx else []
                        if items:
                            # EID 주입: 세션의 마지막 EIDs와 순서를 맞춰 표기
                            if turn_state is not None:
                                eids = (
                                    SESSION_STATE.get(session_id, {})
                                    .get("last_eids_map", {})
                                    .get(turn_state.turn_id, [])
                                )
                            else:
                                eids = SESSION_STATE.get(session_id, {}).get(
                                    "last_eids", []
                                )
                            lines = ["검색 결과 기반 추천:"]
                            for it in items:
                                title = (it.get("title") or "").strip()
                                desc = (it.get("desc") or "").strip()
                                url = (it.get("url") or "").strip()
                                if title and url:
                                    eid_tag = ""
                                    if eids:
                                        eid_tag = f" [{eids.pop(0)}]" if eids else ""
                                    lines.append(f"- {title}{eid_tag}")
                                    if desc:
                                        lines.append(f"  {desc}")
                                    lines.append(f"  {url}")
                            det = "\n".join(lines)
                            if det.strip():
                                final_answer = det
                                parsed_response = AssistantResponse(
                                    internal_analysis=internal_analysis,
                                    final_answer=final_answer,
                                    turn_summary=TurnSummary(
                                        user_intent=user_intent, ai_summary=ai_summary
                                    ),
                                )
                    except Exception:
                        # 실패 시 기존 응답 유지
                        pass

            # 로컬 추천 일관성 가드: 웹 컨텍스트 결과 기반으로 안전한 추천으로 보정
            try:
                from backend.generation.formatters import (
                    enforce_web_results_in_answer as _enforce,
                )

                # 웹 경로에서만 웹 결과 기반 보정을 허용
                if (
                    (turn_state is not None)
                    and (turn_state.route == "web")
                    and (web_ctx or "").strip()
                ):
                    enforced = _enforce(final_answer, web_ctx or "", top_k=3)
                    if enforced and enforced != final_answer:
                        final_answer = enforced
                        parsed_response = AssistantResponse(
                            internal_analysis=internal_analysis,
                            final_answer=final_answer,
                            turn_summary=TurnSummary(
                                user_intent=user_intent, ai_summary=ai_summary
                            ),
                        )
            except Exception:
                pass

            # [참고] 섹션 보강(링크 포함)
            try:
                from backend.generation.formatters import ensure_references_in_answer
                from backend.search_engine.formatter import blocks_to_items as _bti2

                # 참고 링크는 프롬프트 주입 여부와 무관하게, 별도의 refs 컨텍스트를 우선 사용한다.
                # 웹 경로에서만 참고 섹션(웹 링크) 삽입을 허용
                refs_ctx = (web_refs_ctx or "").strip() or (web_ctx or "").strip()
                if (
                    (turn_state is not None)
                    and (turn_state.route == "web")
                    and refs_ctx
                ):
                    # 1차: URL 보장 참고 섹션
                    final_answer = ensure_references_in_answer(final_answer, refs_ctx)
                    # 2차: EID 주석 포함 참고 섹션으로 교체 (가능할 때)
                    if turn_state is not None:
                        eids2 = (
                            SESSION_STATE.get(session_id, {})
                            .get("last_eids_map", {})
                            .get(turn_state.turn_id, [])
                        )
                    else:
                        eids2 = SESSION_STATE.get(session_id, {}).get("last_eids", [])
                    if eids2:
                        items2 = _bti2(refs_ctx) or []
                        if items2:
                            # 기존 [참고] 섹션 제거
                            import re as _re

                            final_answer = _re.sub(
                                r"\n?\[참고\][\s\S]*$", "", final_answer
                            ).rstrip()
                            lines = ["[참고]"]
                            for i, it in enumerate(items2):
                                if i >= len(eids2):
                                    break
                                eid = eids2[i]
                                title = (it.get("title") or "").strip()
                                desc = (it.get("desc") or "").strip() or "-"
                                url = (it.get("url") or "").strip()
                                if title and url and eid:
                                    lines.append(f"- {title} [{eid}]: {desc}")
                                    lines.append(f"  {url}")
                            if len(lines) > 1:
                                sep = (
                                    "\n\n"
                                    if final_answer and not final_answer.endswith("\n")
                                    else "\n"
                                )
                                final_answer = final_answer + sep + "\n".join(lines)
                    parsed_response = AssistantResponse(
                        internal_analysis=internal_analysis,
                        final_answer=final_answer,
                        turn_summary=TurnSummary(
                            user_intent=user_intent, ai_summary=ai_summary
                        ),
                    )
            except Exception:
                pass

            # 최종 정리 및 전송
            final_answer = cleanup_final_answer(final_answer)
            try:
                await websocket.send_text(final_answer)
            except Exception as se:
                # 클라이언트 단절 또는 이미 close 전송된 경우
                log_event(
                    "ws_send_failed",
                    {"error": repr(se), "phase": "final_answer"},
                    level=logging.WARNING,
                )
                # 더 이상 전송 시도하지 않고 정상 종료 경로로 이동
                return full_response, AssistantResponse(
                    internal_analysis=internal_analysis,
                    final_answer=final_answer,
                    turn_summary=TurnSummary(
                        user_intent=user_intent, ai_summary=ai_summary
                    ),
                )

            log_event(
                "llm_parsed_json",
                {
                    "internal_analysis": internal_analysis,
                    "final_answer": final_answer,
                    "user_intent": user_intent,
                    "ai_summary": ai_summary,
                },
            )
        except Exception as e:
            log_event(
                "llm_json_parse_error",
                {"error": repr(e), "raw": full_response[:1000]},
                level=logging.ERROR,
            )
            try:
                await websocket.send_text(full_response)
            except Exception as se:
                log_event(
                    "ws_send_failed",
                    {"error": repr(se), "phase": "json_parse_fallback"},
                    level=logging.WARNING,
                )
                # 전송 실패 시에도 파싱 실패 응답 객체를 리턴하여 상위 로직이 계속 진행되도록 함
            parsed_response = AssistantResponse(
                internal_analysis="파싱 실패",
                final_answer=full_response,
                turn_summary=TurnSummary(user_intent="불명", ai_summary="불명"),
            )
    except Exception as e:
        log_event("llm_call_error", {"error": repr(e)}, level=logging.ERROR)
        error_msg = "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        try:
            await websocket.send_text(error_msg)
        except Exception as se:
            # 이미 클로즈되었거나 단절된 경우: 로깅만 하고 종료
            from backend.utils.logger import log_event as _log_event2

            try:
                _log_event2(
                    "ws_send_failed",
                    {"error": repr(se), "phase": "llm_call_error"},
                    level=logging.WARNING,
                )
            except Exception:
                pass
        parsed_response = AssistantResponse(
            internal_analysis="LLM 호출 실패",
            final_answer=error_msg,
            turn_summary=TurnSummary(user_intent="불명", ai_summary="오류 발생"),
        )

    return full_response, parsed_response
