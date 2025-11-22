"""
backend.generation.message_builder - 메시지 조립

프롬프트 조립 및 최종 메시지 생성을 담당합니다.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

logger = logging.getLogger("message_builder")

from backend.utils.tracing import traceable


@traceable(name="Gen: build_messages", run_type="chain", tags=["gen", "prompt"])
async def build_messages(
    hist_msgs: List[Dict[str, str]],
    session_id: str,
    user_input: str,
    rag_ctx: str,
    web_ctx: str,
    mobile_ctx: str,
    conv_ctx: str,
    realtime_ctx: str,
    prev_turn_ctx: str = "",
) -> List[Dict[str, str]]:
    """
    프롬프트 조립 및 최종 메시지 생성.

    Args:
        hist_msgs: 선별된 히스토리 메시지
        session_id: 세션 ID
        user_input: 사용자 입력
        rag_ctx: RAG 컨텍스트
        web_ctx: WEB 컨텍스트
        mobile_ctx: 모바일 컨텍스트
        conv_ctx: 대화 컨텍스트
        realtime_ctx: 실시간 컨텍스트
        prev_turn_ctx: 이전 턴 요약

    Returns:
        List[Dict]: LLM에 전달할 최종 메시지 배열
    """
    from backend.directives.compiler import compile_unified_prompt_split
    from backend.directives.store import (
        acquire_compile_lock,
    )
    from backend.directives.store import get_compiled as get_compiled_directives
    from backend.directives.store import (
        get_unified_base,
        get_unified_overlay,
        release_compile_lock,
        set_unified_base,
        set_unified_overlay,
    )
    from backend.policy import SESSION_STATE
    from backend.prompts.system import (
        EVIDENCE_SYS_RULE,
        IDENTITY_PROMPT,
        FINAL_PROMPT,
        NO_EVIDENCE_SYS_RULE,
    )
    from backend.routing.router_context import user_for_session
    from backend.utils.logger import log_event

    # 1) Unified Directives 로드 (2계층 캐시 → 미스 시 컴파일)
    try:
        # 조건부 프로필 로드 플래그 (설정)
        from backend.config import get_settings as _gs

        profile_on_demand_enabled = bool(_gs().PROFILE_TIER_ON_DEMAND)
        # 롤아웃 플래그: 비활성 시 이전(1계층) 경로 사용
        if not bool(_gs().UNIFIED_COMPILER_ENABLED):
            try:
                from backend.directives.compiler import compile_unified_prompt
                from backend.directives.store import (
                    get_compiled_unified,
                    set_compiled_unified,
                )

                real_user_id = user_for_session(session_id) or session_id
                cached_prompt, cached_ver = get_compiled_unified(session_id, user_input)
                if cached_prompt:
                    slot_sys = cached_prompt
                else:
                    slot_sys, new_ver = await compile_unified_prompt(
                        user_id=real_user_id,
                        session_id=session_id,
                        user_query=user_input,
                        top_k=5,
                    )
                    set_compiled_unified(session_id, user_input, slot_sys, new_ver)
            except Exception:
                slot_sys, _ = get_compiled_directives(session_id)
            raise StopIteration

        base_hit = False
        overlay_hit = False

        overlay, base_ver = get_unified_overlay(session_id, user_input)
        if overlay:
            overlay_hit = True

        real_user_id = user_for_session(session_id) or session_id
        base = get_unified_base(real_user_id, base_ver) if base_ver else ""
        if base:
            base_hit = True

        if base and overlay:
            slot_sys = (base + "\n\n" + overlay).strip()
            try:
                log_event(
                    "unified_cache_hits", {"base": base_hit, "overlay": overlay_hit}
                )
            except Exception:
                pass
        else:
            lock_key, ok = acquire_compile_lock(session_id, user_input)
            if not ok:
                from backend.directives.store import get_compiled_unified

                cached_prompt, _cached_ver = get_compiled_unified(
                    session_id, user_input
                )
                if cached_prompt:
                    slot_sys = cached_prompt
                else:
                    slot_sys, _ = get_compiled_directives(session_id)
            else:
                t0 = time.time()
                try:
                    # 조건부 프로필 로드: RAG/Web 증거 존재 여부 계산 후 전달
                    has_evidence = bool(
                        (rag_ctx or "").strip() or (web_ctx or "").strip()
                    )
                    base_new, overlay_new, base_ver_new = (
                        await compile_unified_prompt_split(
                            user_id=real_user_id,
                            session_id=session_id,
                            user_query=user_input,
                            top_k=5,
                            has_evidence=(
                                has_evidence if profile_on_demand_enabled else None
                            ),
                        )
                    )
                    if base_new:
                        set_unified_base(real_user_id, base_ver_new, base_new)
                    set_unified_overlay(
                        session_id, user_input, overlay_new or "", base_ver_new
                    )
                    slot_sys = "\n\n".join([p for p in (base_new, overlay_new) if p])
                    try:
                        log_event(
                            "unified_compile_latency_ms",
                            {"ms": int((time.time() - t0) * 1000)},
                        )
                        log_event(
                            "unified_prompt_token_len",
                            {"approx": max(1, len(slot_sys) // 4)},
                        )
                    except Exception:
                        pass
                except Exception:
                    from backend.directives.store import get_compiled_unified

                    cached_prompt, _cached_ver = get_compiled_unified(
                        session_id, user_input
                    )
                    if cached_prompt:
                        slot_sys = cached_prompt
                    else:
                        slot_sys, _ = get_compiled_directives(session_id)
                finally:
                    release_compile_lock(lock_key)
    except StopIteration:
        pass
    except Exception:
        slot_sys, _ = get_compiled_directives(session_id)

    # 2) 프롬프트 템플릿 적용 (prompts.system의 FINAL_PROMPT 재사용)

    web_summary = ""
    if web_ctx and len(web_ctx) > 400:
        web_summary = "[요약] 검색 결과를 바탕으로 적절한 응답 제공: "

    aux_ctx = SESSION_STATE.get(session_id, {}).get("aux_ctx", "")

    # 증거 계약 힌트 (EID 사용 가이드)
    # (이미 rag_ctx에 포함되어 있으므로 추가 변경 불필요)

    # 가드: 라우팅 요약 태그 프롬프트 혼입 방지
    if "[ROUTING_ONLY]" in (rag_ctx or "") or "[ROUTING_ONLY]" in (conv_ctx or ""):
        log_event("guard_routing_summary_leak", {}, level=logging.ERROR)
        raise ValueError("Routing summary must not appear in prompt")

    conv_ctx_for_prompt = "" if hist_msgs else conv_ctx
    # conv_ctx가 충분히 길면 prev_turn_ctx는 생략하여 중복을 줄인다
    try:
        if len((conv_ctx or "").splitlines()) > 10:
            prev_turn_ctx = ""
    except Exception:
        pass
    prompt = FINAL_PROMPT.format(
        realtime_ctx=realtime_ctx,
        rag_ctx=rag_ctx,
        web_ctx=web_ctx,
        mobile_ctx=mobile_ctx,
        conv_ctx=conv_ctx_for_prompt,
        aux_ctx=aux_ctx,
        question=user_input,
        web_summary=web_summary,
        prev_turn_ctx=prev_turn_ctx,
    )

    # 3) 증거 모드 판별
    evidence_mode = bool(rag_ctx.strip() or web_ctx.strip())
    sys_rule = EVIDENCE_SYS_RULE if evidence_mode else NO_EVIDENCE_SYS_RULE

    # 4) 메시지 조립
    messages = (
        [{"role": "system", "content": IDENTITY_PROMPT}]
        + ([{"role": "system", "content": slot_sys}] if slot_sys else [])
        + [{"role": "system", "content": sys_rule}]
        + hist_msgs
        + [{"role": "user", "content": prompt}]
    )

    # 5) 최종 글로벌 캡 적용
    try:
        from backend.context.message_capper import cap_final_messages
        from backend.memory.summarizer import get_tokenizer

        enc = get_tokenizer()
        from backend.config import get_settings as _gs2

        _s = _gs2()
        model_context_window = int(getattr(_s, "MODEL_CONTEXT_WINDOW", 16384))
        response_reserve = int(getattr(_s, "RESPONSE_RESERVE_TOKENS", 2048))
        final_cap = max(3400, model_context_window - response_reserve)

        messages = cap_final_messages(
            messages,
            total_cap=final_cap,
            tokenizer=enc,
            preserve_system=True,
        )

        log_event(
            "final_messages_capped",
            {
                "session_id": session_id,
                "total_cap": final_cap,
                "message_count": len(messages),
            },
        )
    except Exception as e:
        from backend.utils.logger import log_event

        log_event(
            "final_messages_cap_error",
            {"session_id": session_id, "error": repr(e)},
            level=logging.ERROR,
        )

    return messages
