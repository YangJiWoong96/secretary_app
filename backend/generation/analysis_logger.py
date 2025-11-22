"""
backend.generation.analysis_logger - AnalysisV1 감사 로그 호출기

TurnState(route, allowed_ctx, ctx_present)를 LLM에 전달하여
구조화 JSON만 응답받고, 서버는 일치성 검증 후 로그만 남긴다.
최종 답변은 절대 수정하지 않는다.
"""

from __future__ import annotations

import json
import logging
from typing import List

from backend.routing.turn_state import TurnState

logger = logging.getLogger("analysis_logger")

from backend.utils.tracing import traceable


@traceable(name="AnalysisV1: audit", run_type="chain", tags=["audit", "routing"])
async def run_analysis_v1(
    *,
    session_id: str,
    turn_state: TurnState,
    allowed_ctx: List[str],
    ctx_present: List[str],
) -> None:
    """
    AnalysisV1 구조화 분석 호출(비차단).
    - 입력: TurnState 요약(route), allowed_ctx, ctx_present
    - 출력: JSON(AnalysisV1 스키마) → 서버 검증 후 로그만 기록
    """
    try:
        from backend.config import get_settings
        from backend.utils.logger import safe_log_event
        from backend.utils.retry import openai_chat_with_retry
        from backend.utils.schema_builder import build_json_schema
        from backend.utils.schema_registry import get_analysis_v1_schema

        settings = get_settings()

        sys_msg = {
            "role": "system",
            "content": (
                "너는 감사 분석기다. 주어진 정보로만 판단하여 JSON만 출력한다. "
                "route는 입력 route와 동일해야 한다. "
                "ctx_used는 allowed_ctx ∩ ctx_present의 부분집합이어야 한다. "
                "추가 텍스트를 쓰지 말고 JSON만 출력하라."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"[route]\n{turn_state.route}\n\n"
                f"[allowed_ctx]\n{', '.join(allowed_ctx)}\n\n"
                f"[ctx_present]\n{', '.join(ctx_present)}\n"
            ),
        }

        rf = build_json_schema("AnalysisV1", get_analysis_v1_schema(), strict=True)
        resp = await openai_chat_with_retry(
            model=settings.LLM_MODEL,
            messages=[sys_msg, user_msg],
            temperature=0.0,
            max_tokens=120,
            response_format=rf,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw.startswith("{") else {}

        route_out = str(data.get("route") or "")
        ctx_used = list(data.get("ctx_used") or [])
        reasons = list(data.get("reasons") or [])
        notes = str(data.get("notes") or "")

        # 서버측 일치성 검증(답변 불변, 로그만)
        allowed_set = set(allowed_ctx or [])
        present_set = set(ctx_present or [])
        mask = allowed_set.intersection(present_set)
        used_set = set(ctx_used or [])

        valid = (route_out == turn_state.route) and used_set.issubset(mask)

        safe_log_event(
            "analysis_v1",
            {
                "session_id": session_id,
                "turn_id": turn_state.turn_id,
                "route": turn_state.route,
                "allowed_ctx": list(allowed_set),
                "ctx_present": list(present_set),
                "analysis": {
                    "route": route_out,
                    "ctx_used": ctx_used,
                    "reasons": reasons,
                    "notes": notes,
                },
                "valid": bool(valid),
            },
        )
        if not valid:
            safe_log_event(
                "analysis_v1_invalid",
                {
                    "session_id": session_id,
                    "turn_id": turn_state.turn_id,
                    "mask": list(mask),
                    "violations": {
                        "route_mismatch": route_out != turn_state.route,
                        "ctx_outside_mask": not used_set.issubset(mask),
                    },
                },
            )
    except Exception as e:
        try:
            from backend.utils.logger import safe_log_event

            safe_log_event("analysis_v1_error", {"error": repr(e)})
        except Exception:
            logger.warning(f"[analysis_v1] error: {e}")
