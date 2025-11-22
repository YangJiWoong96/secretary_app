"""
사용자 피드백 감지기 (LLM 기반)
"""

from __future__ import annotations

import json
import logging
from typing import List

from backend.rag.evidence_feedback_schema import EvidenceFeedback

logger = logging.getLogger("feedback_detector")


async def detect_evidence_feedback(
    user_input: str,
    ai_output: str,
    prev_rag_ctx: str,
    prev_web_ctx: str,
    session_id: str,
    turn_id: str,
) -> List[EvidenceFeedback]:
    """
    사용자 발화에서 이전 증거에 대한 피드백 감지 (LLM json_schema)
    """
    if not prev_rag_ctx and not prev_web_ctx:
        return []

    from backend.utils.retry import openai_chat_with_retry
    from backend.utils.schema_registry import get_feedback_analysis_schema

    # JSON Schema는 build_json_schema에 '이름'과 '순수 스키마 본문'을 전달해야 합니다.
    # 기존 코드의 이중 래핑({"name":..., "schema":...})은 OpenAI가 type을 None으로 해석하여 400을 유발했습니다.
    schema = get_feedback_analysis_schema()

    sys_msg = {
        "role": "system",
        "content": (
            "너는 사용자 피드백 분석기다. 이전 AI 답변(증거 포함)에 대해 "
            "긍정/부정/수정/추가 피드백을 분석하고 JSON으로만 출력한다."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"[이전 AI 답변]\n{ai_output[:500]}\n\n"
            f"[이전 RAG 증거]\n{prev_rag_ctx[:300] if prev_rag_ctx else '(없음)'}\n\n"
            f"[이전 웹 증거]\n{prev_web_ctx[:300] if prev_web_ctx else '(없음)'}\n\n"
            f"[현재 사용자 입력]\n{user_input}\n\n"
            "JSON만 출력"
        ),
    }

    try:
        from backend.utils.schema_builder import build_json_schema

        resp = await openai_chat_with_retry(
            model="gpt-4o-mini",
            messages=[sys_msg, user_msg],
            response_format=build_json_schema("FeedbackAnalysis", schema, strict=True),
            temperature=0.0,
            max_tokens=600,
            timeout=1.5,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}

        if not data.get("has_feedback", False):
            return []

        out: List[EvidenceFeedback] = []
        for fb in data.get("feedbacks", []) or []:
            try:
                et = str(fb.get("evidence_type", "")).lower()
                if et not in ("rag", "web"):
                    continue
                import hashlib

                snippet = str(fb.get("evidence_snippet", ""))[:100]
                evidence_id = hashlib.sha256(snippet.encode()).hexdigest()[:16]
                out.append(
                    EvidenceFeedback(
                        session_id=session_id,
                        turn_id=turn_id,
                        evidence_type=et,  # type: ignore[arg-type]
                        evidence_id=evidence_id,
                        original_evidence_snippet=snippet,
                        feedback_type=str(fb.get("feedback_type")),  # type: ignore[arg-type]
                        user_comment=str(fb.get("user_comment", "")),
                        ai_response=ai_output[:500],
                        timestamp=int(__import__("time").time()),
                        confidence_adjustment=float(
                            fb.get("confidence_adjustment", 0.0)
                        ),
                    )
                )
            except Exception as ie:
                logger.warning(f"feedback parse skipped: {ie}")
        logger.info(f"[feedback_detector] Detected {len(out)} feedbacks")
        return out

    except Exception as e:
        logger.error(f"[feedback_detector] Error: {e}")
        return []
