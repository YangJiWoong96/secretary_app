"""
웹 검색 결과 평가 및 선별 적재 판단 (LLM 기반)

규칙 기반 키워드/휴리스틱 필터는 사용하지 않으며, LLM의 구조화 출력을 통해 적재 여부를 판단한다.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

logger = logging.getLogger("web_evidence_evaluator")


async def should_archive_web_result(
    web_ctx: str,
    user_input: str,
    ai_output: str,
    user_next_turn: Optional[str] = None,
) -> Tuple[bool, float, str]:
    """
    웹 검색 결과를 RAG에 적재할지 LLM 기반 평가

    반환: (적재 여부, 신뢰도, 사유)
    """
    from backend.utils.logger import log_event

    if not web_ctx or len(web_ctx) < 50:
        log_event("web_evaluator_skip", {"reason": "web_ctx too short"})
        return False, 0.0, "web_ctx too short"

    from backend.utils.retry import openai_chat_with_retry
    from backend.utils.schema_registry import get_archive_decision_schema

    schema_inner = get_archive_decision_schema()

    sys_msg = {
        "role": "system",
        "content": (
            "너는 웹 검색 결과 적재 판단기다. 다음 기준으로 장기 저장 가치를 평가:\n"
            "1. 사실 정보 (날짜, 장소, 인물, 숫자, 사건 등) 포함 여부\n"
            "2. 신규성 (최신 뉴스, 업데이트된 정보)\n"
            "3. 재사용 가능성 (일반적 지식 vs 일회성 정보)\n"
            "4. 사용자 관심도 (후속 질문, 긍정적 반응)\n"
            "5. 출처 신뢰도(링크 신뢰 추정): 정부/공식/학계/언론 등 공신력, 원문과 제목/본문 정합성, 저자/기관 명시 여부 등을 바탕으로 0~1로 산출\n"
            "출력은 JSON이며 모든 필드를 반드시 포함하라: should_archive, confidence, reason, fact_density, novelty, user_interest, date_candidates, consensus_date_ymd, consensus_confidence."
        ),
    }

    next_turn_info = (
        f"\n\n[사용자 후속 반응]\n{user_next_turn}"
        if user_next_turn
        else "\n\n[사용자 후속 반응]\n(아직 없음)"
    )

    user_msg = {
        "role": "user",
        "content": (
            f"[웹 검색 결과]\n{web_ctx[:1200]}\n\n"
            f"[사용자 질문]\n{user_input}\n\n"
            f"[AI 답변]\n{ai_output[:600]}"
            f"{next_turn_info}\n\n"
            "출력 규칙:\n"
            "1) should_archive, confidence, reason, fact_density, novelty, user_interest를 포함하라.\n"
            "2) 각 블록에서 날짜(있다면)를 YYYYMMDD로 추출해 date_candidates에 모아라.\n"
            "3) 서로 다른 출처 간 일치하는 날짜가 다수이면 consensus_date_ymd로 결정하고, 신뢰도를 0~1로 산출(consensus_confidence).\n"
            "4) 일치가 약하거나 상충하면 consensus_date_ymd=null, consensus_confidence는 낮게.\n"
            "5) 각 링크의 출처 신뢰도를 0~1로 추정하여 평균을 confidence에 반영(단순 평균 가중).\n"
            "JSON만 출력"
        ),
    }

    try:
        import asyncio

        from backend.utils.schema_builder import build_json_schema

        resp = await asyncio.wait_for(
            openai_chat_with_retry(
                model="gpt-4o-mini",
                messages=[sys_msg, user_msg],
                response_format=build_json_schema(
                    "ArchiveDecision", schema_inner, strict=True
                ),
                temperature=0.0,
                max_tokens=300,
            ),
            timeout=1.8,
        )

        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}

        should_archive = bool(data.get("should_archive", False))
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))

        # 날짜 합의/후보 로깅(옵션)
        try:
            from backend.utils.logger import log_event

            log_event(
                "web_evidence_dates",
                {
                    "candidates": data.get("date_candidates", []),
                    "consensus_ymd": data.get("consensus_date_ymd"),
                    "consensus_confidence": data.get("consensus_confidence", 0.0),
                },
            )
        except Exception:
            pass

        logger.info(
            f"[web_evidence_evaluator] Archive decision: should_archive={should_archive}, confidence={confidence:.2f}"
        )

        # 로깅
        log_event(
            "web_evidence_evaluated",
            {
                "should_archive": should_archive,
                "confidence": confidence,
                "reason": reason,
                "fact_density": data.get("fact_density", 0.0),
                "novelty": data.get("novelty", 0.0),
                "user_interest": data.get("user_interest", 0.0),
            },
        )

        return should_archive, confidence, reason
    except asyncio.TimeoutError:
        # 타임아웃은 소프트-필: 적재하지 않고 조용히 패스
        try:
            from backend.utils.logger import log_event as _le

            _le("web_evaluator_timeout", {"len": len(web_ctx or "")})
        except Exception:
            pass
        return False, 0.0, "timeout"
    except Exception as e:
        logger.error(f"[web_evidence_evaluator] Error: {e}")
        try:
            from backend.utils.logger import log_event as _le

            _le("web_evaluator_error", {"error": repr(e)}, level=logging.ERROR)
        except Exception:
            pass
        return False, 0.0, f"evaluation_error: {str(e)}"
