"""
backend.generation.evidence_supplement - 증거 지연 보강

타임아웃된 증거 수집을 백그라운드에서 완료 후 보강 메시지를 전송합니다.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

logger = logging.getLogger("evidence_supplement")


async def supplement_evidence(
    session_id: str,
    evidence_task: asyncio.Task,
    websocket: WebSocket,
    original_query: str,
    initial_answer: str,
    timeout_s: float = 2.0,
) -> None:
    """
    지연된 증거를 후속 메시지로 보강.

    Args:
        session_id: 세션 ID
        evidence_task: 증거 수집 비동기 태스크
        websocket: WebSocket 연결
        original_query: 원본 사용자 질문
        initial_answer: 초기 답변
        timeout_s: 최대 대기 시간 (초)
    """
    from backend.utils.logger import log_event

    try:
        # 최대 timeout_s 대기
        evidence = await asyncio.wait_for(evidence_task, timeout=timeout_s)

        log_event(
            "supplement_evidence_ready",
            {
                "session_id": session_id,
                "original_query": (original_query or "")[:100],
                "evidence_len": len(evidence or ""),
            },
        )

        refinement_prompt = (
            f"[초기 답변]\n{initial_answer}\n\n"
            f"[새로 찾은 증거]\n{evidence}\n\n"
            f"[사용자 질문]\n{original_query}\n\n"
            f"위 증거를 바탕으로 초기 답변을 보완/정정하라.\n"
            f"규칙:\n"
            f"1. 증거와 충돌하는 부분은 정정\n"
            f"2. 새로운 사실은 추가\n"
            f"3. 변경 사항만 간결하게 전달 (전체 재작성 X)\n"
            f"4. 형식: '추가 정보를 바탕으로 보완하자면, [보완 내용]'"
        )

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        refinement = await llm.ainvoke(refinement_prompt)

        supplement_msg = f"\n\n{refinement.content}"
        await websocket.send_text(supplement_msg)

        log_event(
            "supplement_evidence_applied",
            {"session_id": session_id, "refinement_len": len(supplement_msg)},
        )
    except asyncio.TimeoutError:
        from backend.utils.logger import log_event

        log_event(
            "supplement_evidence_timeout",
            {"session_id": session_id},
            level=logging.WARNING,
        )
    except Exception as e:
        from backend.utils.logger import log_event

        log_event(
            "supplement_evidence_error",
            {"session_id": session_id, "error": repr(e)},
            level=logging.ERROR,
        )
