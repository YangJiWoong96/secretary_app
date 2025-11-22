"""
backend.generation.evidence_handlers

Evidence 관련 특수 요청 처리 핸들러

역할:
- Evidence 상세 요청 판별 (LLM 기반)
- 미사용 Evidence 재로드 판별 (LLM 기반)
- EID 원문 조회
- 미사용 증거 EID 재로딩
"""

import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("evidence_handlers")


async def detect_evidence_detail_request(
    user_input: str,
    history: str,
    llm_model: str,
) -> Tuple[bool, Optional[str]]:
    """
    사용자 입력이 특정 증거 원문 상세 요청인지 판별한다.

    Args:
        user_input: 사용자 입력 텍스트
        history: 대화 히스토리 (최근 1500자)
        llm_model: 사용할 LLM 모델명

    Returns:
        (want_detail, eid):
            - want_detail: 상세 요청 여부 (bool)
            - eid: 요청된 EID (있으면 문자열, 없으면 None)
    """
    try:
        from backend.utils.retry import openai_chat_with_retry

        hist_for_llm = history[-1500:] if len(history) > 1500 else history

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 증거 참조 시스템 인터프리터다. 입력이 특정 증거 원문 상세 요청인지 판단하라. "
                    "규칙: 사용자가 [E_...] 식별자를 명시하며 '자세히/상세/원문/보여줘' 등 의도로 상세 내용을 요청하면 "
                    "detail_request=true, eid에는 해당 식별자를 넣어라. 여러 개면 가장 명시적인 하나만. "
                    'JSON만 출력: {"detail_request": bool, "eid": string}'
                ),
            },
            {
                "role": "user",
                "content": f"[hist]\n{hist_for_llm}\n\n[input]\n{user_input}",
            },
        ]

        resp = await openai_chat_with_retry(
            model=llm_model,
            messages=msgs,
            response_format={"type": "json_object"},
            max_tokens=60,
            temperature=0.0,
        )

        content = (resp.choices[0].message.content or "").strip()
        data = {}
        if content.startswith("{"):
            try:
                data = json.loads(content)
            except Exception as e:
                logger.warning(
                    f"[detect_evidence_detail_request] JSON parse error: {e}"
                )
                data = {}

        want_detail = bool(data.get("detail_request"))
        eid = str(data.get("eid") or "").strip()

        return (want_detail, eid if want_detail else None)

    except Exception as e:
        logger.error(f"[detect_evidence_detail_request] Error: {e}")
        return (False, None)


async def detect_unused_evidence_reload(
    user_input: str,
    llm_model: str,
) -> bool:
    """
    사용자가 이전 턴의 미사용 증거나 추가/다른 정보를 원하는지 판별한다.

    Args:
        user_input: 사용자 입력 텍스트
        llm_model: 사용할 LLM 모델명

    Returns:
        reload_unused: 미사용 증거 재로드 여부 (bool)
    """
    try:
        from backend.utils.retry import openai_chat_with_retry

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 대화 맥락 관리자다. 사용자가 이전 턴의 미사용 증거나 추가/다른 정보를 원하면 "
                    "reload_unused=true로 설정한다. 키워드: '아까', '다른', '추가', '더', 'more', 'another'. "
                    'JSON만 출력: {"reload_unused": bool}'
                ),
            },
            {"role": "user", "content": user_input or ""},
        ]

        resp = await openai_chat_with_retry(
            model=llm_model,
            messages=msgs,
            response_format={"type": "json_object"},
            max_tokens=20,
            temperature=0.0,
        )

        content = (resp.choices[0].message.content or "").strip()
        reload_unused = False
        if content.startswith("{"):
            try:
                reload_unused = bool(json.loads(content).get("reload_unused"))
            except Exception as e:
                logger.warning(f"[detect_unused_evidence_reload] JSON parse error: {e}")
                reload_unused = False

        return reload_unused

    except Exception as e:
        logger.error(f"[detect_unused_evidence_reload] Error: {e}")
        return False


async def handle_evidence_detail_request(
    eid: str,
    session_id: str,
) -> Optional[str]:
    """
    EID에 해당하는 증거 원문을 조회한다.

    Args:
        eid: Evidence ID
        session_id: 세션 ID

    Returns:
        증거 원문 (있으면 문자열, 없으면 None)
    """
    try:
        from backend.context.evidence_contractor import get_evidence_contractor

        contractor = get_evidence_contractor()
        body = contractor.retrieve_evidence(eid) or "(증거 만료)"
        return body

    except Exception as e:
        logger.error(f"[handle_evidence_detail_request] Error: {e}")
        return None


async def reload_unused_evidence_eids(
    session_id: str,
    last_turn_id: str,
) -> List[str]:
    """
    마지막 턴의 미사용 증거 계약들을 조회하여 EID 리스트를 반환한다.

    Args:
        session_id: 세션 ID
        last_turn_id: 마지막 턴 ID

    Returns:
        미사용 EID 리스트
    """
    try:
        from backend.context.evidence_contractor import get_evidence_contractor

        contractor = get_evidence_contractor()
        unused_contracts = contractor.retrieve_unused_evidence(session_id, last_turn_id)

        if unused_contracts:
            return [c.eid for c in unused_contracts]
        else:
            return []

    except Exception as e:
        logger.error(f"[reload_unused_evidence_eids] Error: {e}")
        return []


__all__ = [
    "detect_evidence_detail_request",
    "detect_unused_evidence_reload",
    "handle_evidence_detail_request",
    "reload_unused_evidence_eids",
]
