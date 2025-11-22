"""
최종 메시지 리스트 글로벌 토큰 캡핑 유틸리티

목표:
- SYSTEM + 히스토리 + 최종 사용자 프롬프트를 합친 전체 메시지 토큰 수가 모델 컨텍스트 창을 초과하지 않도록, 우선순위에 따라 안전하게 절단한다.

절단 우선순위(낮은 것부터 제거/압축):
1) 히스토리 메시지(오래된 것부터 제거)
2) 마지막 사용자 프롬프트의 컨텍스트 영역(질문과 이후 지시문/출력 규칙은 보존)
3) System 메시지(보존 플래그가 꺼져 있을 때만)

주의:
- 마지막 사용자 프롬프트는 XML 출력 규칙 등 후반부 지시문이 반드시 필요하므로, 해당 꼬리(tail)는 보존한다.
- "사용자 질문:" 라인은 반드시 보존한다. (질문 텍스트 자체도 최대한 보존)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("message_capper")


def _count_tokens_messages(messages: List[Dict[str, Any]], tokenizer) -> int:
    """메시지 배열의 총 토큰 수를 계산한다."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(tokenizer.encode(content))
    return total


def cap_final_messages(
    messages: List[Dict[str, Any]],
    total_cap: int,
    tokenizer,
    preserve_system: bool = True,
) -> List[Dict[str, Any]]:
    """
    최종 메시지 리스트를 total_cap 토큰 이내로 조정한다.

    절단 우선순위(낮은 것부터 제거):
    1) 히스토리 메시지 (오래된 것부터)
    2) 마지막 user 메시지의 컨텍스트 영역(질문/지시문 유지)
    3) System 메시지 (preserve_system=False일 때만)

    Args:
        messages: LLM에 전달할 메시지 리스트
        total_cap: 전체 토큰 상한
        tokenizer: tiktoken 등의 토크나이저(encode/decode 제공)
        preserve_system: True면 System 메시지는 최대한 보존

    Returns:
        조정된 메시지 리스트
    """
    try:
        current_tokens = _count_tokens_messages(messages, tokenizer)
    except Exception:
        # 토크나이저 오류 시 원본 반환(안전 폴백)
        return messages

    if current_tokens <= total_cap:
        return messages

    logger.warning(
        f"[message_capper] Messages exceed cap: {current_tokens} > {total_cap}. Trimming..."
    )

    # ── 1) 메시지 분류: system / history / last_user
    system_msgs: List[Dict[str, Any]] = []
    history_msgs: List[Dict[str, Any]] = []
    last_user_msg: Dict[str, Any] | None = None

    for idx, msg in enumerate(messages):
        role = str(msg.get("role") or "")
        if role == "system":
            system_msgs.append(msg)
        elif role in ("user", "assistant"):
            # 마지막 메시지가 user인 경우 별도 보관(최신 사용자 프롬프트)
            if role == "user" and idx == len(messages) - 1:
                last_user_msg = msg
            else:
                history_msgs.append(msg)
        else:
            # 알 수 없는 역할은 히스토리로 취급(안전)
            history_msgs.append(msg)

    # ── 2) 오래된 히스토리부터 제거
    def _assembled_with(user_msg: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        return system_msgs + history_msgs + ([user_msg] if user_msg else [])

    while (
        history_msgs
        and _count_tokens_messages(_assembled_with(last_user_msg), tokenizer)
        > total_cap
    ):
        removed = history_msgs.pop(0)
        logger.debug(
            f"[message_capper] removed history (role={removed.get('role')}) to meet cap"
        )

    # ── 3) 여전히 초과라면 마지막 사용자 프롬프트(Context 영역) 압축
    if (
        last_user_msg
        and _count_tokens_messages(_assembled_with(last_user_msg), tokenizer)
        > total_cap
    ):
        last_user_msg = _trim_last_user_prompt(
            last_user_msg,
            tokenizer,
            target_tokens=total_cap
            - _count_tokens_messages(system_msgs + history_msgs, tokenizer),
        )

    # ── 4) 여전히 초과인데 system 보존이 꺼져 있으면 system 압축
    if (not preserve_system) and _count_tokens_messages(
        _assembled_with(last_user_msg), tokenizer
    ) > total_cap:
        system_msgs = _trim_system_messages(
            system_msgs,
            tokenizer,
            target_tokens=max(
                0,
                total_cap
                - _count_tokens_messages(
                    history_msgs + ([last_user_msg] if last_user_msg else []), tokenizer
                ),
            ),
        )

    final_messages = _assembled_with(last_user_msg)
    final_tokens = _count_tokens_messages(final_messages, tokenizer)

    logger.info(
        f"[message_capper] Trimmed: {current_tokens} → {final_tokens} tokens ({len(messages)} → {len(final_messages)} messages)"
    )

    return final_messages


def _trim_last_user_prompt(
    user_msg: Dict[str, Any],
    tokenizer,
    target_tokens: int,
) -> Dict[str, Any]:
    """
    마지막 사용자 프롬프트 압축(명세 기반 섹션 절단)

    전략:
    - 프롬프트를 [HEAD(컨텍스트+질문)] + [TAIL(지시문/출력 규칙)]로 분리한다.
    - TAIL은 반드시 보존한다.
    - HEAD는 섹션 마커 기반으로 분리하고, 우선순위에 따라 낮은 섹션부터 토큰을 줄인다.
      우선순위(보존 우선): question > realtime > rag > web > memory > mobile > aux
    - 예산이 극히 부족하면 question 일부만 남기고 나머지 제거한다.
    """
    content: str = user_msg.get("content") or ""

    # 지시문 시작 마커(필수 출력 규칙) 탐지 — TAIL은 보존 대상
    tail_marker = "[⚠️ 필수 출력 규칙"
    tail_idx = content.find(tail_marker)

    # 마커가 없으면(비정형) 보수적 토큰 절단으로 폴백
    if tail_idx == -1:
        enc = tokenizer.encode(content)
        if len(enc) <= target_tokens:
            return {"role": "user", "content": content}
        # 질문 라인을 보존하려 시도하되, 불가능하면 토큰 상한 내로 절단
        q_idx_full = content.rfind("사용자 질문:")
        if q_idx_full != -1:
            head = content[:q_idx_full]
            question_and_after = content[q_idx_full:]
            # head는 가능한 한 제거하고 question을 최대한 남긴다.
            tail_tokens = 0
            remain_for_question = max(0, target_tokens - tail_tokens)
            qa_enc = tokenizer.encode(question_and_after)
            qa_trim = tokenizer.decode(qa_enc[:remain_for_question])
            return {"role": "user", "content": qa_trim}
        # 질문 위치를 특정할 수 없으면 단순 절단
        return {
            "role": "user",
            "content": tokenizer.decode(enc[: max(0, target_tokens)]),
        }

    head = content[:tail_idx]
    tail = content[tail_idx:]

    # 섹션 추출(HEAD 영역만 고려)
    # 마커가 없을 수도 있는 rag/web은 비어있을 수 있음
    sections: Dict[str, str] = {
        "realtime": _extract_section(head, "[현재 시각]", "["),
        "rag": _extract_section(head, "[RAG 컨텍스트]", "["),
        "web": _extract_section(head, "[웹 컨텍스트]", "["),
        "mobile": _extract_section(head, "[모바일 컨텍스트]", "["),
        "memory": _extract_section(head, "[대화 컨텍스트]", "["),
        "aux": _extract_section(head, "[맥락 보조]", "["),
        "question": _extract_section(head, "사용자 질문:", "\n\n["),
    }

    # 토큰 수 계산
    section_tokens: Dict[str, int] = {
        k: len(tokenizer.encode(v or "")) for k, v in sections.items()
    }

    tail_tokens = len(tokenizer.encode(tail))
    head_target = max(0, target_tokens - tail_tokens)

    # 현재 HEAD 토큰 합
    head_now = sum(section_tokens.values())
    if head_now <= head_target:
        rebuilt_head = _rebuild_prompt(sections)
        return {"role": "user", "content": rebuilt_head + tail}

    # 절단 우선순위(낮은 것부터 줄임)
    priority_order = ["mobile", "aux", "memory", "web", "rag"]

    def _shrink(key: str, reduce_by: int) -> None:
        nonlocal sections, section_tokens
        if (sections.get(key) or "") == "" or section_tokens.get(key, 0) == 0:
            return
        current = sections[key] or ""
        enc = tokenizer.encode(current)
        new_len = max(0, len(enc) - reduce_by)
        if new_len == 0:
            sections[key] = ""
        else:
            sections[key] = tokenizer.decode(enc[:new_len]) + "\n(... 이하 생략)"
        section_tokens[key] = len(tokenizer.encode(sections[key] or ""))

    # 1차: 낮은 우선순위부터 필요량만큼 줄이기
    while sum(section_tokens.values()) > head_target and any(
        section_tokens[k] > 0 for k in priority_order
    ):
        overflow = sum(section_tokens.values()) - head_target
        # 한 라운드에서 고르게 분산 절감(라운드 로빈)
        for key in priority_order:
            if sum(section_tokens.values()) <= head_target:
                break
            if section_tokens.get(key, 0) == 0:
                continue
            step = max(50, overflow // 3)  # 한 번에 과도 절단 방지
            _shrink(key, step)
            overflow = max(0, sum(section_tokens.values()) - head_target)

    # 2차: 그래도 초과면 realtime/question 일부만 제외하고 추가 절감 불가 → question을 최소 보존
    if sum(section_tokens.values()) > head_target:
        must_keep_q = max(60, len(tokenizer.encode(sections.get("question", ""))) // 2)
        # realtime을 먼저 최소화
        rt_tokens = section_tokens.get("realtime", 0)
        if rt_tokens > 0:
            _shrink(
                "realtime", max(0, rt_tokens - 40)
            )  # 최소 40토큰은 남김(현재 시각 한두 줄)
        # 그 뒤에도 초과면 question을 일부 남기고 절단
        if sum(section_tokens.values()) > head_target:
            q_txt = sections.get("question", "")
            q_enc = tokenizer.encode(q_txt)
            available_for_q = max(
                0,
                head_target
                - (sum(v for k, v in section_tokens.items() if k != "question")),
            )
            keep = max(0, min(len(q_enc), max(must_keep_q, available_for_q)))
            sections["question"] = tokenizer.decode(q_enc[:keep]) + (
                "\n(... 이하 생략)" if keep < len(q_enc) else ""
            )
            section_tokens["question"] = len(tokenizer.encode(sections["question"]))

    rebuilt_head = _rebuild_prompt(sections)
    return {"role": "user", "content": rebuilt_head + tail}


def _extract_section(content: str, start_marker: str, end_marker: str) -> str:
    """프롬프트에서 섹션 추출(간단 마커 기반)"""
    try:
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return ""
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            return content[start_idx:].strip()
        return content[start_idx:end_idx].strip()
    except Exception:
        return ""


def _rebuild_prompt(sections: Dict[str, str]) -> str:
    """섹션들을 다시 프롬프트로 조립(존재하는 섹션만 순서대로)"""
    parts: List[str] = []
    if sections.get("realtime"):
        parts.append(f"[현재 시각]\n{sections['realtime']}")
    if sections.get("rag"):
        parts.append(f"[RAG 컨텍스트]\n{sections['rag']}")
    if sections.get("web"):
        parts.append(f"[웹 컨텍스트]\n{sections['web']}")
    if sections.get("mobile"):
        parts.append(f"[모바일 컨텍스트]\n{sections['mobile']}")
    if sections.get("memory"):
        parts.append(f"[대화 컨텍스트]\n{sections['memory']}")
    if sections.get("aux"):
        parts.append(f"[맥락 보조]\n{sections['aux']}")
    if sections.get("question"):
        parts.append(f"사용자 질문: {sections['question']}")
    return "\n\n".join(parts)


def _trim_system_messages(
    system_msgs: List[Dict[str, Any]],
    tokenizer,
    target_tokens: int,
) -> List[Dict[str, Any]]:
    """
    System 메시지 압축

    전략:
    - 첫 메시지(IDENTITY)와 마지막 메시지(룰)는 최대한 보존
    - 중간(System 다이나믹/Directive)은 토큰 상한 내로 순차 채우거나 절단
    """
    if not system_msgs:
        return system_msgs

    if len(system_msgs) == 1:
        # 단일 System만 있는 경우, 필요 시 토큰 절단
        return [_trim_one_system(system_msgs[0], tokenizer, target_tokens)]

    identity_msg = system_msgs[0]
    rule_msg = system_msgs[-1]
    middle_msgs = system_msgs[1:-1]

    id_tokens = len(tokenizer.encode(identity_msg.get("content") or ""))
    rule_tokens = len(tokenizer.encode(rule_msg.get("content") or ""))

    remain = max(0, target_tokens - id_tokens - rule_tokens)

    trimmed_middle: List[Dict[str, Any]] = []
    for msg in middle_msgs:
        content = msg.get("content") or ""
        tks = len(tokenizer.encode(content))
        if tks <= remain:
            trimmed_middle.append(msg)
            remain -= tks
        else:
            # 남은 예산만큼 절단
            enc = tokenizer.encode(content)
            trimmed_content = tokenizer.decode(enc[:remain]) if remain > 0 else ""
            trimmed_middle.append({"role": "system", "content": trimmed_content})
            remain = 0
            break

    return [identity_msg] + trimmed_middle + [rule_msg]


def _trim_one_system(
    msg: Dict[str, Any], tokenizer, target_tokens: int
) -> Dict[str, Any]:
    content = msg.get("content") or ""
    enc = tokenizer.encode(content)
    if len(enc) <= target_tokens:
        return msg
    return {"role": "system", "content": tokenizer.decode(enc[: max(0, target_tokens)])}
