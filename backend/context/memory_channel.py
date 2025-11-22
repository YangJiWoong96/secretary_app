"""
backend.context.memory_channel - Memory 채널 빌더

최근 4턴 원문을 강제 포함하고(Must include), MTM에서 관련 요약을 Top-K로 추가하여
프롬프트용 대화 컨텍스트를 구성한다. 총 토큰 예산은 token_budget으로 제한한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from backend.utils.tracing import traceable

logger = logging.getLogger("memory_channel")


@traceable(name="Memory: schedule_message_summary", run_type="tool", tags=["memory"])
async def schedule_message_summary(
    session_id: str,
    message_content: str,
    trigger_threshold: int = 800,
):
    """
    메시지 요약 백그라운드 스케줄러.

    - 현재 턴은 원문 사용, 요약은 캐시 미스 시 비동기로 준비해 다음 턴에 활용
    - 규칙 기반 1차 압축 → 임계 미달 시 캐시 저장 → 초과 시 LLM 보조(타임아웃)
    """
    from backend.context.korean_sentence_utils import (
        compress_middle_sentences_rule_based,
        extract_head_tail_sentences,
    )
    from backend.context.summary_cache import get_summary_cache
    from backend.memory.summarizer import get_tokenizer

    enc = get_tokenizer()
    cache = get_summary_cache()

    tokens_original = len(enc.encode(message_content or ""))
    if tokens_original <= int(trigger_threshold):
        return

    cached = await cache.get(session_id, message_content or "")
    if cached is not None:
        return

    async def _background_summarize():
        try:
            # 1) 규칙 기반 압축
            head, middle, tail = extract_head_tail_sentences(message_content or "")
            middle_compressed = compress_middle_sentences_rule_based(middle)
            rule_based_summary = f"{head} ... {middle_compressed} ... {tail}".strip()
            tokens_after_rule = len(enc.encode(rule_based_summary))

            if tokens_after_rule <= int(trigger_threshold):
                await cache.set(
                    session_id,
                    message_content or "",
                    rule_based_summary,
                    tokens_original,
                    tokens_after_rule,
                )
                return

            # 2) LLM 보조 요약 (짧은 타임아웃)
            try:
                from backend.context.compressor import compress_with_llm
                from backend.context.compressor_utils import verify_preservation

                llm_summary = await asyncio.wait_for(
                    compress_with_llm(
                        rule_based_summary,
                        target_tokens=int(trigger_threshold),
                        strategy="summary",
                    ),
                    timeout=0.35,
                )
                tokens_after_llm = len(enc.encode(llm_summary))

                if verify_preservation(message_content or "", llm_summary or ""):
                    await cache.set(
                        session_id,
                        message_content or "",
                        llm_summary,
                        tokens_original,
                        tokens_after_llm,
                    )
                else:
                    await cache.set(
                        session_id,
                        message_content or "",
                        rule_based_summary,
                        tokens_original,
                        tokens_after_rule,
                    )
            except asyncio.TimeoutError:
                await cache.set(
                    session_id,
                    message_content or "",
                    rule_based_summary,
                    tokens_original,
                    tokens_after_rule,
                )
        except Exception as e:
            try:
                logger.error(f"[summary_cache] background summarize error: {e}")
            except Exception:
                pass

    asyncio.create_task(_background_summarize())


@traceable(name="Memory: build_memory_channel", run_type="chain", tags=["memory"])
async def build_memory_channel(
    user_id: str,
    session_id: str,
    user_input: str,
    token_budget: int = 1000,
) -> str:
    """
    Memory 채널 구성 (수정 버전): 최근 4턴 원문/요약 + MTM Top-K

    - 각 메시지에 대해 요약 캐시 히트 시 요약 사용, 미스 시 원문 사용
    - 모든 메시지에 대해 다음 턴 대비 비동기 요약을 스케줄
    - MTM는 예산 내에서만 추가
    """
    from backend.context.summary_cache import get_summary_cache
    from backend.memory import get_memory_coordinator, get_short_term_memory
    from backend.memory.summarizer import get_tokenizer

    enc = get_tokenizer()
    coordinator = get_memory_coordinator()
    cache = get_summary_cache()

    # 1) 최근 4턴(= 8 메시지: user/ai 페어)
    stm = get_short_term_memory(session_id)
    msgs = getattr(stm, "chat_memory", {}).messages or []
    recent_8 = msgs[-8:]

    # 2) 캐시 확인 및 백그라운드 요약 스케줄
    recent_text_parts: List[str] = []
    for m in recent_8:
        role = "사용자" if getattr(m, "type", "") == "human" else "AI"
        content = getattr(m, "content", "") or ""

        summary = await cache.get(session_id, content)
        if summary is not None:
            recent_text_parts.append(f"{role}: {summary}")
        else:
            recent_text_parts.append(f"{role}: {content}")

        # 다음 턴 대비 비동기 요약 예약
        asyncio.create_task(
            schedule_message_summary(session_id, content, trigger_threshold=800)
        )

    recent_text = "[최근 대화]\n" + "\n".join(recent_text_parts)
    recent_tokens = len(enc.encode(recent_text))

    # 3) MTM 유사도 Top-K (예산 내 추가)
    mtm_text = ""
    mtm_budget = int(token_budget or 0) - recent_tokens
    if mtm_budget > 200:
        try:
            mtm_summaries = coordinator.mtm.get_relevant_summaries(
                user_id, session_id, user_input, top_k=5
            )
        except Exception:
            mtm_summaries = []

        if mtm_summaries:
            mtm_text = "[관련 과거 대화]\n"
            for summary in mtm_summaries:
                line = f"- {summary.get('mtm_summary', '')}\n"
                if len(enc.encode(mtm_text + line)) <= mtm_budget:
                    mtm_text += line
                else:
                    break

    # 4) 결합 및 반환 (메모리 채널 내에서 추가 압축은 수행하지 않음)
    combined = (recent_text or "").strip() + "\n\n" + (mtm_text or "").strip()
    return combined


def _compress_sentences(text: str) -> str:
    """[Deprecated] 간이 문장 압축(호환 유지용).

    실제 사용 경로는 한국어 유틸 기반의 비동기 요약/캐시로 대체되었다.
    """
    raw = (text or "").split(". ")
    compressed: List[str] = []
    for sent in raw:
        words = sent.split()
        unique_words: List[str] = []
        seen = set()
        for w in words:
            lw = w.lower()
            if lw not in seen:
                unique_words.append(w)
                seen.add(lw)
        s = " ".join(unique_words).strip()
        if s:
            compressed.append(s)
    return ". ".join(compressed)
