"""
증거 블록 요약 엔진

블록 단위로 LLM을 호출해 사용자 질의와의 관련성이 높은 핵심 정보만 남기도록
병렬 요약을 수행한다. 긴 블록만 요약 대상으로 삼고, 시간/숫자/고유명사 보존을
강하게 지시한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from backend.context.evidence_block_parser import EvidenceBlock
from backend.utils.retry import openai_chat_with_retry

logger = logging.getLogger("block_summarizer")


async def summarize_evidence_blocks(
    blocks: List[EvidenceBlock],
    user_query: str,
    target_tokens_per_block: int = 150,
    max_concurrency: int = 5,
) -> List[EvidenceBlock]:
    """
    증거 블록들을 병렬로 요약한다.

    Args:
        blocks: 원본 블록 리스트
        user_query: 사용자 질의(요약 방향성 지정)
        target_tokens_per_block: 블록당 목표 토큰 수
        max_concurrency: 동시 실행 상한

    Returns:
        List[EvidenceBlock]: 요약된 블록 리스트(짧은 블록은 원본 유지)
    """
    # tiktoken 인코더는 메모리 요약 모듈에서 공용 제공
    from backend.memory.summarizer import get_tokenizer

    enc = get_tokenizer()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _summarize_one(block: EvidenceBlock) -> EvidenceBlock:
        """단일 블록 요약 루틴.

        - 이미 짧은 블록은 원본 유지
        - 요약 실패/타임아웃 시 원본 유지
        - 제목 프리픽스가 사라지지 않도록 간단 검증
        """
        async with semaphore:
            try:
                # 이미 충분히 짧으면 스킵
                if block.original_tokens <= target_tokens_per_block:
                    return block

                sys_msg = {
                    "role": "system",
                    "content": (
                        "너는 증거 요약 전문가다. 사용자 질의와 관련된 핵심 정보만 추출하라.\n"
                        "규칙:\n"
                        "1. 숫자/날짜/고유명사는 반드시 보존\n"
                        "2. 출처 정보([출처: ...])는 유지\n"
                        "3. 불필요한 수식어/부연설명 제거\n"
                        f"4. 목표: {target_tokens_per_block} 토큰 이내"
                    ),
                }

                user_msg = {
                    "role": "user",
                    "content": (
                        f"[사용자 질의]\n{user_query}\n\n"
                        f"[원본 블록]\n{block.content}\n\n"
                        "핵심 정보만 추출하여 요약"
                    ),
                }

                # 과도 지연 방지를 위한 타임아웃
                resp = await asyncio.wait_for(
                    openai_chat_with_retry(
                        model="gpt-4o-mini",
                        messages=[sys_msg, user_msg],
                        temperature=0.0,
                        max_tokens=min(300, target_tokens_per_block * 2),
                    ),
                    timeout=1.2,
                )

                summary = (resp.choices[0].message.content or "").strip()
                if not summary:
                    return block

                # 간단 검증: 제목 프리픽스가 사라지면 보강
                if block.title[:20] and block.title[:20] not in summary:
                    summary = f"{block.title}\n{summary}"

                summary_tokens = len(enc.encode(summary))

                logger.debug(
                    f"[block_summarizer] Block {block.index}: "
                    f"{block.original_tokens} → {summary_tokens} tokens"
                )

                # 반환 시 현재 길이를 original_tokens 필드에 기록(후속 랭커 호환)
                return EvidenceBlock(
                    index=block.index,
                    title=block.title,
                    content=summary,
                    source=block.source,
                    original_tokens=summary_tokens,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[block_summarizer] Timeout for block {block.index}, using original"
                )
                return block
            except Exception as e:
                logger.error(
                    f"[block_summarizer] Error summarizing block {block.index}: {e}"
                )
                return block

    # 병렬 요약 실행
    summarized_blocks = await asyncio.gather(*[_summarize_one(b) for b in blocks])

    total_original = sum(b.original_tokens for b in blocks)
    total_summarized = sum(b.original_tokens for b in summarized_blocks)
    try:
        ratio = (total_summarized / max(1, total_original)) * 100.0
    except Exception:
        ratio = 100.0

    logger.info(
        f"[block_summarizer] Summarized {len(blocks)} blocks: "
        f"{total_original} → {total_summarized} tokens ({ratio:.1f}%)"
    )

    return list(summarized_blocks)
