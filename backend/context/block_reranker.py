"""
증거 블록 재순위화 및 Top-K 선택기

요약된 증거 블록을 사용자 질의와의 의미 유사도 중심으로 재순위화하고,
주어진 토큰 예산 내에서 상위 블록을 선택/결합한다.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from backend.context.evidence_block_parser import EvidenceBlock
from backend.rag.embeddings import embed_query_openai

logger = logging.getLogger("block_reranker")


def _safe_norm(vec: np.ndarray) -> float:
    try:
        n = float(np.linalg.norm(vec))
        return n if n > 0 else 1.0
    except Exception:
        return 1.0


def rerank_and_select_blocks(
    blocks: List[EvidenceBlock],
    user_query: str,
    token_budget: int,
    tokenizer,
) -> str:
    """
    요약된 블록들을 재순위화하여 예산 내 Top-K 선택

    순위화 기준(가중 합):
    1) 쿼리와의 임베딩 유사도 (70%)
    2) 블록 길이 (짧을수록 유리) (15%)
    3) 압축/간결성 근사치(짧을수록 유리) (15%)

    Args:
        blocks: 요약된 블록 리스트
        user_query: 사용자 질의
        token_budget: 토큰 예산(선택 합산 상한)
        tokenizer: 토크나이저

    Returns:
        str: 선택된 블록들을 결합한 최종 텍스트(빈 문자열 가능)
    """
    if not blocks:
        return ""

    # 1) 쿼리 임베딩
    try:
        query_emb = embed_query_openai(user_query)
    except Exception:
        # 임베딩 실패 시, 모든 블록을 원래 순서로 예산 내에서만 선택
        logger.warning(
            "[block_reranker] embed_query failed; falling back to order-only selection"
        )
        selected_blocks: list[tuple[EvidenceBlock, float, int]] = []
        current_tokens = 0
        for blk in blocks:
            tk = len(tokenizer.encode(blk.content))
            if current_tokens + tk <= token_budget:
                selected_blocks.append((blk, 0.0, tk))
                current_tokens += tk
            else:
                break
        selected_blocks.sort(key=lambda x: x[0].index)
        return "\n\n".join([b.content for b, _, __ in selected_blocks])

    query_norm = _safe_norm(query_emb)

    # 2) 각 블록 스코어 계산
    block_scores: list[tuple[EvidenceBlock, float, int]] = []
    for block in blocks:
        try:
            blk_emb = embed_query_openai(block.content)
            blk_norm = _safe_norm(blk_emb)
            similarity = float(np.dot(query_emb, blk_emb) / (query_norm * blk_norm))
        except Exception:
            similarity = 0.0

        # 길이 기반 점수: 짧을수록 유리 (100토큰 스케일)
        block_tokens = len(tokenizer.encode(block.content))
        length_score = 1.0 / (1.0 + (block_tokens / 100.0))

        # 압축/간결성 근사: 짧을수록 높은 점수 (1000토큰 기준 캡)
        compression_score = 1.0 - (min(block_tokens, 1000) / 1000.0)
        compression_score = max(0.0, min(1.0, compression_score))

        final_score = 0.70 * similarity + 0.15 * length_score + 0.15 * compression_score

        block_scores.append((block, final_score, block_tokens))

    # 3) 스코어 기준 정렬
    block_scores.sort(key=lambda x: x[1], reverse=True)

    # 4) 예산 내에서 Top-K 선택
    selected: list[tuple[EvidenceBlock, float, int]] = []
    used_tokens = 0
    for blk, score, tk in block_scores:
        if used_tokens + tk <= token_budget:
            selected.append((blk, score, tk))
            used_tokens += tk
        else:
            break

    # 5) 선택이 0개라면, 가장 높은 스코어 1개라도 넣어 주는 안전 장치
    if not selected and block_scores:
        best_blk, best_score, best_tk = block_scores[0]
        selected = [(best_blk, best_score, best_tk)]
        used_tokens = best_tk

    # 6) 원래 순서(index) 복원
    selected.sort(key=lambda x: x[0].index)

    # 7) 결합
    final_text = "\n\n".join([blk.content for blk, _, __ in selected])

    logger.info(
        f"[block_reranker] Selected {len(selected)}/{len(blocks)} blocks, "
        f"{used_tokens}/{token_budget} tokens used"
    )

    return final_text
