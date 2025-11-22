"""
증거 블록 파싱 유틸리티
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("evidence_block_parser")


@dataclass
class EvidenceBlock:
    """증거 블록"""

    index: int
    title: str
    content: str
    source: str  # "rag" 또는 "web"
    original_tokens: int


def parse_evidence_blocks(
    evidence_text: str,
    source_type: str,
    tokenizer,
) -> List[EvidenceBlock]:
    """
    증거 텍스트를 블록 단위로 파싱

    블록 구분:
    - RAG: [출처: ...] 마커로 구분(실제 구현에서는 \n\n 기준으로 최소 블록화)
    - WEB: 빈 줄(\n\n)로 구분된 3줄 블록 (제목/설명/URL)

    Args:
        evidence_text: 증거 전체 텍스트
        source_type: "rag" 또는 "web"
        tokenizer: 토크나이저

    Returns:
        List[EvidenceBlock]: 파싱된 블록 리스트
    """
    blocks: List[EvidenceBlock] = []

    if not (evidence_text or "").strip():
        return blocks

    if source_type == "rag":
        # RAG 블록: [출처: ...] 마커로 구분. 실사용 포맷 변동을 고려해 \n\n 기반 분할
        raw_blocks = (evidence_text or "").split("\n\n")

        for idx, raw_block in enumerate(raw_blocks):
            lines = [l.strip() for l in (raw_block or "").split("\n") if l.strip()]
            if not lines:
                continue

            # 첫 줄에서 제목/출처 추출
            first_line = lines[0]
            if first_line.startswith("[출처:"):
                # [출처: 타입 | 도메인 | YYYY-MM] 제목
                parts = first_line.split("]", 1)
                # source_info = parts[0] + "]"  # 필요 시 사용
                title = parts[1].strip() if len(parts) > 1 else "(제목 없음)"
            else:
                # source_info = "(출처 없음)"
                title = first_line[:100]

            content = "\n".join(lines)
            tokens = len(tokenizer.encode(content))

            blocks.append(
                EvidenceBlock(
                    index=idx,
                    title=title,
                    content=content,
                    source=source_type,
                    original_tokens=tokens,
                )
            )

    elif source_type == "web":
        # 웹 블록: 빈 줄로 구분된 3줄 블록(제목/설명/URL) 가정
        raw_blocks = (evidence_text or "").split("\n\n")

        for idx, raw_block in enumerate(raw_blocks):
            lines = [l.strip() for l in (raw_block or "").split("\n") if l.strip()]
            if len(lines) < 2:
                continue

            title = lines[0][:100]
            content = "\n".join(lines)
            tokens = len(tokenizer.encode(content))

            blocks.append(
                EvidenceBlock(
                    index=idx,
                    title=title,
                    content=content,
                    source=source_type,
                    original_tokens=tokens,
                )
            )

    logger.info(f"[evidence_block_parser] Parsed {len(blocks)} {source_type} blocks")

    return blocks
