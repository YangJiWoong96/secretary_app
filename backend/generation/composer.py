from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ComposeInput:
    question: str
    persona10: str
    web_ctx_blocks: str
    rag_ctx_blocks: str
    citations_policy: str = "strict"


def compose_fact_answer(ci: ComposeInput) -> str:
    """
    Evidence-Gated Composer: 근거가 있을 때만 사실을 출력.
    - web_ctx가 있으면 각 블록을 그대로 나열(이름/설명/링크)
    - rag_ctx가 있으면 그 범위 내에서만 요약/인용
    - 근거가 없으면 빈 문자열 반환(상위 레이어에서 보강질문 처리)
    """
    has_web = bool(ci.web_ctx_blocks and ci.web_ctx_blocks.strip())
    # RAG는 LLM 요약이 필요하므로 직접 출력하지 않는다.
    if not has_web:
        return ""
    header = ""
    if len(ci.web_ctx_blocks) > 400:
        header = "[요약] 아래 검색 결과를 한 문장으로 요약: "
    body = ci.web_ctx_blocks.strip()
    out = f"{header}{body}" if header else body
    return out


def apply_style_wrapper(persona10: str, content: str) -> str:
    """
    스타일 래퍼: 말투/형식만 적용. 내용 변경 금지.
    간단 구현: 페르소나 선호가 있으면 첫 문장 톤만 조정.
    """
    if not content:
        return content
    if not persona10:
        return content
    # 최소 포매팅: 한 줄 프리앰블을 붙이지 않고 내용만 온화한 말투로 유지
    return content
