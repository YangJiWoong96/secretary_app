from typing import List


def build_capabilities_card() -> str:
    lines: List[str] = [
        "[능력 카드]",
        "- 대화: 일상 질의 응답, 톤/스타일 적용(동적 지시문)",
        "- 검색: MCP 기반 웹 검색(Naver) 결과 요약/인용",
        "- RAG: 개인 프로필/회차 요약(Milvus) 의미 검색 및 날짜 필터",
        "- 재작성: RAG/WEB 쿼리 재작성(쿼리 중심 단서 기반, 오염 방지)",
        "- 메모리: STWM(단기) 앵커/엔티티, Ko-NER+Gazetteer 지명 보정",
        "- 증거: 웹/라그 참조 포인터를 인덱싱해 후속 회상 지원",
        "- 안전: 근거 없을 때 리스트/링크 생성 억제, 사후 검토",
    ]
    return "\n".join(lines)
