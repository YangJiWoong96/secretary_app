from __future__ import annotations

"""
문단 기반 입력 파이프라인

역할:
- 사용자 자연어 문단(200~1000자)을 간단 분석하여 검색 쿼리/요약을 생성
- 생성된 쿼리/요약을 Proactive 파이프라인 상태에 주입한 뒤 기존 LangGraph를 실행

주의:
- 운영 의존성(예: koNLPy/kss)이 없을 수 있으므로, 초기 버전은 폴백 중심의 경량 구현
- 실패 시에도 기존 파이프라인과 호환되는 최소 키(web_queries, rag_query)를 보장
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from backend.proactive.multi_agent_graph import run_proactive_pipeline


class ParagraphAnalyzerAgent:
    """
    문단 분석 통합 에이전트(경량 구현).
    - 실제 상세 구현은 docs 지시서를 따르되, 환경 의존성 문제를 피하기 위해
      기본 키워드/요약 기반의 폴백을 제공한다.
    """

    async def analyze(self, paragraph: str) -> Dict[str, Any]:
        text = (paragraph or "").strip()
        # 1) 간단 요약(100자 클리핑)
        summary = text[:100]

        # 2) 간단 키워드(공백 기준 상위 단어 일부)
        try:
            words = [w for w in text.replace("\n", " ").split(" ") if len(w) > 1]
            key_phrases = list(dict.fromkeys(words))[:5]
        except Exception:
            key_phrases = []

        # 3) 검색 쿼리 (요약 + '최신 동향' 1개, 키워드 조합 1개)
        queries = []
        if summary:
            queries.append(f"{summary} 최신 동향")
        if len(key_phrases) >= 2:
            queries.append(f"{key_phrases[0]} {key_phrases[1]}")
        if not queries:
            queries = [text[:32] + " 최신 동향"] if text else []

        # 4) 필터(보수적 기본값)
        filters = {
            "date_range": "any",
            "source_types": ["news", "paper"],
            "trust_threshold": 0.7,
        }

        return {
            "key_topics": [],
            "key_phrases": key_phrases,
            "intents": [],
            "urgency": 5,
            "search_queries": queries[:5],
            "rag_query": summary,
            "filters": filters,
            "original_text": text,
            "summary": summary,
            "confidence": 0.6,
        }


async def run_paragraph_based_pipeline(paragraph: str, user_id: str) -> Dict[str, Any]:
    """
    문단 기반 멀티에이전트 파이프라인 실행기.
    - ParagraphAnalyzerAgent 결과를 ProactiveState에 주입
    - 기존 LangGraph 파이프라인을 동기 호출(스레드)로 실행
    """
    analyzer = ParagraphAnalyzerAgent()
    analysis = await analyzer.analyze(paragraph)

    initial_state: Dict[str, Any] = {
        "user_id": user_id,
        "session_id": user_id,
        "trigger_type": "paragraph_input",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # 문단 분석 결과 주입
        "paragraph_analysis": {
            "key_topics": analysis.get("key_topics", []),
            "key_phrases": analysis.get("key_phrases", []),
            "intents": analysis.get("intents", []),
            "urgency": analysis.get("urgency", 5),
            "confidence": analysis.get("confidence", 0.6),
        },
        # 검색 쿼리/요약
        "web_queries": analysis.get("search_queries", []),
        "rag_query": analysis.get("rag_query", ""),
        "search_filters": analysis.get("filters", {}),
        # 원문/요약
        "original_paragraph": analysis.get("original_text", paragraph),
        "paragraph_summary": analysis.get("summary", paragraph[:100]),
    }

    final_state = await asyncio.to_thread(run_proactive_pipeline, initial_state)  # type: ignore
    return final_state


__all__ = ["ParagraphAnalyzerAgent", "run_paragraph_based_pipeline"]
