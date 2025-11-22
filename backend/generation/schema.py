"""
backend.generation.schema - LLM 응답 스키마 정의

멀티턴 맥락 유지를 위한 구조화된 응답 모델을 정의합니다.
"""

from pydantic import BaseModel, Field


class TurnSummary(BaseModel):
    """
    턴 요약 구조

    사용자 의도와 AI 답변을 간략하게 요약하여
    다음 턴의 맥락 유지에 활용합니다.
    """

    user_intent: str = Field(description="사용자 질문의 핵심 의도를 10자 내외로 요약")
    ai_summary: str = Field(description="AI 자신의 답변 내용을 10자 내외로 요약")


class AssistantResponse(BaseModel):
    """
    LLM 최종 응답 구조

    내부 분석, 최종 답변, 턴 요약을 포함하는 구조화된 응답입니다.
    - internal_analysis: 사용자에게 보여지지 않는 내부 분석 과정
    - final_answer: 사용자에게 전달되는 최종 답변
    - turn_summary: 다음 턴의 맥락 유지를 위한 요약
    """

    internal_analysis: str = Field(
        description="사용자에게 보여주지 않을, 답변 생성을 위한 내부 분석 과정. "
        "사용자의 요청과 사용할 컨텍스트를 명시."
    )
    final_answer: str = Field(
        description="사용자에게 최종적으로 보여줄 친근하고 간결한 답변."
    )
    turn_summary: TurnSummary = Field(description="현재 턴(turn)에 대한 구조화된 요약.")
