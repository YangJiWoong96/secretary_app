# 지시문 JSON 스키마 정의

from typing import Dict, List, Literal, Optional, TypedDict

Tone = Literal["neutral", "friendly", "professional", "warm"]
Formality = Literal["banmal", "jondaemal", "neutral"]  # 반말/존댓말/중립
Emotion = Literal["calm", "empathetic", "encouraging", "matter_of_fact"]


class Directives(TypedDict, total=False):
    # === 사용자 고정 취향(주제 무관) ===
    tone: Tone  # 전체 말투
    formality: Formality  # 존댓말/반말
    emotion: Emotion  # 정서적 기조
    style: List[str]  # ["두괄식","간결","근거중심","불필요한 서사 배제"] 등
    verbosity: int  # 1(아주짧게)~5(상세히), 기본 2
    emojis: bool  # 이모지 사용 여부
    markdown: bool  # 마크다운 사용 여부(목록/굵게 정도)
    language: Literal["ko"]  # 고정 한국어
    taboo_phrases: List[str]  # 싫어하는 표현(예: "친애하는", "안녕하세요~" 등)
    do: List[str]  # 꼭 지킬 습관(예: "두괄식 한줄 요약 선행")
    dont: List[str]  # 하지 말 것(예: "키워드 나열 금지")


class DirectiveSnapshot(TypedDict, total=False):
    directives: Directives
    confidence: float  # 0~1, 에이전트 확신도
    reasons: List[str]  # 왜 그렇게 판단했는지(감사/로그용)


# === Signals: 언어/주제/메타/모바일/정서 분포 등 동적 지표(프롬프트 보조) ===
class Signals(TypedDict, total=False):
    # --- v2: LLM 기반 동적 신호 ---
    v: int  # 스키마 버전(현재 2)
    sentiment: Dict[
        str, float
    ]  # {"positive_ratio":0~1, "negative_ratio":0~1, "neutral_ratio":0~1}
    topics: List[Dict[str, float]]  # [{"label":"음식","weight":0.22}, ...] (합=1)
    communication_style: str  # "direct" | "indirect" | "formal" | "casual" | "mixed"
    emotional_intensity: float  # 0~1

    # --- v1(기존 휴리스틱)과의 하위호환 ---
    language: Dict[str, float]
    style: Dict[str, float]
    meta: Dict[str, float]
    affect: Dict[str, float]
    mobile: Dict[str, object]


# === Persona: BigFive/MBTI 등 장기 성향(보수적 업데이트) ===
class Persona(TypedDict, total=False):
    bigfive: Dict[
        str, float
    ]  # {"openness":0~1, "conscientiousness":..., "extraversion":..., "agreeableness":..., "neuroticism":...}
    mbti: Optional[str]  # 예: "INTJ", "ENFP" 등 (확신 낮으면 생략)


# === 에이전트 반환: directives + signals + persona ===
class DirectiveReport(TypedDict, total=False):
    directives: Directives
    signals: Signals
    persona: Persona
    confidence: float
    reasons: List[str]
