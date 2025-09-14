# 지시문 JSON 스키마 정의

from typing import Literal, TypedDict, List, Optional, Dict

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
    # 언어 패턴
    language: Dict[
        str, float
    ]  # {"positive_ratio":0~1, "negative_ratio":0~1, "jondaemal_ratio":0~1, ...}
    # 주제 선호 분포 상위 N
    topics: List[Dict[str, float]]  # [{"label":"음식","weight":0.22}, ...]
    # 스타일 특징
    style: Dict[
        str, float
    ]  # {"prefers_short":0/1, "profanity_ratio":0~1, "emotional_intensity":0~1}
    # 대화 메타데이터
    meta: Dict[
        str, float
    ]  # {"avg_turn_chars":..., "avg_session_turns":..., "repeat_topic_ratio":...}
    # 감정/정서 분포
    affect: Dict[
        str, float
    ]  # {"positive":..., "negative":..., "anger":..., "joy":..., "sadness":...}
    # 모바일/캘린더/위치 요약(ingest 결합)
    mobile: Dict[
        str, object
    ]  # {"prime_time":"late_night", "prime_time_share":0.6, ...}


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
