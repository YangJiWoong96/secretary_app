# LLM 기반 지시문 추출(배치)

from dotenv import load_dotenv, find_dotenv
import os, json, asyncio

load_dotenv(find_dotenv(usecwd=True))

from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI
from backend.rag import retrieve_from_rag  # optional reuse by callers
from .schema import DirectiveSnapshot, DirectiveReport

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5")
THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")
EXTRACT_TIMEOUT_S = float(os.getenv("DIR_EXTRACT_TIMEOUT_S", "5.0"))

# 키가 없다면...
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not loaded. Check C:\\My_Business\\.env and dotenv loading."
    )

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 함수콜 강제 JSON 스키마(지시문 전용, 하위호환)
JSON_SCHEMA = {
    "name": "DirectiveSnapshot",
    "schema": {
        "type": "object",
        "properties": {
            "directives": {
                "type": "object",
                "properties": {
                    "tone": {
                        "type": "string",
                        "enum": ["neutral", "friendly", "professional", "warm"],
                    },
                    "formality": {
                        "type": "string",
                        "enum": ["banmal", "jondaemal", "neutral"],
                    },
                    "emotion": {
                        "type": "string",
                        "enum": ["calm", "empathetic", "encouraging", "matter_of_fact"],
                    },
                    "style": {"type": "array", "items": {"type": "string"}},
                    "verbosity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "emojis": {"type": "boolean"},
                    "markdown": {"type": "boolean"},
                    "language": {"type": "string", "enum": ["ko"]},
                    "taboo_phrases": {"type": "array", "items": {"type": "string"}},
                    "do": {"type": "array", "items": {"type": "string"}},
                    "dont": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["directives", "confidence"],
        "additionalProperties": False,
    },
}

SYS = (
    "넌 비서앱의 '고정 취향'만 추출하는 분석기다. 주제별/순간적 요구는 무시하고, "
    "일관된 말투/어투/정서/서술방식/이모지/길이 등 '오래 지속될 선호'만 요약해라. "
    "대화에 없는 값은 비워두고, 모순되면 최근 경향을 우선하되 과격하게 바꾸지 마라."
)
USR_TMPL = (
    "[대화 로그(과거→최근) 일부]:\n{conv}\n\n"
    "출력: JSON (스키마 준수). 지시문은 간결히. 코딩 관련 규칙은 넣지 말 것."
)


def _msgs_to_text(msgs: List[BaseMessage], cap_chars: int = 16000) -> str:
    parts = []
    for m in msgs:
        role = "user" if m.type == "human" else "assistant"
        ts = getattr(m, "additional_kwargs", {}).get("ts", "")
        parts.append(f"{role}({ts}): {m.content}")
    s = "\n".join(parts)
    return s[-cap_chars:]  # 뒤에서 캡


async def extract_directives(messages: List[BaseMessage]) -> DirectiveSnapshot:
    text = _msgs_to_text(messages)
    try:
        kwargs: Dict[str, Any] = {
            "model": THINKING_MODEL,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USR_TMPL.format(conv=text)},
            ],
            "max_tokens": 600,
        }
        m = (THINKING_MODEL or "").lower()
        supports = any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
        if supports:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA,
            }
        resp = await asyncio.wait_for(
            client.chat.completions.create(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {"directives": {}, "confidence": 0.0, "reasons": ["extract_failed"]}


# ---------------------- 휴리스틱 기반 신호 추출 ----------------------

_POS_WORDS = [
    "좋",
    "행복",
    "고마",
    "감사",
    "기쁘",
    "재밌",
    "즐겁",
    "멋지",
]
_NEG_WORDS = [
    "싫",
    "나쁘",
    "불편",
    "짜증",
    "화나",
    "빡치",
    "우울",
    "슬프",
]
_ANGER = ["화나", "빡치", "열받", "분노"]
_JOY = ["행복", "기쁘", "즐겁", "재밌", "좋"]
_SAD = ["우울", "슬프", "눈물", "ㅠ", "ㅜ"]
_PROFANITY = ["씨발", "좆", "병신", "fuck", "shit"]

_TOPIC_LEX = {
    "음식": ["맛집", "식당", "요리", "메뉴", "점심", "저녁", "카페"],
    "게임": ["게임", "랭크", "플레이", "캐릭터"],
    "경제": ["주식", "환율", "코스피", "코스닥", "나스닥", "금리", "물가"],
    "여행": ["여행", "항공", "호텔", "예약", "관광"],
    "운동": ["운동", "헬스", "러닝", "축구", "야구"],
    "IT": ["GPU", "AI", "모델", "서버", "프로그래밍", "코딩"],
    "정치": ["정치", "대선", "국회", "정당"],
    "문화": ["영화", "음악", "드라마", "공연"],
}


def _ratio(dividend: int, divisor: int) -> float:
    return (dividend / divisor) if divisor > 0 else 0.0


def _analyze_signals(messages: List[BaseMessage]) -> Dict[str, Any]:
    # 사용자 발화만 분석
    user_texts = [
        m.content or "" for m in messages if getattr(m, "type", "") == "human"
    ]
    if not user_texts:
        return {}
    all_text = "\n".join(user_texts)
    total_chars = sum(len(t) for t in user_texts)
    turns = len(user_texts)

    # 존댓말/반말 간이 비율
    jondaemal_hits = sum(t.count("요") + t.count("니다") for t in user_texts)
    banmal_hits = sum(t.count("야") + t.count("해") for t in user_texts)

    # 감성 단어 카운트
    pos = sum(sum(1 for w in _POS_WORDS if w in t) for t in user_texts)
    neg = sum(sum(1 for w in _NEG_WORDS if w in t) for t in user_texts)

    anger = sum(sum(1 for w in _ANGER if w in t) for t in user_texts)
    joy = sum(sum(1 for w in _JOY if w in t) for t in user_texts)
    sad = sum(sum(1 for w in _SAD if w in t) for t in user_texts)

    profanity = sum(sum(1 for w in _PROFANITY if w in t) for t in user_texts)
    excl = all_text.count("!")
    emoticons = all_text.count("ㅠ") + all_text.count("ㅜ") + all_text.count("ㅋㅋ")

    # 주제 분포(가중치)
    topic_scores: Dict[str, int] = {k: 0 for k in _TOPIC_LEX}
    for label, kws in _TOPIC_LEX.items():
        for kw in kws:
            topic_scores[label] += all_text.count(kw)
    # 상위 4개만 정규화
    top = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    total_score = sum(v for _, v in top) or 1
    topics = [
        {"label": k, "weight": round(v / total_score, 3)} for k, v in top if v > 0
    ]

    # 반복 주제(상위 토픽 비중으로 근사)
    repeat_topic_ratio = max([t["weight"] for t in topics], default=0.0)

    avg_turn_chars = _ratio(total_chars, turns)
    prefers_short = 1.0 if avg_turn_chars <= 35 else 0.0
    profanity_ratio = _ratio(profanity, turns)
    emotional_intensity = min(1.0, _ratio(excl + emoticons, turns))

    language_block = {
        "positive_ratio": round(_ratio(pos, pos + neg), 3),
        "negative_ratio": round(_ratio(neg, pos + neg), 3),
        "jondaemal_ratio": round(
            _ratio(jondaemal_hits, jondaemal_hits + banmal_hits), 3
        ),
    }
    style_block = {
        "prefers_short": prefers_short,
        "profanity_ratio": round(profanity_ratio, 3),
        "emotional_intensity": round(emotional_intensity, 3),
    }
    meta_block = {
        "avg_turn_chars": round(avg_turn_chars, 1),
        "avg_session_turns": float(turns),
        "repeat_topic_ratio": round(repeat_topic_ratio, 3),
    }
    affect_block = {
        "positive": round(_ratio(pos, turns), 3),
        "negative": round(_ratio(neg, turns), 3),
        "anger": round(_ratio(anger, turns), 3),
        "joy": round(_ratio(joy, turns), 3),
        "sadness": round(_ratio(sad, turns), 3),
    }

    return {
        "language": language_block,
        "topics": topics,
        "style": style_block,
        "meta": meta_block,
        "affect": affect_block,
    }


# ---------------------- 페르소나(LLM 보수적 추정) ----------------------

PERSONA_SCHEMA = {
    "name": "Persona",
    "schema": {
        "type": "object",
        "properties": {
            "bigfive": {
                "type": "object",
                "properties": {
                    "openness": {"type": "number", "minimum": 0, "maximum": 1},
                    "conscientiousness": {"type": "number", "minimum": 0, "maximum": 1},
                    "extraversion": {"type": "number", "minimum": 0, "maximum": 1},
                    "agreeableness": {"type": "number", "minimum": 0, "maximum": 1},
                    "neuroticism": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                ],
                "additionalProperties": False,
            },
            "mbti": {
                "type": ["string", "null"],
                "pattern": "^(?:[EI][SN][TF][JP])$",
            },
        },
        "required": ["bigfive"],
        "additionalProperties": False,
    },
}


async def _infer_persona(messages: List[BaseMessage]) -> Dict[str, Any]:
    text = _msgs_to_text(messages, cap_chars=12000)
    try:
        kwargs: Dict[str, Any] = {
            "model": THINKING_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "주어진 대화 텍스트를 바탕으로 Big Five(0~1 스케일)와 선택적 MBTI를 추정하라. "
                        "근거가 부족하면 MBTI는 null로 두어라. 추측 금지. 한국어 유지 불필요(키만 반환)."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 300,
        }
        m = (THINKING_MODEL or "").lower()
        supports = any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
        if supports:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": PERSONA_SCHEMA,
            }
        resp = await asyncio.wait_for(
            client.chat.completions.create(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content) if content.startswith("{") else {}
    except Exception:
        return {}


# ---------------------- 통합 리포트 ----------------------


async def extract_report(messages: List[BaseMessage]) -> DirectiveReport:
    """
    지시문(directives): LLM 기반(기존 스키마)
    신호(signals): 휴리스틱 기반(언어/주제/스타일/메타/감정)
    페르소나(persona): LLM 보수 추정(BigFive/선택적 MBTI)
    """
    # 1) 병렬 실행: 지시문(LLM) + 페르소나(LLM) + 휴리스틱
    heur = _analyze_signals(messages)
    d_task = asyncio.create_task(extract_directives(messages))
    p_task = asyncio.create_task(_infer_persona(messages))
    d = await d_task
    p = await p_task

    report: DirectiveReport = {
        "directives": d.get("directives") or {},
        "signals": heur,
        "persona": p or {},
        "confidence": float(d.get("confidence", 0.0)),
        "reasons": d.get("reasons", []),
    }
    return report
