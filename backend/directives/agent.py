# LLM 기반 지시문 추출(배치)

import asyncio
import json
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
from backend.utils.retry import openai_chat_with_retry

from backend.config import get_settings
from backend.rag import retrieve_from_rag  # optional reuse by callers
from backend.utils.logger import get_logger, log_event

from .schema import DirectiveReport, DirectiveSnapshot

from backend.utils.tracing import traceable

_settings = get_settings()
LLM_MODEL = _settings.LLM_MODEL
THINKING_MODEL = _settings.THINKING_MODEL
EXTRACT_TIMEOUT_S = float(getattr(_settings, "DIR_EXTRACT_TIMEOUT_S", 5.0))

logger = get_logger("directives")

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


@traceable(name="Directives: extract_directives", run_type="chain", tags=["directives"])
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
            from backend.utils.schema_builder import build_json_schema

            kwargs["response_format"] = build_json_schema(
                "DirectiveSnapshot", JSON_SCHEMA, strict=True
            )
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {"directives": {}, "confidence": 0.0, "reasons": ["extract_failed"]}


# ---------------------- 다중 샘플 합의 ----------------------
def _merge_snapshots(snaps: list[dict]) -> dict:
    """
    다중 샘플 스냅샷을 키별로 안정 합성.
    - 분류형: 최빈값
    - 정수/실수: 중앙값
    - 리스트: 교집합(없으면 최빈 리스트)
    - confidence: 중앙값
    """
    import statistics as _stats
    from collections import Counter as _Counter

    if not snaps:
        return {"directives": {}, "confidence": 0.0, "reasons": ["empty_samples"]}
    ds = [s.get("directives") or {} for s in snaps]
    keys = set().union(*[d.keys() for d in ds]) if ds else set()
    out: dict = {}
    for k in keys:
        vals = [d.get(k) for d in ds if k in d]
        if not vals:
            continue
        v0 = vals[0]
        try:
            if isinstance(v0, (int, float)):
                out[k] = (
                    int(_stats.median(vals))
                    if isinstance(v0, int)
                    else float(_stats.median(vals))
                )
            elif isinstance(v0, list):
                inter = set(vals[0])
                for v in vals[1:]:
                    try:
                        inter = inter.intersection(set(v))
                    except Exception:
                        inter = set()
                        break
                if inter:
                    out[k] = list(inter)[:5]
                else:
                    flat = [x for v in vals for x in (v or [])]
                    out[k] = [x for x, _ in _Counter(flat).most_common(5)]
            else:
                out[k] = _Counter(vals).most_common(1)[0][0]
        except Exception:
            out[k] = v0
    confs = [float(s.get("confidence", 0.0)) for s in snaps]
    try:
        conf_med = float(_stats.median(confs)) if confs else 0.0
    except Exception:
        conf_med = max(confs) if confs else 0.0
    return {"directives": out, "confidence": conf_med, "reasons": ["consensus"]}


async def extract_directives_consensus(
    messages: List[BaseMessage],
) -> DirectiveSnapshot:
    n = int(getattr(_settings, "DIR_EXTRACT_SAMPLES", 3))
    n = max(1, min(5, n))
    tasks = [asyncio.create_task(extract_directives(messages)) for _ in range(n)]
    snaps: list[dict] = []
    for t in tasks:
        try:
            snaps.append(await t)
        except Exception:
            snaps.append(
                {"directives": {}, "confidence": 0.0, "reasons": ["extract_failed"]}
            )
    try:
        return _merge_snapshots(snaps)
    except Exception:
        # 합의 실패 시 첫 성공 샘플 또는 빈 스냅샷 반환
        for s in snaps:
            if s.get("directives"):
                return s
        return {"directives": {}, "confidence": 0.0, "reasons": ["consensus_failed"]}


SIGNALS_MODEL = getattr(_settings, "SIGNALS_MODEL", THINKING_MODEL)


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


@traceable(
    name="Directives: infer_persona", run_type="chain", tags=["directives", "persona"]
)
async def _infer_persona(messages: List[BaseMessage]) -> Dict[str, Any]:
    text = _msgs_to_text(messages, cap_chars=12000)
    try:
        kwargs: Dict[str, Any] = {
            "model": THINKING_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "주어진 대화 텍스트를 바탕으로 장기적 성향(Trait)인 Big Five(0~1 스케일)와 선택적 MBTI를 보수적으로 추정하라. "
                        "단기 감정/상태는 무시하고, 일관된 언어습관/반복적 선택 근거가 있을 때만 반영하라. "
                        "근거가 부족하면 MBTI는 null로 두고, Big Five는 0.5±0.1 범위의 보수 추정치를 사용하라(추측 금지). "
                        "JSON 키만 반환하고 불필요한 텍스트는 포함하지 마라."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 300,
        }
        m = (THINKING_MODEL or "").lower()
        supports = any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
        if supports:
            from backend.utils.schema_builder import build_json_schema

            kwargs["response_format"] = build_json_schema(
                "Persona", PERSONA_SCHEMA, strict=True
            )
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content) if content.startswith("{") else {}
    except Exception:
        return {}


# ---------------------- 통합 리포트 ----------------------


def _normalize_sentiment(raw: Dict[str, Any]) -> Dict[str, float]:
    """감정 비율 합=1 불변식 보장. 결측/비정상값은 균등분배로 복원."""
    pos = float(raw.get("positive_ratio", 0) or 0)
    neg = float(raw.get("negative_ratio", 0) or 0)
    neu = float(raw.get("neutral_ratio", 0) or 0)
    # 음수 방지 및 NaN 대비
    pos = max(0.0, pos)
    neg = max(0.0, neg)
    neu = max(0.0, neu)
    s = pos + neg + neu
    if s <= 0:
        return {
            "positive_ratio": 1 / 3,
            "negative_ratio": 1 / 3,
            "neutral_ratio": 1 - 2 / 3,
        }
    pos /= s
    neg /= s
    neu = 1.0 - pos - neg  # 누적 오차 방지
    return {
        "positive_ratio": round(pos, 6),
        "negative_ratio": round(neg, 6),
        "neutral_ratio": round(neu, 6),
    }


def _normalize_topics(
    topics: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, float]]:
    """토픽 상위 K 정규화(합=1). 잘못된 항목/음수는 제거."""
    cleaned: List[Dict[str, float]] = []
    for t in topics or []:
        label = t.get("label")
        w = t.get("weight")
        try:
            w = float(w)
        except Exception:
            continue
        if not label or w <= 0:
            continue
        cleaned.append({"label": str(label), "weight": float(w)})
    cleaned = cleaned[:top_k]
    s = sum(t["weight"] for t in cleaned)
    if s <= 0:
        return []
    acc = 0.0
    result: List[Dict[str, float]] = []
    for i, t in enumerate(cleaned):
        if i < len(cleaned) - 1:
            nw = round(t["weight"] / s, 6)
            acc += nw
        else:
            nw = round(max(0.0, 1.0 - acc), 6)
        result.append({"label": t["label"], "weight": nw})
    return result


# ---------------------- Signals(LLM) 스키마 ----------------------
SIGNALS_SCHEMA = {
    "name": "Signals",
    "schema": {
        "type": "object",
        "properties": {
            "v": {"type": "integer", "const": 2},
            "sentiment": {
                "type": "object",
                "properties": {
                    "positive_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "negative_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "neutral_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["positive_ratio", "negative_ratio", "neutral_ratio"],
                "additionalProperties": False,
            },
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["label", "weight"],
                    "additionalProperties": False,
                },
                "maxItems": 5,
            },
            "communication_style": {
                "type": "string",
                "enum": ["direct", "indirect", "formal", "casual", "mixed"],
            },
            "emotional_intensity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["sentiment", "topics", "communication_style"],
        "additionalProperties": False,
    },
    "strict": True,
}


@traceable(
    name="Directives: analyze_signals_llm",
    run_type="chain",
    tags=["directives", "signals"],
)
async def analyze_signals_llm(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    LLM 기반 동적 신호(Signals) 추출

    - sentiment: 긍정/부정/중립 비율(합=1)
    - topics: Top-5 라벨+가중치(합=1)
    - communication_style: direct/indirect/formal/casual/mixed
    - emotional_intensity: 0~1
    """
    text = _msgs_to_text(messages, cap_chars=12000)
    try:
        sys_prompt = (
            "너는 대화 분석 전문가다. 아래 대화를 분석하여 다음을 JSON으로 출력하라:\n"
            "1) sentiment: 사용자 발화의 감정 분포(합=1.0)\n"
            "2) topics: Top-5 주제(가중치 합=1.0)\n"
            "3) communication_style: direct/indirect/formal/casual/mixed 중 택1\n"
            "4) emotional_intensity: 0~1\n\n"
            "규칙:\n- AI(assistant) 발화는 무시하고 human 발화만 분석\n- 근거 부족 시 중립/낮은 값으로 보수 추정\n- topics 라벨은 구체적으로 기술"
        )

        kwargs: Dict[str, Any] = {
            "model": SIGNALS_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": 500,
            "temperature": 0.0,
        }
        m = (SIGNALS_MODEL or "").lower()
        supports = any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
        if supports:
            from backend.utils.schema_builder import build_json_schema

            kwargs["response_format"] = build_json_schema(
                "Signals", SIGNALS_SCHEMA, strict=True
            )

        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}

        # 기본값 보정 및 정규화
        sentiment_raw = data.get("sentiment", {})
        topics_raw = data.get("topics", [])
        comm_style = data.get("communication_style", "mixed") or "mixed"
        intensity = data.get("emotional_intensity", 0.5)
        try:
            intensity = float(intensity)
        except Exception:
            intensity = 0.5
        intensity = max(0.0, min(1.0, intensity))

        sentiment = _normalize_sentiment(sentiment_raw)
        topics = _normalize_topics(topics_raw, top_k=5)

        result = {
            "v": 2,
            "sentiment": sentiment,
            "topics": topics,
            "communication_style": comm_style,
            "emotional_intensity": intensity,
        }

        # 로깅(구조화 + 가독 메시지)
        log_event("signals_llm", data={"result": result})
        try:
            logger.info(f"[signals_llm] extracted: {result}")
        except Exception:
            pass

        return result
    except Exception as e:
        logger.exception(f"[signals_llm] extraction failed: {repr(e)}")
        fallback = {
            "v": 2,
            "sentiment": {
                "positive_ratio": 1 / 3,
                "negative_ratio": 1 / 3,
                "neutral_ratio": 1 - 2 / 3,
            },
            "topics": [],
            "communication_style": "mixed",
            "emotional_intensity": 0.5,
        }
        log_event("signals_llm_error", data={"error": repr(e)})
        return fallback


@traceable(
    name="Directives: extract_report", run_type="chain", tags=["directives", "report"]
)
async def extract_report(messages: List[BaseMessage]) -> DirectiveReport:
    """
    통합 리포트 생성: Directives + Signals(LLM) + Persona

    - 3개 작업을 병렬로 수행해 대기시간을 최소화한다.
    """
    d_task = asyncio.create_task(extract_directives_consensus(messages))
    s_task = asyncio.create_task(analyze_signals_llm(messages))
    p_task = asyncio.create_task(_infer_persona(messages))

    d = await d_task
    s = await s_task
    p = await p_task

    report: DirectiveReport = {
        "directives": d.get("directives") or {},
        "signals": s or {},
        "persona": p or {},
        "confidence": float(d.get("confidence", 0.0)),
        "reasons": d.get("reasons", []),
    }
    return report
