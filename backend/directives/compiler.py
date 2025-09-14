# # JSON → system 주입용 미니 프롬프트
import json
from .schema import Directives

# 시스템 프롬프트는 "JSON 지시문 + 짧은 고정 헤더"만 넣습니다.
HEADER = (
    "You are an assistant for a Korean user. "
    "Follow the following JSON directives consistently across topics. "
    "If the user explicitly asks to deviate, confirm once and then adapt temporarily."
)


def _compact_signals(sig: dict) -> dict:
    """
    시스템 프롬프트에 넣을 만큼만 축약. 토큰 사용을 최소화한다.
    - language: positive/negative/jondaemal만 유지(소수점 1~2자리)
    - topics: 상위 3개
    - style/meta/affect: 핵심 1~2개만
    - mobile: prime_time/avg_calendar_events_per_day만
    """
    if not sig:
        return {}
    out = {}
    lang = sig.get("language") or {}
    if lang:
        out["language"] = {
            "positive": round(float(lang.get("positive_ratio", 0.0)), 2),
            "negative": round(float(lang.get("negative_ratio", 0.0)), 2),
            "jondaemal": round(float(lang.get("jondaemal_ratio", 0.0)), 2),
        }
    topics = sig.get("topics") or []
    if topics:
        out["topics"] = topics[:3]
    style = sig.get("style") or {}
    if style:
        out["style"] = {
            "prefers_short": style.get("prefers_short", 0.0),
            "emotional_intensity": style.get("emotional_intensity", 0.0),
        }
    meta = sig.get("meta") or {}
    if meta:
        out["meta"] = {"repeat_topic_ratio": meta.get("repeat_topic_ratio", 0.0)}
    affect = sig.get("affect") or {}
    if affect:
        out["affect"] = {
            "positive": affect.get("positive", 0.0),
            "negative": affect.get("negative", 0.0),
        }
    mobile = sig.get("mobile") or {}
    if mobile:
        out["mobile"] = {
            "prime_time": mobile.get("prime_time"),
            "avg_calendar_events_per_day": mobile.get("avg_calendar_events_per_day"),
        }
    return out


def compile_prompt_from_json(
    d: Directives, signals: dict | None = None, persona: dict | None = None
) -> str:
    # 꼭 필요한 키만 유지(토큰 절약)
    allow = [
        "tone",
        "formality",
        "emotion",
        "style",
        "verbosity",
        "emojis",
        "markdown",
        "language",
        "taboo_phrases",
        "do",
        "dont",
    ]
    slim = {
        k: v for k, v in (d or {}).items() if k in allow and v not in (None, [], "")
    }
    body = {"directives": slim}
    sig_comp = _compact_signals(signals or {})
    if sig_comp:
        body["signals"] = sig_comp
    if persona:
        # persona는 프롬프트 오염 방지를 위해 bigfive만 축약 반영
        bf = (persona.get("bigfive") or {}) if isinstance(persona, dict) else {}
        if bf:
            body["persona"] = {"bigfive": bf}
    return HEADER + "\n\n" + json.dumps(body, ensure_ascii=False, separators=(",", ":"))
