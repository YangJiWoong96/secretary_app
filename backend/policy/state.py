import os
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PolicyState:
    pii_allowed: bool
    consent_long_term: bool
    retention: Dict[str, int]
    redaction: Dict[str, object]


@dataclass
class OpsConfig:
    timebox_ms: int
    cache_minutes: int
    topk_summaries: int
    bm25_weight: float
    emb_weight: float
    thresholds: Dict[str, float]
    stwm: Dict[str, int]
    summarize_trigger: Dict[str, float]


def get_policy_state() -> PolicyState:
    return PolicyState(
        pii_allowed=False,
        consent_long_term=bool(int(os.getenv("CONSENT_LONG_TERM", "1"))),
        retention={
            "stwm_ttl_min": int(os.getenv("STWM_TTL_MIN", "10")),
            "turn_summary_max": int(os.getenv("TURN_SUMMARY_MAX", "60")),
            "cache_minutes": int(os.getenv("CACHE_MINUTES", "3")),
        },
        redaction={
            "apply_on_store": True,
            "patterns": [
                r"[0-9]{2,4}-[0-9]{3,4}-[0-9]{4}",  # 전화번호
                r"[\w\.-]+@[\w\.-]+",  # 이메일
                r"\b\d{6}-\d{7}\b",  # 주민번호 형태(예시)
            ],
        },
    )


def get_ops_config() -> OpsConfig:
    return OpsConfig(
        timebox_ms=int(os.getenv("TIMEBOX_MS", "1200")),
        cache_minutes=int(os.getenv("CACHE_MINUTES", "3")),
        topk_summaries=int(os.getenv("TOPK_SUMMARIES", "3")),
        bm25_weight=float(os.getenv("BM25_WEIGHT", "0.6")),
        emb_weight=float(os.getenv("EMB_WEIGHT", "0.4")),
        thresholds={
            "tau": float(os.getenv("TAU", "0.55")),
            "delta": float(os.getenv("DELTA", "0.08")),
            "low_conf_margin": float(os.getenv("LOW_CONF_MARGIN", "0.18")),
            "fastpath_margin": float(os.getenv("FASTPATH_MARGIN", "0.15")),
        },
        stwm={
            "ttl_min": int(os.getenv("STWM_TTL_MIN", "10")),
            "max_slots": int(os.getenv("STWM_MAX_SLOTS", "30")),
            "flush_batch": int(os.getenv("STWM_FLUSH_BATCH", "10")),
        },
        summarize_trigger={
            "buf_tokens": float(os.getenv("SUM_BUF_TOKENS", "400")),
            "min_turns": float(os.getenv("SUM_MIN_TURNS", "4")),
            "topic_shift_cos": float(os.getenv("SUM_TOPIC_COS", "0.7")),
        },
    )


def redact_text(text: str) -> str:
    ps = get_policy_state()
    if not ps.redaction.get("apply_on_store", True):
        return text
    out = text or ""
    for pat in ps.redaction.get("patterns", []) or []:
        try:
            out = re.sub(pat, "[REDACTED]", out)
        except Exception:
            continue
    return out
