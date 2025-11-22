import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class PolicyState:
    """개인정보/보존/마스킹 정책 상태(전역 기본값)."""

    pii_allowed: bool
    consent_long_term: bool
    retention: Dict[str, int]
    redaction: Dict[str, object]


@dataclass
class OpsConfig:
    """운영 파라미터(토큰 한도·캐시·임계 등). 외부 계약 불변."""

    timebox_ms: int
    cache_minutes: int
    topk_summaries: int
    bm25_weight: float
    emb_weight: float
    thresholds: Dict[str, float]
    stwm: Dict[str, int]
    summarize_trigger: Dict[str, float]


def get_policy_state() -> PolicyState:
    """정책 상태 기본값을 환경변수 기반으로 구성해 반환한다."""
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
    """운영 파라미터를 환경변수에서 로드해 반환한다."""
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
    """정책에 정의된 패턴 기반으로 텍스트를 마스킹한다(저장 시 적용)."""
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


# ===== 전역 상태 관리 =====


class GlobalState:
    """
    애플리케이션 전역 상태 관리

    프로필 DB, Bot Profile DB, 세션 상태 등을 중앙에서 관리합니다.
    스레드 안전하게 구현되어 있습니다.
    """

    def __init__(self):
        """GlobalState 초기화"""
        self._lock = Lock()
        self._profile_db: Dict[str, Dict] = {}

    @property
    def profile_db(self) -> Dict[str, Dict]:
        """사용자 프로필 DB"""
        return self._profile_db

    def get_profile(self, user_id: str) -> Dict:
        """
        사용자 프로필 가져오기

        Args:
            user_id: 사용자 ID

        Returns:
            Dict: 사용자 프로필 (없으면 빈 딕셔너리)
        """
        return self._profile_db.get(user_id, {})

    def set_profile(self, user_id: str, profile: Dict) -> None:
        """
        사용자 프로필 설정

        Args:
            user_id: 사용자 ID
            profile: 프로필 딕셔너리
        """
        with self._lock:
            self._profile_db[user_id] = profile


# ===== 싱글톤 인스턴스 =====
_global_state_instance: Optional[GlobalState] = None


def get_global_state() -> GlobalState:
    """
    전역 GlobalState 싱글톤 인스턴스 반환

    Returns:
        GlobalState: 전역 상태 관리자
    """
    global _global_state_instance

    if _global_state_instance is None:
        _global_state_instance = GlobalState()

    return _global_state_instance


# 호환성을 위한 전역 변수 (싱글톤 속성으로 리다이렉트)
PROFILE_DB: Dict[str, Dict] = {}


def _init_profile_dbs():
    """프로필 DB 전역 변수 초기화"""
    global PROFILE_DB
    state = get_global_state()
    PROFILE_DB = state.profile_db
