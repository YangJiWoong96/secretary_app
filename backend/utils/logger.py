"""
backend.utils.logger - 중앙 총괄 로거(단일 진입점)

특징:
- JSON 콘솔 로깅(개발 기본값: 풀덤프 허용)
- 컨텍스트(session_id, turn_id) 주입: contextvars 기반
- uvicorn/fastapi 접근/에러 로그 통합(핸들러 단일화)
- 편의 함수: get_logger, set_context, clear_context, log_event, init_logging

주의:
- 이 모듈만이 로깅 설정을 관장한다. 타 모듈에서 핸들러/포맷터 생성 금지.
- 데이터 마스킹 비활성(개발/테스트 전용). 원문 그대로 출력.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# =============================
# 컨텍스트 (세션/턴)
# =============================
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)
_turn_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "turn_id", default=None
)


# =============================
# 관찰 가능성 스키마/샘플링 설정
# =============================
# 이벤트 스키마 레지스트리 (필수/옵션 필드 및 샘플링 비율)
EVENT_SCHEMAS: dict[str, dict[str, object]] = {
    # 01A
    "history_recency_bias": {
        "required": ["recent_count", "older_count", "total_count"],
        "optional": ["lambda", "recent_turns"],
        "sample_rate": 1.0,
    },
    # 01B
    "evidence.stored": {
        "required": ["session_id", "eid_count"],
        "optional": ["ttl"],
        "sample_rate": 1.0,
    },
    # 01C
    "evidence.pruned": {
        "required": ["session_id", "cited_count", "unused_count"],
        "optional": [],
        "sample_rate": 1.0,
    },
    # 01D
    "archive.pending": {
        "required": ["session_id", "turn_id"],
        "optional": ["expiry_turns"],
        "sample_rate": 1.0,
    },
    "archive.promoted": {
        "required": ["session_id", "turn_id"],
        "optional": [],
        "sample_rate": 1.0,
    },
    "archive.dropped": {
        "required": ["session_id", "turn_id"],
        "optional": [],
        "sample_rate": 1.0,
    },
    "archive.expired": {
        "required": ["session_id", "turn_id"],
        "optional": [],
        "sample_rate": 1.0,
    },
    # 01E
    "profile.loaded": {
        "required": ["guard_count", "core_count", "dynamic_count"],
        "optional": ["has_evidence", "on_demand_enabled"],
        "sample_rate": 1.0,
    },
    # 01F (고빈도 → 1%)
    "adaptive_budget_allocated": {
        "required": ["evidence_cap", "memory_cap", "profile_cap"],
        "optional": ["caps"],
        "sample_rate": 0.01,
    },
    # 기타 참조 이벤트
    "evidence.retrieved": {
        "required": ["eid"],
        "optional": [],
        "sample_rate": 1.0,
    },
    "evidence.expired": {
        "required": ["eid"],
        "optional": [],
        "sample_rate": 1.0,
    },
    "evidence.reused": {
        "required": ["session_id", "active_eid_count"],
        "optional": [],
        "sample_rate": 1.0,
    },
}

# 샘플링 재현성을 위한 시드(옵션)
try:
    from backend.config import get_settings as _gs

    _seed_raw = getattr(_gs(), "OBSERVABILITY_SAMPLE_SEED", None)
    if _seed_raw is not None and str(_seed_raw).strip() != "":
        random.seed(int(_seed_raw))
except Exception:
    pass


def set_context(
    session_id: Optional[str] = None, turn_id: Optional[str] = None
) -> None:
    """세션/턴 컨텍스트를 설정한다(부분 갱신 허용)."""
    if session_id is not None:
        _session_id_var.set(session_id)
    if turn_id is not None:
        _turn_id_var.set(turn_id)


def clear_context() -> None:
    """세션/턴 컨텍스트를 초기화한다."""
    _session_id_var.set(None)
    _turn_id_var.set(None)


class _ContextFilter(logging.Filter):
    """logging.Filter: record에 session_id/turn_id를 삽입"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            setattr(record, "session_id", _session_id_var.get())
            setattr(record, "turn_id", _turn_id_var.get())
        except Exception:
            # 컨텍스트 주입 실패 시 무시
            setattr(record, "session_id", None)
            setattr(record, "turn_id", None)
        return True


class _JsonFormatter(logging.Formatter):
    """간결 JSON 포맷터(원문 풀덤프 허용)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        try:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        except Exception:
            ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

        payload: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None),
            "session_id": getattr(record, "session_id", None),
            "turn_id": getattr(record, "turn_id", None),
        }

        # 메시지는 event와 별개로 보관(필요 시)
        msg = record.getMessage()
        if msg:
            payload["message"] = msg

        # data(임의 구조체) 병합
        data = getattr(record, "data", None)
        if data is not None:
            # JSON 직렬화 가능한 형태로 보정
            try:
                json.dumps(data, ensure_ascii=False, default=str)
                payload["data"] = data
            except Exception:
                payload["data"] = str(data)

        # 예외 정보 포함
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            # 최후 방어: 문자열로 강제
            return f"{payload}"


# 전역 초기화 플래그
_initialized = False


def _ensure_root_config(level: int) -> None:
    """루트 핸들러/포맷터 단일화 구성."""
    global _initialized
    if _initialized:
        return

    # 루트 로거 정리
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)

    # uvicorn 계열 로거 연결
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "uvicorn.asgi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(level)
        lg.propagate = True  # 루트로 전파하여 단일 핸들러 사용

    _initialized = True


def init_logging(
    default_level: str | int = None, full_dump_default: bool = True
) -> None:
    """
    중앙 로깅 초기화(앱 시작 시 1회 호출).

    Args:
        default_level: 문자열 또는 정수 레벨. 없으면 환경변수 LOG_LEVEL 사용.
        full_dump_default: 개발 기본 풀덤프 토글(현재는 실제 동작엔 영향 없음).
    """
    # 레벨 결정
    if isinstance(default_level, str):
        level_name = default_level.upper().strip()
        level = getattr(logging, level_name, logging.INFO)
    elif isinstance(default_level, int):
        level = default_level
    else:
        try:
            from backend.config import get_settings as _gs2

            env = str(getattr(_gs2(), "LOG_LEVEL", "INFO")).upper().strip()
        except Exception:
            env = os.getenv("LOG_LEVEL", "INFO").upper().strip()
        level = getattr(logging, env, logging.INFO)

    # 풀덤프 토글(현재는 정보용. 필요 시 전역 변수로 관리 가능)
    os.environ.setdefault("LOG_FULL_DUMP", "1" if full_dump_default else "0")

    _ensure_root_config(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """이름 기반 로거를 반환(설정은 중앙에서 일괄 적용됨)."""
    return logging.getLogger(name or "app")


def log_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
    *,
    force: bool = False,
) -> None:
    """
    구조화 이벤트 로깅 도우미.

    Args:
        event: 이벤트 이름(예: "routing_scores")
        data: 추가 구조 데이터(원문 포함 가능)
        level: 로깅 레벨
        logger: 사용할 로거(미지정 시 루트)
    """
    lg = logger or get_logger()
    try:
        # 관찰 가능성 토글
        try:
            from backend.config import get_settings as _gs3

            observability_enabled = bool(getattr(_gs3(), "OBSERVABILITY_ENABLED", True))
        except Exception:
            observability_enabled = os.getenv(
                "OBSERVABILITY_ENABLED", "true"
            ).lower() in (
                "true",
                "1",
                "yes",
            )
        if not observability_enabled and not force:
            return

        # 입력 데이터 준비
        payload: Dict[str, Any] = dict(data or {})

        # PII/민감 텍스트 필드 정제: 원문 로깅 금지(길이만 보존)
        try:
            _sensitive_keys = {
                "text",
                "user_input",
                "final_answer",
                "hist_text",
                "rag_ctx",
                "web_ctx",
                "answer",
                "mobile_ctx",
                "prev_turn_ctx",
                "routing_ctx",
                "rag_query_text",
                "web_query",
            }
            for k in list(payload.keys()):
                v = payload.get(k)
                if isinstance(v, str) and (k in _sensitive_keys):
                    payload[k] = {"len": len(v)}
        except Exception:
            pass

        # 스키마 검증 및 샘플링(정의된 이벤트에 한함)
        schema = EVENT_SCHEMAS.get(event)
        if schema is not None:
            # 필수 필드 검증(누락 시 WARNING)
            try:
                required = list(schema.get("required", []))  # type: ignore[assignment]
                for field in required:
                    if field not in payload:
                        logging.getLogger("app").warning(
                            f"[log_event] Missing required field '{field}' in event '{event}'"
                        )
            except Exception:
                pass

            # 샘플링: force=True는 샘플링 무시
            if not force:
                try:
                    sample_rate = float(schema.get("sample_rate", 1.0))  # type: ignore[arg-type]
                except Exception:
                    sample_rate = 1.0
                if sample_rate < 1.0:
                    if random.random() > max(0.0, min(1.0, sample_rate)):
                        return

        # 공통 필드 자동 주입
        try:
            # ns 단위 타임스탬프
            payload.setdefault("timestamp", int(time.time_ns()))
            # 호출 레벨 문자열
            payload.setdefault("level", logging.getLevelName(level))
            # 세션 ID 주입(컨텍스트 기반, 명시값이 있으면 보존)
            sess = _session_id_var.get()
            if sess is not None and "session_id" not in payload:
                payload["session_id"] = sess
        except Exception:
            pass

        # 음수 레이턴시 방지: took_ms가 있으면 0 하한 적용
        if "took_ms" in payload:
            try:
                tm = float(payload.get("took_ms", 0.0))
                if tm < 0:
                    payload["took_ms"] = 0.0
            except Exception:
                pass

        lg.log(level, event, extra={"event": event, "data": payload})
    except Exception:
        # 마지막 방어: 최소한의 문자열 로그
        try:
            lg.log(level, f"{event} | {data}")
        except Exception:
            pass


# === 컨텍스트 가드 유틸 ===


def _ctx_brief(txt: Optional[str], head: int = 160) -> Dict[str, Any]:
    s = txt or ""
    return {
        "len": len(s),
        "empty": (len(s) == 0),
        "has_http": ("http://" in s) or ("https://" in s),
        "has_rag_scheme": ("rag://" in s),
        "head": s[:head],
    }


def log_ctx_guard(
    stage: str,
    user_input: str = "",
    web_ctx: Optional[str] = None,
    rag_ctx: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    컨텍스트가 뒤바뀌었는지 빠르게 확인하기 위한 스냅샷 로깅.
    - stage: 'context_filter_before' / 'context_filter_after' / 'refs_store' / 'post_verify' 등 호출 지점 이름
    """
    try:
        from backend.config import get_settings as _gs4

        if not bool(getattr(_gs4(), "LOG_GUARD", True)):
            return
    except Exception:
        if os.getenv("LOG_GUARD", "1") != "1":
            return
    data = {
        "stage": stage,
        "user_input_head": (user_input or "")[:160],
        "web_ctx": _ctx_brief(web_ctx),
        "rag_ctx": _ctx_brief(rag_ctx),
    }
    if extra:
        data.update(extra)
    # 뒤바뀐 의심 패턴 자동 표시(휴리스틱)
    swapped_hint = False
    try:
        # 정상이라면 web은 http(s)多 / rag는 rag://多 혹은 둘 다 없을 수 있음
        w = data["web_ctx"]
        r = data["rag_ctx"]
        if (not w["has_http"] and r["has_http"]) or (
            w["has_rag_scheme"] and not r["has_rag_scheme"]
        ):
            swapped_hint = True
    except Exception:
        pass
    data["suspect_swapped"] = swapped_hint
    log_event("ctx_guard", data)


__all__ = [
    "init_logging",
    "get_logger",
    "set_context",
    "clear_context",
    "log_event",
    "log_ctx_guard",
    "safe_log_event",
]


def safe_log_event(
    event: str,
    data: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
    *,
    force: bool = False,
) -> None:
    """
    log_event 안전 래퍼.
    - 외부 호출부에서 이중 try/except를 제거하기 위해 제공한다.
    - 이벤트 이름/데이터/레벨/포맷은 log_event와 동일하며, 내부 예외는 항상 삼킨다.
    """
    try:
        log_event(event, data, level=level, logger=logger, force=force)
    except Exception:
        # 완전 침묵이 기존 패턴과 동일한 동작이다.
        pass
