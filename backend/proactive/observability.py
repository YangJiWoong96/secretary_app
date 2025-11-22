from __future__ import annotations

"""
세션4 관측성 유틸: Trace ID 전파, 구조화 로깅 헬퍼, 메트릭 배치 전송

주의:
- PII(원문) 로깅 금지. user_id는 해시만 사용할 것.
- send_metrics_batch는 실제 전송 구현체가 외부에 있을 수 있으므로 훅으로 남긴다.
"""

import asyncio
import hashlib
import queue
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.utils.logger import log_event

# ===============
# Trace Context
# ===============
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id or "")


def get_trace_id() -> str:
    val = _trace_id_var.get() or ""
    return val or str(uuid.uuid4())


def ensure_trace_id(state: Dict[str, Any]) -> str:
    tid = (state or {}).get("trace_id") or get_trace_id()
    set_trace_id(tid)
    return tid


# ===============
# Hash helpers (PII 최소화)
# ===============
def hash_user_id(user_id: str) -> str:
    try:
        return hashlib.sha256((user_id or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def hash_text(text: str) -> str:
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


# ===============
# Metrics (비동기 배치)
# ===============
_metric_queue: queue.Queue = queue.Queue(maxsize=1000)


def emit_metric(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
) -> None:
    try:
        _metric_queue.put_nowait(
            {
                "name": name,
                "value": float(value),
                "labels": dict(labels or {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except queue.Full:
        # 성능 우선: 메트릭 드롭
        pass


async def metric_batch_sender(send_metrics_batch):
    """주기적으로 메트릭을 배치 전송.

    Args:
        send_metrics_batch: Callable[[list[dict]], Awaitable[None]]
    """

    while True:
        batch = []
        for _ in range(200):  # 최대 200개 묶음
            try:
                m = _metric_queue.get_nowait()
                batch.append(m)
            except queue.Empty:
                break
        if batch:
            try:
                await send_metrics_batch(batch)
            except Exception:
                # 전송 실패는 무시(관측성보다는 본 기능 우선)
                pass
        await asyncio.sleep(10)


# ===============
# Log helpers
# ===============
def log_pipeline_event(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    payload = dict(data or {})
    payload.setdefault("trace_id", get_trace_id())
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    log_event(event_type, payload)


__all__ = [
    "set_trace_id",
    "get_trace_id",
    "ensure_trace_id",
    "hash_user_id",
    "hash_text",
    "emit_metric",
    "metric_batch_sender",
    "log_pipeline_event",
]
