"""
backend.utils.tracing - LangSmith 트레이싱 유틸

- langsmith 미설치/비활성 시에도 안전하게 동작하도록 폴백을 제공합니다.
- 각 모듈에서 `from backend.utils.tracing import traceable` 만 임포트해 사용하세요.
"""

from __future__ import annotations

from typing import Any, Callable
import os

try:
    # LangSmith SDK가 존재하면 실제 데코레이터를 사용
    from langsmith import traceable as _ls_traceable  # type: ignore
except Exception:  # pragma: no cover
    _ls_traceable = None  # type: ignore


def _noop_decorator(
    *dargs: Any, **dkwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    langsmith 부재 시 사용할 NO-OP 데코레이터.
    함수 시그니처를 그대로 유지하며, 호출 동작에는 영향을 주지 않습니다.
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _wrap


def traceable(
    *dargs: Any, **dkwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    LangSmith traceable 데코레이터 안전 래퍼.

    사용 예:
    @traceable(name="RAG: retrieve", run_type="chain", tags=["rag"])
    async def retrieve(...):
        ...
    """
    enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if not enabled or _ls_traceable is None:
        return _noop_decorator(*dargs, **dkwargs)
    return _ls_traceable(*dargs, **dkwargs)  # type: ignore
