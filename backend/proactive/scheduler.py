"""
프로액티브 스케줄러

메인 대화 파이프라인과 분리된 주기 작업으로, 활성 사용자 집합에 대해
멀티에이전트 시스템의 푸시 생성 루틴을 기동한다.
- 외부 계약(API/설정/로그 키)은 변경하지 않는다.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import List

from backend.directives.store import get_active_users

from .agent import select_and_send

_SCHED_STARTED = False


async def _run_once() -> int:
    """활성 사용자에 대해 1회 스캔 및 푸시 시도 실행.

    Returns:
        int: 처리된 사용자 수(스케줄 상태 관찰용)
    """
    users: List[str] = get_active_users() or []
    if not users:
        return 0
    max_conc = int(os.getenv("PROACTIVE_CONCURRENCY", "3"))
    sem = asyncio.Semaphore(max_conc)

    async def _task(uid: str):
        async with sem:
            try:
                await select_and_send(uid, max_send=1)
            except Exception:
                pass

    await asyncio.gather(*[asyncio.create_task(_task(u)) for u in users])
    return len(users)


async def _loop(interval_sec: float) -> None:
    """주기 실행 루프(백그라운드).

    Args:
        interval_sec: 반복 간격(초)
    """
    while True:
        try:
            await _run_once()
        except Exception:
            pass
        await asyncio.sleep(interval_sec)


async def ensure_proactive_scheduler() -> None:
    """앱 시작 시 1회만 호출하여 주기 실행 스케줄러를 작동시킨다."""
    global _SCHED_STARTED
    if _SCHED_STARTED:
        return
    _SCHED_STARTED = True
    interval = float(os.getenv("PROACTIVE_INTERVAL_SEC", "1800"))  # 기본 30분
    asyncio.create_task(_loop(interval))
