import asyncio
import os
from datetime import datetime, timezone
from typing import List

from .agent import select_and_send
from backend.directives.store import get_active_users


_SCHED_STARTED = False


async def _run_once() -> int:
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


async def _loop(interval_sec: float):
    while True:
        try:
            await _run_once()
        except Exception:
            pass
        await asyncio.sleep(interval_sec)


async def ensure_proactive_scheduler():
    """앱 시작 시 1회만 호출하여 주기 실행 스케줄러를 작동시킨다."""
    global _SCHED_STARTED
    if _SCHED_STARTED:
        return
    _SCHED_STARTED = True
    interval = float(os.getenv("PROACTIVE_INTERVAL_SEC", "1800"))  # 기본 30분
    asyncio.create_task(_loop(interval))
