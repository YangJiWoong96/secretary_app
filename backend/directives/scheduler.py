# 일일 스케줄러(KST 03:00)로 directives/signals/persona 배치 업데이트 예약

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

from .store import get_active_users
from .pipeline import schedule_directive_update

KST = timezone(timedelta(hours=9))


DIR_SCHEDULE_ENABLED = os.getenv("DIR_SCHEDULE_ENABLED", "1") == "1"
DIR_SCHEDULE_HOUR_KST = int(os.getenv("DIR_SCHEDULE_HOUR_KST", "3"))  # 03:00
DIR_BATCH_DELAY_MS = int(os.getenv("DIR_BATCH_DELAY_MS", "50"))  # 큐 과도 적체 방지

_sched_started = False


def _next_run(after: datetime | None = None) -> datetime:
    now = after or datetime.now(KST)
    today_target = now.replace(
        hour=DIR_SCHEDULE_HOUR_KST, minute=0, second=0, microsecond=0
    )
    if now < today_target:
        return today_target
    return today_target + timedelta(days=1)


async def _daily_loop():
    while True:
        if not DIR_SCHEDULE_ENABLED:
            await asyncio.sleep(3600)
            continue
        now = datetime.now(KST)
        nxt = _next_run(now)
        sleep_s = max(1.0, (nxt - now).total_seconds())
        await asyncio.sleep(sleep_s)
        # 실행 시점
        try:
            users: List[str] = get_active_users()
            # 각 유저에 대해 배치 업데이트 예약(force)
            for idx, sid in enumerate(users):
                try:
                    schedule_directive_update(sid, force=True)
                except Exception:
                    pass
                if DIR_BATCH_DELAY_MS > 0:
                    await asyncio.sleep(DIR_BATCH_DELAY_MS / 1000.0)
        except Exception:
            # 에러 발생 시 다음 주기까지 대기
            await asyncio.sleep(60)


async def ensure_daily_scheduler():
    global _sched_started
    if _sched_started:
        return
    _sched_started = True
    asyncio.create_task(_daily_loop())
