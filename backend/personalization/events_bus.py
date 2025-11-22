from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional


def _get_redis(url: str):
    """Redis 클라이언트를 지연 생성한다."""
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        return None


STREAM_KEY = "pref:events"
GROUP = "scoreboard_consumers"
CONSUMER = "scoreboard_worker"


def publish_preference_event(redis_url: str, event: Dict) -> Optional[str]:
    """
    선호 이벤트를 Redis Stream에 발행한다.
    event 스키마(표준):
      {user_id, norm_key, event_type(explicit|choose|positive|negative),
       intensity: float, ts: int, source: str (turn_id|ui_action)}
    """
    r = _get_redis(redis_url)
    if not r:
        return None
    try:
        ev = {
            k: (
                json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list))
                else str(v)
            )
            for k, v in (event or {}).items()
        }
        return r.xadd(STREAM_KEY, ev, maxlen=10000, approximate=True)
    except Exception:
        return None


async def ensure_scoreboard_consumer(redis_url: str) -> None:
    """
    Scoreboard 업데이트용 Stream Consumer를 보장한다.
    - 그룹이 없으면 생성
    - 블로킹 루프를 백그라운드 태스크로 실행
    """
    r = _get_redis(redis_url)
    if not r:
        return
    try:
        # 그룹 보장
        try:
            r.xgroup_create(name=STREAM_KEY, groupname=GROUP, id="0-0", mkstream=True)
        except Exception:
            # 이미 존재
            pass

        async def _loop():
            from backend.personalization.preference_scoreboard import (
                PreferenceScoreboard,
            )

            sb = PreferenceScoreboard(redis_url)
            while True:
                try:
                    resp = r.xreadgroup(
                        GROUP, CONSUMER, streams={STREAM_KEY: ">"}, count=32, block=2000
                    )
                    if not resp:
                        await asyncio.sleep(0.2)
                        continue
                    for _stream, entries in resp:
                        for entry_id, fields in entries:
                            try:
                                user_id = str(fields.get("user_id") or "")
                                norm_key = str(fields.get("norm_key") or "")
                                etype = str(fields.get("event_type") or "")
                                intensity = float(fields.get("intensity") or 0.0)
                                if user_id and norm_key and etype:
                                    sb.update(
                                        user_id,
                                        norm_key,
                                        {etype: 1},
                                        intensity=intensity,
                                    )
                            except Exception:
                                # 개별 이벤트 실패는 건너뜀
                                pass
                            finally:
                                try:
                                    r.xack(STREAM_KEY, GROUP, entry_id)
                                except Exception:
                                    pass
                except Exception:
                    await asyncio.sleep(0.5)

        asyncio.create_task(_loop())
    except Exception:
        return
