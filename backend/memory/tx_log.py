"""
backend.memory.tx_log - Redis Streams 기반 트랜잭션 로그

단일 진입점 코디네이터의 정확히 한 번(Exactly-Once) 처리를 위해
트랜잭션 시작/커밋/롤백 이벤트를 Redis Streams에 기록하고,
턴 멱등성 키(tx:done:{turn_id})를 활용해 중복 처리를 방지한다.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import redis

from backend.config import get_settings
from backend.memory.redis_index import register_key

logger = logging.getLogger("tx_log")


class RedisTransactionLog:
    """
    Redis Streams 기반 트랜잭션 로그 관리기

    - 스트림 키: txlog:{user_id}:{session_id}
    - 멱등성 키: tx:done:{turn_id} (String "1", TTL 기본 7일)
    - 보존 정책: MAXLEN ~ {maxlen}
    """

    def __init__(self, maxlen: int = 1000):
        self.settings = get_settings()
        self._r = redis.Redis.from_url(self.settings.REDIS_URL, decode_responses=True)
        self.maxlen = maxlen

    # -----------------------------
    # 멱등성 키
    # -----------------------------
    def is_turn_done(self, turn_id: str) -> bool:
        try:
            return bool(self._r.exists(f"tx:done:{turn_id}"))
        except Exception:
            return False

    def mark_turn_done(self, turn_id: str, ttl_sec: int = 7 * 24 * 3600) -> None:
        try:
            key = f"tx:done:{turn_id}"
            self._r.set(key, "1", ex=ttl_sec)
        except Exception as e:
            logger.warning(f"[tx] mark_turn_done error: {e}")

    # -----------------------------
    # 스트림 이벤트
    # -----------------------------
    def _stream_key(self, user_id: str, session_id: str) -> str:
        return f"txlog:{user_id}:{session_id}"

    def begin(self, user_id: str, session_id: str, turn_id: str) -> str:
        """트랜잭션 시작 이벤트 기록 및 엔트리 ID 반환"""
        fields = {
            "event": "begin",
            "turn_id": turn_id,
            "ts": str(int(time.time() * 1000)),
        }
        try:
            # 인덱스 세트 등록 (STREAM 키)
            try:
                register_key(user_id, self._stream_key(user_id, session_id))
            except Exception:
                pass
            xid = self._r.xadd(
                self._stream_key(user_id, session_id),
                fields,
                maxlen=self.maxlen,
                approximate=True,
            )
            return xid
        except Exception as e:
            logger.warning(f"[tx] begin error: {e}")
            return "0-0"

    def commit(self, user_id: str, session_id: str, turn_id: str, tx_id: str) -> None:
        """커밋 이벤트 기록 및 멱등성 키 마킹"""
        fields = {
            "event": "commit",
            "turn_id": turn_id,
            "tx_id": tx_id,
            "ts": str(int(time.time() * 1000)),
        }
        try:
            self._r.xadd(
                self._stream_key(user_id, session_id),
                fields,
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as e:
            logger.warning(f"[tx] commit log error: {e}")
        # 멱등성 마킹은 로그 실패와 무관하게 시도
        self.mark_turn_done(turn_id)

    def rollback(
        self,
        user_id: str,
        session_id: str,
        turn_id: str,
        tx_id: str,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """롤백 이벤트 기록 (감사 추적용)"""
        fields = {
            "event": "rollback",
            "turn_id": turn_id,
            "tx_id": tx_id,
            "reason": reason[:500],
            "ts": str(int(time.time() * 1000)),
        }
        if meta:
            # 간단 직렬화(필요 최소한만 기록)
            for k, v in list(meta.items())[:10]:
                fields[f"m:{k}"] = str(v)[:200]
        try:
            self._r.xadd(
                self._stream_key(user_id, session_id),
                fields,
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as e:
            logger.warning(f"[tx] rollback log error: {e}")
