"""
메시지 요약 캐시 시스템 (세션별 인메모리 TTL 캐시)

특징:
- 세션별 로컬 캐시
- TTL 만료(기본 1시간)
- 세션당 최대 항목 제한(기본 50개)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("summary_cache")


@dataclass
class SummaryEntry:
    """요약 캐시 엔트리"""

    original_hash: str
    summary: str
    created_at: float
    tokens_original: int
    tokens_summary: int


class SummaryCache:
    """
    메시지 요약 캐시(인메모리)
    """

    def __init__(self, ttl_seconds: int = 3600, max_per_session: int = 50):
        self._cache: Dict[str, Dict[str, SummaryEntry]] = {}
        self._ttl = int(ttl_seconds)
        self._max_per_session = int(max_per_session)
        self._lock = asyncio.Lock()

    def _make_key(self, session_id: str, text: str) -> tuple[str, str]:
        """세션ID와 텍스트 해시로 키를 생성한다."""
        text_hash = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]
        return str(session_id or ""), text_hash

    async def get(self, session_id: str, text: str) -> Optional[str]:
        """요약 캐시 조회. 만료 시 None 반환."""
        sess_id, text_hash = self._make_key(session_id, text)
        async with self._lock:
            session_cache = self._cache.get(sess_id, {})
            entry = session_cache.get(text_hash)
            if entry is None:
                return None
            if (time.time() - entry.created_at) > self._ttl:
                try:
                    del session_cache[text_hash]
                except Exception:
                    pass
                return None
            return entry.summary

    async def set(
        self,
        session_id: str,
        text: str,
        summary: str,
        tokens_original: int,
        tokens_summary: int,
    ) -> None:
        """요약 결과를 캐시에 저장한다."""
        sess_id, text_hash = self._make_key(session_id, text)
        async with self._lock:
            if sess_id not in self._cache:
                self._cache[sess_id] = {}
            session_cache = self._cache[sess_id]

            # 용량 제한: 가장 오래된 항목 제거
            if len(session_cache) >= self._max_per_session:
                try:
                    oldest_key = min(
                        session_cache.keys(), key=lambda k: session_cache[k].created_at
                    )
                    del session_cache[oldest_key]
                except Exception:
                    # 제거 실패 시 무시하고 덮어씀
                    pass

            session_cache[text_hash] = SummaryEntry(
                original_hash=text_hash,
                summary=summary,
                created_at=time.time(),
                tokens_original=int(tokens_original),
                tokens_summary=int(tokens_summary),
            )


# 전역 싱글톤 인스턴스
_summary_cache: Optional[SummaryCache] = None


def get_summary_cache() -> SummaryCache:
    """전역 요약 캐시 인스턴스 반환"""
    global _summary_cache
    if _summary_cache is None:
        _summary_cache = SummaryCache()
    return _summary_cache
