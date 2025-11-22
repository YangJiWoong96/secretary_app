from __future__ import annotations

"""
Google Programmable Search(JSON API) 클라이언트

역할:
- CSE를 통해 일반 웹/도메인 제한(site:) 검색을 수행한다.
- 키/엔진 미설정 시 비활성(빈 결과 반환)로 안전하게 동작한다.
- Redis 캐시 및 간단 레이트리밋으로 비용/지연을 관리한다.

주의:
- 본 모듈은 본문 추출을 수행하지 않는다. URL/제목/스니펫 등 메타만 반환하며,
  본문은 상위 레이어(WebResearchAgent)의 추출 단계에서 처리한다.
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class CSEClient:
    """
    Google CSE(JSON API) 래퍼.

    - api_key/cx가 없으면 enabled=False로 동작하며, 이 경우 항상 빈 결과를 반환한다.
    - qps/ttl은 합리적 기본값을 가지며, 환경/설정에 따라 상향/하향 조정 가능하다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        ttl_sec: int = 900,
        qps: float = 2.0,
        redis_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or ""
        self.cx = cx or os.getenv("CSE_CX") or ""
        self.enabled = bool(self.api_key and self.cx)
        self.ttl_sec = int(ttl_sec)
        self.qps = max(0.5, float(qps))
        self._last_ts = 0.0
        self._redis = (
            redis.Redis.from_url(redis_url)
            if (redis is not None and redis_url)
            else None
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        site: Optional[str] = None,
        date_restrict: Optional[str] = None,  # "d7","m1" 등
        lr: str = "lang_ko",
        gl: str = "kr",
        safe: str = "off",
    ) -> List[Dict[str, Any]]:
        """
        CSE 검색 실행.
        - site: 특정 도메인 제한(예: "reddit.com")
        - date_restrict: 최신성 필터("d7"=7일, "m1"=1개월)
        - lr/gl/safe: 언어/지역/세이프 설정

        Returns:
            List[Dict]: WebSearchResult 호환 dict 목록(본문 content는 비워둔다)
        """
        if not self.enabled:
            return []

        q = (query or "").strip()
        if site:
            q = f"site:{site} {q}"

        key_raw = f"cse:{q}:{limit}:{date_restrict}:{lr}:{gl}:{safe}"
        key = hashlib.md5(key_raw.encode()).hexdigest()
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        await self._rate_limit()

        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": q,
            "num": min(10, max(1, int(limit))),
            "lr": lr,
            "gl": gl,
            "safe": safe,
        }
        if date_restrict:
            params["dateRestrict"] = date_restrict

        out: List[Dict[str, Any]] = []
        from backend.utils.http_client import get_async_client
        client = get_async_client()
        r = await client.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=httpx.Timeout(10.0))
        if r.status_code != 200:
            return []
        data = r.json()
            for it in data.get("items", []) or []:
                url = it.get("link") or ""
                title = it.get("title") or ""
                snippet = it.get("snippet") or ""
                domain = urlparse(url).netloc
                published = None
                try:
                    meta = (it.get("pagemap") or {}).get("metatags") or []
                    if meta and isinstance(meta, list):
                        mt = meta[0] or {}
                        published = mt.get("article:published_time") or mt.get(
                            "og:updated_time"
                        )
                except Exception:
                    published = None

                out.append(
                    {
                        "url": url,
                        "title": title,
                        "excerpt": snippet,
                        "content": "",
                        "trust_score": 0.0,
                        "domain": domain,
                        "published_at": published,
                        "source_type": "web",
                        "cross_validated": False,
                        "evidence_ids": [],
                    }
                )

        self._set_cache(key, out, self.ttl_sec)
        return out

    async def _rate_limit(self) -> None:
        """
        간단한 QPS 레이트리밋.
        """
        if self.qps <= 0:
            return
        now = time.time()
        min_interval = 1.0 / self.qps
        elapsed = now - self._last_ts
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_ts = time.time()

    def _get_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _set_cache(self, key: str, value: List[Dict[str, Any]], ttl: int) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass


__all__ = ["CSEClient"]
