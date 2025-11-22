import time
from typing import Any, Dict

import httpx
from backend.utils.http_client import get_async_client


class MCPClient:
    def __init__(self, base_url: str, timeout_s: float = 2.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_s)

    async def naver_search(
        self,
        query: str,
        display: int = 5,
        endpoint: str | None = None,
        freshness_days: int | None = None,
    ) -> Dict[str, Any]:
        try:
            t0 = time.time()
            client = get_async_client(self.timeout)
            payload = {"query": query, "display": display}
            if endpoint:
                payload["endpoint"] = endpoint
            if freshness_days is not None:
                payload["freshness_days"] = int(freshness_days)
            r = await client.post(
                f"{self.base_url}/mcp/search/naver", json=payload, timeout=self.timeout
            )
            took = (time.time() - t0) * 1000
            if r.status_code != 200:
                return {
                    "kind": "error",
                    "status": r.status_code,
                    "took_ms": took,
                    "data": {},
                }
            data = r.json() or {}
            data["took_ms"] = took
            return data
        except Exception:
            return {"kind": "error", "status": 599, "data": {}}


class NaverDirectClient:
    def __init__(self, client_id: str, client_secret: str, timeout_s: float = 2.5):
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = httpx.Timeout(timeout_s)

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    async def search(self, kind: str, query: str, display: int = 5) -> Dict[str, Any]:
        base = {
            "local": "https://openapi.naver.com/v1/search/local.json",
            "news": "https://openapi.naver.com/v1/search/news.json",
            "webkr": "https://openapi.naver.com/v1/search/webkr.json",
        }.get(kind, "https://openapi.naver.com/v1/search/webkr.json")
        try:
            client = get_async_client(self.timeout)
            r = await client.get(
                base,
                params={"query": query, "display": display},
                headers=self._headers(),
                timeout=self.timeout,
            )
            data = r.json() if r.status_code == 200 else {}
            return {"status": r.status_code, "data": data}
        except Exception:
            return {"status": 599, "data": {}}
