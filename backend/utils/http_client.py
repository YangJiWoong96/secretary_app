"""
backend.utils.http_client - 공유형 HTTPX AsyncClient 및 간단 재시도 유틸

- 프로세스 전역에서 재사용 가능한 AsyncClient 싱글톤을 제공한다.
- 요청 단위 timeout은 client.request(..., timeout=...) 인자로 지정할 수 있다.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any

import httpx

_CLIENT: Optional[httpx.AsyncClient] = None


def get_async_client(
    timeout: Optional[float | httpx.Timeout] = None,
) -> httpx.AsyncClient:
    """
    공유형 AsyncClient 십글톤을 반환한다.

    Args:
        timeout: 기본 타임아웃. 지정하지 않으면 10초로 초기화한다.
                 주의: 반환된 클라이언트는 재사용되므로, 개별 요청의 타임아웃은
                 client.request(..., timeout=...) 형태로 지정하는 것을 권장한다.
    """
    global _CLIENT
    if _CLIENT is None:
        default_timeout: httpx.Timeout
        if isinstance(timeout, (int, float)):
            default_timeout = httpx.Timeout(float(timeout))
        elif isinstance(timeout, httpx.Timeout):
            default_timeout = timeout
        else:
            default_timeout = httpx.Timeout(10.0)
        _CLIENT = httpx.AsyncClient(
            timeout=default_timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _CLIENT


async def http_request_with_retry(
    method: str,
    url: str,
    *,
    retries: int = 2,
    backoff_s: float = 0.25,
    timeout: Optional[float | httpx.Timeout] = None,
    **kwargs: Any,
) -> httpx.Response:
    """
    간단한 비동기 HTTP 재시도 유틸리티.

    - 연결/읽기 타임아웃, 일시적 네트워크 오류에 대해 지정된 횟수만큼 재시도한다.
    - 개별 호출마다 timeout 파라미터를 지정할 수 있다.
    """
    client = get_async_client()
    last_exc: Optional[BaseException] = None
    for attempt in range(max(0, retries) + 1):
        try:
            return await client.request(
                method=method.upper(), url=url, timeout=timeout, **kwargs
            )
        except (
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as e:
            last_exc = e
            if attempt >= retries:
                break
            await asyncio.sleep(backoff_s * (2**attempt))
        except Exception as e:
            # 비예상 예외는 즉시 전파
            raise
    assert last_exc is not None
    raise last_exc


async def aclose() -> None:
    """
    프로세스 종료 시 호출하여 내부 AsyncClient를 종료한다.
    (선택 사항)
    """
    global _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None
