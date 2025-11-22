"""
backend.utils.retry - 재시도 및 네트워크 유틸리티

OpenAI API 및 HTTP 요청에 대한 재시도 로직을 제공합니다.
지수 백오프 + 지터를 사용하여 안정적인 재시도를 구현합니다.
"""

import asyncio
import logging
import random
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("retry")


class RetryManager:
    """
    재시도 로직 관리 클래스

    OpenAI API 호출, HTTP 요청 등에 대한 재시도 로직을 통합 관리합니다.
    지수 백오프 + 랜덤 지터를 사용하여 안정적인 재시도를 구현합니다.
    """

    def __init__(self):
        """
        RetryManager 초기화

        설정 및 클라이언트는 지연 로딩하여 순환 의존성을 방지합니다.
        """
        self._settings = None
        self._client = None

    @property
    def settings(self):
        """설정 인스턴스 (지연 로딩)"""
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def client(self):
        """OpenAI 클라이언트 인스턴스 (지연 로딩)"""
        if self._client is None:
            from backend.config import get_openai_client

            self._client = get_openai_client()
        return self._client

    async def backoff_sleep(self, attempt: int) -> None:
        """
        지수 백오프 + 랜덤 지터를 사용한 대기

        재시도 간 대기 시간을 지수적으로 증가시키며,
        랜덤 지터를 추가하여 thundering herd 문제를 방지합니다.

        Args:
            attempt: 현재 시도 횟수 (0부터 시작)

        Example:
            >>> retry_manager = RetryManager()
            >>> await retry_manager.backoff_sleep(0)  # ~0.125-0.25초 대기
            >>> await retry_manager.backoff_sleep(1)  # ~0.25-0.5초 대기
            >>> await retry_manager.backoff_sleep(2)  # ~0.5-1.0초 대기
        """
        base_delay = self.settings.RETRY_BASE_DELAY
        delay = base_delay * (2**attempt) * (0.5 + random.random())
        await asyncio.sleep(delay)

    def _normalize_parameters(self, create_kwargs: Dict[str, Any]) -> None:
        """
        OpenAI API 파라미터 호환성 교정

        Chat Completions API의 파라미터를 모델별로 호환되도록 정규화합니다:
        - max_completion_tokens → max_tokens 변환
        - response_format 호환성 체크 (구형 모델에서는 제거)

        Args:
            create_kwargs: API 호출 파라미터 딕셔너리 (in-place 수정됨)
        """
        try:
            model_name = str(create_kwargs.get("model", self.settings.LLM_MODEL) or "")

            # 1) max_completion_tokens → max_tokens 정규화
            if "max_completion_tokens" in create_kwargs:
                try:
                    mt = int(create_kwargs.pop("max_completion_tokens"))
                    create_kwargs["max_tokens"] = mt
                except Exception:
                    create_kwargs.pop("max_completion_tokens", None)

            # 2) response_format 호환성 체크 (gpt-4o/4.1/o3 등 일부 모델만 안정 지원)
            rf = create_kwargs.get("response_format")
            if isinstance(rf, dict):
                rf_type = str(rf.get("type", "")).lower()
                # 지원 모델 힌트 키워드
                supports = any(
                    k in model_name for k in ("gpt-4o", "gpt-4.1", "o3", "o4", "4o")
                )
                if not supports:
                    create_kwargs.pop("response_format", None)
                elif rf_type not in ("json_object", "json_schema"):
                    create_kwargs.pop("response_format", None)
                else:
                    # json_schema의 필수 파라미터(name) 보정
                    if rf_type == "json_schema":
                        try:
                            js = rf.get("json_schema")
                            if isinstance(js, dict) and not js.get("name"):
                                js["name"] = "AutoSchema"
                                rf["json_schema"] = js
                                create_kwargs["response_format"] = rf
                        except Exception:
                            # 보정 실패 시 제거하여 400 방지
                            create_kwargs.pop("response_format", None)
        except Exception as e:
            logger.warning(f"[retry] Parameter normalization failed: {e}")

    def _auto_correct_on_error(
        self, create_kwargs: Dict[str, Any], error_msg: str
    ) -> None:
        """
        에러 메시지 기반 자동 교정

        API 호출 실패 시 에러 메시지를 분석하여 파라미터를 자동으로 교정합니다.

        Args:
            create_kwargs: API 호출 파라미터 딕셔너리 (in-place 수정됨)
            error_msg: 에러 메시지 문자열
        """
        try:
            # 남아있는 비호환 매개변수 정리
            if "max_completion_tokens" in create_kwargs:
                mt = int(create_kwargs.pop("max_completion_tokens"))
                create_kwargs["max_tokens"] = mt

            # json_schema.name 누락 등으로 인한 400 발생 시 보정
            rf = create_kwargs.get("response_format")
            if isinstance(rf, dict):
                if (
                    "json_schema.name" in error_msg
                    or "json_schema.name" in error_msg.replace(" ", "").lower()
                ):
                    try:
                        js = rf.get("json_schema")
                        if isinstance(js, dict) and not js.get("name"):
                            js["name"] = "AutoSchema"
                            rf["json_schema"] = js
                            create_kwargs["response_format"] = rf
                        else:
                            create_kwargs.pop("response_format", None)
                    except Exception:
                        create_kwargs.pop("response_format", None)
                else:
                    # 기타 불명 400이면 제거
                    create_kwargs.pop("response_format", None)

            # 스트리밍 관련 에러 시 비활성화
            if "stream" in error_msg and create_kwargs.get("stream") is True:
                create_kwargs["stream"] = False
        except Exception:
            pass

    async def openai_chat_with_retry(self, **create_kwargs) -> Any:
        """
        OpenAI Chat Completions API 호출 (재시도 포함)

        OpenAI API 호출 시 발생할 수 있는 일시적 오류에 대해 자동으로 재시도합니다.
        파라미터 호환성 문제를 자동으로 교정하며, 지수 백오프를 사용합니다.

        Args:
            **create_kwargs: OpenAI API chat.completions.create()에 전달할 키워드 인자

        Returns:
            OpenAI ChatCompletion 응답 객체

        Raises:
            Exception: 모든 재시도가 실패한 경우 마지막 에러를 발생시킴

        Example:
            >>> retry_manager = RetryManager()
            >>> response = await retry_manager.openai_chat_with_retry(
            ...     model="gpt-4o-mini",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     temperature=0.7
            ... )
        """
        last_err = None

        # 사전 파라미터 정규화
        self._normalize_parameters(create_kwargs)

        max_retries = self.settings.MAX_RETRIES_OPENAI

        for attempt in range(max_retries + 1):
            try:
                return await self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[retry] OpenAI API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

                # 400 파라미터 호환 이슈에 대한 자동 교정 후 재시도
                if attempt < max_retries:
                    error_msg = str(e)
                    self._auto_correct_on_error(create_kwargs, error_msg)
                    await self.backoff_sleep(attempt)
                    continue
                break

        # 모든 재시도 실패
        logger.error(f"[retry] OpenAI API call failed after {max_retries + 1} attempts")
        raise last_err

    async def rewrite_with_retries(
        self,
        messages: list[dict],
        base_timeout_s: float,
        attempts: int = 1,
        delta_s: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        """
        쿼리 재작성 전용 재시도 함수 (타임아웃 점진적 증가)

        재작성 호출에 대해 타임아웃 발생 시 타임아웃을 점진적으로 늘려가며 재시도합니다.

        Args:
            messages: OpenAI API 메시지 리스트
            base_timeout_s: 기본 타임아웃(초)
            attempts: 최대 시도 횟수
            delta_s: 재시도마다 증가시킬 타임아웃(초)
            max_tokens: 최대 토큰 수 (None이면 설정값 사용)
            response_format: 응답 포맷 (예: {"type": "json_object"})

        Returns:
            성공 시 응답 문자열, 실패 시 None

        Example:
            >>> retry_manager = RetryManager()
            >>> result = await retry_manager.rewrite_with_retries(
            ...     messages=[{"role": "user", "content": "Rewrite: ..."}],
            ...     base_timeout_s=1.25,
            ...     attempts=2,
            ...     delta_s=1.0
            ... )
        """
        for i in range(attempts):
            timeout = base_timeout_s + i * delta_s
            try:
                resp = await asyncio.wait_for(
                    self.openai_chat_with_retry(
                        model=self.settings.REWRITE_MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=(
                            max_tokens
                            if max_tokens is not None
                            else self.settings.REWRITE_MAX_TOKENS
                        ),
                        **(
                            {"response_format": response_format}
                            if response_format
                            else {}
                        ),
                    ),
                    timeout=timeout,
                )
                return (resp.choices[0].message.content or "").strip()
            except asyncio.TimeoutError:
                if i + 1 >= attempts:
                    logger.warning(
                        f"[retry] Rewrite timeout after {attempts} attempts (final timeout: {timeout:.2f}s)"
                    )
                    break
                logger.warning(
                    f"[retry] Rewrite timeout ({timeout:.2f}s) -> retry {i + 1}"
                )
            except Exception as e:
                logger.warning(f"[retry] Rewrite error: {e}")
                break

        return None

    async def http_get_with_retry(
        self,
        url: str,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout: Optional[httpx.Timeout] = None,
    ) -> httpx.Response:
        """
        HTTP GET 요청 (재시도 포함)

        HTTP GET 요청 시 발생할 수 있는 일시적 오류에 대해 자동으로 재시도합니다.

        Args:
            url: 요청 URL
            headers: HTTP 헤더 (선택)
            params: 쿼리 파라미터 (선택)
            timeout: 타임아웃 설정 (선택)

        Returns:
            httpx.Response 객체

        Raises:
            Exception: 모든 재시도가 실패한 경우 마지막 에러를 발생시킴

        Example:
            >>> retry_manager = RetryManager()
            >>> response = await retry_manager.http_get_with_retry(
            ...     url="https://api.example.com/data",
            ...     headers={"Authorization": "Bearer ..."},
            ...     timeout=httpx.Timeout(5.0)
            ... )
        """
        last_err = None
        max_retries = self.settings.MAX_RETRIES_HTTP

        from backend.utils.http_client import get_async_client

        for attempt in range(max_retries + 1):
            try:
                client_http = get_async_client()
                return await client_http.get(
                    url, headers=headers, params=params, timeout=timeout
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[retry] HTTP GET failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

                if attempt >= max_retries:
                    break

                await self.backoff_sleep(attempt)

        # 모든 재시도 실패
        logger.error(f"[retry] HTTP GET failed after {max_retries + 1} attempts: {url}")
        raise last_err


# ===== 싱글톤 인스턴스 =====
_retry_manager_instance: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """
    전역 RetryManager 싱글톤 인스턴스 반환

    애플리케이션 전역에서 동일한 재시도 관리자를 공유합니다.

    Returns:
        RetryManager: 전역 재시도 관리자 인스턴스

    Example:
        >>> from backend.utils.retry import get_retry_manager
        >>> retry_manager = get_retry_manager()
        >>> response = await retry_manager.openai_chat_with_retry(...)
    """
    global _retry_manager_instance

    if _retry_manager_instance is None:
        _retry_manager_instance = RetryManager()
        logger.info("[retry] RetryManager instance created")

    return _retry_manager_instance


# ===== 기존 함수명 호환성을 위한 래퍼 =====


async def backoff_sleep(attempt: int) -> None:
    """
    지수 백오프 대기 (호환성 래퍼)

    Args:
        attempt: 현재 시도 횟수
    """
    retry_manager = get_retry_manager()
    await retry_manager.backoff_sleep(attempt)


async def openai_chat_with_retry(**create_kwargs) -> Any:
    """
    OpenAI Chat Completions API 호출 (호환성 래퍼)

    Args:
        **create_kwargs: OpenAI API 파라미터

    Returns:
        OpenAI ChatCompletion 응답
    """
    retry_manager = get_retry_manager()
    return await retry_manager.openai_chat_with_retry(**create_kwargs)


async def rewrite_with_retries(
    messages: list[dict],
    base_timeout_s: float,
    attempts: int = 1,
    delta_s: float = 1.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """
    쿼리 재작성 재시도 (호환성 래퍼)

    Args:
        messages: API 메시지
        base_timeout_s: 기본 타임아웃
        attempts: 시도 횟수
        delta_s: 타임아웃 증가량
        max_tokens: 최대 토큰
        response_format: 응답 포맷

    Returns:
        재작성된 문자열 또는 None
    """
    retry_manager = get_retry_manager()
    return await retry_manager.rewrite_with_retries(
        messages, base_timeout_s, attempts, delta_s, max_tokens, response_format
    )


async def http_get_with_retry(
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: Optional[httpx.Timeout] = None,
) -> httpx.Response:
    """
    HTTP GET 요청 재시도 (호환성 래퍼)

    Args:
        url: 요청 URL
        headers: HTTP 헤더
        params: 쿼리 파라미터
        timeout: 타임아웃

    Returns:
        HTTP 응답
    """
    retry_manager = get_retry_manager()
    return await retry_manager.http_get_with_retry(url, headers, params, timeout)
