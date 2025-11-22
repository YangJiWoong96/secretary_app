"""
backend.utils.datetime_utils - 날짜/시간 유틸리티

KST 타임존 기반 날짜 처리 및 상대시제(오늘, 내일, 이번주 등) 파싱 기능을 제공합니다.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple

# ===== KST 타임존 정의 =====
KST = timezone(timedelta(hours=9))


class DateTimeHelper:
    """
    날짜/시간 관련 헬퍼 함수 모음

    KST(한국 표준시) 기준으로 날짜/시간을 처리하며,
    YYYYMM, YYYYMMDD 등의 정수 형식 변환을 제공합니다.
    """

    @staticmethod
    def now_kst() -> datetime:
        """
        현재 KST 시간 반환

        Returns:
            datetime: KST 타임존이 설정된 현재 시간

        Example:
            >>> now = DateTimeHelper.now_kst()
            >>> print(now.tzinfo)  # UTC+09:00
        """
        return datetime.now(KST)

    @staticmethod
    def ym(dt: datetime) -> int:
        """
        datetime을 YYYYMM 정수로 변환

        Args:
            dt: 변환할 datetime

        Returns:
            int: YYYYMM 형식 정수 (예: 202508)

        Example:
            >>> dt = datetime(2025, 8, 17)
            >>> DateTimeHelper.ym(dt)
            202508
        """
        return dt.year * 100 + dt.month

    @staticmethod
    def ymd(dt: datetime) -> int:
        """
        datetime을 YYYYMMDD 정수로 변환

        Args:
            dt: 변환할 datetime

        Returns:
            int: YYYYMMDD 형식 정수 (예: 20250817)

        Example:
            >>> dt = datetime(2025, 8, 17)
            >>> DateTimeHelper.ymd(dt)
            20250817
        """
        return dt.year * 10000 + dt.month * 100 + dt.day

    @staticmethod
    def week_range(base: datetime, offset_weeks: int = 0) -> Tuple[datetime, datetime]:
        """
        주 범위 계산 (월요일 시작)

        Args:
            base: 기준 날짜
            offset_weeks: 주 오프셋 (0: 이번주, -1: 지난주, 1: 다음주)

        Returns:
            Tuple[datetime, datetime]: (주 시작일, 주 종료일) - 월요일~일요일

        Example:
            >>> base = datetime(2025, 8, 15)  # 금요일
            >>> start, end = DateTimeHelper.week_range(base, 0)
            >>> # start: 2025-08-11 (월요일), end: 2025-08-17 (일요일)
        """
        d0 = base + timedelta(weeks=offset_weeks)
        start = d0 - timedelta(days=d0.weekday())  # 월요일
        end = start + timedelta(days=6)
        return start.replace(tzinfo=KST), end.replace(tzinfo=KST)

    @staticmethod
    def month_range(
        base: datetime, offset_months: int = 0
    ) -> Tuple[datetime, datetime]:
        """
        월 범위 계산 (1일 ~ 말일)

        Args:
            base: 기준 날짜
            offset_months: 월 오프셋 (0: 이번달, -1: 지난달, 1: 다음달)

        Returns:
            Tuple[datetime, datetime]: (월 첫날, 월 마지막날)

        Example:
            >>> base = datetime(2025, 8, 15)
            >>> start, end = DateTimeHelper.month_range(base, 0)
            >>> # start: 2025-08-01, end: 2025-08-31
        """
        y, m = base.year, base.month + offset_months
        y += (m - 1) // 12
        m = ((m - 1) % 12) + 1
        start = datetime(y, m, 1, tzinfo=KST)
        y2, m2 = y + (m // 12), (m % 12) + 1
        next_first = datetime(y2, m2, 1, tzinfo=KST)
        end = next_first - timedelta(days=1)
        return start, end

    @staticmethod
    def ym_minus_months(base: datetime, months: int) -> int:
        """
        과거 월을 YYYYMM 형식으로 계산

        Args:
            base: 기준 날짜
            months: 뺄 개월 수

        Returns:
            int: YYYYMM 형식 정수

        Example:
            >>> base = datetime(2025, 3, 15)
            >>> DateTimeHelper.ym_minus_months(base, 5)
            202410  # 2024년 10월
        """
        y, m = base.year, base.month - months
        while m <= 0:
            y -= 1
            m += 12
        return y * 100 + m


class RelativeDateParser:
    """
    상대시제 → 절대 날짜 파서

    한국어 상대시제 표현(오늘, 내일, 이번주, 지난달 등)을
    절대 날짜 범위로 변환합니다.
    """

    # ===== 상대시제 패턴 정의 =====

    # 일 단위 패턴
    RELATIVE_PATTERNS_DAY: List[
        Tuple[str, Callable[[datetime], Tuple[datetime, datetime]]]
    ] = [
        (r"\b오늘\b", lambda now: (now, now)),
        (r"\b내일\b", lambda now: (now + timedelta(days=1), now + timedelta(days=1))),
        (r"\b모레\b", lambda now: (now + timedelta(days=2), now + timedelta(days=2))),
        (r"\b글피\b", lambda now: (now + timedelta(days=3), now + timedelta(days=3))),
        (r"\b내글피\b", lambda now: (now + timedelta(days=4), now + timedelta(days=4))),
        (r"\b어제\b", lambda now: (now - timedelta(days=1), now - timedelta(days=1))),
        (
            r"\b그제\b|\b그저께\b|\b엊그제\b",
            lambda now: (now - timedelta(days=2), now - timedelta(days=2)),
        ),
        (r"\b그끄제\b", lambda now: (now - timedelta(days=3), now - timedelta(days=3))),
    ]

    # 주 단위 패턴
    RELATIVE_PATTERNS_WEEK: List[
        Tuple[str, Callable[[datetime], Tuple[datetime, datetime]]]
    ] = [
        (r"\b이번\s*주말\b", lambda now: DateTimeHelper.week_range(now, 0)),
        (
            r"\b지난\s*주말\b|\b저번\s*주말\b",
            lambda now: DateTimeHelper.week_range(now, -1),
        ),
        (r"\b다음\s*주말\b", lambda now: DateTimeHelper.week_range(now, 1)),
        (r"\b이번\s*주\b", lambda now: DateTimeHelper.week_range(now, 0)),
        (
            r"\b지난\s*주\b|\b저번\s*주\b|저번주",
            lambda now: DateTimeHelper.week_range(now, -1),
        ),
        (r"\b다음\s*주\b", lambda now: DateTimeHelper.week_range(now, 1)),
    ]

    # 월/년 단위 패턴
    RELATIVE_PATTERNS_MONTH_YEAR: List[
        Tuple[str, Callable[[datetime], Tuple[datetime, datetime]]]
    ] = [
        (r"\b이번\s*달\b|\b이달\b", lambda now: DateTimeHelper.month_range(now, 0)),
        (
            r"\b지난\s*달\b|\b저번\s*달\b",
            lambda now: DateTimeHelper.month_range(now, -1),
        ),
        (r"\b다음\s*달\b", lambda now: DateTimeHelper.month_range(now, 1)),
        (
            r"\b올해\b",
            lambda now: (
                datetime(now.year, 1, 1, tzinfo=KST),
                datetime(now.year, 12, 31, tzinfo=KST),
            ),
        ),
        (
            r"\b작년\b",
            lambda now: (
                datetime(now.year - 1, 1, 1, tzinfo=KST),
                datetime(now.year - 1, 12, 31, tzinfo=KST),
            ),
        ),
        (
            r"\b재작년\b",
            lambda now: (
                datetime(now.year - 2, 1, 1, tzinfo=KST),
                datetime(now.year - 2, 12, 31, tzinfo=KST),
            ),
        ),
        (
            r"\b내년\b",
            lambda now: (
                datetime(now.year + 1, 1, 1, tzinfo=KST),
                datetime(now.year + 1, 12, 31, tzinfo=KST),
            ),
        ),
        (
            r"\b내후년\b",
            lambda now: (
                datetime(now.year + 2, 1, 1, tzinfo=KST),
                datetime(now.year + 2, 12, 31, tzinfo=KST),
            ),
        ),
    ]

    @classmethod
    def extract_date_range_for_rag(
        cls, text: str, now: Optional[datetime] = None
    ) -> Optional[Tuple[int, int]]:
        """
        RAG 검색용 날짜 범위 추출 (YYYYMMDD 형식)

        텍스트에서 상대시제 표현을 찾아 절대 날짜 범위로 변환합니다.
        여러 표현이 있을 경우 가장 넓은 범위를 반환합니다.

        Args:
            text: 분석할 텍스트 (예: "오늘 일정 알려줘", "지난주 회의록")
            now: 기준 시간 (None이면 현재 KST 시간 사용)

        Returns:
            Optional[Tuple[int, int]]: (시작일, 종료일) YYYYMMDD 형식, 없으면 None

        Example:
            >>> text = "오늘 일정 알려줘"
            >>> date_range = RelativeDateParser.extract_date_range_for_rag(text)
            >>> print(date_range)  # (20250817, 20250817)

            >>> text = "이번주 회의록"
            >>> date_range = RelativeDateParser.extract_date_range_for_rag(text)
            >>> print(date_range)  # (20250811, 20250817) - 월~일
        """
        now = now or DateTimeHelper.now_kst()
        start_dt, end_dt = None, None

        def _apply(patterns: List[Tuple[str, Callable]]):
            nonlocal start_dt, end_dt
            for pat, fn in patterns:
                m = re.search(pat, text)
                if m:
                    s, e = fn(now)
                    start_dt = s if start_dt is None else min(start_dt, s)
                    end_dt = e if end_dt is None else max(end_dt, e)

        _apply(cls.RELATIVE_PATTERNS_DAY)
        _apply(cls.RELATIVE_PATTERNS_WEEK)
        _apply(cls.RELATIVE_PATTERNS_MONTH_YEAR)

        if start_dt and end_dt:
            return (DateTimeHelper.ymd(start_dt), DateTimeHelper.ymd(end_dt))
        return None

    @classmethod
    def month_tokens_for_web(
        cls, text: str, now: Optional[datetime] = None
    ) -> Optional[Tuple[int, int]]:
        """
        웹 검색용 월 토큰 추출 (YYYYMM 형식)

        텍스트에서 상대시제 표현을 찾아 월 단위 범위로 변환합니다.
        뉴스/시황성 질의에 적합한 월 단위 필터링에 사용됩니다.

        Args:
            text: 분석할 텍스트 (예: "이번달 뉴스", "작년 기사")
            now: 기준 시간 (None이면 현재 KST 시간 사용)

        Returns:
            Optional[Tuple[int, int]]: (시작월, 종료월) YYYYMM 형식, 없으면 None

        Example:
            >>> text = "이번달 뉴스"
            >>> month_range = RelativeDateParser.month_tokens_for_web(text)
            >>> print(month_range)  # (202508, 202508)

            >>> text = "올해 기사"
            >>> month_range = RelativeDateParser.month_tokens_for_web(text)
            >>> print(month_range)  # (202501, 202512)
        """
        now = now or DateTimeHelper.now_kst()
        start_dt, end_dt = None, None

        def _apply(patterns: List[Tuple[str, Callable]]):
            nonlocal start_dt, end_dt
            for pat, fn in patterns:
                m = re.search(pat, text)
                if m:
                    s, e = fn(now)
                    start_dt = s if start_dt is None else min(start_dt, s)
                    end_dt = e if end_dt is None else max(end_dt, e)

        _apply(cls.RELATIVE_PATTERNS_DAY)
        _apply(cls.RELATIVE_PATTERNS_WEEK)
        _apply(cls.RELATIVE_PATTERNS_MONTH_YEAR)

        if start_dt and end_dt:
            return (DateTimeHelper.ym(start_dt), DateTimeHelper.ym(end_dt))
        return None


# ===== 호환성을 위한 함수형 인터페이스 =====


def now_kst() -> datetime:
    """현재 KST 시간 (호환성 래퍼)"""
    return DateTimeHelper.now_kst()


def ym(dt: datetime) -> int:
    """datetime → YYYYMM (호환성 래퍼)"""
    return DateTimeHelper.ym(dt)


def ymd(dt: datetime) -> int:
    """datetime → YYYYMMDD (호환성 래퍼)"""
    return DateTimeHelper.ymd(dt)


def week_range(base: datetime, offset_weeks: int = 0) -> Tuple[datetime, datetime]:
    """주 범위 계산 (호환성 래퍼)"""
    return DateTimeHelper.week_range(base, offset_weeks)


def month_range(base: datetime, offset_months: int = 0) -> Tuple[datetime, datetime]:
    """월 범위 계산 (호환성 래퍼)"""
    return DateTimeHelper.month_range(base, offset_months)


def ym_minus_months(base: datetime, months: int) -> int:
    """과거 월 계산 (호환성 래퍼)"""
    return DateTimeHelper.ym_minus_months(base, months)


def extract_date_range_for_rag(
    text: str, now: Optional[datetime] = None
) -> Optional[Tuple[int, int]]:
    """RAG용 날짜 범위 추출 (호환성 래퍼)"""
    return RelativeDateParser.extract_date_range_for_rag(text, now)


def month_tokens_for_web(
    text: str, now: Optional[datetime] = None
) -> Optional[Tuple[int, int]]:
    """웹 검색용 월 토큰 추출 (호환성 래퍼)"""
    return RelativeDateParser.month_tokens_for_web(text, now)


# ===== 추가 유틸리티 함수 =====


def kst_day_bounds(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    KST 기준 하루 범위 (00:00 ~ 23:59:59)

    Args:
        now: 기준 시간 (None이면 현재 KST 시간)

    Returns:
        Tuple[datetime, datetime]: (당일 00:00, 다음날 00:00)

    Example:
        >>> start, end = kst_day_bounds()
        >>> # start: 2025-08-17 00:00:00+09:00
        >>> # end: 2025-08-18 00:00:00+09:00
    """
    now = now or DateTimeHelper.now_kst()
    start = datetime(now.year, now.month, now.day, tzinfo=KST)
    end = start + timedelta(days=1)
    return start, end


def to_utc(dt: datetime) -> datetime:
    """
    datetime을 UTC로 변환

    Args:
        dt: 변환할 datetime (타임존 없으면 KST로 간주)

    Returns:
        datetime: UTC 타임존으로 변환된 datetime

    Example:
        >>> kst_time = datetime(2025, 8, 17, 15, 0, tzinfo=KST)
        >>> utc_time = to_utc(kst_time)
        >>> print(utc_time)  # 2025-08-17 06:00:00+00:00
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(timezone.utc)


def safe_parse_iso(dt_str: str) -> Optional[datetime]:
    """
    ISO 형식 문자열을 datetime으로 안전하게 파싱

    Args:
        dt_str: ISO 형식 날짜 문자열 (예: "2025-08-17T15:00:00Z")

    Returns:
        Optional[datetime]: 파싱된 datetime, 실패 시 None

    Example:
        >>> dt = safe_parse_iso("2025-08-17T15:00:00Z")
        >>> print(dt)  # 2025-08-17 15:00:00+00:00

        >>> dt = safe_parse_iso("invalid")
        >>> print(dt)  # None
    """
    if not dt_str:
        return None

    s = dt_str.strip()
    try:
        # "Z" 접미사를 "+00:00"으로 변환
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        dt = datetime.fromisoformat(s)

        # 타임존 없으면 UTC로 간주
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except Exception:
        return None


def fmt_hm_kst(dt: datetime) -> str:
    """
    datetime을 HH:MM 형식 문자열로 포맷 (KST 기준)

    Args:
        dt: 포맷팅할 datetime

    Returns:
        str: "HH:MM" 형식 문자열, 실패 시 빈 문자열

    Example:
        >>> dt = datetime(2025, 8, 17, 15, 30, tzinfo=timezone.utc)
        >>> fmt_hm_kst(dt)
        '00:30'  # UTC 15:30 = KST 00:30 (다음날)
    """
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        dt_kst = dt.astimezone(KST)
        return dt_kst.strftime("%H:%M")
    except Exception:
        return ""


def msg_ts_dt(message) -> Optional[datetime]:
    """
    메시지 객체에서 타임스탬프 추출

    LangChain 메시지 객체의 additional_kwargs["ts"]에서
    타임스탬프를 추출하여 datetime으로 변환합니다.

    Args:
        message: LangChain 메시지 객체 (HumanMessage/AIMessage)

    Returns:
        Optional[datetime]: KST로 정규화된 datetime, 실패 시 None

    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> msg = HumanMessage(
        ...     content="Hello",
        ...     additional_kwargs={"ts": "2025-08-17T15:00:00+09:00"}
        ... )
        >>> dt = msg_ts_dt(msg)
        >>> print(dt)  # 2025-08-17 15:00:00+09:00
    """
    try:
        ts = getattr(message, "additional_kwargs", {}).get("ts")
        if not ts:
            return None

        # ISO 문자열 → datetime (tz 포함)
        dt = datetime.fromisoformat(ts)

        # KST로 정규화
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        else:
            dt = dt.astimezone(KST)

        return dt
    except Exception:
        return None


def extract_ts_bounds(
    messages, fallback_now: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    """
    메시지 리스트에서 타임스탬프 범위 추출

    Args:
        messages: LangChain 메시지 리스트
        fallback_now: 타임스탬프가 없을 때 사용할 기본값

    Returns:
        Tuple[datetime, datetime]: (최소 시간, 최대 시간)

    Example:
        >>> messages = [msg1, msg2, msg3]  # 타임스탬프 포함 메시지들
        >>> start, end = extract_ts_bounds(messages)
        >>> print(f"대화 범위: {start} ~ {end}")
    """
    fallback_now = fallback_now or DateTimeHelper.now_kst()
    dts = [msg_ts_dt(m) for m in messages]
    dts = [d for d in dts if d is not None]

    if not dts:
        # 과거 메시지에 ts가 없을 수 있으므로 안전한 폴백
        return fallback_now, fallback_now

    return min(dts), max(dts)
