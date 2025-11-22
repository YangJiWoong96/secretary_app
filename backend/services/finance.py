from __future__ import annotations

import json
from typing import Any, Dict, Optional


async def _av_get(function: str, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
        from backend.config import get_settings

        key = (get_settings().ALPHA_VANTAGE_API_KEY or "").strip()
        if not key:
            return None
        import httpx
        from backend.utils.http_client import get_async_client

        url = "https://www.alphavantage.co/query"
        p = {"function": function, "apikey": key, **params}
        client = get_async_client()
        r = await client.get(url, params=p, timeout=httpx.Timeout(10.0))
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


async def get_finance_data(symbol: str) -> Optional[Dict[str, Any]]:
    """GLOBAL_QUOTE 데이터(원본 JSON) 반환."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    data = await _av_get("GLOBAL_QUOTE", {"symbol": sym})
    if not data:
        return None
    return data.get("Global Quote") or None


async def get_finance_text(symbol: str) -> str:
    """
    멀티에이전트용 금융 전문 텍스트:
    - GLOBAL_QUOTE + pretty JSON
    """
    try:
        sym = (symbol or "").strip().upper()
        if not sym:
            return ""
        data = await get_finance_data(sym)
        if not data:
            return ""
        lines = []
        lines.append(f"[티커] {sym}")
        lines.append(
            f"[가격] {data.get('05. price','-')} · [변화] {data.get('09. change','-')} "
            f"({data.get('10. change percent','-')}) · [거래일] {data.get('07. latest trading day','-')}"
        )
        try:
            pretty = json.dumps({"quote": data}, ensure_ascii=False, indent=2)
            lines.append(pretty)
        except Exception:
            pass
        from backend.config import get_settings

        cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
        text = "\n".join(lines)
        return text[:cap] if cap > 0 else text
    except Exception:
        return ""


async def get_finance_earnings_text(symbol: str) -> str:
    """Alpha Vantage EARNINGS 요약(최신 분기 2~3개)"""
    sym = (symbol or "").strip().upper()
    if not sym:
        return ""
    data = await _av_get("EARNINGS", {"symbol": sym})
    if not data:
        return ""
    q = data.get("quarterlyEarnings") or []
    head = q[:3]
    lines = [f"[실적] {sym} 최근 분기"]
    for it in head:
        lines.append(
            f"- {it.get('fiscalDateEnding','')}: EPS {it.get('reportedEPS','-')} "
            f"(서프라이즈 {it.get('surprise','-')} | {it.get('surprisePercentage','-')}%)"
        )
    try:
        pretty = json.dumps({"earnings_head": head}, ensure_ascii=False, indent=2)
        lines.append(pretty)
    except Exception:
        pass
    try:
        from backend.config import get_settings

        cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
    except Exception:
        cap = 12000
    text = "\n".join(lines)
    return text[:cap] if cap > 0 else text


async def get_finance_news_text(symbol: str) -> str:
    """Alpha Vantage NEWS_SENTIMENT (상위 3건 헤드라인)"""
    sym = (symbol or "").strip().upper()
    if not sym:
        return ""
    data = await _av_get(
        "NEWS_SENTIMENT", {"tickers": sym, "sort": "LATEST", "limit": "5"}
    )
    if not data:
        return ""
    feed = data.get("feed") or []
    head = feed[:3]
    lines = [f"[뉴스] {sym} 최신"]
    for it in head:
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        rel = it.get("relevance_score") or ""
        lines.append(f"- {title} ({url}) rel={rel}")
    try:
        from backend.config import get_settings

        cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
    except Exception:
        cap = 12000
    text = "\n".join(lines)
    return text[:cap] if cap > 0 else text


async def get_finance_sector_text(symbol: str) -> str:
    """OVERVIEW로 섹터 추출 + SECTOR 실시간 성과 비교"""
    sym = (symbol or "").strip().upper()
    if not sym:
        return ""
    overview = await _av_get("OVERVIEW", {"symbol": sym}) or {}
    sector = (overview.get("Sector") or "").strip()
    sector_perf = await _av_get("SECTOR", {}) or {}
    perf = (
        (sector_perf.get("Rank A: Real-Time Performance") or {}) if sector_perf else {}
    )
    lines = [f"[섹터] {sym} - {sector or 'N/A'}"]
    if sector and perf:
        # 상위/하위 3개 섹터 요약
        try:
            items = list(perf.items())

            # items: [("Information Technology", "+0.23%"), ...]
            def _to_num(s: str) -> float:
                try:
                    return float(s.replace("%", "").strip())
                except Exception:
                    return 0.0

            sortd = sorted(items, key=lambda x: _to_num(x[1]), reverse=True)
            top3 = sortd[:3]
            bot3 = sortd[-3:]
            lines.append("[상위 섹터] " + ", ".join([f"{k}({v})" for k, v in top3]))
            lines.append("[하위 섹터] " + ", ".join([f"{k}({v})" for k, v in bot3]))
            if sector in perf:
                lines.append(f"[해당 섹터] {sector} = {perf.get(sector)}")
        except Exception:
            pass
    try:
        from backend.config import get_settings

        cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
    except Exception:
        cap = 12000
    text = "\n".join(lines)
    return text[:cap] if cap > 0 else text


__all__ = [
    "get_finance_text",
    "get_finance_data",
    "get_finance_earnings_text",
    "get_finance_news_text",
    "get_finance_sector_text",
]

from __future__ import annotations

"""
backend.services.finance - 금융/시세 유틸리티

기능:
- 금융 의도 감지(LLM 구조화 출력)
- 티커 해석(LLM 구조화 출력, 보수적 폴백 포함)
- 실시간 시세 조회(야후 파이낸스)
- 3줄 블록 포맷 생성(타 모듈과 일관)

주의:
- 외부 호출은 짧은 타임아웃 사용
- 실패 시 단계적 폴백: 실시간 → 전일종가 → 링크 제공
"""

import asyncio
import re
from typing import Any, Dict, Optional, Tuple

import httpx


async def detect_finance_intent(user_input: str, realtime_ctx: str) -> Dict[str, Any]:
    """LLM으로 금융 의도 감지.

    반환: {"is_finance": bool, "intent": str, "entity": str, "ticker_hint": str, "exchange_hint": str}
    intent: "realtime_price" | "historical_price" | "news" | "unknown"
    """
    from backend.memory import model_supports_response_format
    from backend.utils.retry import openai_chat_with_retry
    from backend.utils.schema_builder import build_json_schema
    from backend.utils.schema_registry import get_finance_intent_schema

    schema = build_json_schema(
        "FinanceIntent", get_finance_intent_schema(), strict=True
    )

    msgs = [
        {
            "role": "system",
            "content": (
                "너는 금융 의도 감지기다. 질문이 실시간 가격/시세/종가/환율/지수 등 금융 데이터 조회인지 판별하라. "
                "질문 내 기업/지수/티커/거래소를 식별하면 entity/ticker_hint/exchange_hint에 채워라. JSON만 출력."
            ),
        },
        {"role": "system", "content": f"[현재 시각]\n{(realtime_ctx or '').strip()}"},
        {"role": "user", "content": (user_input or "").strip()},
    ]

    try:
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 80,
        }
        if model_supports_response_format("gpt-4o-mini"):
            kwargs["response_format"] = schema
        resp = await openai_chat_with_retry(**kwargs)
        txt = (resp.choices[0].message.content or "").strip()
        import json as _json

        return (
            _json.loads(txt)
            if txt.startswith("{")
            else {"is_finance": False, "intent": "unknown"}
        )
    except Exception:
        return {"is_finance": False, "intent": "unknown"}


_FAMOUS_TICKERS: Dict[str, Tuple[str, str]] = {
    # name(lower) -> (ticker, exchange)
    "엔비디아": ("NVDA", "NASDAQ"),
    "nvidia": ("NVDA", "NASDAQ"),
    "nvda": ("NVDA", "NASDAQ"),
    "삼성전자": ("005930.KS", "KRX"),
    "애플": ("AAPL", "NASDAQ"),
    "apple": ("AAPL", "NASDAQ"),
    "msft": ("MSFT", "NASDAQ"),
    "마이크로소프트": ("MSFT", "NASDAQ"),
    "tsla": ("TSLA", "NASDAQ"),
    "테슬라": ("TSLA", "NASDAQ"),
}


async def resolve_ticker(name_or_ticker: str, realtime_ctx: str = "") -> Dict[str, Any]:
    """이름/별칭/티커 → 정규 티커 해석(LLM 우선, 실패 시 소규모 사전 폴백)."""
    from backend.memory import model_supports_response_format
    from backend.utils.retry import openai_chat_with_retry
    from backend.utils.schema_builder import build_json_schema
    from backend.utils.schema_registry import get_ticker_resolve_schema

    base = (name_or_ticker or "").strip()
    if not base:
        return {"ticker": "", "exchange": "", "name": ""}

    msgs = [
        {
            "role": "system",
            "content": (
                "너는 금융 대상 식별기다. 입력이 회사명/별칭/티커 중 무엇이든 규격화된 티커를 결정하라. "
                "코인/선물/지수는 is_crypto로 구분. 확실한 거래소 표기(예: .KS, .KQ, .T, .NS 등)를 제공하라. JSON만."
            ),
        },
        {"role": "system", "content": f"[현재 시각]\n{(realtime_ctx or '').strip()}"},
        {"role": "user", "content": base},
    ]

    try:
        schema = build_json_schema(
            "TickerResolve", get_ticker_resolve_schema(), strict=True
        )
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 60,
        }
        if model_supports_response_format("gpt-4o-mini"):
            kwargs["response_format"] = schema
        resp = await openai_chat_with_retry(**kwargs)
        txt = (resp.choices[0].message.content or "").strip()
        import json as _json

        data = _json.loads(txt) if txt.startswith("{") else {}
        if data.get("ticker"):
            return data
    except Exception:
        pass

    key = base.lower()
    if key in _FAMOUS_TICKERS:
        t, ex = _FAMOUS_TICKERS[key]
        return {"ticker": t, "exchange": ex, "name": name_or_ticker, "is_crypto": False}
    return {
        "ticker": base.upper(),
        "exchange": "",
        "name": name_or_ticker,
        "is_crypto": False,
    }


async def fetch_realtime_quote(ticker: str, timeout_s: float = 1.8) -> Dict[str, Any]:
    """야후 파이낸스 실시간 호가(유사) 조회. 실패 시 빈 dict.

    참고: 비공식 엔드포인트. 과도한 호출은 피해야 하며, 상용 환경에서는 정식 데이터 공급자를 권장.
    """
    if not ticker:
        return {}
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": ticker}
    try:
        to = httpx.Timeout(timeout_s)
        from backend.utils.http_client import get_async_client
        client = get_async_client()
        r = await client.get(url, params=params, timeout=to)
        r.raise_for_status()
        data = r.json()
            q = ((data or {}).get("quoteResponse", {}) or {}).get("result", [])
            if not q:
                return {}
            it = q[0]
            return {
                "price": it.get("regularMarketPrice"),
                "previous_close": it.get("regularMarketPreviousClose"),
                "currency": it.get("currency"),
                "exchange": it.get("fullExchangeName") or it.get("exchange") or "",
                "ts": it.get("regularMarketTime"),
                "change": it.get("regularMarketChange"),
                "change_percent": it.get("regularMarketChangePercent"),
            }
    except Exception:
        return {}


def _format_change(change: Optional[float], change_pct: Optional[float]) -> str:
    try:
        if change is None or change_pct is None:
            return ""
        sign = "▲" if change >= 0 else "▼"
        return f" {sign} {change:.2f} ({change_pct:.2f}%)"
    except Exception:
        return ""


def build_price_block(entity_name: str, ticker: str, quote: Dict[str, Any]) -> str:
    """시세를 3줄 블록으로 구성한다."""
    name = (entity_name or ticker or "").strip() or ticker
    px = quote.get("price")
    exch = (quote.get("exchange") or "").strip()
    cur = (quote.get("currency") or "").strip()
    ch = _format_change(quote.get("change"), quote.get("change_percent"))
    title = (
        f"{name}({ticker}) 실시간 가격: {px if px is not None else 'N/A'} {cur} {ch}"
    )
    desc = f"거래소: {exch or '-'} | 전일종가: {quote.get('previous_close', '-') } | 업데이트: epoch {quote.get('ts', '-') }"
    url = f"https://finance.yahoo.com/quote/{ticker}"
    return "\n".join([title, desc, url])


async def build_finance_block(user_input: str, realtime_ctx: str) -> Tuple[str, str]:
    """금융 질의에 대해 (이름/티커) → 시세 블록을 생성. (title/desc/url)

    반환: (block, reason)
    """
    # 1) 후보 엔티티 추출(간단): 따옴표/한글/영문/티커 패턴에서 가장 긴 토큰
    base = (user_input or "").strip()
    cand = "".join(re.findall(r"[A-Za-z가-힣0-9\.]+", base))[:40] or base[:40]
    resolved = await resolve_ticker(cand, realtime_ctx)
    ticker = (resolved.get("ticker") or "").strip()
    name = (resolved.get("name") or cand or ticker).strip()

    # 2) 실시간 시세
    quote = await fetch_realtime_quote(ticker)
    if quote and quote.get("price") is not None:
        return build_price_block(name, ticker, quote), "realtime_price"

    # 3) 폴백: 전일 종가만
    if quote and quote.get("previous_close") is not None:
        # previous_close를 price로 사용
        q2 = dict(quote)
        q2["price"] = quote.get("previous_close")
        return build_price_block(name, ticker, q2), "previous_close"

    # 4) 최종 폴백: 링크만 제공
    url = (
        f"https://finance.yahoo.com/quote/{ticker}"
        if ticker
        else "https://finance.yahoo.com/"
    )
    block = "\n".join(
        [
            f"{name or ticker or '가격 정보'}",
            "실시간 가격을 확인할 수 없습니다. 링크를 참고하세요.",
            url,
        ]
    )
    return block, "link_only"
