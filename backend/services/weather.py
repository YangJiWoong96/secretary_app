from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional


async def _http_get_json(url: str, timeout_s: float = 1.0) -> Optional[Dict[str, Any]]:
    try:
        import httpx
        from backend.utils.http_client import get_async_client

        to = httpx.Timeout(timeout_s)
        client = get_async_client()
        r = await client.get(url, timeout=to)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


async def _geocode(query: str, timeout_s: float = 1.0) -> Optional[Dict[str, float]]:
    # OpenStreetMap Nominatim (rate limits apply; for prod use an API key/hosted plan)
    import urllib.parse as _u

    q = _u.quote((query or "").strip())
    url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1"
    data = await _http_get_json(url, timeout_s)
    try:
        if data and isinstance(data, list) and data:
            lat = float(data[0]["lat"])  # type: ignore[index]
            lon = float(data[0]["lon"])  # type: ignore[index]
            return {"lat": lat, "lon": lon}
    except Exception:
        return None
    return None


async def _open_meteo(
    lat: float, lon: float, timeout_s: float = 1.0
) -> Optional[Dict[str, Any]]:
    url = (
        "https://api.open-meteo.com/v1/forecast?"  # free, no key
        f"latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m,wind_speed_10m"
        "&timezone=auto"
    )
    return await _http_get_json(url, timeout_s)


def _fmt_num(v: Optional[float], unit: str) -> str:
    try:
        if v is None:
            return "-"
        if unit == "°C":
            return f"{float(v):.0f}°C"
        if unit == "mm":
            return f"{float(v):.1f}mm"
        if unit == "%":
            return f"{float(v):.0f}%"
        if unit == "m/s":
            return f"{float(v):.1f}m/s"
        return str(v)
    except Exception:
        return "-"


async def get_weather_block(query: str) -> Optional[Dict[str, str]]:
    """
    지명 텍스트를 받아 현재 날씨를 조회하고 3줄 블록에 들어갈 title/desc/url을 반환한다.
    실패 시 None.
    """
    # 1) 지명→좌표
    geo = await asyncio.wait_for(_geocode(query), timeout=1.0)
    if not geo:
        return None

    # 2) 현재 날씨
    meteo = await asyncio.wait_for(_open_meteo(geo["lat"], geo["lon"]), timeout=1.0)
    if not meteo:
        return None

    cur = (meteo.get("current") or {}) if isinstance(meteo, dict) else {}
    t = cur.get("temperature_2m")
    ta = cur.get("apparent_temperature")
    pr = cur.get("precipitation")
    rh = cur.get("relative_humidity_2m")
    ws = cur.get("wind_speed_10m")
    time_str = str(cur.get("time") or "")

    desc = (
        f"기온 { _fmt_num(t, '°C') } · 체감 { _fmt_num(ta, '°C') } · 강수 { _fmt_num(pr, 'mm') } "
        f"· 습도 { _fmt_num(rh, '%') } · 풍속 { _fmt_num(ws, 'm/s') }"
    )

    # title과 url 구성
    title = f"현재 날씨 ({time_str})"
    url = "https://www.open-meteo.com/en/"

    return {"title": title, "desc": desc, "url": url}


async def get_weather_text(query: str) -> str:
    """
    멀티에이전트용: 3줄 제한 없이 구조화 데이터를 최대한 포함한 텍스트를 반환.
    - 현재는 current 섹션 중심으로 상세 수치와 원본 JSON 일부를 포함
    """
    try:
        geo = await asyncio.wait_for(_geocode(query), timeout=1.0)
        if not geo:
            return ""
        meteo = await asyncio.wait_for(_open_meteo(geo["lat"], geo["lon"]), timeout=1.0)
        if not meteo:
            return ""
        cur = (meteo.get("current") or {}) if isinstance(meteo, dict) else {}
        lines = []
        lines.append(f"[위치] lat={geo['lat']:.5f}, lon={geo['lon']:.5f}")
        lines.append(f"[시각] {cur.get('time')}")
        lines.append(
            f"[기온] {cur.get('temperature_2m')} °C (체감 {cur.get('apparent_temperature')} °C)"
        )
        lines.append(
            f"[강수] {cur.get('precipitation')} mm · [습도] {cur.get('relative_humidity_2m')} % · [풍속] {cur.get('wind_speed_10m')} m/s"
        )
        # 원본 JSON 일부(가독을 위해 current만 pretty)
        try:
            pretty = json.dumps({"current": cur}, ensure_ascii=False, indent=2)
            lines.append(pretty)
        except Exception:
            pass
        text = "\n".join(lines)
        try:
            from backend.config import get_settings

            cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
        except Exception:
            cap = 12000
        return text[:cap] if cap > 0 else text
    except Exception:
        return ""


async def get_weather_data(query: str) -> Optional[Dict[str, Any]]:
    """
    멀티에이전트용: 구조화 데이터 반환(lat/lon/current).
    """
    try:
        geo = await asyncio.wait_for(_geocode(query), timeout=1.0)
        if not geo:
            return None
        meteo = await asyncio.wait_for(_open_meteo(geo["lat"], geo["lon"]), timeout=1.0)
        if not meteo:
            return None
        cur = (meteo.get("current") or {}) if isinstance(meteo, dict) else {}
        return {"lat": geo["lat"], "lon": geo["lon"], "current": cur}
    except Exception:
        return None
