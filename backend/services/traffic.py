from __future__ import annotations

import re
from typing import Optional, Tuple


def _parse_two_coords(
    s: str,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    문자열에서 (lat,lng) 패턴 2개를 찾아 (출발, 도착) 좌표쌍을 반환.
    """
    try:
        matches = re.findall(r"\(([^,]+),\s*([^)]+)\)", s)
        if len(matches) >= 2:
            lat1 = float(matches[0][0].strip())
            lon1 = float(matches[0][1].strip())
            lat2 = float(matches[1][0].strip())
            lon2 = float(matches[1][1].strip())
            return (lat1, lon1), (lat2, lon2)
        return None
    except Exception:
        return None


async def get_traffic_text(query_or_context: str) -> str:
    """
    멀티에이전트용 교통 전문 텍스트:
    - 공개 OSRM 라우팅 API를 사용해 (lat,lng)→(lat,lng) 경로의 거리/시간을 조회
    - 좌표가 2쌍 이상 추출되지 않으면 빈 문자열 반환
    """
    coords = _parse_two_coords(query_or_context or "")
    if not coords:
        return ""
    (lat1, lon1), (lat2, lon2) = coords
    try:
        import httpx
        from backend.utils.http_client import get_async_client

        # OSRM 공개 서버(테스트 용): 경도,위도 순서 주의(lon,lat;lon,lat)
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        client = get_async_client()
        r = await client.get(url, timeout=httpx.Timeout(10.0))
        if r.status_code != 200:
            return ""
        data = r.json()
            routes = (data or {}).get("routes") or []
            if not routes:
                return ""
            rt = routes[0]
            distance_m = float(rt.get("distance") or 0.0)
            duration_s = float(rt.get("duration") or 0.0)
            km = distance_m / 1000.0
            min = duration_s / 60.0
            text = (
                f"[교통 경로]\n"
                f"출발: ({lat1:.5f}, {lon1:.5f}) → 도착: ({lat2:.5f}, {lon2:.5f})\n"
                f"거리: {km:.1f} km · 소요: {min:.0f} 분\n"
                f"원본: OSRM (router.project-osrm.org)\n"
            )
        # 상한 적용
        try:
            from backend.config import get_settings

            cap = int(get_settings().MA_WEB_CONTENT_MAX_CHARS)
        except Exception:
            cap = 12000
        return text[:cap] if cap > 0 else text
    except Exception:
        return ""


__all__ = ["get_traffic_text"]


async def get_traffic_data(query_or_context: str):
    """
    멀티에이전트용: 구조화 라우트 데이터(distance_km, duration_min, origin, destination) 반환.
    """
    coords = _parse_two_coords(query_or_context or "")
    if not coords:
        return None
    (lat1, lon1), (lat2, lon2) = coords
    try:
        import httpx
        from backend.utils.http_client import get_async_client

        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        client = get_async_client()
        r = await client.get(url, timeout=httpx.Timeout(10.0))
        if r.status_code != 200:
            return None
        data = r.json()
            routes = (data or {}).get("routes") or []
            if not routes:
                return None
            rt = routes[0]
            distance_m = float(rt.get("distance") or 0.0)
            duration_s = float(rt.get("duration") or 0.0)
            km = distance_m / 1000.0
            min = duration_s / 60.0
            return {
                "origin": {"lat": lat1, "lon": lon1},
                "destination": {"lat": lat2, "lon": lon2},
                "distance_km": km,
                "duration_min": min,
            }
    except Exception:
        return None
