import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

# Firestore는 선택 의존성. 환경/권한 없을 경우 자동 비활성화
try:
    from google.cloud import firestore as gcf  # type: ignore
except Exception:
    gcf = None

KST = timezone(timedelta(hours=9))


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(timezone.utc)


def _safe_parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    s = dt_str.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _ensure_fs_db():
    firestore_enable = bool(int(os.getenv("FIRESTORE_ENABLE", "1")))
    if not firestore_enable or gcf is None:
        return None
    try:
        return gcf.Client()
    except Exception:
        return None


def _hour_bucket_kst(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_kst = dt.astimezone(KST)
    return dt_kst.hour


def _prime_time_from_hist(hour_hist: List[int]) -> Tuple[str, float]:
    """
    시간대 히스토그램(0~23 카운트)에서 대표 성향 라벨과 점유율을 추출.
    - late_night: 23~03, morning: 05~10, day: 11~18, evening: 19~22
    """

    def share(indices: List[int]) -> float:
        total = sum(hour_hist) or 1
        return sum(hour_hist[i] for i in indices) / total

    late_night_idx = [23, 0, 1, 2, 3]
    morning_idx = [5, 6, 7, 8, 9, 10]
    day_idx = list(range(11, 19))
    evening_idx = [19, 20, 21, 22]

    scores = {
        "late_night": share(late_night_idx),
        "morning": share(morning_idx),
        "day": share(day_idx),
        "evening": share(evening_idx),
    }
    best = max(scores, key=scores.get)
    return best, scores[best]


def summarize_ingest_profile(
    session_id: str, lookback_days: int = 14
) -> Dict[str, Any]:
    """
    Firestore의 모바일 이벤트(캘린더/위치)를 N일 범위로 집계하여
    지시문/페르소나 보조 신호로 사용할 요약을 생성한다.

    반환 예시:
    {
      "prime_time": "late_night",
      "prime_time_share": 0.62,
      "avg_calendar_events_per_day": 2.1,
      "top_addresses": ["서울 강남구 ...", ...][:3]
    }
    실패 시 빈 dict.
    """
    db = _ensure_fs_db()
    if not db:
        return {}

    try:
        now_kst = datetime.now(KST)
        start_kst = now_kst - timedelta(days=lookback_days)
        start_utc = _to_utc(start_kst)
        end_utc = _to_utc(now_kst)

        q = (
            db.collection(os.getenv("FIRESTORE_USERS_COLL", "users"))
            .document(session_id)
            .collection(os.getenv("FIRESTORE_EVENTS_SUB", "unified_events"))
            .where("recordTimestamp", ">=", start_utc)
            .where("recordTimestamp", "<", end_utc)
        )

        docs = list(q.stream())
        if not docs:
            return {}

        events = [d.to_dict() for d in docs]

        # 시간대 히스토그램(0~23)
        hour_hist = [0] * 24
        # 주소 상위 3개
        addr_count: Dict[str, int] = {}
        # 캘린더 이벤트 수
        cal_cnt = 0

        for e in events:
            ts = e.get("recordTimestamp")
            if ts:
                hour_hist[_hour_bucket_kst(ts)] += 1

            dt = (e.get("dataType") or "").upper()
            if dt == "LOCATION":
                p = e.get("payload", {}) or {}
                addr = (p.get("address") or "").strip()
                if addr:
                    addr_count[addr] = addr_count.get(addr, 0) + 1
            elif dt == "CALENDAR_UPDATE":
                p = e.get("payload", {}) or {}
                items = p.get("events", []) or []
                for ev in items:
                    st_raw = ev.get("startTime")
                    if _safe_parse_iso(st_raw):
                        cal_cnt += 1

        prime_time, share = _prime_time_from_hist(hour_hist)
        days = max(1, lookback_days)
        avg_cal_per_day = cal_cnt / days
        top_addresses = sorted(addr_count.items(), key=lambda x: x[1], reverse=True)
        top_addresses = [a for a, _ in top_addresses[:3]]

        return {
            "prime_time": prime_time,
            "prime_time_share": round(share, 3),
            "avg_calendar_events_per_day": round(avg_cal_per_day, 2),
            "top_addresses": top_addresses,
            "hour_hist": hour_hist,
        }
    except Exception:
        return {}
