"""
backend.ingest.mobile_context - 모바일 컨텍스트 빌더

Firestore에서 사용자의 모바일 앱 데이터(일정, 위치)를 조회하여 컨텍스트를 생성합니다.
"""

import asyncio
import datetime
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo  # <-- 이 줄을 사용

logger = logging.getLogger("mobile_context")


def get_current_time_context():
    """현재 한국 시간 컨텍스트 문자열을 생성합니다."""
    # 시간대 설정 (ZoneInfo 사용)
    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    # 요일 한글 변환
    weekday_kor = ["월", "화", "수", "목", "금", "토", "일"]
    # 최종 시간 문자열 포맷팅
    time_str = now.strftime(f"%Y년 %m월 %d일 {weekday_kor[now.weekday()]}요일 %p %I:%M")
    # 시간 정보만 간결하게 반환
    return f"현재 시각: {time_str} (KST)"


class MobileContextBuilder:
    """모바일 컨텍스트 빌더"""

    def __init__(self):
        self._settings = None
        self._fs_db = None

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def build_context(self, user_id: str) -> str:
        """모바일 컨텍스트 빌드 (비동기)"""
        return await asyncio.to_thread(self._build_context_sync, user_id)

    def _build_context_sync(self, user_id: str) -> str:
        """Firestore 동기 조회"""
        from backend.config import get_firestore_client

        try:
            # Firestore v1 API: FieldFilter 권장 방식 사용
            from google.cloud.firestore_v1 import FieldFilter  # type: ignore
        except Exception:
            FieldFilter = None  # type: ignore
        from backend.utils.datetime_utils import (
            fmt_hm_kst,
            kst_day_bounds,
            safe_parse_iso,
            to_utc,
        )

        db = get_firestore_client()
        if not db:
            return ""

        try:
            kst_start, kst_end = kst_day_bounds()
            start_utc = to_utc(kst_start)
            end_utc = to_utc(kst_end)

            coll = (
                db.collection(self.settings.FIRESTORE_USERS_COLL)
                .document(user_id)
                .collection(self.settings.FIRESTORE_EVENTS_SUB)
            )
            if FieldFilter is not None:
                q = (
                    coll.where(filter=FieldFilter("recordTimestamp", ">=", start_utc))
                    .where(filter=FieldFilter("recordTimestamp", "<", end_utc))
                    .order_by("recordTimestamp", direction="DESCENDING")
                    .limit(200)
                )
            else:
                # 구버전 호환(경고는 가능)
                q = (
                    coll.where("recordTimestamp", ">=", start_utc)
                    .where("recordTimestamp", "<", end_utc)
                    .order_by("recordTimestamp", direction="DESCENDING")
                    .limit(200)
                )

            docs = list(q.stream())
            if not docs:
                return ""

            events = [d.to_dict() for d in docs]

            # 위치 추출
            latest_loc = None
            for e in events:
                if (e.get("dataType") or "").upper() == "LOCATION":
                    latest_loc = e
                    break

            # 일정 추출
            today_events = []
            for e in events:
                if (e.get("dataType") or "").upper() != "CALENDAR_UPDATE":
                    continue
                payload = e.get("payload", {}) or {}
                for ev in payload.get("events", []) or []:
                    st_raw = ev.get("startTime")
                    if not st_raw:
                        continue
                    st_dt = safe_parse_iso(st_raw)
                    if not st_dt:
                        continue
                    if kst_start <= st_dt.astimezone(kst_start.tzinfo) < kst_end:
                        today_events.append(ev)

            # 포맷팅
            lines_cal = []
            if today_events:
                today_events.sort(
                    key=lambda ev: safe_parse_iso(ev.get("startTime") or "")
                    or datetime.now()
                )
                for ev in today_events[:5]:
                    st = safe_parse_iso(ev.get("startTime") or "")
                    hm = fmt_hm_kst(st) if st else ""
                    title = (ev.get("title") or "").strip() or "(제목 없음)"
                    loc = (ev.get("location") or "").strip()
                    if loc:
                        lines_cal.append(f"- {hm} {title} @ {loc}")
                    else:
                        lines_cal.append(f"- {hm} {title}")

            line_loc = ""
            if latest_loc:
                p = latest_loc.get("payload", {}) or {}
                addr = (p.get("address") or "").strip()
                if addr:
                    line_loc = f"현재 위치: {addr}"
                else:
                    lat, lng = p.get("latitude"), p.get("longitude")
                    if lat is not None and lng is not None:
                        line_loc = f"현재 위치: ({lat:.5f}, {lng:.5f})"

            blocks = []
            if lines_cal:
                blocks.append("[오늘 일정]\n" + "\n".join(lines_cal))
            if line_loc:
                blocks.append("[현재 위치]\n" + line_loc)

            return "\n\n".join(blocks)
        except Exception as e:
            logger.warning(f"[mobile] Fetch error: {e}")
            return ""


_builder_instance = None


def get_mobile_context_builder():
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = MobileContextBuilder()
    return _builder_instance


async def build_mobile_ctx(user_id: str) -> str:
    """호환성 래퍼"""
    builder = get_mobile_context_builder()
    return await builder.build_context(user_id)
