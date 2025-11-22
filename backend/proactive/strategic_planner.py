"""
backend.proactive.strategic_planner - 세션 1 전략 플래너 노드

역할:
- MTM/캘린더/프로필 요약을 읽기 전용으로 결합하여 정보 필요(info_needs)를 산출
- 이벤트 근접도(Urgency) 계산
- 선호도/근접도/시즌성 가중 스코어로 우선순위 결정

주의:
- 절대 쓰기 금지(읽기 전용 접근만 허용)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from backend.proactive.schemas import ProactiveState
from backend.utils.datetime_utils import fmt_hm_kst, kst_day_bounds, safe_parse_iso
from backend.utils.logger import log_event

logger = logging.getLogger("proactive.planner")


def _calculate_urgency(event_time: datetime, now: datetime) -> float:
    """이벤트 근접도 계산 (세션 명세의 지수 감쇠 수식 적용).

    - 24시간 이내: 1.0
    - 24시간~7일: 0.6 * exp(-Δt/72)
    - 그 외: 0.1
    """
    delta_hours = (event_time - now).total_seconds() / 3600.0
    if delta_hours <= 24:
        return 1.0
    if delta_hours <= 168:
        return float(0.6 * math.exp(-delta_hours / 72.0))
    return 0.1


def _seasonal_factor(now: datetime) -> float:
    """간단한 시즌성 팩터.
    - 프라임타임(11:30~13:30, 19:00~21:30): 1.2x
    - 주말: 1.1x
    - 기본: 1.0x
    """
    h = now.hour + now.minute / 60.0
    prime = 1.2 if (11.5 <= h <= 13.5) or (19.0 <= h <= 21.5) else 1.0
    weekend = 1.1 if now.weekday() >= 5 else 1.0
    return float(prime * weekend)


def _weighted_priority(
    preference_score: float, urgency: float, seasonal: float
) -> float:
    """가중 우선순위 스코어.
    α=0.4(선호), β=0.5(근접), γ=0.1(시즌) — 합 1.0
    """
    return float(0.4 * preference_score + 0.5 * urgency + 0.1 * seasonal)


def _load_user_preferences(user_id: str) -> Dict[str, float]:
    """프로필 RAG를 통해 사용자 선호도를 읽기 전용으로 조회.
    - 가용 데이터가 없으면 중립값(0.5)로 폴백
    """
    try:
        from backend.rag.profile_rag import get_profile_rag

        rag = get_profile_rag()
        # 질의 텍스트는 선호 일반 요약 — 내부적으로 guard/core/dynamic를 조회
        q = "사용자 일반 선호 요약"
        prof = rag.profile_coll.query(
            expr=f"user_id == '{user_id}' and status == 'active'",
            output_fields=["key_path", "value", "confidence", "tier", "category"],
            limit=100,
        )
        prefs: Dict[str, float] = {}
        for r in prof or []:
            kp = (r.get("key_path") or "").strip()
            if kp.startswith("preferences."):
                # 간단 매핑: 키 존재 → 선호치 0.7, confidence로 보정
                base = 0.7
                conf = float(r.get("confidence", 0.5))
                prefs[kp.split(".", 1)[-1]] = min(
                    1.0, max(0.0, base * (0.5 + 0.5 * conf))
                )
        return prefs
    except Exception:
        return {}


def _load_today_calendar(user_id: str) -> List[Dict]:
    """Firestore에서 오늘 일정 이벤트를 읽기 전용으로 조회.
    schema: ingest/mobile_context 참고
    """
    try:
        from google.cloud.firestore_v1 import FieldFilter  # type: ignore

        from backend.config import get_firestore_client
    except Exception:
        get_firestore_client = None  # type: ignore
        FieldFilter = None  # type: ignore

    if get_firestore_client is None:
        return []
    db = get_firestore_client()
    if not db:
        return []

    start_kst, end_kst = kst_day_bounds()
    try:
        coll = db.collection("users").document(user_id).collection("unified_events")
        if FieldFilter is not None:
            q = (
                coll.where(filter=FieldFilter("recordTimestamp", ">=", start_kst))
                .where(filter=FieldFilter("recordTimestamp", "<", end_kst))
                .order_by("recordTimestamp", direction="DESCENDING")
                .limit(200)
            )
        else:
            q = (
                coll.where("recordTimestamp", ">=", start_kst)
                .where("recordTimestamp", "<", end_kst)
                .order_by("recordTimestamp", direction="DESCENDING")
                .limit(200)
            )
        docs = list(q.stream())
    except Exception:
        docs = []

    events = []
    for d in docs:
        m = d.to_dict() or {}
        if (m.get("dataType") or "").upper() != "CALENDAR_UPDATE":
            continue
        payload = m.get("payload", {}) or {}
        for ev in payload.get("events", []) or []:
            st_raw = ev.get("startTime")
            if not st_raw:
                continue
            st_dt = safe_parse_iso(st_raw)
            if not st_dt:
                continue
            events.append(
                {
                    "id": ev.get("id") or ev.get("eventId") or st_raw,
                    "title": (ev.get("title") or "").strip() or "(제목 없음)",
                    "location": (ev.get("location") or "").strip(),
                    "start": st_dt,
                }
            )
    return events


def strategic_planner_node(state: ProactiveState) -> ProactiveState:
    """Strategic Planner: info_needs + urgency + 우선순위 산출.

    입력: user_id, session_id, timestamp
    출력: planner_analysis, info_needs, urgency_scores
    """
    user_id = state.get("user_id") or state.get("session_id") or ""
    if not user_id:
        return state

    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # 1) 오늘 일정 로드 → 근접도 계산
    cal = _load_today_calendar(user_id)
    urgencies: Dict[str, float] = {}
    for ev in cal:
        try:
            urgencies[ev["id"]] = _calculate_urgency(
                ev["start"].astimezone(now.tzinfo), now
            )
        except Exception:
            continue

    # 2) 사용자 선호도 로드
    prefs = _load_user_preferences(user_id)
    seasonal = _seasonal_factor(now.astimezone())

    # 3) 정보 필요 카테고리 후보 (건강/관계/재무/여가/업무)
    candidates = ["건강", "관계", "재무", "여가", "업무"]
    scores: List[Tuple[str, float]] = []
    for cat in candidates:
        pref = float(prefs.get(cat, 0.5))
        # 카테고리별 대표 이벤트 근접도(달리 없으면 0.1)
        # 간단히: 일정 제목에 키워드가 있으면 매핑 강화
        cat_urg = 0.1
        try:
            for ev in cal:
                t = (ev.get("title") or "").lower()
                if (
                    (cat == "업무" and any(k in t for k in ["회의", "미팅", "업무"]))
                    or (cat == "건강" and any(k in t for k in ["병원", "운동", "헬스"]))
                    or (cat == "관계" and any(k in t for k in ["약속", "만남", "생일"]))
                    or (cat == "여가" and any(k in t for k in ["여행", "영화", "취미"]))
                    or (cat == "재무" and any(k in t for k in ["결제", "납부", "세금"]))
                ):
                    eid = ev["id"]
                    cat_urg = max(cat_urg, float(urgencies.get(eid, 0.1)))
        except Exception:
            pass

        score = _weighted_priority(pref, cat_urg, seasonal)
        scores.append((cat, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    info_needs = [c for c, _ in scores]

    planner_analysis = {
        "priorities": info_needs[:3],
        "reasoning": "선호도/근접도/시즌성 가중치 기반 산출",
        "seasonal": seasonal,
        "pref_head": {k: round(v, 3) for k, v in list(prefs.items())[:5]},
        "calendar_today": [
            {
                "title": ev.get("title"),
                "time": fmt_hm_kst(ev.get("start")) if ev.get("start") else "",
                "location": ev.get("location", ""),
                "urgency": round(float(urgencies.get(ev.get("id", ""), 0.1)), 3),
            }
            for ev in cal[:5]
        ],
    }

    out: ProactiveState = {
        **state,
        "planner_analysis": planner_analysis,
        "info_needs": info_needs,
        "urgency_scores": {k: round(float(v), 4) for k, v in urgencies.items()},
    }

    try:
        log_event(
            "proactive.planner_complete",
            {
                "user_id": user_id,
                "priorities": planner_analysis.get("priorities", [])[:3],
                "calendar_count": len(cal),
            },
        )
    except Exception:
        pass

    return out


__all__ = ["strategic_planner_node"]
