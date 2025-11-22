from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.proactive.data_contracts import (
    CausalLink,
    ContextAnalyzerInput,
    ContextAnalyzerOutput,
    ContextBundle,
    InsightSignal,
    get_conversation_texts,
    get_latest_mtm_summary,
    hash_user_id,
)
from backend.utils.logger import log_event

try:
    from zoneinfo import ZoneInfo  # Python 3.9+

    KST = ZoneInfo("Asia/Seoul")
except Exception:  # pragma: no cover - 환경에 따라 zoneinfo 미존재 시 UTC 폴백
    KST = timezone.utc


def _now_kst() -> datetime:
    return datetime.now(KST)


class ContextAnalyzerAgent:
    """
    다층 컨텍스트 수집 및 이상 탐지/인과 추론을 수행하는 에이전트.
    - 4개 소스 병렬 수집(rag/mobile/conversation/memory)
    - 600ms 타임아웃, 미완료 소스는 빈 문자열로 폴백
    - 구조화 로깅 이벤트 발행
    """

    TIMEOUT_S: float = 0.6

    async def analyze(self, state: ContextAnalyzerInput) -> ContextAnalyzerOutput:
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")

        log_event(
            "context.collection_start",
            {
                "session_id": session_id,
                "user_id": hash_user_id(user_id),
                "sources": ["rag", "mobile", "conversation", "memory"],
            },
        )

        t0 = time.time()
        tasks = {
            "rag": asyncio.create_task(self._fetch_rag(session_id)),
            "mobile": asyncio.create_task(self._fetch_mobile(user_id)),
            "conversation": asyncio.create_task(self._fetch_conversation(session_id)),
            "memory": asyncio.create_task(self._fetch_mtm(user_id, session_id)),
        }

        done, pending = await asyncio.wait(
            tasks.values(), timeout=self.TIMEOUT_S, return_when=asyncio.ALL_COMPLETED
        )

        latencies: Dict[str, int] = {}
        sizes: Dict[str, int] = {}
        results: Dict[str, str] = {
            "rag": "",
            "mobile": "",
            "conversation": "",
            "memory": "",
        }
        timeouts: List[str] = []

        for key, task in tasks.items():
            start = getattr(task, "_start_ts", None) or t0
            if task in done:
                try:
                    val = await task
                except Exception:
                    val = ""
                results[key] = val or ""
                latencies[key] = int((time.time() - start) * 1000)
                sizes[f"{key}_len"] = len(results[key])
            else:
                try:
                    task.cancel()
                except Exception:
                    pass
                timeouts.append(key)
                results[key] = ""
                latencies[key] = int((time.time() - start) * 1000)
                sizes[f"{key}_len"] = 0

        took_ms = int((time.time() - t0) * 1000)

        bundle: ContextBundle = ContextBundle(
            rag_ctx=results["rag"],
            mobile_ctx=results["mobile"],
            conversation_ctx=results["conversation"],
            memory_ctx=results["memory"],
        )

        # 이상 탐지/충돌/정서
        insights: List[InsightSignal] = []
        try:
            insights.extend(self._detect_health_signals(session_id, bundle))
        except Exception:
            pass
        try:
            insights.extend(await self._detect_schedule_conflicts(bundle))
        except Exception:
            pass
        try:
            insights.extend(self._detect_sentiment_patterns(bundle))
        except Exception:
            pass

        # 인과관계 추론
        causal_links: List[CausalLink] = []
        try:
            causal_links = self._infer_causality(bundle, insights)
        except Exception:
            causal_links = []

        # 관측성 로그
        log_event(
            "context.collection_complete",
            {
                "session_id": session_id,
                "latencies": latencies,
                "sizes": sizes,
                "timeouts": timeouts,
                "took_ms": took_ms,
            },
        )

        for s in insights:
            try:
                log_event(
                    "context.insight_detected",
                    {
                        "session_id": session_id,
                        "type": s.get("type"),
                        "severity": s.get("severity"),
                        "confidence": s.get("confidence"),
                        "evidence_count": len(s.get("evidence", [])),
                    },
                )
            except Exception:
                pass

        for c in causal_links:
            try:
                log_event(
                    "context.causality_inferred",
                    {
                        "session_id": session_id,
                        "cause": c.get("cause"),
                        "effect": c.get("effect"),
                        "implication": c.get("implication"),
                        "confidence": c.get("confidence"),
                    },
                )
            except Exception:
                pass

        return ContextAnalyzerOutput(
            internal_contexts=bundle,
            context_insights=insights,
            causal_links=causal_links,
            collection_latencies=latencies,
        )

    # -----------------------------
    # 병렬 수집
    # -----------------------------
    async def _fetch_rag(self, session_id: str) -> str:
        """
        RAG 컨텍스트 수집: 최신 MTM 요약 또는 라우팅 요약을 질의로 사용.
        실패 시 빈 문자열 반환.
        """

        try:
            from backend.rag.retrieval import retrieve_enhanced

            query = get_latest_mtm_summary(session_id, session_id) or ""
            if not query:
                return ""
            ctx = await retrieve_enhanced(
                query=query, route="rag", session_id=session_id
            )
            return ctx or ""
        except Exception:
            return ""

    async def _fetch_mobile(self, user_id: str) -> str:
        try:
            from backend.ingest.mobile_context import build_mobile_ctx

            return await build_mobile_ctx(user_id)
        except Exception:
            return ""

    async def _fetch_conversation(self, session_id: str) -> str:
        """
        최근 대화 컨텍스트: 최신 N개 메시지를 블록 문자열로 구성.
        - PII 최소화는 로깅 단계에서만 적용(원문은 State에만 존재).
        """

        try:
            texts = get_conversation_texts(session_id, max_messages=40)
            if not texts:
                return ""
            # 최근 12개만 간단 요약형(헤드)로 구성
            tail = texts[-12:]
            lines = [t.strip() for t in tail if t and t.strip()]
            return "\n".join(lines)
        except Exception:
            return ""

    async def _fetch_mtm(self, user_id: str, session_id: str) -> str:
        try:
            return get_latest_mtm_summary(user_id, session_id) or ""
        except Exception:
            return ""

    # -----------------------------
    # 탐지 로직
    # -----------------------------
    def _detect_health_signals(
        self, session_id: str, contexts: ContextBundle
    ) -> List[InsightSignal]:
        """
        Z-score 기반 건강 시그널 탐지. 기준선이 없거나 표준편차가 0이면 감지하지 않음.
        """

        recent_texts = (contexts.get("conversation_ctx") or "").split("\n")
        if not recent_texts:
            return []

        negative_keywords = ["피곤", "힘들", "아프", "스트레스", "불안"]
        recent_last3 = recent_texts[-3:]
        recent_count = sum(
            t.count(kw) for t in recent_last3 for kw in negative_keywords
        )

        mean, std = self._load_baseline_neg_kw(session_id)
        if std <= 0:
            return []

        z = (float(recent_count) - float(mean)) / float(std)
        if z <= 2.0:
            return []

        sev = "medium" if z < 3.0 else "high"
        conf = min(1.0, z / 3.0)
        sig: InsightSignal = InsightSignal(
            type="health_alert",
            pattern="repeated_negative_sentiment",
            severity=sev,
            confidence=conf,
            evidence=[
                f"최근 3개 발화 부정 키워드 {recent_count}회 (평균 대비 {z:.2f}σ)",
            ],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return [sig]

    async def _detect_schedule_conflicts(
        self, contexts: ContextBundle
    ) -> List[InsightSignal]:
        """
        현재 위치와 다음 일정 장소 간 이동 시간을 추정하고 버퍼 부족 시 경고.
        - 위치가 위/경도 쌍으로 제공될 때만 계산(지오코딩 비사용).
        """

        mobile = contexts.get("mobile_ctx") or ""
        if not mobile:
            return []

        events = self._parse_calendar_events(mobile)
        if not events:
            return []

        next_ev = events[0]
        event_location = (next_ev.get("location") or "").strip()
        start_dt: Optional[datetime] = next_ev.get("start_time")
        if not event_location or not start_dt:
            return []

        cur_loc = self._parse_current_location(mobile)
        if not cur_loc:
            return []

        travel_min = self._estimate_travel_minutes(cur_loc, event_location)
        if travel_min is None:
            return []

        now = _now_kst()
        time_until_min = (start_dt - now).total_seconds() / 60.0
        required_buffer = travel_min * 0.3
        actual_buffer = time_until_min - travel_min

        if actual_buffer < required_buffer:
            sev = "high" if actual_buffer < 0 else "medium"
            sig: InsightSignal = InsightSignal(
                type="schedule_conflict",
                pattern="tight_schedule",
                severity=sev,
                confidence=0.9,
                evidence=[
                    f"현재 위치: {cur_loc}",
                    f"일정 장소: {event_location}",
                    f"이동 시간: {travel_min:.0f}분",
                    f"남은 시간: {time_until_min:.0f}분",
                    f"버퍼 부족: {(-actual_buffer):.0f}분",
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return [sig]
        return []

    def _detect_sentiment_patterns(
        self, contexts: ContextBundle
    ) -> List[InsightSignal]:
        """
        정서 패턴 감지는 현재 단계에서 보수적으로 비활성화(추후 ML 교체).
        """

        return []

    def _infer_causality(
        self, contexts: ContextBundle, insights: List[InsightSignal]
    ) -> List[CausalLink]:
        links: List[CausalLink] = []

        mobile = contexts.get("mobile_ctx", "")
        rag_ctx = contexts.get("rag_ctx", "")

        # 규칙 1: 비 예보 + 외출 일정 → 방수 장비
        if ("비" in mobile) and ("외출" in mobile):
            links.append(
                CausalLink(
                    cause="비 예보",
                    effect="외출 일정 존재",
                    implication="방수 장비 준비 필요 (우산, 방수화)",
                    confidence=0.9,
                )
            )

        # 규칙 2: 건강 악화 + PT → 휴식 권유
        health_alerts = [s for s in insights if s.get("type") == "health_alert"]
        if health_alerts and ("PT" in mobile):
            links.append(
                CausalLink(
                    cause="반복되는 피로 증상",
                    effect="오늘 저녁 고강도 운동 계획",
                    implication="컨디션 고려하여 운동 강도 조절 또는 연기 권장",
                    confidence=0.85,
                )
            )

        # 규칙 3: 중요 미팅 + 선물 관심사 → 관계 유지
        if ("미팅" in mobile) and ("선물" in rag_ctx):
            links.append(
                CausalLink(
                    cause="다가오는 미팅",
                    effect="과거 선물 관심사 이력",
                    implication="미팅 전 선물 준비로 관계 강화 기회",
                    confidence=0.75,
                )
            )
        return links

    # -----------------------------
    # 보조 유틸
    # -----------------------------
    def _load_baseline_neg_kw(self, session_id: str) -> tuple[float, float]:
        """
        Redis에서 baseline 통계를 읽기 전용으로 조회.
        키가 없으면 (0.0, 0.0) 반환하여 감지 비활성화.
        """

        try:
            # settings 우선, 실패 시 환경변수 사용
            try:
                from backend.config import get_settings

                REDIS_URL = get_settings().REDIS_URL
            except Exception:
                REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

            import redis  # type: ignore

            r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            m = r.get(f"baseline:neg_kw_mean:{session_id}")
            s = r.get(f"baseline:neg_kw_std:{session_id}")
            mean = float(m) if m is not None else 0.0
            std = float(s) if s is not None else 0.0
            return mean, std
        except Exception:
            return 0.0, 0.0

    def _parse_calendar_events(self, mobile_ctx: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in (mobile_ctx or "").split("\n")]
        events: List[Dict[str, Any]] = []
        in_today = False
        for ln in lines:
            if ln.startswith("[오늘 일정]"):
                in_today = True
                continue
            if ln.startswith("[") and "]" in ln and not ln.startswith("[오늘 일정]"):
                in_today = False
            if in_today and ln.startswith("- "):
                # 포맷: "- HH:MM 제목 @ 장소" 또는 "- HH:MM 제목"
                body = ln[2:].strip()
                # 시간 추출
                m = re.match(r"(?P<hh>\d{2}):(\d{2})\s+(?P<title>.+)", body)
                if not m:
                    continue
                hh = int(m.group("hh"))
                mm = int(body[3:5]) if ":" in body[:5] else 0
                title_and_rest = body[6:].strip() if len(body) > 5 else ""
                loc = ""
                if " @ " in title_and_rest:
                    try:
                        _, loc = title_and_rest.rsplit(" @ ", 1)
                    except Exception:
                        loc = ""
                # 오늘 날짜 KST 기준으로 조합(서버 UTC 기준 보수 처리)
                now = _now_kst()
                start_dt = datetime(
                    now.year, now.month, now.day, hour=hh, minute=mm, tzinfo=now.tzinfo
                )
                events.append({"start_time": start_dt, "location": loc})
        events.sort(key=lambda e: e.get("start_time") or datetime.max)
        return events

    def _parse_current_location(self, mobile_ctx: str) -> Optional[str]:
        # "[현재 위치]\n현재 위치: (lat, lng)" 또는 "현재 위치: 주소"
        lines = [ln.strip() for ln in (mobile_ctx or "").split("\n")]
        for i, ln in enumerate(lines):
            if ln.startswith("[현재 위치]"):
                # 다음 라인 탐색
                if i + 1 < len(lines):
                    val = lines[i + 1]
                    if val.startswith("현재 위치:"):
                        return val.split("현재 위치:", 1)[1].strip()
        return None

    def _estimate_travel_minutes(
        self, current_location: str, event_location: str
    ) -> Optional[float]:
        # 위/경도 쌍으로 계산 가능한 경우에만 처리
        cur = self._parse_latlng(current_location)
        ev = self._parse_latlng(event_location)
        if not cur or not ev:
            return None
        lat1, lng1 = cur
        lat2, lng2 = ev
        dist_km = self._haversine_km(lat1, lng1, lat2, lng2)
        # 차량 이동 평균 15km/h 가정(도심 혼잡 보정), 분 환산
        minutes = (dist_km / 15.0) * 60.0
        return max(1.0, minutes)

    @staticmethod
    def _parse_latlng(s: str) -> Optional[tuple[float, float]]:
        try:
            m = re.search(r"\(([^,]+),\s*([^\)]+)\)", s)
            if not m:
                return None
            lat = float(m.group(1).strip())
            lng = float(m.group(2).strip())
            return lat, lng
        except Exception:
            return None

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import atan2, cos, radians, sin, sqrt

        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
