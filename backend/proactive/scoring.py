from __future__ import annotations

"""
ProactiveScoringEngine — 컨텍스트 기반 필요성 점수 계산기

설명:
- 입력 컨텍스트의 요약 상태(activity/location/device/calendar)를 바탕으로 0~1 점수를 산출한다.
- 억제 규칙: DND/NIGHT/IN_MEETING 신호가 있으면 0점으로 강제하여 전송을 차단한다.
- 임계치는 환경변수 PROACTIVE_SCORE_THRESHOLD(기본 0.55)로 제어한다.

주의:
- CPU-only 규칙형. 외부 호출 없음. 예외 시 보수적(Fail-Closed) 사용을 권장.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

from backend.config import get_settings


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reasons: List[str]


class ProactiveScoringEngine:
    def __init__(self) -> None:
        self.threshold: float = float(get_settings().PROACTIVE_SCORE_THRESHOLD)

    def score(self, contexts: Dict[str, str]) -> ScoreResult:
        """
        단순 규칙형 점수 산출.
        - 억제: DND_ON/NIGHT/IN_MEETING → 0.0
        - 가중 합: sa + sl + sd + sc (0~1 클램프)
        """
        device = (contexts.get("device_state") or "").upper()
        calendar = (contexts.get("calendar_state") or "").upper()
        reasons: List[str] = []

        # (신규) 디바이스 상태 규칙 기반 억제
        try:
            if bool(get_settings().FEATURE_DEVICE_STATE):
                from backend.context.device_state import decide_action

                act = decide_action(device)
                if act == "block":
                    reasons.append("suppressed: device_rule")
                    return ScoreResult(score=0.0, reasons=reasons)
        except Exception:
            pass

        # 기본 강제 억제 규칙(캘린더)
        if "IN_MEETING" in calendar:
            reasons.append("suppressed: in_meeting")
            return ScoreResult(score=0.0, reasons=reasons)

        activity = (contexts.get("activity_state") or "").upper()
        location = (contexts.get("location_state") or "").upper()

        # 간단 가중. 결측은 보수적 중간값 처리
        sa = 0.3 if activity in {"SITTING", "FOCUS"} else 0.15 if activity else 0.2
        sl = 0.3 if location in {"AT_HOME", "AT_WORK"} else 0.2
        sd = 0.2
        sc = 0.2 if "MEETING_SOON" in calendar else 0.1

        score = max(0.0, min(1.0, sa + sl + sd + sc))
        reasons.extend([f"sa={sa}", f"sl={sl}", f"sd={sd}", f"sc={sc}"])
        return ScoreResult(score=score, reasons=reasons)


__all__ = ["ProactiveScoringEngine", "ScoreResult"]
