from __future__ import annotations

"""
디바이스 컨텍스트 매니저

역할:
- 간단한 상태 → 액션(block/tone_shift/allow) 규칙 테이블 제공
- ProactiveScoringEngine/에이전트 게이트에서 억제/톤 전환 결정에 사용

주의:
- 상태 문자열은 상위 모듈에서 상향 정규화(대문자)하여 전달한다고 가정
"""

from typing import Dict, Literal

Action = Literal["block", "tone_shift", "allow"]


RULES: Dict[str, Action] = {
    "DND_ON": "block",
    "NIGHT": "block",
    "IN_MEETING": "block",
    "CHARGING": "tone_shift",
    "HEADPHONES": "tone_shift",
}


def decide_action(device_state: str) -> Action:
    return RULES.get((device_state or "").upper(), "allow")


__all__ = ["decide_action", "Action"]
