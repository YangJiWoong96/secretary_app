"""
증거 피드백 스키마 정의
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EvidenceFeedback:
    """
    증거에 대한 사용자 피드백 및 동적 신뢰도 관리

    Note:
        - 동일 evidence_id에 대한 피드백은 시간 가중 평균으로 신뢰도 산출
        - 재랭킹 시 trust(d) = Σ(confidence_adjustment × time_decay)
    """

    user_id: str
    session_id: str
    turn_id: str
    evidence_type: Literal["rag", "web"]
    evidence_id: str
    original_evidence_snippet: str
    feedback_type: Literal["positive", "negative", "correction", "elaboration"]
    user_comment: str
    ai_response: str
    timestamp: int
    confidence_adjustment: float = 0.0
