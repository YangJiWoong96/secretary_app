"""
backend.validation.temporal_validator - 시간 정합성 검증기

상대 시점 표현을 절대 날짜로 해석하고, 증거와 답변의 날짜 일치를 검증합니다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger("temporal_validator")


class TemporalValidator:
    """시간 정합성 검증기"""

    def __init__(self):
        self._now: Optional[datetime] = None

    @property
    def now(self) -> datetime:
        """현재 시각 (KST)"""
        if self._now is None:
            try:
                from backend.utils.datetime_utils import now_kst

                self._now = now_kst()
            except Exception:
                self._now = datetime.now()
        return self._now

    def extract_absolute_dates(
        self, text: str, context: str = ""
    ) -> List[Tuple[str, str]]:
        """
        텍스트에서 절대 날짜 추출.

        Returns: [(원본 표현, YYYY-MM-DD), ...]
        """
        if not text:
            return []

        patterns = {
            # 상대 시점
            r"\b올해\b": lambda: f"{self.now.year}-01-01",
            r"\b작년\b": lambda: f"{self.now.year - 1}-01-01",
            r"\b내년\b": lambda: f"{self.now.year + 1}-01-01",
            r"\b이번\s*달\b": lambda: f"{self.now.year}-{self.now.month:02d}-01",
            r"\b지난\s*달\b": lambda: (
                self.now.replace(day=1) - timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            r"\b다음\s*달\b": lambda: (self.now.replace(day=28) + timedelta(days=4))
            .replace(day=1)
            .strftime("%Y-%m-%d"),
            r"\b어제\b": lambda: (self.now - timedelta(days=1)).strftime("%Y-%m-%d"),
            r"\b오늘\b": lambda: self.now.strftime("%Y-%m-%d"),
            r"\b내일\b": lambda: (self.now + timedelta(days=1)).strftime("%Y-%m-%d"),
            # 절대 날짜
            r"(20\d{2})[년\-\./](0?[1-9]|1[0-2])[월\-\./](0?[1-9]|[12]\d|3[01])": lambda m: f"{int(m.group(1))}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            r"(20\d{2})[년\-\./](0?[1-9]|1[0-2])": lambda m: f"{int(m.group(1))}-{int(m.group(2)):02d}-01",
        }

        results: List[Tuple[str, str]] = []
        for pattern, resolver in patterns.items():
            for match in re.finditer(pattern, text):
                original = match.group(0)
                try:
                    if callable(resolver):
                        if match.groups():
                            resolved = resolver(match)  # type: ignore[arg-type]
                        else:
                            resolved = resolver()  # type: ignore[call-arg]
                    else:
                        resolved = resolver  # type: ignore[assignment]
                    results.append((original, resolved))
                except Exception as e:
                    logger.warning(f"Date resolution failed: {original} - {e}")

        return results

    def validate_date_consistency(
        self, answer: str, evidence: str, strict: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        답변과 증거의 날짜 일치 검증.
        Returns: (일치 여부, 불일치 이유)
        """
        try:
            answer_dates = self.extract_absolute_dates(answer)
            evidence_dates = self.extract_absolute_dates(evidence)

            if not answer_dates or not evidence_dates:
                return True, None

            answer_years = {int(d.split("-")[0]) for _, d in answer_dates}
            evidence_years = {int(d.split("-")[0]) for _, d in evidence_dates}

            if strict and answer_years and evidence_years:
                if not answer_years.intersection(evidence_years):
                    return (
                        False,
                        f"날짜 불일치: 답변={answer_years} vs 증거={evidence_years}",
                    )
            return True, None
        except Exception as e:
            logger.warning(f"validate_date_consistency error: {e}")
            return True, None


_validator_instance: Optional[TemporalValidator] = None


def get_temporal_validator() -> TemporalValidator:
    """전역 시간 검증기 인스턴스"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TemporalValidator()
    return _validator_instance
