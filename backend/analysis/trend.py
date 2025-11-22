from __future__ import annotations

"""
스파이크(바이럴) 검출

역할:
- 전일 대비 급증한 키워드/타이틀을 간단한 배수 기준(threshold)으로 검출한다.
- 상위 로직(프로액티브 에이전트)이 알림 후보로 사용할 수 있도록 최소한의 인터페이스만 제공한다.
"""

from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import re


def detect_viral(
    today: Dict[str, int], yesterday: Dict[str, int], threshold: float = 2.0
) -> List[str]:
    """
    전일 대비 급증 항목(키/타이틀) 검출.

    Args:
        today: 금일 카운트 맵(항목 → 등장 횟수)
        yesterday: 전일 카운트 맵(항목 → 등장 횟수)
        threshold: 급증 배수 임계값(기본 2.0배)

    Returns:
        List[str]: 급증으로 판단된 항목 리스트(등장 순서 보장 없음)
    """
    results: List[str] = []
    for k, v in (today or {}).items():
        prev = max(0, int(yesterday.get(k, 0)))
        if prev > 0:
            if (float(v) / float(prev)) >= float(threshold):
                results.append(k)
        else:
            # 완전 신규(전일 0)인 경우는 이 함수에서는 제외한다.
            # 신규 토픽 감지는 별도 정책(최소 등장 횟수 기준 등)에서 다루는 것을 권장.
            continue
    return results


def _snapshot_root() -> Path:
    # backend/analysis/trend.py → backend/evidence/websnap
    return Path(__file__).resolve().parents[1] / "evidence" / "websnap"


def _normalize_title_text(title: str) -> str:
    t = str(title or "")
    t = re.sub(r"</?b>", "", t, flags=re.I)
    t = " ".join(t.split())
    return t


def _count_titles_for_day(day: str, kinds: List[str]) -> Dict[str, int]:
    """
    스냅샷 기준 특정 일자(day: YYYYMMDD)에서 제목별 등장 횟수를 합산한다.
    """
    counts: Dict[str, int] = {}
    root = _snapshot_root() / day
    if not root.exists():
        return counts
    for kind in kinds or []:
        for fp in sorted(root.glob(f"{kind}-*.jsonl")):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            title = _normalize_title_text(rec.get("title", ""))
                            if not title:
                                continue
                            counts[title] = counts.get(title, 0) + 1
                        except Exception:
                            continue
            except Exception:
                continue
    return counts


def detect_viral_from_snapshots(
    kinds: List[str] | None = None, threshold: float = 2.0
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    스냅샷 폴더(evidence/websnap)에서 오늘/어제의 제목 등장 횟수를 비교해 스파이크 항목을 검출한다.

    Returns:
        (viral_list, today_counts, yesterday_counts)
    """
    kinds = kinds or ["news", "webkr", "blog"]
    now = datetime.now()
    day_today = now.strftime("%Y%m%d")
    day_yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
    today = _count_titles_for_day(day_today, kinds)
    yesterday = _count_titles_for_day(day_yesterday, kinds)
    viral = detect_viral(today, yesterday, threshold=threshold)
    return viral, today, yesterday


__all__ = ["detect_viral", "detect_viral_from_snapshots"]
