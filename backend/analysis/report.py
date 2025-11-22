from __future__ import annotations

"""
일/주간 Markdown 리포트 생성기

역할:
- 검색 스냅샷(evidence/websnap)을 기준으로 일/주간 상위 이슈(제목)와 분포를 요약해 Markdown으로 반환한다.
"""

from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import re


def _snapshot_root() -> Path:
    return Path(__file__).resolve().parents[1] / "evidence" / "websnap"


def _normalize_title_text(title: str) -> str:
    t = str(title or "")
    t = re.sub(r"</?b>", "", t, flags=re.I)
    t = " ".join(t.split())
    return t


def _collect_counts_for_range(
    start_day: str, end_day: str, kinds: List[str]
) -> Dict[str, int]:
    """
    날짜 범위(YYYYMMDD~YYYYMMDD) 동안 제목별 등장 횟수 합산.
    """
    counts: Dict[str, int] = {}
    root = _snapshot_root()
    try:
        sd = datetime.strptime(start_day, "%Y%m%d")
        ed = datetime.strptime(end_day, "%Y%m%d")
        cur = sd
        while cur <= ed:
            day_dir = root / cur.strftime("%Y%m%d")
            if day_dir.exists():
                for kind in kinds or []:
                    for fp in sorted(day_dir.glob(f"{kind}-*.jsonl")):
                        try:
                            with fp.open("r", encoding="utf-8") as f:
                                for line in f:
                                    try:
                                        rec = json.loads(line)
                                        title = _normalize_title_text(
                                            rec.get("title", "")
                                        )
                                        if not title:
                                            continue
                                        counts[title] = counts.get(title, 0) + 1
                                    except Exception:
                                        continue
                        except Exception:
                            continue
            cur += timedelta(days=1)
    except Exception:
        return counts
    return counts


def _format_markdown(
    title: str, period_desc: str, top_items: List[Tuple[str, int]]
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**기간**: {period_desc}")
    lines.append(f"**생성시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 🔥 TOP 이슈")
    lines.append("")
    if not top_items:
        lines.append("- (자료 없음)")
    else:
        for i, (t, c) in enumerate(top_items, 1):
            lines.append(f"{i}. {t} — {c}회")
    lines.append("")
    lines.append("---")
    lines.append("*본 리포트는 자동 생성되었습니다.*")
    return "\n".join(lines)


def generate_daily_report(kinds: List[str] | None = None, top_n: int = 10) -> str:
    kinds = kinds or ["news", "webkr", "blog"]
    today = datetime.now().strftime("%Y%m%d")
    counts = _collect_counts_for_range(today, today, kinds)
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    md = _format_markdown("📅 일간 이슈 요약", today, top)
    return md


def generate_weekly_report(kinds: List[str] | None = None, top_n: int = 15) -> str:
    kinds = kinds or ["news", "webkr", "blog"]
    now = datetime.now()
    end_day = now.strftime("%Y%m%d")
    start_day = (now - timedelta(days=6)).strftime("%Y%m%d")
    counts = _collect_counts_for_range(start_day, end_day, kinds)
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    md = _format_markdown("🗓️ 주간 이슈 요약", f"{start_day} ~ {end_day}", top)
    return md


__all__ = ["generate_daily_report", "generate_weekly_report"]
