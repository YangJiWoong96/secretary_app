from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

try:
    from google.cloud import firestore
except Exception:
    firestore = None

# Optional: joblib for persistence
try:
    from joblib import dump as _joblib_dump
except Exception:
    _joblib_dump = None

# Optional: sklearn
try:
    from sklearn.ensemble import GradientBoostingClassifier

    excepted_sklearn = False
except Exception:
    GradientBoostingClassifier = None
    excepted_sklearn = True

KST = timezone(timedelta(hours=9))

_FEATURE_ORDER = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "web_len",
    "rag_len",
    "mob_len",
    "minutes_since_last_open",
]


def _to_float(x, default=0.0) -> float:
    """안전 부동소수 변환(실패 시 기본값)."""
    try:
        return float(x)
    except Exception:
        return float(default)


def _nz_len(x: str) -> float:
    """문자열 길이의 정규화(0~1), 비문자열/빈값 안전 처리."""
    if not isinstance(x, (str, bytes)):
        return 0.0
    l = len(x)
    return min(1.0, float(l) / 4000.0)


def _make_features(
    ts: datetime, ctx_meta: Dict[str, Any], minutes_since_last_open: float
) -> List[float]:
    """시간/컨텍스트/오픈 간격 기반 특성 벡터 생성."""
    h = ts.astimezone(KST).hour + ts.astimezone(KST).minute / 60.0
    dow = ts.astimezone(KST).weekday()
    h_s = math.sin(2 * math.pi * h / 24.0)
    h_c = math.cos(2 * math.pi * h / 24.0)
    d_s = math.sin(2 * math.pi * dow / 7.0)
    d_c = math.cos(2 * math.pi * dow / 7.0)
    w = _to_float(ctx_meta.get("web_len", 0.0))
    r = _to_float(ctx_meta.get("rag_len", 0.0))
    m = _to_float(ctx_meta.get("mob_len", 0.0))
    feat = [
        h_s,
        h_c,
        d_s,
        d_c,
        min(1.0, w / 4000.0),
        min(1.0, r / 4000.0),
        min(1.0, m / 4000.0),
        min(720.0, float(minutes_since_last_open)),
    ]
    return feat


async def retrain_once(lookback_days: int = 7) -> Tuple[int, int]:
    """
    최근 lookback_days 동안의 푸시 로그를 수집하여 GBDT를 재학습한다.
    라벨: opened_at 존재 여부(1) vs 없음(0)
    특징: 시간 사인/코사인, ctx_meta(web/rag/mob 길이), 마지막 오픈 후 경과분
    """
    if firestore is None or GradientBoostingClassifier is None or _joblib_dump is None:
        return 0, 0
    try:
        db = firestore.Client()
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        # collection group query
        q = db.collection_group("proactive_push_logs").where("timestamp", ">=", since)
        rows: List[Tuple[List[float], int]] = []
        # per-user last open tracking
        last_open_at: Dict[str, datetime] = {}
        for d in q.stream():
            m = d.to_dict() or {}
            uid = (
                (d.reference.parent.parent.id if d.reference.parent.parent else None)
                or m.get("user_id")
                or ""
            )
            ts = m.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            opened_at = m.get("opened_at")
            label = 1 if isinstance(opened_at, datetime) else 0
            ctx_meta = m.get("ctx_meta") or {}
            # minutes since last open in history for this uid
            prev_open = last_open_at.get(uid)
            if isinstance(opened_at, datetime):
                last_open_at[uid] = opened_at
            mins_since = 120.0
            if isinstance(prev_open, datetime):
                mins_since = max(0.0, (ts - prev_open).total_seconds() / 60.0)
            feat = _make_features(ts, ctx_meta, mins_since)
            rows.append((feat, label))
        n = len(rows)
        if n < 50:
            return n, 0
        X = [f for f, _ in rows]
        y = [int(y) for _, y in rows]
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X, y)
        path = os.getenv(
            "PROACTIVE_RANKER_SKLEARN_PATH",
            "c:/My_Business/models/proactive_ranker.joblib",
        )
        _joblib_dump((clf, _FEATURE_ORDER), path)
        return n, 1
    except Exception:
        return 0, 0


_started = False


async def _daily_loop():
    """매일 03:30 KST에 1회 재학습을 실행하는 루프."""
    while True:
        try:
            now = datetime.now(KST)
            next_run = now.replace(hour=3, minute=30, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run + timedelta(days=1)
            sleep_s = (next_run - now).total_seconds()
            await asyncio.sleep(sleep_s)
            n, ok = await retrain_once()
        except Exception:
            pass
        # 안전 대기
        await asyncio.sleep(60.0)


async def ensure_daily_retrainer() -> None:
    global _started
    if _started:
        return
    _started = True
    asyncio.create_task(_daily_loop())
