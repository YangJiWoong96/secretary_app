"""
backend.routing.mctx_store - 라우팅 보조 신호(EMA 컨텍스트 임베딩) 저장/조회

설계 요약:
- 키: routing:mctx:{session_id}
- 값(JSON): {"vec":[float...](L2 norm), "updated_at": float, "turn_id": str}
- TTL: 600초(10분)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import redis

from backend.config import get_settings
from backend.rag.embeddings import embed_query_cached


def _redis() -> redis.Redis:
    s = get_settings()
    return redis.Redis.from_url(s.REDIS_URL, decode_responses=True)


def _key(session_id: str) -> str:
    return f"routing:mctx:{session_id}"


def load_mctx(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = _redis()
        raw = r.get(_key(session_id))
        if not raw:
            return None
        data = json.loads(raw)
        vec = data.get("vec") or []
        if not vec:
            return None
        return {
            "vec": np.array([float(x) for x in vec], dtype=np.float32),
            "updated_at": float(data.get("updated_at") or 0.0),
            "turn_id": str(data.get("turn_id") or ""),
        }
    except Exception:
        return None


def save_mctx(
    session_id: str, vec: np.ndarray, turn_id: str, ttl_sec: int = 600
) -> None:
    try:
        r = _redis()
        v = np.array(vec, dtype=np.float32)
        norm = float(np.linalg.norm(v) or 1.0)
        v = (v / norm).astype(np.float32)
        payload = {
            "vec": [float(x) for x in v.tolist()],
            "updated_at": time.time(),
            "turn_id": str(turn_id or ""),
        }
        r.setex(_key(session_id), ttl_sec, json.dumps(payload))
    except Exception:
        pass


def update_mctx_with_summary(session_id: str, summary_text: str, turn_id: str) -> None:
    """
    요약 텍스트를 임베딩하여 m_ctx를 갱신.
    - 동일 토픽(align≥ρ=0.35)이면 EMA 업데이트(α=0.16), 아니면 교체
    """
    try:
        prev = load_mctx(session_id)
        new_vec = embed_query_cached((summary_text or "").strip())
        if prev and isinstance(prev.get("vec"), np.ndarray):
            v_prev = prev["vec"]
            v_new = np.array(new_vec, dtype=np.float32)
            # 정규화
            v_prev = v_prev / (float(np.linalg.norm(v_prev)) or 1.0)
            v_new = v_new / (float(np.linalg.norm(v_new)) or 1.0)
            # 정합(토픽 유사) 판정
            align = float(np.dot(v_prev, v_new))
            if align >= 0.35:
                alpha = 0.16
                v_mix = (1.0 - alpha) * v_prev + alpha * v_new
                save_mctx(session_id, v_mix, turn_id, ttl_sec=600)
            else:
                save_mctx(session_id, v_new, turn_id, ttl_sec=600)
        else:
            save_mctx(
                session_id, np.array(new_vec, dtype=np.float32), turn_id, ttl_sec=600
            )
    except Exception:
        # 실패해도 본 플로우 영향 없음
        pass
