"""
동적 신뢰도 기반 재랭킹 알고리즘 (trust + recency + similarity)

현재 파이프라인과의 호환을 위해 독립 유틸로 제공. 후보는 다음 스키마를 기대한다:
{
  "id": str,              # evidence id (또는 유니크 키)
  "embedding": list[float],
  "date_end": int,        # YYYYMMDD 정수
}
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List

import numpy as np

from backend.rag.milvus import ensure_collections

logger = logging.getLogger("reranker")


def compute_trust_score(
    evidence_id: str, session_id: str, time_decay_lambda: float = 0.01
) -> float:
    """
    특정 증거에 대한 동적 신뢰도 점수 계산
    trust(d) = Σ(confidence_boost × exp(-λ × age_days))
    """
    try:
        _, log_coll = ensure_collections()
        # 스키마 호환: log 컬렉션에는 'original_evidence_id'나 'confidence_boost' 필드가 없다.
        # 따라서 텍스트 메타([meta] ...)를 파싱해서 필터링/부스트를 추출한다.
        rows = []
        try:
            rows = log_coll.query(
                expr=f"type == 'evidence_feedback' and user_id == '{session_id}'",
                output_fields=["created_at", "text"],
                limit=200,
            )
        except Exception:
            rows = []

        # evidence_id와 매칭되는 행만 선별 (텍스트 메타에서 original_evidence_id=... 파싱)
        import re as _re

        evid_id = str(evidence_id or "")

        def _match_row(r) -> bool:
            try:
                t = str(r.get("text", ""))
                m = _re.search(r"original_evidence_id=([^\s]+)", t)
                return bool(m and m.group(1) == evid_id)
            except Exception:
                return False

        results = [r for r in (rows or []) if _match_row(r)]
        if not results:
            return 0.0
        now_ns = time.time_ns()
        acc = 0.0
        for fb in results:
            # 텍스트 메타에서 confidence_boost 추출
            boost = 0.0
            try:
                t = str(fb.get("text", ""))
                m = _re.search(r"confidence_boost=([+-]?[0-9]*\.?[0-9]+)", t)
                if m:
                    boost = float(m.group(1))
            except Exception:
                boost = 0.0
            created_ns = int(fb.get("created_at", now_ns))
            age_days = max(0, (now_ns - created_ns) / (1e9 * 86400))
            acc += boost * math.exp(-time_decay_lambda * age_days)
        return max(-1.0, min(1.0, acc))
    except Exception as e:
        logger.warning(f"compute_trust_score error: {e}")
        return 0.0


def rerank_with_confidence(
    candidates: List[Dict],
    query_emb: np.ndarray,
    session_id: str,
    alpha: float = 0.7,
    beta: float = 0.1,
    gamma: float = 0.2,
) -> List[Dict]:
    """
    Score = α·sim(q,d) + β·recency(d) + γ·trust(d)
    """
    from backend.utils.logger import log_event

    try:
        scored = []
        for cand in candidates:
            # sim
            cemb = np.array(cand.get("embedding", []) or [], dtype=np.float32)
            if cemb.size == 0:
                sim = 0.0
            else:
                sim = float(
                    np.dot(query_emb, cemb)
                    / ((np.linalg.norm(query_emb) * np.linalg.norm(cemb)) + 1e-9)
                )
            # recency
            try:
                date_end = int(cand.get("date_end", 0))
                if date_end > 0:
                    y, m, d = (
                        date_end // 10000,
                        (date_end % 10000) // 100,
                        date_end % 100,
                    )
                    from datetime import datetime

                    doc_date = datetime(y, m, d)
                    age_days = (datetime.now() - doc_date).days
                    recency = math.exp(-0.01 * max(0, age_days))
                else:
                    recency = 0.5
            except Exception:
                recency = 0.5
            # trust
            evid = str(cand.get("id", ""))
            trust = compute_trust_score(evid, session_id)
            final = alpha * sim + beta * recency + gamma * trust
            scored.append(
                {
                    **cand,
                    "final_score": final,
                    "sim_score": sim,
                    "recency_score": recency,
                    "trust_score": trust,
                }
            )

        scored.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        log_event(
            "reranking_applied",
            {
                "session_id": session_id,
                "candidates_count": len(candidates),
                "top_3_scores": [
                    {
                        "id": s.get("id", "")[:50],
                        "final": round(s.get("final_score", 0.0), 3),
                        "sim": round(s.get("sim_score", 0.0), 3),
                        "recency": round(s.get("recency_score", 0.0), 3),
                        "trust": round(s.get("trust_score", 0.0), 3),
                    }
                    for s in scored[:3]
                ],
            },
        )
        return scored
    except Exception as e:
        logger.warning(f"rerank_with_confidence error: {e}")
        return candidates
