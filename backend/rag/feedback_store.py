"""
증거 피드백 저장/업서트 및 로깅
"""

from __future__ import annotations

import logging
import time

from backend.rag.embeddings import embed_query_openai
from backend.rag.evidence_feedback_schema import EvidenceFeedback
from backend.rag.milvus import ensure_collections
from backend.utils.datetime_utils import now_kst, ym, ymd
from backend.utils.logger import safe_log_event

logger = logging.getLogger("feedback_store")


async def store_feedback_enhanced_evidence(
    feedback: EvidenceFeedback, original_evidence: str
):
    """
    피드백이 포함된 증거를 RAG DB에 저장하고 동적 신뢰도 점수 갱신을 위한 로그를 남긴다.
    type="evidence_feedback" 문서를 log_coll에 업서트한다.
    """
    from backend.utils.logger import log_event

    try:
        _, log_coll = ensure_collections()

        enriched_text = (
            f"[원본 증거 ({feedback.evidence_type})]\n{(original_evidence or '')[:500]}\n\n"
            f"[사용자 피드백 ({feedback.feedback_type})]\n{feedback.user_comment}\n\n"
            f"[맥락]\n{feedback.ai_response[:300]}"
        )

        emb = embed_query_openai(enriched_text)
        now = now_kst()
        doc_id = (
            f"{feedback.user_id}:feedback:{feedback.evidence_id}:{feedback.turn_id}"
        )

        # Milvus log 스키마 비호환 필드(text에 메타 보존)
        safe_text = (
            enriched_text
            + f"\n[meta] feedback_type={feedback.feedback_type}"
            + f"\n[meta] evidence_type={feedback.evidence_type}"
            + f"\n[meta] original_evidence_id={feedback.evidence_id}"
            + f"\n[meta] confidence_boost={float(feedback.confidence_adjustment or 0.0):.2f}"
        )
        log_coll.upsert(
            [
                {
                    "id": doc_id,
                    "embedding": emb,
                    "text": safe_text,
                    "user_id": feedback.user_id,
                    "type": "evidence_feedback",
                    "created_at": int(time.time_ns()),
                    "date_start": ymd(now),
                    "date_end": ymd(now),
                    "date_ym": ym(now),
                }
            ]
        )

        logger.info(
            f"[feedback_store] Stored feedback: type={feedback.feedback_type}, evidence={feedback.evidence_type}, id={feedback.evidence_id}"
        )
        log_event(
            "feedback_stored_for_reranking",
            {
                "doc_id": doc_id,
                "evidence_id": feedback.evidence_id,
                "evidence_type": feedback.evidence_type,
                "feedback_type": feedback.feedback_type,
                "confidence_boost": float(feedback.confidence_adjustment or 0.0),
            },
        )
        try:
            safe_log_event(
                "rag.log_upsert",
                {
                    "user_id": feedback.user_id,
                    "collection": "logs",
                    "chunk_count": 1,
                    "vector_dim": len(emb or []),
                    "reason": "evidence_feedback",
                },
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"[feedback_store] Error: {e}")
        try:
            from backend.utils.logger import log_event as _le

            _le("feedback_store_error", {"error": repr(e)}, level=logging.ERROR)
        except Exception:
            pass
