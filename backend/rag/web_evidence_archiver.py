"""
웹 검색 결과 선별 적재 (Milvus upsert)

상태머신: pending → active/drop/expire
사용자 피드백 후에만 web_archived로 승격한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Dict, List

from backend.rag.embeddings import embed_query_openai
from backend.rag.milvus import ensure_collections
from backend.utils.datetime_utils import now_kst, ym, ymd
from backend.utils.logger import safe_log_event

logger = logging.getLogger("web_evidence_archiver")


# 환경변수 플래그
DELAYED_ARCHIVE_ENABLED = os.getenv("DELAYED_ARCHIVE_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
ARCHIVE_EXPIRY_TURNS = int(os.getenv("ARCHIVE_EXPIRY_TURNS", "3"))


async def enqueue_pending_evidence(
    session_id: str,
    turn_id: str,
    web_ctx: str,
    user_context: str,
    confidence: float,
) -> None:
    """
    웹 증거를 pending 상태로 Redis 큐에 저장 (즉시 Milvus 적재 금지)

    키: pending_evidence:{session}:{turn}
    TTL: 600초 (10분)
    """
    # 플래그 OFF: 기존 즉시 적재 경로
    if not DELAYED_ARCHIVE_ENABLED:
        await _archive_immediately(session_id, web_ctx, user_context, confidence)
        return

    import redis  # 지역 임포트로 런타임 의존 최소화

    from backend.config import get_settings
    from backend.utils.logger import log_event

    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    pending: Dict[str, object] = {
        "turn_id": turn_id,
        "web_ctx": web_ctx,
        "user_context": user_context,
        "confidence": float(confidence or 0.0),
        "created_at": int(time.time()),
        "expiry_turns": ARCHIVE_EXPIRY_TURNS,
    }

    key = f"pending_evidence:{session_id}:{turn_id}"

    try:
        redis_client.setex(key, 600, json.dumps(pending, ensure_ascii=False))
        logger.info(f"[web_evidence_archiver] Enqueued pending evidence: {key}")
        log_event(
            "archive.pending",
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "expiry_turns": ARCHIVE_EXPIRY_TURNS,
            },
        )
    except Exception as e:
        logger.error(f"[web_evidence_archiver] Enqueue error: {e}")


async def evaluate_with_feedback(
    session_id: str,
    current_turn_id: str,
    feedbacks: List,
) -> None:
    """
    사용자 피드백 기반 pending 증거 평가 및 승격/폐기/만료 카운트 다운

    Args:
        session_id: 세션 ID
        current_turn_id: 현재 턴 ID
        feedbacks: 피드백 리스트 (backend.rag.feedback_detector 결과)
    """
    import redis

    from backend.config import get_settings
    from backend.utils.logger import log_event

    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # 세션 내 모든 pending 키 조회
    pattern = f"pending_evidence:{session_id}:*"
    try:
        keys = redis_client.keys(pattern)
    except Exception as e:
        logger.error(f"[web_evidence_archiver] Redis keys error: {e}")
        return

    for key in keys:
        try:
            pending_raw = redis_client.get(key)
            if not pending_raw:
                continue

            pending: Dict = json.loads(pending_raw)
            prev_turn_id = str(pending.get("turn_id", ""))

            # 긍정 피드백 → 승격
            has_positive = any(
                getattr(fb, "feedback_type", None) == "positive"
                and getattr(fb, "evidence_type", None) == "web"
                for fb in (feedbacks or [])
            )
            if has_positive:
                await _promote_to_active(session_id, pending)
                redis_client.delete(key)
                log_event(
                    "archive.promoted",
                    {"session_id": session_id, "turn_id": prev_turn_id},
                )
                continue

            # 부정 피드백 → 폐기
            has_negative = any(
                getattr(fb, "feedback_type", None) == "negative"
                and getattr(fb, "evidence_type", None) == "web"
                for fb in (feedbacks or [])
            )
            if has_negative:
                try:
                    await drop_evidence(session_id, prev_turn_id)
                except Exception:
                    redis_client.delete(key)
                log_event(
                    "archive.dropped",
                    {"session_id": session_id, "turn_id": prev_turn_id},
                )
                continue

            # 무반응 → 만료 카운트 감소
            expiry_turns = int(pending.get("expiry_turns", ARCHIVE_EXPIRY_TURNS)) - 1
            if expiry_turns <= 0:
                redis_client.delete(key)
                log_event(
                    "archive.expired",
                    {"session_id": session_id, "turn_id": prev_turn_id},
                )
            else:
                pending["expiry_turns"] = expiry_turns
                redis_client.setex(key, 600, json.dumps(pending, ensure_ascii=False))
        except Exception as e:
            logger.error(f"[web_evidence_archiver] Evaluate error: {e}")


async def _promote_to_active(session_id: str, pending: Dict) -> None:
    """pending 증거를 web_archived로 Milvus에 승격"""
    await _archive_immediately(
        session_id,
        str(pending.get("web_ctx", "")),
        str(pending.get("user_context", "")),
        float(pending.get("confidence", 0.0)),
    )


async def drop_evidence(session_id: str, turn_id: str) -> None:
    """pending 증거를 삭제(drop)한다."""
    import redis

    from backend.config import get_settings

    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    key = f"pending_evidence:{session_id}:{turn_id}"
    try:
        redis_client.delete(key)
    except Exception as e:
        logger.error(f"[web_evidence_archiver] Drop error: {e}")


async def archive_web_evidence(
    session_id: str,
    web_ctx: str,
    user_context: str,
    confidence: float,
):
    """
    웹 검색 결과를 즉시 적재하는 기존 로직(회귀/승격 시 사용)
    """
    await _archive_immediately(session_id, web_ctx, user_context, confidence)


async def _archive_immediately(
    session_id: str,
    web_ctx: str,
    user_context: str,
    confidence: float,
) -> None:
    """
    즉시 적재 (기존 로직)

    type="web_archived"로 Milvus에 저장한다.
    """
    from backend.utils.logger import log_event

    try:
        _, log_coll = ensure_collections()

        blocks = (web_ctx or "").split("\n\n")
        archived_count = 0

        for block in blocks:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if len(lines) < 2:
                continue

            title = lines[0][:200]
            description = " ".join(lines[1:-1]) if len(lines) > 2 else lines[1]
            url = lines[-1] if lines[-1].startswith("http") else ""

            if not url:
                url_hash = hashlib.sha256((title + description).encode()).hexdigest()[
                    :16
                ]
                url_clean = ""
            else:
                import re

                url_clean = re.sub(r"[?&]utm_[^&]*", "", url)
                url_hash = hashlib.sha256(url_clean.encode()).hexdigest()[:16]

            enriched_text = (
                f"[웹 문서]\n제목: {title}\n출처: {url or '(URL 없음)'}\n내용: {description[:300]}\n\n"
                f"[수집 맥락]\n{(user_context or '')[:200]}"
            )

            emb = embed_query_openai(enriched_text)
            now = now_kst()
            doc_id = f"{session_id}:web_archived:{url_hash}:{int(time.time())}"

            # Milvus 스키마(Log 컬렉션)에 없는 필드는 text에 보존
            safe_text = (
                enriched_text
                + (f"\n[meta] url={url_clean}" if url_clean else "")
                + f"\n[meta] confidence={float(confidence or 0.0):.2f}"
            )
            log_coll.upsert(
                [
                    {
                        "id": doc_id,
                        "embedding": emb,
                        "text": safe_text,
                        "user_id": session_id,
                        "type": "web_archived",
                        "created_at": int(time.time_ns()),
                        "date_start": ymd(now),
                        "date_end": ymd(now),
                        "date_ym": ym(now),
                    }
                ]
            )

            archived_count += 1
            log_event(
                "web_evidence_archived_block",
                {
                    "doc_id": doc_id,
                    "url": url_clean or "(no URL)",
                    "title": title[:100],
                    "confidence": float(confidence or 0.0),
                },
            )

        logger.info(
            f"[web_evidence_archiver] Archived {archived_count}/{len(blocks)} web blocks"
        )
        log_event(
            "web_evidence_archived_batch",
            {
                "session_id": session_id,
                "archived_count": archived_count,
                "total_blocks": len(blocks),
                "confidence": float(confidence or 0.0),
            },
        )
        # 구조화 로깅(rag.log_upsert) - 집계
        try:
            safe_log_event(
                "rag.log_upsert",
                {
                    "user_id": session_id,
                    "collection": "logs",
                    "chunk_count": archived_count,
                    "vector_dim": 0,
                    "reason": "web_archived",
                },
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"[web_evidence_archiver] Error: {e}")
        try:
            from backend.utils.logger import log_event as _le

            _le("web_archiver_error", {"error": repr(e)}, level=logging.ERROR)
        except Exception:
            pass
