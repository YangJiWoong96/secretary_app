# -*- coding: utf-8 -*-
"""
backend.behavior.behavior_extractor - Behavior Slots → EWMA/Scoreboard 갱신기

요구사항:
- 규칙 기반 분류 결과를 PreferenceScoreboard(EWMA)에 반영
- (옵션 B) Milvus behavior_slots 컬렉션에 upsert
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.config import get_settings
from backend.personalization.preference_scoreboard import PreferenceScoreboard


def _norm_key_for_scoreboard(norm_key: str) -> str:
    """
    Scoreboard 키 스키마: 그대로 사용(behavior.* 네임스페이스)
    """
    return norm_key.strip()


async def update_from_slots(
    user_id: str,
    slots: List[Dict[str, Any]],
    intensity: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    분류 결과를 EWMA 점수기에 반영하고, RAG behavior_slots에 항상 upsert한다.

    신규 동작:
        - BEHAVIOR_UPSERT_TO_RAG 플래그 제거
        - Scoreboard와 Milvus를 항상 함께 업데이트
        - 트랜잭션 ID 기반 에러 추적

    Args:
        user_id: 사용자 ID
        slots: 분류된 행동 슬롯 리스트
        intensity: 감정 강도 (0.0~1.0)

    Returns:
        List[Dict]: 업데이트 결과 (Scoreboard 상태 포함)
    """
    import uuid
    import logging

    logger = logging.getLogger("behavior_extractor")

    if not user_id or not slots:
        return []

    settings = get_settings()
    sb = PreferenceScoreboard(settings.REDIS_URL)
    results: List[Dict[str, Any]] = []

    # 트랜잭션 ID 생성 (에러 추적용)
    tx_id = str(uuid.uuid4())[:8]
    logger.info(f"[behavior] tx_id={tx_id} user={user_id} slots={len(slots)}")

    # 메트릭 임포트
    try:
        from backend.utils.metrics import (
            behavior_slots_detected,
            behavior_scoreboard_updates,
            behavior_rag_upserts,
        )
    except Exception:
        behavior_slots_detected = None
        behavior_scoreboard_updates = None
        behavior_rag_upserts = None

    for it in slots:
        try:
            norm_key = _norm_key_for_scoreboard(str(it.get("norm_key") or ""))
            if not norm_key:
                continue

            # 메트릭: 슬롯 감지
            if behavior_slots_detected is not None:
                try:
                    behavior_slots_detected.labels(
                        user_id=user_id, slot_key=norm_key
                    ).set(1)
                except Exception:
                    pass

            # 이벤트 1회(positive) + 강도
            entry = {}
            if sb.available():
                entry = sb.update(
                    user_id=user_id,
                    norm_key=norm_key,
                    events={"positive": 1},
                    intensity=float(intensity or 0.0),
                )
            else:
                logger.warning(f"[behavior] tx_id={tx_id} Scoreboard unavailable")

            result = {
                "norm_key": norm_key,
                "scoreboard": entry or {},
                "slot_key": str(it.get("slot_key") or ""),
                "value": it.get("value"),
                "confidence": float(it.get("confidence") or 0.5),
                "evidence": str(it.get("evidence") or ""),
            }
            results.append(result)

            # 메트릭: Scoreboard 업데이트 성공
            if behavior_scoreboard_updates is not None:
                try:
                    status = entry.get("status", "pending")
                    behavior_scoreboard_updates.labels(
                        user_id=user_id, status=status
                    ).inc()
                except Exception:
                    pass
        except Exception as e:
            logger.error(
                f"[behavior] tx_id={tx_id} Scoreboard update failed: norm_key={it.get('norm_key')} err={e}"
            )
            continue

    # RAG behavior_slots 업서트 (항상 수행)
    if results:
        try:
            from backend.rag.behavior_writer import upsert_slots as _upsert_beh  # type: ignore

            await _upsert_beh(user_id=user_id, items=results)
            logger.info(
                f"[behavior] tx_id={tx_id} RAG upsert success: {len(results)} items"
            )

            # 메트릭: RAG 업서트 성공
            if behavior_rag_upserts is not None:
                try:
                    behavior_rag_upserts.labels(user_id=user_id, status="success").inc()
                except Exception:
                    pass
        except Exception as e:
            logger.error(
                f"[behavior] tx_id={tx_id} RAG upsert failed: user={user_id} items={len(results)} err={e}",
                exc_info=True,
            )
            # 메트릭: RAG 업서트 실패
            if behavior_rag_upserts is not None:
                try:
                    behavior_rag_upserts.labels(user_id=user_id, status="failure").inc()
                except Exception:
                    pass
            # Scoreboard는 이미 업데이트되었으므로, 롤백하지 않음
            # 다음 턴에서 재시도 또는 수동 보정 필요

    return results


__all__ = ["update_from_slots"]
