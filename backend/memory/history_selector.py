"""
backend.memory.history_selector - 히스토리 선별

유사도 기반 히스토리 메시지 선별을 담당합니다.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Dict, List

import numpy as np

from backend.config import get_settings
from backend.utils.tracing import traceable

logger = logging.getLogger("history_selector")

# Recency Bias 동작 플래그/파라미터 (환경변수 기반)
# - RECENCY_BIAS_ENABLED: 기능 온/오프 (기본: true)
# - RECENCY_BIAS_LAMBDA: 시간 감쇠 람다 (기본: 0.05)
# - HISTORY_RECENT_TURNS: 최근 N턴 강제 포함 (기본: 6 → 12개 메시지)
_settings = get_settings()
RECENCY_BIAS_ENABLED = bool(getattr(_settings, "RECENCY_BIAS_ENABLED", True))
RECENCY_BIAS_LAMBDA = float(getattr(_settings, "RECENCY_BIAS_LAMBDA", 0.05))
HISTORY_RECENT_TURNS = int(getattr(_settings, "HISTORY_RECENT_TURNS", 6))


@traceable(name="Memory: prepare_history", run_type="chain", tags=["memory"])
async def prepare_history(session_id: str, user_input: str) -> List[Dict[str, str]]:
    """
    히스토리 로드 및 Recency-Biased Sliding Window 선별.

    알고리즘:
    1. 최근 N턴(기본 6턴=12메시지) 무조건 포함 (시간적 연속성 보장)
    2. [SUMMARIZED@...] 블록 제외
    3. 과거 턴(13~40번째 메시지 범위) 중 시간감쇠 유사도 Top-K(기본 4) 선별
    4. 시간순 정렬 유지(원래 순서 복원)

    플래그 OFF 시(RECENCY_BIAS_ENABLED=false), 기존 레거시 로직으로 폴백:
    - 최근 12 메시지 중 유사도 Top-8 + 최근 4개 강제 포함

    Args:
        session_id: 세션 ID
        user_input: 사용자 입력

    Returns:
        List[Dict]: 선별된 히스토리 메시지 리스트
            [{"role": "user"|"assistant", "content": str}, ...]
    """
    from backend.memory import get_short_term_memory
    from backend.rag.embeddings import embed_documents, embed_query_openai
    from backend.utils.logger import log_event

    hist_msgs_chat: List[Dict[str, str]] = []
    try:
        stm = get_short_term_memory(session_id)
        raw_msgs = list(getattr(stm, "chat_memory", {}).messages or [])

        # [SUMMARIZED@...] 블록 제외
        def _not_summarized(m):
            try:
                txt = (getattr(m, "content", "") or "").strip()
                return not txt.startswith("[SUMMARIZED@")
            except Exception:
                return True

        filtered = [
            m
            for m in raw_msgs
            if getattr(m, "type", "") in ("human", "ai") and _not_summarized(m)
        ]

        # 연속 user 방지
        if filtered and getattr(filtered[-1], "type", "") == "human":
            filtered = filtered[:-1]

        # Recency Bias 비활성화 → 레거시 로직으로 폴백
        if not RECENCY_BIAS_ENABLED:
            tail = filtered[-12:]
            lines = [
                (getattr(m, "type", ""), (getattr(m, "content", "") or "").strip())
                for m in tail
            ]
            lines = [(r, t) for (r, t) in lines if t]

            try:
                qv = await asyncio.to_thread(embed_query_openai, user_input)
                texts = [t for _, t in lines]
                tv_list = await asyncio.to_thread(embed_documents, texts)

                scored: list[tuple[float, tuple[str, str]]] = []
                for (role, text), text_vec in zip(lines, tv_list):
                    if isinstance(text_vec, Exception):
                        continue
                    denom = float(
                        (np.linalg.norm(qv) * np.linalg.norm(text_vec)) or 1.0
                    )
                    sim = float(np.dot(qv, text_vec) / denom)
                    scored.append((sim, (role, text)))

                scored.sort(key=lambda x: -x[0])

                recent_4 = lines[-4:] if len(lines) >= 4 else lines
                top_8 = [role_text for _, role_text in scored[:8]]

                for role_text in recent_4:
                    if role_text not in top_8:
                        top_8.append(role_text)

                picked = top_8[:12]
            except Exception:
                picked = lines[-8:]

            for role, text in picked:
                if role == "human":
                    hist_msgs_chat.append({"role": "user", "content": text})
                elif role == "ai":
                    hist_msgs_chat.append({"role": "assistant", "content": text})

            return hist_msgs_chat

        # ===== Recency-Biased Sliding Window (신규 로직) =====
        recent_n_msgs = max(0, HISTORY_RECENT_TURNS) * 2  # N턴 = 2N 메시지

        # Phase 1: 최근 N개 메시지 강제 포함 (원문 유지)
        recent_msgs = (
            filtered[-recent_n_msgs:] if len(filtered) >= recent_n_msgs else filtered
        )

        # Phase 2: 과거 메시지 창 (최근 이후 최대 28개 메시지 = 14턴)에서 시간감쇠 유사도 선별
        older_msgs = (
            filtered[-(recent_n_msgs + 28) : -recent_n_msgs]
            if len(filtered) > recent_n_msgs
            else []
        )

        if not older_msgs:
            # 과거 창이 비어있으면 최근만 반환
            for msg in recent_msgs:
                role = getattr(msg, "type", "")
                text = (getattr(msg, "content", "") or "").strip()
                if not text:
                    continue
                if role == "human":
                    hist_msgs_chat.append({"role": "user", "content": text})
                elif role == "ai":
                    hist_msgs_chat.append({"role": "assistant", "content": text})

            try:
                log_event(
                    "history_recency_bias",
                    {
                        "recent_count": len(recent_msgs),
                        "older_count": 0,
                        "total_count": len(hist_msgs_chat),
                    },
                )
            except Exception:
                pass

            return hist_msgs_chat

        # 시간 감쇠 적용 유사도 계산: final_score = sim * exp(-λ * age_turns)
        try:
            query_vec = await asyncio.to_thread(embed_query_openai, user_input)
            # 원본 인덱스를 보존하면서 공백 메시지는 제외하여 임베딩 실행
            older_indexed_texts: list[tuple[int, object, str]] = []
            for i, msg in enumerate(older_msgs):
                text = (getattr(msg, "content", "") or "").strip()
                if text:
                    older_indexed_texts.append((i, msg, text))

            texts_for_embed = [t for (_, _, t) in older_indexed_texts]
            doc_vecs = await asyncio.to_thread(embed_documents, texts_for_embed)

            scored_older: list[tuple[float, object, int]] = []
            for (orig_idx, msg, _text), doc_vec in zip(older_indexed_texts, doc_vecs):
                if isinstance(doc_vec, Exception):
                    continue

                denom = float(
                    (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)) or 1.0
                )
                sim = float(np.dot(query_vec, doc_vec) / denom)

                # 최근에 가까울수록 age_turns가 작도록 정의 (원본 인덱스 기반)
                age_turns = len(older_msgs) - orig_idx
                decay = math.exp(-RECENCY_BIAS_LAMBDA * age_turns)
                final_score = sim * decay

                scored_older.append((final_score, msg, orig_idx))

            # Top-K=4 선별 후, 원래 순서로 복원
            scored_older.sort(key=lambda x: -x[0])
            selected_older_with_idx = scored_older[:4]
            selected_older_with_idx.sort(key=lambda x: x[2])
            selected_older = [msg for _, msg, _ in selected_older_with_idx]
        except Exception as e:
            logger.warning(
                f"prepare_history time_decay failed: {e}, fallback to recent only"
            )
            selected_older = []

        # Phase 3: 결합 (시간순 정렬 유지)
        combined_msgs = selected_older + recent_msgs

        for msg in combined_msgs:
            role = getattr(msg, "type", "")
            text = (getattr(msg, "content", "") or "").strip()
            if not text:
                continue
            if role == "human":
                hist_msgs_chat.append({"role": "user", "content": text})
            elif role == "ai":
                hist_msgs_chat.append({"role": "assistant", "content": text})

        # 텔레메트리 로깅 (성공 경로)
        try:
            log_event(
                "history_recency_bias",
                {
                    "recent_count": len(recent_msgs),
                    "older_count": len(selected_older),
                    "total_count": len(hist_msgs_chat),
                    "lambda": RECENCY_BIAS_LAMBDA,
                    "recent_turns": HISTORY_RECENT_TURNS,
                },
            )
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"prepare_history failed: {e}")
        hist_msgs_chat = []

    return hist_msgs_chat
