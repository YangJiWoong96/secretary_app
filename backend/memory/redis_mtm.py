"""
backend.memory.redis_mtm - MTM(중기 메모리) Redis 모듈 (Cross-Session)

요약 스냅샷을 Redis에 보관하여 단기(STM)와 장기(LTM) 사이의 중간 계층을 형성한다.

구성 (Cross-Session):
- 인덱스: ZSET mtm:{user_id} (score=timestamp, member=item_id)
  → 사용자별 단일 키로 모든 세션의 MTM을 통합 관리
- 항목:  HASH mtm:item:{item_id}
  - { id, user_id, session_id, mtm_summary, routing_summary, emb_full, emb_routing, ts, access_count, last_accessed }

제약:
- TTL: MTM_TTL_DAYS (기본 7일)
- 최대 항목수: MTM_MAX_SUMMARIES (초과 시 Hybrid 정리: 시간 기반 + LRU)
- 주제 중복 제거: 임베딩 유사도 기반 (MTM_TOPIC_SIMILARITY_THRESHOLD, 기본 0.85)
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import redis

from backend.config import get_settings
from backend.memory.redis_index import register_key
from backend.rag.embeddings import (
    embed_query_cached,
    embed_query_gemma,
    embed_query_openai,
)

logger = logging.getLogger("redis_mtm")


class RedisMTM:
    """MTM 요약 보관소 (Cross-Session 지원)"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._r = redis.Redis.from_url(self.settings.REDIS_URL, decode_responses=True)

        # 주제 중복 제거 임계값 (임베딩 유사도)
        self.topic_similarity_threshold = float(
            getattr(self.settings, "MTM_TOPIC_SIMILARITY_THRESHOLD", 0.85)
        )

    # -----------------------------
    # 키 유틸
    # -----------------------------
    def _zkey(self, user_id: str) -> str:
        """
        Cross-Session MTM 키 스키마

        신규: mtm:{user_id}
        레거시 호환: mtm:{user_id}:{session_id}는 마이그레이션 후 완전 삭제
        """
        return f"mtm:{user_id}"

    def _hkey(self, item_id: str) -> str:
        return f"mtm:item:{item_id}"

    # -----------------------------
    # 기본 연산
    # -----------------------------
    def add_summary(
        self,
        user_id: str,
        session_id: str,
        mtm_summary: str,
        routing_summary: str,
    ) -> str:
        """
        MTM 요약 추가 (Cross-Session, 세션 정보는 메타데이터)

        Args:
            user_id: 사용자 ID (키 스코핑)
            session_id: 세션 ID (메타데이터로만 저장)
            mtm_summary: 상세 요약 (200~350 토큰)
            routing_summary: 라우팅용 초경량 요약 (20토큰)

        Returns:
            item_id: 생성된 요약 ID
        """
        now = int(time.time())
        item_id = str(uuid.uuid4())

        # 임베딩 생성 (OpenAI 우선, Gemma 폴백)
        try:
            routing_text = (
                (routing_summary or "")
                .replace("[ROUTING_ONLY]", "")
                .replace("[/ROUTING_ONLY]", "")
            )

            v_full = embed_query_openai(mtm_summary or "")
            v_routing = embed_query_openai(routing_text or "")

            # L2 정규화
            def _l2norm(v: np.ndarray) -> np.ndarray:
                n = float(np.linalg.norm(v) or 1.0)
                return v / n

            v_full = _l2norm(v_full)
            v_routing = _l2norm(v_routing)

            emb_full_l = [float(x) for x in v_full]
            emb_routing_l = [float(x) for x in v_routing]
        except Exception as e:
            logger.warning(f"[mtm] OpenAI embedding failed, using Gemma: {e}")
            try:
                v_full = embed_query_gemma(mtm_summary or "")
                v_routing = embed_query_gemma(
                    (routing_summary or "")
                    .replace("[ROUTING_ONLY]", "")
                    .replace("[/ROUTING_ONLY]", "")
                )
                emb_full_l = [float(x) for x in v_full]
                emb_routing_l = [float(x) for x in v_routing]
            except Exception as e2:
                logger.error(f"[mtm] All embeddings failed: {e2}")
                emb_full_l = []
                emb_routing_l = []

        # HASH 페이로드 (session_id를 명시적으로 포함)
        payload = {
            "id": item_id,
            "user_id": user_id,  # 명시적 user_id 필드 추가
            "session_id": session_id,  # 메타데이터로 보존
            "mtm_summary": mtm_summary,
            "routing_summary": routing_summary,
            # 임베딩 (신규/구 필드 모두 저장)
            "embedding_full": json.dumps(emb_full_l, ensure_ascii=False),
            "embedding_routing": json.dumps(emb_routing_l, ensure_ascii=False),
            "emb_full": json.dumps(emb_full_l, ensure_ascii=False),  # 하위 호환
            "emb_routing": json.dumps(emb_routing_l, ensure_ascii=False),
            "ts": str(now),
            # LRU 추적 필드 추가
            "access_count": "0",
            "last_accessed": str(now),
        }

        hkey = self._hkey(item_id)
        zkey = self._zkey(user_id)  # Cross-Session 키

        # 인덱스 등록 (user_id 기준)
        try:
            register_key(user_id, zkey)
        except Exception as e:
            logger.warning(f"[mtm] Index registration failed: {e}")

        # Redis 저장
        p = self._r.pipeline()
        p.hset(hkey, mapping=payload)
        p.zadd(zkey, {item_id: now})

        # TTL 설정 (HASH만, ZSET은 정리 로직에서 관리)
        ttl_sec = int(self.settings.MTM_TTL_DAYS) * 24 * 3600
        p.expire(hkey, ttl_sec)
        p.execute()

        # Hybrid 정리 (30일 하드리밋 + LRU)
        try:
            self._hybrid_cleanup(user_id)
        except Exception as e:
            logger.warning(f"[mtm] Cleanup error: {e}")

        return item_id

    def get_latest(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        최신 MTM 요약 조회 (라우팅용)

        Args:
            user_id: 사용자 ID
            session_id: 특정 세션 필터링 (선택적)

        Returns:
            최신 MTM 요약 (없으면 None)

        동작:
            - session_id 제공 시: 해당 세션의 최신 요약
            - session_id 미제공 시: 전체 최신 요약 (Cross-Session)
        """
        zkey = self._zkey(user_id)
        try:
            # 전체 최신 조회
            last_id = self._r.zrevrange(zkey, 0, 0)
            if not last_id:
                return None

            item = self._r.hgetall(self._hkey(last_id[0]))

            # 세션 필터링 (선택적)
            if session_id and item.get("session_id") != session_id:
                # 해당 세션의 최신 요약 찾기
                all_ids = self._r.zrevrange(zkey, 0, 100)
                for iid in all_ids:
                    it = self._r.hgetall(self._hkey(iid))
                    if it.get("session_id") == session_id:
                        # 접근 횟수 증가 (LRU 추적)
                        self._update_access(iid)
                        return it
                return None

            # 접근 횟수 증가
            self._update_access(last_id[0])
            return item
        except Exception as e:
            logger.warning(f"[mtm] get_latest error: {e}")
            return None

    def get_relevant_summaries(
        self, user_id: str, session_id: str, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Cross-Session MTM 검색 (Multi-Signal 필터링 + 임베딩 기반 중복 제거)

        현업 검증된 방법 (Netflix/Spotify 방식):
        1. 시간 감쇠 (Temporal Decay) - 신선도 보장
        2. 임베딩 유사도 (Semantic Similarity) - 관련성
        3. 세션 다양성 (Session Diversity) - Cross-Session 맥락 확장
        4. 임베딩 기반 주제 중복 제거 (Topic Deduplication) - 노이즈 감소

        Args:
            user_id: 사용자 ID
            session_id: 현재 세션 ID (다양성 계산용)
            query: 검색 쿼리
            top_k: 반환 개수

        Returns:
            관련 MTM 요약 리스트 (점수 순)
        """
        zkey = self._zkey(user_id)

        # 최근 100개 후보 조회
        ids = self._r.zrevrange(zkey, 0, 100)
        if not ids:
            return []

        items = [self._r.hgetall(self._hkey(i)) for i in ids]

        # 쿼리 임베딩 (OpenAI 우선, Gemma 폴백)
        try:
            qv = embed_query_openai(query or "")
            qv = np.array(qv, dtype=np.float32)
            qn = float(np.linalg.norm(qv) or 1.0)
            qv = qv / qn  # 정규화
        except Exception:
            try:
                qv = embed_query_gemma(query or "")
                qv = np.array(qv, dtype=np.float32)
                qn = float(np.linalg.norm(qv) or 1.0)
                qv = qv / qn
            except Exception as e:
                logger.error(f"[mtm] Query embedding failed: {e}")
                return []

        # Multi-Signal Scoring
        now = time.time()
        scored = []
        selected_embeddings = []  # 임베딩 기반 중복 제거용

        for it in items:
            try:
                # 1) 시간 감쇠 (7일 반감기)
                ts = int(it.get("ts", 0))
                age_days = (now - ts) / 86400
                time_decay = math.exp(-0.1 * age_days)  # 7일 후 약 50%

                # 2) 임베딩 유사도
                emb_txt = it.get("embedding_full") or it.get("emb_full", "[]")
                ev = np.array(json.loads(emb_txt), dtype=np.float32)
                ev_norm = float(np.linalg.norm(ev) or 1.0)
                sim = float((ev @ qv) / (ev_norm * qn or 1.0))

                # 3) 세션 다양성 보너스 (현재 세션이 아닌 경우)
                sess_id = it.get("session_id", "")
                diversity_bonus = 1.2 if sess_id != session_id else 1.0

                # 4) 임베딩 기반 주제 중복 제거
                if self._is_topic_duplicate(ev, selected_embeddings):
                    continue  # 이미 유사한 주제가 선택됨

                # 최종 점수 = 유사도 × 시간감쇠 × 다양성
                final_score = sim * time_decay * diversity_bonus

                scored.append((final_score, it, ev))
            except Exception as e:
                logger.warning(f"[mtm] Scoring item failed: {e}")
                continue

        # 점수 순 정렬
        scored.sort(key=lambda x: -x[0])

        # Top-K 선택 및 임베딩 추적
        results = []
        for score, item, emb in scored[:top_k]:
            results.append(item)
            selected_embeddings.append(emb)

            # 접근 횟수 증가 (LRU 추적)
            try:
                self._update_access(item["id"])
            except Exception:
                pass

        return results

    def _is_topic_duplicate(
        self, candidate_emb: np.ndarray, selected_embeddings: List[np.ndarray]
    ) -> bool:
        """
        임베딩 기반 주제 중복 검사 (현업 검증 방법)

        Netflix/Spotify 방식:
        - 이미 선택된 임베딩과의 코사인 유사도 계산
        - 임계값(settings.MTM_TOPIC_SIMILARITY_THRESHOLD, 기본 0.85) 이상이면 중복으로 판단

        Args:
            candidate_emb: 후보 임베딩
            selected_embeddings: 이미 선택된 임베딩 리스트

        Returns:
            True if 중복, False otherwise

        장점:
        - "첫 5단어" 같은 휴리스틱 불필요
        - 의미적 유사도 기반 (문장 표현이 달라도 주제가 같으면 감지)
        - 계산 비용 낮음 (이미 저장된 임베딩 재사용)
        """
        if not selected_embeddings:
            return False

        try:
            # 임계값 로드
            threshold = self.topic_similarity_threshold

            # 정규화
            c_norm = float(np.linalg.norm(candidate_emb) or 1.0)
            c_normalized = candidate_emb / c_norm

            # 이미 선택된 임베딩들과 유사도 계산
            for sel_emb in selected_embeddings:
                sel_norm = float(np.linalg.norm(sel_emb) or 1.0)
                sel_normalized = sel_emb / sel_norm

                # 코사인 유사도
                similarity = float(np.dot(c_normalized, sel_normalized))

                # 임계값 이상이면 중복
                if similarity >= threshold:
                    return True

            return False
        except Exception as e:
            logger.warning(f"[mtm] Topic duplicate check failed: {e}")
            return False  # 실패 시 보수적으로 중복 아닌 것으로 처리

    def _update_access(self, item_id: str) -> None:
        """
        접근 횟수 및 시각 업데이트 (LRU 추적)
        """
        try:
            hkey = self._hkey(item_id)
            p = self._r.pipeline()
            p.hincrby(hkey, "access_count", 1)
            p.hset(hkey, "last_accessed", str(int(time.time())))
            p.execute()
        except Exception as e:
            logger.warning(f"[mtm] Access update failed: {e}")

    def _hybrid_cleanup(self, user_id: str) -> None:
        """
        Hybrid 정리 전략:
        1. 30일 이상 → 무조건 삭제 (하드 리밋)
        2. 7일 이상 + 접근 0회 → 삭제 (비활성)
        3. 남은 개수가 여전히 초과 → LRU 기반 삭제

        현업 검증된 방법 (Twitter/LinkedIn):
        - 시간 기반 정리 + 사용 빈도 기반 정리 조합
        """
        zkey = self._zkey(user_id)
        now = int(time.time())

        # 모든 아이템 조회
        all_ids = self._r.zrange(zkey, 0, -1, withscores=True)
        to_delete = []

        for iid, created_ts in all_ids:
            age_days = (now - created_ts) / 86400
            hkey = self._hkey(iid)

            try:
                access_count = int(self._r.hget(hkey, "access_count") or 0)
            except Exception:
                access_count = 0

            # 규칙 1: 30일 이상 → 무조건 삭제
            if age_days > 30:
                to_delete.append(iid)
                continue

            # 규칙 2: 7일 이상 + 접근 0회 → 삭제
            if age_days > 7 and access_count == 0:
                to_delete.append(iid)

        # 삭제 실행
        if to_delete:
            p = self._r.pipeline()
            for iid in to_delete:
                p.zrem(zkey, iid)
                p.delete(self._hkey(iid))
            p.execute()
            logger.info(f"[mtm] Cleaned {len(to_delete)} items (time-based)")

        # 남은 개수 확인
        max_n = int(self.settings.MTM_MAX_SUMMARIES)
        cnt = int(self._r.zcard(zkey))

        if cnt <= max_n:
            return

        # 규칙 3: LRU 기반 정리 (접근 횟수가 적은 순)
        remaining_ids = self._r.zrange(zkey, 0, -1)
        access_scores = []
        for iid in remaining_ids:
            hkey = self._hkey(iid)
            try:
                access_count = int(self._r.hget(hkey, "access_count") or 0)
                access_scores.append((access_count, iid))
            except Exception:
                access_scores.append((0, iid))

        # 접근 횟수가 적은 순 정렬
        access_scores.sort()

        # 초과분만큼 삭제
        to_delete_lru = cnt - max_n
        p = self._r.pipeline()
        for _, iid in access_scores[:to_delete_lru]:
            p.zrem(zkey, iid)
            p.delete(self._hkey(iid))
        p.execute()
        logger.info(f"[mtm] Cleaned {to_delete_lru} items (LRU-based)")
