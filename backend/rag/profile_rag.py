# C:\My_Business\backend\rag\profile_rag.py
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from backend.rag.retrieval_utils import (
    age_days,
    boost_by_interests,
    calculate_priority_multiplier,
    effective_ts_ns,
    final_score,
    tier_decay,
)

if TYPE_CHECKING:  # 타입 검사 시에만 참조하여 순환/런타임 의존성 회피
    from backend.rag.profile_schema import ProfileTier


class ProfileRAG:
    """
    RAG 기반 동적 프로필 시스템

    - explicit 우선 (가중치 2.0)
    - inferred 후순위
    - norm_key 중복 제거
    - MMR 다양성 적용
    - Bot 프로필(guard/hints) 2계층 조회
    """

    def __init__(self):
        from backend.config import get_settings
        from backend.rag.embeddings import embed_query_openai
        from backend.rag.milvus import ensure_profile_collection
        from backend.rag.retrieval_utils import (
            age_days,
            boost_by_interests,
            calculate_priority_multiplier,
            effective_ts_ns,
            final_score,
            tier_decay,
        )
        from backend.utils.logger import log_event

        self.settings = get_settings()
        self.profile_coll = ensure_profile_collection()
        self.embed_query = embed_query_openai
        self._log_event = log_event

        # 세션5 스코어 조합 파라미터 캐시
        self._alpha = float(self.settings.SCORE_ALPHA)
        self._beta = float(self.settings.SCORE_BETA)
        self._gamma = float(self.settings.SCORE_GAMMA)
        self._delta = float(self.settings.SCORE_DELTA)
        self._priority_cap = float(self.settings.PRIORITY_CAP)

        # Redis 클라이언트 (봇 가드 캐시용) — 선택적 의존성
        try:
            import redis

            self.redis = redis.Redis.from_url(
                self.settings.REDIS_URL, decode_responses=True
            )
        except Exception:
            self.redis = None

    async def query_by_tier(
        self,
        user_id: str,
        tier: "ProfileTier",
        user_input: str = "",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        계층별 프로필 조회

        - GUARD: Redis 영구 캐시 (명시적 갱신만), Milvus 정합 쿼리
        - CORE: Redis 7일 캐시, Milvus 정합 쿼리
        - DYNAMIC: 캐시 없음, 임베딩 벡터 검색
        """

        from backend.rag.profile_schema import TIER_TTL, ProfileTier

        if tier == ProfileTier.GUARD:
            cache_key = f"profile:guard:{user_id}"
            if self.redis is not None:
                try:
                    cached = self.redis.get(cache_key)
                    if cached:
                        try:
                            self._log_event("profile.guard_hit", {"cached": True})
                        except Exception:
                            pass
                        return json.loads(cached)
                except Exception:
                    pass

            # tier='guard' 우선, 구버전 호환: category='bot_guard' 도 포함
            expr = (
                f"user_id == '{user_id}' and status == 'active' and "
                f"(tier == 'guard' or category == 'bot_guard')"
            )
            # 파티션 우선 조회 (tier_guard)
            results = None
            try:
                results = self.profile_coll.query(
                    expr=expr,
                    output_fields=[
                        "key_path",
                        "value",
                        "confidence",
                        "source",
                        "tier",
                        "scope",
                    ],
                    limit=100,
                    partition_names=["tier_guard"],
                )
            except Exception:
                try:
                    results = self.profile_coll.query(
                        expr=expr,
                        output_fields=[
                            "key_path",
                            "value",
                            "confidence",
                            "source",
                            "tier",
                            "scope",
                        ],
                        limit=100,
                    )
                except Exception:
                    results = []
            items = results or []
            if self.redis is not None and items:
                try:
                    # 영구 캐시(ex=None)
                    self.redis.set(
                        cache_key, json.dumps(items, ensure_ascii=False), ex=None
                    )
                except Exception:
                    pass
            try:
                self._log_event(
                    "profile.guard_hit", {"cached": False, "count": len(items)}
                )
            except Exception:
                pass
            return items

        if tier == ProfileTier.CORE:
            cache_key = f"profile:core:{user_id}"
            if self.redis is not None:
                try:
                    cached = self.redis.get(cache_key)
                    if cached:
                        try:
                            self._log_event("profile.core_hit", {"cached": True})
                        except Exception:
                            pass
                        return json.loads(cached)
                except Exception:
                    pass

            # Core: communication./response./language. 키, 또는 tier='core'
            core_expr = " or ".join(
                [
                    f"key_path LIKE '{p}%'"
                    for p in ["communication.", "response.", "language."]
                ]
            )
            expr = (
                f"user_id == '{user_id}' and status == 'active' and "
                f"(tier == 'core' or ({core_expr}))"
            )
            results = None
            try:
                results = self.profile_coll.query(
                    expr=expr,
                    output_fields=[
                        "key_path",
                        "value",
                        "confidence",
                        "source",
                        "tier",
                        "scope",
                    ],
                    limit=50,
                    partition_names=["tier_core"],
                )
            except Exception:
                try:
                    results = self.profile_coll.query(
                        expr=expr,
                        output_fields=[
                            "key_path",
                            "value",
                            "confidence",
                            "source",
                            "tier",
                            "scope",
                        ],
                        limit=50,
                    )
                except Exception:
                    results = []
            items = results or []
            if self.redis is not None and items:
                try:
                    self.redis.set(
                        cache_key,
                        json.dumps(items, ensure_ascii=False),
                        ex=TIER_TTL[ProfileTier.CORE] or 604800,
                    )
                except Exception:
                    pass
            try:
                self._log_event(
                    "profile.core_hit", {"cached": False, "count": len(items)}
                )
            except Exception:
                pass
            return items

        # DYNAMIC: 벡터 검색, Guard 항목 제외, 캐시 없음
        qv = await self._embed_and_norm(user_input or "")
        results = None
        try:
            results = self.profile_coll.search(
                data=[qv.tolist()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=(
                    f"user_id == '{user_id}' and status == 'active' and "
                    f"(tier != 'guard' and category != 'bot_guard')"
                ),
                output_fields=[
                    "key_path",
                    "value",
                    "confidence",
                    "category",
                    "tier",
                    "scope",
                ],
                partition_names=["tier_dynamic"],
            )
        except Exception:
            try:
                results = self.profile_coll.search(
                    data=[qv.tolist()],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=top_k,
                    expr=(
                        f"user_id == '{user_id}' and status == 'active' and "
                        f"(tier != 'guard' and category != 'bot_guard')"
                    ),
                    output_fields=[
                        "key_path",
                        "value",
                        "confidence",
                        "category",
                        "tier",
                        "scope",
                    ],
                )
            except Exception:
                results = []
        items = [r.entity for r in (results[0] if results else [])]
        return items

    async def query_relevant_profile(
        self,
        user_id: str,
        user_input: str,
        top_k: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        쿼리 기반 동적 User 프로필 조회

        Returns:
            Dict: {"preferences": [...], "traits": [...], "constraints": [...]} (없으면 빈 배열)
        """

        mmr_lambda = float(os.getenv("PROFILE_MMR_LAMBDA", "0.7"))
        rag_cand_k = int(os.getenv("RAG_CAND_K", str(top_k * 3)))

        # 1) 쿼리 임베딩 (L2 정규화)
        qv = await self._embed_and_norm(user_input)

        # 2) explicit 소스 검색 (guard/bot_guard 제외)
        explicit_results = self.profile_coll.search(
            data=[qv.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=rag_cand_k,
            expr=(
                f"user_id == '{user_id}' and "
                f"source == 'explicit' and "
                f"status == 'active' and "
                f"(tier != 'guard' and category != 'bot_guard')"
            ),
            output_fields=[
                "category",
                "key_path",
                "norm_key",
                "value",
                "confidence",
                "source",
                "status",
                "tags",
                "embedding",
                "tier",
                "created_at",
                "updated_at",
                "extras",
            ],
        )

        # 3) inferred 소스 검색 (guard/bot_guard 제외)
        inferred_results = self.profile_coll.search(
            data=[qv.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=rag_cand_k,
            expr=(
                f"user_id == '{user_id}' and "
                f"source == 'inferred' and "
                f"status == 'active' and "
                f"confidence >= 0.6 and "
                f"(tier != 'guard' and category != 'bot_guard')"
            ),
            output_fields=[
                "category",
                "key_path",
                "norm_key",
                "value",
                "confidence",
                "source",
                "embedding",
                "tier",
                "created_at",
                "updated_at",
                "tags",
                "extras",
            ],
        )

        # 4) 재스코어링 (시간 감쇠 + 우선순위 + 관심사 부스트)
        now_ns = int(time.time_ns())
        user_interests = await self._load_user_interests(user_id)

        rescored: List[Tuple[Dict, float]] = []

        def _rescore_hit(hit) -> Optional[Tuple[Dict, float]]:
            try:
                entity = hit.entity
                sim = float(1.0 - float(getattr(hit, "distance", 0.0)))
                tr = (entity.get("tier") or "dynamic").strip().lower()
                dc = tier_decay(tr, entity, now_ns)
                pr = calculate_priority_multiplier(
                    source=entity.get("source"),
                    tier=tr,
                    confidence=float(entity.get("confidence", 0.5)),
                    cap=self._priority_cap,
                )
                ib = boost_by_interests(entity, user_interests)
                fs = final_score(
                    sim,
                    dc,
                    pr,
                    ib,
                    self._alpha,
                    self._beta,
                    self._gamma,
                    self._delta,
                    cap=self._priority_cap,
                )

                # 안정성/점수 보너스(있을 때만): extras.scoreboard.{stability,score}
                try:
                    extras = entity.get("extras") or {}
                    sb = extras.get("scoreboard") or {}
                    stab = float(sb.get("stability", 0.5))
                    pref_score = float(sb.get("score", 0.5))
                    fs = (
                        fs
                        * (0.8 + 0.4 * max(0.0, min(1.0, stab)))
                        * (0.85 + 0.3 * max(0.0, min(1.0, pref_score)))
                    )
                except Exception:
                    pass

                # 상세 로깅(옵션)
                try:
                    age = age_days(effective_ts_ns(entity, now_ns), now_ns)
                    self._log_event(
                        "profile.rescore",
                        {
                            "sim": round(sim, 6),
                            "decay": round(dc, 6),
                            "priority": round(pr, 6),
                            "interest": round(ib, 6),
                            "final": round(fs, 6),
                            "age_days": round(float(age), 3),
                            "tier": tr,
                            "source": (entity.get("source") or ""),
                            "alpha": self._alpha,
                            "beta": self._beta,
                            "gamma": self._gamma,
                            "delta": self._delta,
                        },
                    )
                except Exception:
                    pass

                return (entity, fs)
            except Exception:
                return None

        for r in explicit_results[0] if explicit_results else []:
            item = _rescore_hit(r)
            if item is not None:
                rescored.append(item)

        for r in inferred_results[0] if inferred_results else []:
            item = _rescore_hit(r)
            if item is not None:
                rescored.append(item)

        rescored.sort(key=lambda x: -x[1])

        # 5) 멀티홉 재질의(스켈레톤) — 점수가 낮으면 쿼리 확장 재질의 시도(현재는 패스스루)
        try:
            signals_topics = await self._load_signals_topics(user_id)
            rescored = await self._maybe_multihop(rescored, user_input, signals_topics)
        except Exception:
            pass

        # 6) norm_key 중복 제거
        deduped = self._dedupe_by_norm_key(rescored)

        # 7) MMR 다양성 필터링
        selected = self._apply_mmr(deduped, lambda_param=mmr_lambda, top_k=top_k)

        # 8) 카테고리별 그룹화 (키 미존재 시 무시)
        result = {"preferences": [], "traits": [], "constraints": []}
        for entity, _ in selected:
            cat = (entity.get("category") or "").strip()
            if cat in result:
                result[cat].append(entity)
        return result

    async def _load_signals_topics(self, user_id: str) -> List[str]:
        """
        세션1 Signals와 프로필 interests를 합쳐 사용할 토픽 집합 로더(스켈레톤).
        현재 세션5 범위에서는 빈 목록을 반환하고, 후속 세션에서 구현.
        """
        try:
            # TODO(세션1 연계): signals 서비스에서 토픽 로드
            return []
        except Exception:
            return []

    async def _maybe_multihop(
        self,
        selected: List[Tuple[Dict, float]],
        query_text: str,
        signals_topics: List[str],
    ) -> List[Tuple[Dict, float]]:
        """
        멀티홉 재질의 스켈레톤.
        - 최상위 점수가 낮을 때(query 확장 → 재검색) 시도할 자리표시자.
        - 현재 세션5에서는 기존 결과를 그대로 반환.
        """
        try:
            if not selected:
                return selected
            top_score = float(selected[0][1]) if selected and selected[0] else 0.0
            # 규칙 예시: top_score < 0.6이면 재질의 고려 (미구현)
            _ = query_text, signals_topics
            return selected
        except Exception:
            return selected

    async def _load_user_interests(self, user_id: str) -> List[Dict[str, Any]]:
        """
        사용자 관심사 로드(프로필에서 추출)
        Returns: [{"topic": str, "confidence": float}, ...]
        """
        try:
            results = self.profile_coll.query(
                expr=(
                    f"user_id == '{user_id}' and "
                    f"category == 'interests' and "
                    f"status == 'active'"
                ),
                output_fields=["value", "confidence"],
                limit=50,
            )
        except Exception:
            results = []

        interests: List[Dict[str, Any]] = []
        for r in results or []:
            try:
                val = r.get("value")
                conf = float(r.get("confidence", 0.5))
                topic = None
                if isinstance(val, str):
                    # JSON 또는 평문 토픽 처리
                    try:
                        obj = json.loads(val)
                        if isinstance(obj, dict) and "topic" in obj:
                            topic = str(obj.get("topic"))
                        else:
                            topic = str(val)
                    except Exception:
                        topic = str(val)
                elif isinstance(val, dict):
                    if "topic" in val:
                        topic = str(val.get("topic"))
                if topic:
                    interests.append({"topic": topic, "confidence": conf})
            except Exception:
                continue
        return interests

    async def query_bot_profile(
        self,
        user_id: str,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Bot 프로필 2계층 조회

        1. BotGuard: 항상 주입 (Redis 캐시 24h)
        2. BotHints: 조건부 동적 조회 (쿼리 기반)
        """
        from backend.rag.profile_ids import bot_user_id_for

        bot_uid = bot_user_id_for(user_id)
        bot_guard = await self._load_bot_guard_cached(bot_uid)

        qv = await self._embed_and_norm(user_input)
        # 1) 개인화 봇 힌트
        bot_hints = self.profile_coll.search(
            data=[qv.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=3,
            expr=(
                f"user_id == '{bot_uid}' and "
                f"category == 'bot_hint' and "
                f"status == 'active'"
            ),
            output_fields=["key_path", "value"],
        )
        hint_entities = [h.entity for h in (bot_hints[0] if bot_hints else [])]
        # 2) 개인화가 없으면 전역 폴백
        if not hint_entities:
            bot_hints_global = self.profile_coll.search(
                data=[qv.tolist()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 32}},
                limit=3,
                expr=(
                    f"user_id == 'bot:global' and "
                    f"category == 'bot_hint' and "
                    f"status == 'active'"
                ),
                output_fields=["key_path", "value"],
            )
            hint_entities = [
                h.entity for h in (bot_hints_global[0] if bot_hints_global else [])
            ]

        return {"guard": bot_guard, "hints": hint_entities}

    async def _load_bot_guard_cached(self, user_id: str) -> Dict:
        """BotGuard 캐싱 로드 (Redis 24h TTL). Redis 미사용 시 Milvus 직접 조회."""
        from backend.rag.profile_ids import bot_user_id_for

        bot_uid = bot_user_id_for(user_id)
        cache_key = f"bot:guard:{bot_uid}"
        # Redis 사용 가능 시 캐시 조회
        if self.redis is not None:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # Milvus 조회
        results = self.profile_coll.query(
            expr=(
                f"user_id == '{bot_uid}' and "
                f"category == 'bot_guard' and "
                f"status == 'active'"
            ),
            output_fields=["key_path", "value"],
        )
        guard = {r.get("key_path"): r.get("value") for r in (results or [])}
        # 전역 폴백
        if not guard:
            results = self.profile_coll.query(
                expr=(
                    "user_id == 'bot:global' and "
                    "category == 'bot_guard' and "
                    "status == 'active'"
                ),
                output_fields=["key_path", "value"],
            )
            guard = {r.get("key_path"): r.get("value") for r in (results or [])}

        # 캐시 저장 (있을 때만)
        if self.redis is not None:
            try:
                self.redis.setex(
                    cache_key, 86400, json.dumps(guard, ensure_ascii=False)
                )
            except Exception:
                pass
        return guard

    async def _embed_and_norm(self, text: str) -> np.ndarray:
        v = await self._embed_async(text)
        v = np.array(v, dtype=np.float32)
        den = float(np.linalg.norm(v) or 1.0)
        return v / den

    async def _embed_async(self, text: str):
        # embed_query_openai는 동기 함수이므로 스레드로 오프로딩하지 않고 직접 호출 (비용 매우 작음)
        return self.embed_query(text)

    def _dedupe_by_norm_key(self, candidates: List[Tuple[Dict, float]]):
        seen = set()
        deduped: List[Tuple[Dict, float]] = []
        for entity, score in candidates:
            nk = (entity.get("norm_key") or "").strip()
            if nk and nk not in seen:
                deduped.append((entity, score))
                seen.add(nk)
        return deduped

    def _apply_mmr(
        self, candidates: List[Tuple[Dict, float]], lambda_param: float, top_k: int
    ) -> List[Tuple[Dict, float]]:
        """MMR 다양성 필터링 (임베딩 필드가 없는 엔티티는 다양성 항으로 0 취급)"""
        if not candidates:
            return []

        selected: List[Tuple[Dict, float]] = []
        remaining = list(candidates)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr = -1e9

            for i, (cand, score) in enumerate(remaining):
                rel = float(score)

                # 다양성: 선택된 것들과의 최대 코사인 유사도
                max_sim = 0.0
                cand_emb = np.array(cand.get("embedding") or [], dtype=np.float32)
                for sel, _ in selected:
                    sel_emb = np.array(sel.get("embedding") or [], dtype=np.float32)
                    if cand_emb.size and sel_emb.size:
                        den = float(
                            (np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb)) or 1.0
                        )
                        sim = float(np.dot(cand_emb, sel_emb) / den)
                        if sim > max_sim:
                            max_sim = sim

                mmr_score = lambda_param * rel - (1.0 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected


# 싱글톤 접근자
_PROFILE_RAG_INSTANCE: Optional[ProfileRAG] = None


def get_profile_rag() -> ProfileRAG:
    """프로세스 전역에서 재사용 가능한 ProfileRAG 인스턴스 반환"""
    global _PROFILE_RAG_INSTANCE
    if _PROFILE_RAG_INSTANCE is None:
        _PROFILE_RAG_INSTANCE = ProfileRAG()
    return _PROFILE_RAG_INSTANCE
