"""
backend.memory.ltm_promoter - LTM 승격 판단기

방법론:
- 제외 패턴 우선 필터링(센트로이드 임베딩 유사도)
- 카테고리별 SEED 센트로이드와의 유사도 계산
- 최고 유사도와 임계값으로 승격 결정

주의:
- 라우팅(SEED)과 독립적으로 동작
- AI 응답은 참고용으로만 받고, 기본 결정은 사용자 입력 기반
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from backend.rag.embeddings import embed_query_openai
from backend.config import get_settings


class LTMPromoter:
    """
    LTM 승격 판단기

    내부적으로 LTM 승격 SEED를 임베딩하여 카테고리별 센트로이드를 구성하고,
    사용자 발화와의 코사인 유사도로 승격 여부를 결정한다.
    """

    def __init__(self) -> None:
        self.seed_embeddings: Dict[str, np.ndarray] = {}
        self.exclusion_embedding: np.ndarray | None = None
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """첫 호출 시 지연 초기화."""
        if self._initialized:
            return

        from backend.memory.ltm_promotion_seed import (
            LTM_EXCLUSION_PATTERNS,
            LTM_PROMOTION_SEED,
        )

        # 1) 카테고리별 센트로이드 계산
        for category, examples in LTM_PROMOTION_SEED.items():
            if not examples:
                continue
            embs = [embed_query_openai(ex) for ex in examples]
            embs_arr = np.stack(embs, axis=0).astype(np.float32)
            centroid = np.mean(embs_arr, axis=0)
            self.seed_embeddings[category] = centroid

        # 2) 제외 패턴 센트로이드 계산
        if LTM_EXCLUSION_PATTERNS:
            excl_embs = [embed_query_openai(ex) for ex in LTM_EXCLUSION_PATTERNS]
            excl_arr = np.stack(excl_embs, axis=0).astype(np.float32)
            self.exclusion_embedding = np.mean(excl_arr, axis=0)

        self._initialized = True

    def should_promote(
        self, user_input: str, ai_output: str
    ) -> Tuple[bool, str, float]:
        """
        LTM 승격 여부 판단

        Args:
            user_input: 사용자 입력 텍스트
            ai_output: AI 응답(참고용)

        Returns:
            (should_promote: bool, category: str, confidence: float)
        """
        self._ensure_initialized()

        # 빈 입력 방어
        text = (user_input or "").strip()
        if not text:
            return False, "empty", 0.0

        q_emb = embed_query_openai(text)
        q_arr = np.array(q_emb, dtype=np.float32)

        # 1) 제외 패턴 우선 필터
        if self.exclusion_embedding is not None:
            excl_sim = self._cosine_sim(q_arr, self.exclusion_embedding)
            if excl_sim > 0.65:
                return False, "excluded", float(excl_sim)

        # 2) 카테고리별 유사도
        sims: Dict[str, float] = {}
        for cat, seed_emb in self.seed_embeddings.items():
            sims[cat] = float(self._cosine_sim(q_arr, seed_emb))

        if not sims:
            return False, "no_seed", 0.0

        # 3) 최고 카테고리 선택
        best_cat = max(sims, key=sims.get)
        best_sim = float(sims[best_cat])

        # 4) 캘리브레이션(Platt/temperature scaling 유사): 사용자별 파라미터 지원
        try:
            import math as _math

            _s = get_settings()
            a_default = float(getattr(_s, "CAL_LTM_A", 10.0))
            b_default = float(getattr(_s, "CAL_LTM_B", -5.0))
            tau_default = float(getattr(_s, "LTM_PROMOTION_THR", 0.55))

            # 사용자별 보정 파라미터 (전역/사용자 키 순서로 조회)
            a, b, tau_u = a_default, b_default, tau_default
            try:
                import redis  # type: ignore

                r = redis.Redis.from_url(_s.REDIS_URL, decode_responses=True)
                raw = r.hgetall("cal:ltm:global") or {}
                a = float(raw.get("a", a))
                b = float(raw.get("b", b))
                tau_u = float(raw.get("tau", tau_u))
            except Exception:
                pass

            p = 1.0 / (1.0 + _math.exp(-(a * best_sim + b)))
            if p >= tau_u:
                return True, best_cat, best_sim
            return False, "low_confidence", best_sim
        except Exception:
            # 폴백: 기존 임계값
            if best_sim >= 0.55:
                return True, best_cat, best_sim
            return False, "low_confidence", best_sim

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
        return float(np.dot(a, b) / denom)


# ===== 싱글톤 제공 =====
_PROMOTER_INSTANCE: LTMPromoter | None = None


def get_ltm_promoter() -> LTMPromoter:
    global _PROMOTER_INSTANCE
    if _PROMOTER_INSTANCE is None:
        _PROMOTER_INSTANCE = LTMPromoter()
    return _PROMOTER_INSTANCE
