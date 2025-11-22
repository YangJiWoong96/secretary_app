from __future__ import annotations

import math
import time
from typing import Dict, Optional


def _get_redis(url: str):
    """
    Redis 클라이언트를 지연 로딩한다.
    - 운영 환경에 따라 redis 모듈 부재 가능성이 있으므로 예외를 흡수한다.
    """
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        return None


class TraitAggregator:
    """
    사용자 단위 트레이트(Big Five/MBTI) 누적기

    - Big Five: EMA + EWVar (안정성 추정)
    - MBTI: 관측 다수결 + 안정성 임계치 충족 시 확정
    """

    def __init__(self, redis_url: str):
        self._r = _get_redis(redis_url)

    def _k(self, user_id: str, name: str) -> str:
        return f"trait:{name}:{user_id}"

    def available(self) -> bool:
        return self._r is not None

    def update_bigfive(
        self,
        user_id: str,
        estimate: Dict[str, float],
        confidence: float,
        now: Optional[int] = None,
    ) -> None:
        """
        Big Five 추정치를 사용자 범위로 누적한다.
        - EMA 가중치는 신뢰도(confidence)로 조절
        """
        if not self._r or not user_id or not isinstance(estimate, dict):
            return
        now = now or int(time.time())
        # 기본 학습률(너무 크지 않게) + Huber 기반 동적 축소
        alpha0 = 0.15
        conf = max(0.0, min(1.0, float(confidence or 0.0)))

        for dim, val in estimate.items():
            try:
                v = float(val)
            except Exception:
                continue
            old = float(self._r.hget(self._k(user_id, "bf"), dim) or 0.5)
            # 잔차
            r = v - old
            # Huber 스케일(δ): 신뢰도가 낮을수록 작은 δ → 이상치에 민감
            delta = 0.5 * max(0.1, conf)
            # Huber 가중
            if abs(r) <= delta:
                weight = 1.0
            else:
                weight = delta / max(1e-6, abs(r))
            # 동적 alpha (신뢰도 × Huber 가중)
            alpha = max(0.02, min(0.6, alpha0 * max(0.25, conf) * weight))
            new = (1.0 - alpha) * old + alpha * v
            self._r.hset(self._k(user_id, "bf"), dim, f"{new:.4f}")

            # EWVar
            var_key = self._k(user_id, "bf_var")
            old_var = float(self._r.hget(var_key, dim) or 0.0)
            ewvar = (1.0 - alpha) * old_var + alpha * ((v - new) ** 2)
            self._r.hset(var_key, dim, f"{ewvar:.6f}")

        self._r.hset(self._k(user_id, "bf_meta"), "ts", str(now))

    def get_bigfive(self, user_id: str) -> Dict[str, float]:
        if not self._r or not user_id:
            return {}
        try:
            raw = self._r.hgetall(self._k(user_id, "bf")) or {}
            return {k: float(v) for k, v in raw.items()}
        except Exception:
            return {}

    def stability(self, user_id: str) -> float:
        """
        Big Five 안정성 추정치(0~1): EWVar 평균의 역함수 기반.
        """
        if not self._r or not user_id:
            return 0.0
        try:
            items = self._r.hgetall(self._k(user_id, "bf_var")) or {}
            if not items:
                return 0.0
            mean_var = sum(float(x) for x in items.values()) / max(1, len(items))
            return float(math.exp(-min(5.0, mean_var * 10.0)))
        except Exception:
            return 0.0

    def record_mbti_observation(self, user_id: str, mbti: Optional[str]) -> None:
        """
        MBTI 관측값을 관찰 리스트에 추가한다(강제 갱신 아님).
        """
        if not self._r or not user_id or not mbti:
            return
        try:
            self._r.lpush(self._k(user_id, "mbti_obs"), mbti)
            self._r.ltrim(self._k(user_id, "mbti_obs"), 0, 49)
        except Exception:
            return

    def maybe_finalize_mbti(
        self,
        user_id: str,
        min_obs: int = 8,
        min_count: int = 5,
        min_stability: float = 0.6,
    ) -> Optional[str]:
        """
        다수결 + 안정성 임계치로 MBTI를 확정/갱신한다.
        """
        if not self._r or not user_id:
            return None
        try:
            obs = self._r.lrange(self._k(user_id, "mbti_obs"), 0, 49)
            if len(obs) < min_obs:
                return None
            from collections import Counter

            top, cnt = Counter(obs).most_common(1)[0]
            if cnt >= int(min_count) and self.stability(user_id) >= float(
                min_stability
            ):
                self._r.set(self._k(user_id, "mbti"), top)
                return top
            return None
        except Exception:
            return None

    def get_mbti(self, user_id: str) -> Optional[str]:
        if not self._r or not user_id:
            return None
        try:
            val = self._r.get(self._k(user_id, "mbti"))
            return str(val) if val else None
        except Exception:
            return None
