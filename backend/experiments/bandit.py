from __future__ import annotations

"""
간단한 Epsilon-Greedy 멀티암드 밴딧 구현

목표:
- 알림 문안/톤/이모지 등 변이에 대해 온라인으로 선택/업데이트
- 저장소는 프로세스 메모리 기본, 필요 시 Redis 사용(REDIS_URL 존재 시)

보상 정의(초기 버전):
- open: +1.0
- dwell_ms >= 3000: +0.5
- answer_started(True): +0.5

주의:
- 실서비스에서는 사용자/세션별로 분리 키를 두는 것이 바람직하나, 초기 단계에서는 전역 스코프로 계측
"""

import os
import random
from typing import Dict, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class _KV:
    """키-값 저장소 추상화(메모리/Redis).

    - 메모리: 프로세스 범위에서만 유지(개발/테스트 기본)
    - Redis: REDIS_URL이 유효할 때 선택 사용
    """

    def __init__(self) -> None:
        self._mem: Dict[str, str] = {}
        self._redis = None
        try:
            from backend.config import get_settings as _gs

            url = _gs().REDIS_URL
        except Exception:
            url = os.getenv("REDIS_URL")
        if url and redis is not None:
            try:
                self._redis = redis.Redis.from_url(url, decode_responses=True)
            except Exception:
                self._redis = None

    def get(self, key: str) -> Optional[str]:
        if self._redis is not None:
            try:
                return self._redis.get(key)  # type: ignore
            except Exception:
                return self._mem.get(key)
        return self._mem.get(key)

    def set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                self._redis.set(key, value)  # type: ignore
                return
            except Exception:
                pass
        self._mem[key] = value


_kv = _KV()


def _k_counts() -> str:
    return "bandit:counts"


def _k_rewards() -> str:
    return "bandit:rewards"


class Bandit:
    def __init__(self, epsilon: float | None = None) -> None:
        try:
            if epsilon is not None:
                self.epsilon = float(epsilon)
            else:
                from backend.config import get_settings as _gs2

                self.epsilon = float(getattr(_gs2(), "BANDIT_EPSILON", 0.1))
        except Exception:
            self.epsilon = 0.1

    def _load(self) -> tuple[Dict[str, float], Dict[str, int]]:
        import json

        try:
            r_json = _kv.get(_k_rewards()) or "{}"
            c_json = _kv.get(_k_counts()) or "{}"
            rewards: Dict[str, float] = json.loads(r_json)
            counts: Dict[str, int] = json.loads(c_json)
            return rewards, counts
        except Exception:
            return {}, {}

    def _save(self, rewards: Dict[str, float], counts: Dict[str, int]) -> None:
        import json

        try:
            _kv.set(_k_rewards(), json.dumps(rewards, ensure_ascii=False))
            _kv.set(_k_counts(), json.dumps(counts, ensure_ascii=False))
        except Exception:
            pass

    def select(self, arms: Dict[str, Dict]) -> str:
        """
        가용 arms 중 하나를 선택:
        - epsilon 확률로 무작위 탐색
        - 그 외에는 평균 보상(reward/count)이 최대인 arm 선택
        """
        rewards, counts = self._load()
        if not arms:
            return "baseline"
        keys = list(arms.keys())
        if random.random() < self.epsilon:
            return random.choice(keys)

        # 평균 보상 기준 최대값 선택(0-division 보호)
        def _avg(a: str) -> float:
            r = float(rewards.get(a, 0.0))
            c = max(1, int(counts.get(a, 0)))
            return r / c

        return max(keys, key=_avg)

    def update(self, arm: str, reward: float) -> None:
        rewards, counts = self._load()
        if not arm:
            return
        counts[arm] = int(counts.get(arm, 0)) + 1
        rewards[arm] = float(rewards.get(arm, 0.0)) + float(reward)
        self._save(rewards, counts)


def reward_from_event(dwell_ms: int = 0, answer_started: bool = False) -> float:
    """오픈 이벤트 기준 보상 함수(초기 버전)."""
    base = 1.0
    if int(dwell_ms) >= 3000:
        base += 0.5
    if bool(answer_started):
        base += 0.5
    return base


__all__ = ["Bandit", "reward_from_event"]
