from __future__ import annotations

"""
검색 랭킹 스코어링 모듈

역할:
- 일반 검색 결과(랭킹 값이 있건 없건)에 대해 '랭킹 역수 + 빈도 + 핫니스' 가중합 점수를 부여한다.
- TrendRadar의 rank/frequency/hotness 개념을 WHOAMI 표준 인터페이스로 단순화했다.

설계 노트:
- rank: 1이 가장 좋음. 역수(1/rank)로 정규화해 높은 순위에 큰 보상을 준다.
- freq: 최근 스냅샷에서의 등장 빈도(0~1 권장). 호출 측에서 min(count/K, 1.0) 등으로 정규화해 전달한다.
- hot: 상위 순위(예: top-5) 등장 비율 등(0~1 권장). 호출 측에서 비율로 전달한다.
- 가중치 w는 배포 환경에서 조정 가능하게 기본값만 제공한다.
"""

from typing import Tuple


def score_item(
    rank: int,
    freq: float,
    hot: float,
    w: Tuple[float, float, float] = (0.55, 0.25, 0.20),
) -> float:
    """
    항목 스코어 계산(가중합).

    Args:
        rank: 자연수 랭킹(1이 최상). 값이 없을 경우 호출 측에서 큰 값(예: 999)로 전달 권장.
        freq: 최근 스냅샷 기반 정규화 빈도(0~1 범위 권장).
        hot: 상위 순위 비율 등 정규화 핫니스(0~1 범위 권장).
        w: (rank 역수, freq, hot) 가중치 튜플.

    Returns:
        float: 최종 스코어(클수록 우선).
    """
    # 0으로 나눔 방지 및 순위 역수 기반 스코어
    rank_score = 1.0 / (float(rank) + 1e-6)
    return w[0] * rank_score + w[1] * float(freq) + w[2] * float(hot)


__all__ = ["score_item"]
