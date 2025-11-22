# C:\My_Business\backend\rag\retrieval_utils.py
"""
backend.rag.retrieval_utils - RAG 검색 유틸리티

Milvus 검색 결과 처리 및 컨텍스트 생성 헬퍼 함수들을 제공합니다.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional

from backend.config import get_settings

logger = logging.getLogger("rag_retrieval")


def milvus_hits_to_ctx(hits, score_min: float = 0.45, top_k: int = 3) -> str:
    """
    Milvus 검색 hits를 텍스트 컨텍스트로 변환

    검색 결과에서 유사도가 임계값 이상인 상위 K개만 선택하여
    텍스트로 결합합니다.

    Args:
        hits: Milvus 검색 결과 리스트
        score_min: 최소 유사도 임계값 (이 값 미만은 제외)
        top_k: 최대 선택 개수

    Returns:
        str: 결합된 텍스트 컨텍스트 (줄바꿈으로 구분)

    Example:
        >>> hits = collection.search(...)
        >>> context = milvus_hits_to_ctx(hits[0], score_min=0.5, top_k=3)
        >>> print(context)
        >>> # 사용자는 서울에 거주하며...
        >>> # 프로젝트 A는...
        >>> # 2025년 10월 15일까지...
    """
    if not hits:
        return ""

    from backend.rag.utils import hit_similarity

    picked = []

    for hit in hits[:top_k]:
        sim = hit_similarity(hit)

        # 상세 로깅
        logger.info(
            f"[rag] hit_id={getattr(hit, 'id', None)} "
            f"sim={sim:.3f} "
            f"dist={getattr(hit, 'distance', None)}"
        )

        # 유사도 필터링
        if sim >= score_min:
            text = hit.entity.get("text")
            if text:
                picked.append(text)

    # 줄바꿈으로 결합
    return "\n".join(p for p in picked if p)


def format_rag_blocks(
    hits, score_min: float = 0.45, top_k: int = 3, include_metadata: bool = False
) -> str:
    """
    Milvus 검색 결과를 포맷팅된 블록으로 변환 (확장 버전)

    Args:
        hits: Milvus 검색 결과
        score_min: 최소 유사도
        top_k: 최대 개수
        include_metadata: 메타데이터 포함 여부 (date, score 등)

    Returns:
        str: 포맷팅된 텍스트
    """
    if not hits:
        return ""

    from backend.rag.utils import hit_similarity

    blocks = []

    for i, hit in enumerate(hits[:top_k], 1):
        sim = hit_similarity(hit)

        if sim < score_min:
            continue

        text = hit.entity.get("text", "")
        if not text:
            continue

        if include_metadata:
            # 메타데이터 포함
            hit_id = getattr(hit, "id", "unknown")
            date_ym = hit.entity.get("date_ym", "")

            block = (
                f"[Block {i}] (score: {sim:.3f}, id: {hit_id}, ym: {date_ym})\n"
                f"{text}"
            )
        else:
            # 텍스트만
            block = text

        blocks.append(block)

    return "\n\n".join(blocks)


def filter_by_date(
    hits, date_start: Optional[int] = None, date_end: Optional[int] = None
) -> List:
    """
    검색 결과를 날짜 범위로 필터링

    Args:
        hits: Milvus 검색 결과
        date_start: 시작 날짜 (YYYYMMDD)
        date_end: 종료 날짜 (YYYYMMDD)

    Returns:
        List: 필터링된 hits

    Example:
        >>> filtered = filter_by_date(hits, 20250801, 20250831)
        >>> # 8월 데이터만 남음
    """
    if not hits:
        return []

    if date_start is None and date_end is None:
        return hits

    filtered = []

    for hit in hits:
        hit_date_start = hit.entity.get("date_start")
        hit_date_end = hit.entity.get("date_end")

        # 날짜 겹침 체크
        if date_start is not None and hit_date_end is not None:
            if hit_date_end < date_start:
                continue

        if date_end is not None and hit_date_start is not None:
            if hit_date_start > date_end:
                continue

        filtered.append(hit)

    return filtered


# ===== 호환성을 위한 함수형 인터페이스 =====


def hit_similarity_wrapper(hit) -> float:
    """
    Hit 유사도 계산 (호환성 래퍼)

    backend.rag.utils.hit_similarity를 직접 사용하되,
    실패 시 METRIC 기반 폴백을 제공합니다.

    Args:
        hit: Milvus 검색 hit 객체

    Returns:
        float: 유사도 스코어 (높을수록 유사)
    """
    try:
        from backend.rag.utils import hit_similarity

        return hit_similarity(hit)
    except Exception:
        # 폴백: METRIC 기반 직접 계산
        from backend.rag import METRIC

        d = getattr(hit, "distance", None)
        s = getattr(hit, "score", None)

        if METRIC == "IP":
            return float(d if d is not None else s)
        elif METRIC == "COSINE":
            dist = float(d if d is not None else 1.0)
            return 1.0 - dist
        else:
            return -float(d if d is not None else 1e9)


# ===== Session 5: 시간 감쇠/우선순위/관심사 부스트 유틸 =====


def effective_ts_ns(entity: Dict[str, Any], now_ns: int) -> int:
    """
    효과 시각을 통일: updated_at 또는 created_at. 둘 다 없으면 now_ns.
    단위는 나노초(ns)로 가정.
    """
    try:
        u = entity.get("updated_at")
        c = entity.get("created_at")
        return int(u or c or now_ns)
    except Exception:
        return int(now_ns)


def age_days(ts_ns: int, now_ns: int) -> float:
    """
    ns → days 변환된 경과 시간(일).
    음수 방지 및 안전한 float 변환.
    """
    try:
        return max(0.0, float(now_ns - int(ts_ns)) / (1e9 * 86400.0))
    except Exception:
        return 0.0


def calculate_temporal_decay(
    created_at_ns: int,
    current_time_ns: Optional[int] = None,
    half_life_days: float = 30.0,
    min_decay: float = 0.3,
) -> float:
    """
    에빙하우스 망각 곡선 기반 시간 감쇠 계수.
    exp(-days/half_life)를 사용하며, 최소 감쇠(min_decay)를 보장.
    """
    if current_time_ns is None:
        current_time_ns = int(time.time_ns())
    days = age_days(created_at_ns, current_time_ns)
    try:
        decay = math.exp(-float(days) / float(half_life_days))
    except Exception:
        decay = 1.0
    return max(float(min_decay), float(decay))


def tier_decay(tier: Optional[str], entity: Dict[str, Any], now_ns: int) -> float:
    """
    티어별 시간 감쇠 계산.
    - guard: 검색 제외가 원칙이나, 유입 시 decay=1.0 처리
    - core: 긴 반감기, 비교적 높은 바닥값
    - dynamic: 짧은 반감기, 기본 바닥값
    """
    s = get_settings()
    t = (tier or "").strip().lower()
    if t == "guard":
        return 1.0
    eff_ns = effective_ts_ns(entity, now_ns)
    if t == "core":
        return calculate_temporal_decay(
            eff_ns,
            current_time_ns=now_ns,
            half_life_days=float(s.PROFILE_DECAY_HALF_LIFE_CORE),
            min_decay=float(s.PROFILE_DECAY_MIN_CORE),
        )
    # default: dynamic
    return calculate_temporal_decay(
        eff_ns,
        current_time_ns=now_ns,
        half_life_days=float(s.PROFILE_DECAY_HALF_LIFE_DYNAMIC),
        min_decay=float(s.PROFILE_DECAY_MIN_DYNAMIC),
    )


def calculate_priority_multiplier(
    source: Optional[str],
    tier: Optional[str] = None,
    confidence: float = 0.5,
    cap: Optional[float] = None,
) -> float:
    """
    프로필 우선순위 가중치.
    - base: source에 따른 가중치
    - tier 보너스
    - 신뢰도 반영: (0.5 + 0.5*confidence)
    - 상한 캡 적용
    """
    base_weights = {
        "explicit": 2.0,
        "directives": 1.5,
        "inferred": 1.0,
    }
    tier_bonus = {
        "guard": 0.5,
        "core": 0.3,
        "dynamic": 0.0,
    }
    src = (source or "inferred").strip().lower()
    tr = (tier or "dynamic").strip().lower()
    weight = float(base_weights.get(src, 1.0)) + float(tier_bonus.get(tr, 0.0))
    # 신뢰도 반영(보수적): 0.5 ~ 1.0 배
    conf_term = 0.5 + 0.5 * float(confidence or 0.0)
    weight *= conf_term
    # 캡 적용
    if cap is None:
        try:
            cap = float(get_settings().PRIORITY_CAP)
        except Exception:
            cap = 2.2
    return min(float(weight), float(cap))


def _to_lower_set(values) -> set[str]:
    out = set()
    try:
        if isinstance(values, dict):
            # JSON 오브젝트일 경우 값들을 문자열로 변환
            for v in values.values():
                out.add(str(v).lower())
        elif isinstance(values, list):
            for v in values:
                out.add(str(v).lower())
        elif values is not None:
            out.add(str(values).lower())
    except Exception:
        pass
    return out


def boost_by_interests(
    entity: Dict[str, Any],
    user_interests: List[Dict[str, Any]],
    boost_ratio: float = 0.3,
) -> float:
    """
    사용자 관심사 일치 시 부스팅.
    - 관심사 신뢰도(confidence)를 반영하여: 1 + boost_ratio * max(topic_conf)
    - 태그는 entity.tags(JSON/list/str), norm_key, category 등에서 추출하여 교집합 확인
    """
    try:
        # 엔티티 태그 집합 수집
        tag_set = set()
        tag_set |= _to_lower_set(entity.get("tags"))
        tag_set |= _to_lower_set(entity.get("category"))
        tag_set |= _to_lower_set(entity.get("norm_key"))
        tag_set |= _to_lower_set(entity.get("key_path"))

        if not tag_set:
            return 1.0

        # 사용자 관심 토픽 집합 및 신뢰도 추출
        best_conf = 0.0
        for it in user_interests or []:
            topic = str(it.get("topic", "")).lower()
            if not topic:
                continue
            if topic in tag_set:
                c = float(it.get("confidence", 0.5))
                if c > best_conf:
                    best_conf = c

        if best_conf > 0.0:
            return float(1.0 + boost_ratio * best_conf)
        return 1.0
    except Exception:
        return 1.0


def final_score(
    sim: float,
    decay: float,
    priority: float,
    interest: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    cap: float = 2.2,
) -> float:
    """
    최종 스코어 결합: sim^α * decay^β * priority^γ * interest^δ
    - priority는 상한 캡 적용
    - 0^0, 음수/NaN 방지, 범위 클램프
    """
    try:
        pr = min(max(float(priority), 1e-6), float(cap))
        sm = min(max(float(sim), 1e-6), 1.0)
        dc = min(max(float(decay), 1e-6), 1.0)
        ib = max(float(interest), 1.0)
        return (
            (sm ** float(alpha))
            * (dc ** float(beta))
            * (pr ** float(gamma))
            * (ib ** float(delta))
        )
    except Exception:
        return float(sim or 0.0)
