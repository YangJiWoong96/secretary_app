from __future__ import annotations

"""
TrustScorer - 웹 문서 신뢰도 계산기

구성 요소(가중치 합 1.0):
- base_trust (α=0.5): 도메인 기본 신뢰도
- cross_validation_bonus (β=0.2): 동일 핵심 구문이 여러 출처에서 반복
- community_bonus (γ=0.15): 커뮤니티(예: reddit) 언급 횟수
- freshness (δ=0.1): 발행 시점 최신성
- quality (ε=0.05): 콘텐츠 길이/구조 품질

입력: 단일 문서와 전체 결과 리스트(교차 검증 산정용)
출력: 0.0~1.0 스코어(float)

주의: 외부 네트워크 호출 없음. 순수 함수적 계산으로 유지.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.utils.logger import log_event

BASE_TRUST_SCORES: Dict[str, float] = {
    # 최고 신뢰 (공공 기관/학술)
    "nih.gov": 0.95,
    "who.int": 0.95,
    "wikipedia.org": 0.85,
    # 고신뢰 (주요 언론)
    "nytimes.com": 0.90,
    "reuters.com": 0.90,
    "apnews.com": 0.90,
    "bbc.com": 0.88,
    # 중신뢰 (일반 뉴스/블로그)
    "naver.com": 0.75,
    "yonhapnews.co.kr": 0.85,
    "joongang.co.kr": 0.80,
    "hankyoreh.com": 0.78,
    "chosun.com": 0.80,
    "khan.co.kr": 0.78,
    "medium.com": 0.65,
    # 커뮤니티 (평판 기반)
    "reddit.com": 0.70,
    # 저신뢰 (개인 블로그)
    "blogspot.com": 0.40,
    "tistory.com": 0.40,
}


def get_base_trust(domain: str) -> float:
    return float(BASE_TRUST_SCORES.get((domain or "").lower(), 0.50))


def _freshness_factor(published_at: Optional[str]) -> float:
    if not published_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    days = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).days
    if days <= 7:
        return 0.1
    if days <= 30:
        return 0.05
    return 0.0


def _quality_factor(content: str) -> float:
    length = len(content or "")
    if length > 1000:
        return 0.1
    if length > 500:
        return 0.05
    return 0.0


def _cross_validation_bonus(key_phrases: List[str], all_contents: List[str]) -> float:
    if not key_phrases:
        return 0.0
    count = 0
    for c in all_contents:
        if not c:
            continue
        if any((kp and kp in c) for kp in key_phrases):
            count += 1
    return min(0.3, 0.1 * count)


def _community_bonus(domain_mentions: int) -> float:
    # Reddit/커뮤니티 언급 수를 0~4 범위로 클램프
    return min(0.2, 0.05 * max(0, int(domain_mentions)))


def _cross_validation_bonus_v2(
    key_phrases: List[str], all_results: List[Dict]
) -> float:
    """
    교차 검증 보너스(v2, 0.0~1.0 스케일)
    - 핵심 구문이 다른 출처의 제목/본문에 등장하는 출처 수를 근사적으로 측정
    - 3개 이상 출처 일치 시 1.0로 상한
    - 실제 가중 적용은 calculate_trust_score에서 0.2 배수로 반영
    """
    if not key_phrases:
        return 0.0
    matched_sources = 0
    for other in all_results:
        try:
            blob = (
                (other.get("content") or "") + " " + (other.get("title") or "")
            ).lower()
            if any((kp or "").lower() in blob for kp in key_phrases):
                matched_sources += 1
        except Exception:
            continue
    return min(1.0, matched_sources / 3.0)


def _source_type_bonus(source_type: str) -> float:
    """
    출처 유형 보너스(0.0 또는 1.0):
    - "paper" | "official" → 1.0 (가중 0.1 적용)
    - 그 외 → 0.0
    """
    t = (source_type or "").lower()
    if t in {"paper", "official"}:
        return 1.0
    return 0.0


def calculate_trust_score(
    item: Dict,
    all_results: List[Dict],
    key_phrases: Optional[List[str]] = None,
    community_mentions: Optional[Dict[str, int]] = None,
) -> float:
    """TrustScore 계산(0.0~1.0).

    Args:
        item: 단일 결과({"domain","content","published_at",...})
        all_results: 전체 결과 리스트(교차 검증용)
        key_phrases: 핵심 구문 후보(교차 검증 매칭 기준)
        community_mentions: 도메인→언급 수 매핑(없으면 0)
    """
    domain = (item.get("domain") or "").lower()
    content = item.get("content") or ""
    published_at = item.get("published_at")
    source_type = (item.get("source_type") or "").lower()

    base = get_base_trust(domain)
    # v1 대비: 교차 검증을 전체 결과에서 계산(제목/본문 포함)하고 0~1.0 스케일로 정규화
    cv_bonus_norm = _cross_validation_bonus_v2(key_phrases or [], all_results)
    comm_bonus = _community_bonus((community_mentions or {}).get(domain, 0))
    fresh = _freshness_factor(published_at)
    qual = _quality_factor(content)
    src_bonus = _source_type_bonus(source_type)

    # 가중치: α=0.5, β=0.2(교차검증), γ=0.15(커뮤니티), δ=0.1(최신), ε=0.05(품질), ζ=0.1(출처유형)
    score = (
        0.5 * base
        + 0.2 * cv_bonus_norm
        + 0.15 * comm_bonus
        + 0.1 * fresh
        + 0.05 * qual
        + 0.1 * src_bonus
    )
    score = max(0.0, min(1.0, float(score)))

    try:
        log_event(
            "web.trust_score_calculated",
            {
                "domain": domain,
                "trust_score": round(score, 4),
                "base_trust": base,
                "cross_validation_bonus": cv_bonus_norm,
                "community_bonus": comm_bonus,
                "freshness": fresh,
                "quality": qual,
                "source_type": source_type,
                "source_type_bonus": src_bonus,
            },
        )
    except Exception:
        pass

    return score


__all__ = ["calculate_trust_score", "get_base_trust"]
