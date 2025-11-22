from __future__ import annotations

"""
도메인 카탈로그

역할:
- 버티컬/카테고리별로 선호/신뢰/응답성 기준의 시드 도메인을 제공한다.
- seed_domains/seed_urls는 Discovery 및 CSE site: 제한 검색에 사용된다.

설계:
- 최소 동작 버전: 정적 맵 + 간단 가중치. 추후 관측성 지표(p95/에러율/채택률)로 동적 갱신 가능.
"""

from typing import Dict, List, Tuple, Optional

# 카테고리 → (도메인, 기본 신뢰) 목록
_CATALOG: Dict[str, List[Tuple[str, float]]] = {
    # 뉴스/공식
    "뉴스": [
        ("news.google.com", 0.80),
        ("reuters.com", 0.90),
        ("apnews.com", 0.90),
        ("nytimes.com", 0.90),
        ("bbc.com", 0.88),
        ("yonhapnews.co.kr", 0.85),
        ("joongang.co.kr", 0.80),
        ("hankyoreh.com", 0.78),
        ("chosun.com", 0.80),
        ("khan.co.kr", 0.78),
    ],
    # 커뮤니티/포럼
    "커뮤니티": [
        ("reddit.com", 0.70),
        ("stackoverflow.com", 0.80),
        ("okky.kr", 0.65),
        ("clien.net", 0.60),
    ],
    # 재무/마켓
    "재무": [
        ("alphavantage.co", 0.85),
        ("investing.com", 0.75),
        ("finance.yahoo.com", 0.78),
        ("wsj.com", 0.86),
    ],
    # 과학/논문
    "과학": [
        ("arxiv.org", 0.90),
        ("nature.com", 0.92),
        ("science.org", 0.92),
        ("acm.org", 0.90),
        ("ieee.org", 0.90),
    ],
    # 공식/문서
    "공식": [
        ("huggingface.co", 0.85),
        ("openai.com", 0.85),
        ("docs.python.org", 0.90),
        ("pytorch.org", 0.88),
    ],
    # 지역/생활
    "여가": [
        ("naver.com", 0.75),
        ("kakao.com", 0.75),
        ("mangoplate.com", 0.70),
        ("map.kakao.com", 0.75),
    ],
}

# 카테고리 ↔ 기본 매핑(플래너 카테고리 → 카탈로그 키)
_CATEGORY_MAP: Dict[str, str] = {
    "건강": "뉴스",
    "관계": "여가",
    "재무": "재무",
    "여가": "여가",
    "업무": "공식",
}


def get_seed_domains(
    info_needs: List[str],
    topk: int = 6,
    recent_domains: Optional[List[str]] = None,
    preferred_regions: Optional[List[str]] = None,
) -> List[str]:
    """
    info_needs(우선순위 카테고리) 기반으로 시드 도메인 목록을 반환.
    - 상위 니즈를 우선 반영하되, 중복 없이 합집합으로 구성.
    - 최근 소비 도메인은 우선순위를 약간 낮춘다(신규성 확보).
    - 지역 선호(preferred_regions)는 향후 확장 훅으로 남겨둔다.
    """
    out: List[str] = []
    seen = set()
    rec = set([d.lower() for d in (recent_domains or [])])
    for cat in info_needs or []:
        key = _CATEGORY_MAP.get(cat, "뉴스")
        for dom, _w in _CATALOG.get(key, []):
            if dom not in seen:
                seen.add(dom)
                # 최근 소비 도메인은 맨 뒤에 붙여 우선순위 소폭 낮춤
                if dom.lower() in rec:
                    out.append(dom)  # 뒤로
                else:
                    out.insert(0, dom)  # 앞으로
            if len(out) >= topk:
                return out
    # 보수적 백업: 뉴스 기본 시드
    if not out:
        for dom, _w in _CATALOG.get("뉴스", []):
            if dom not in seen:
                out.append(dom)
                if len(out) >= topk:
                    break
    return out


__all__ = ["get_seed_domains"]
