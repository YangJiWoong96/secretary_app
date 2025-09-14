import re
from typing import List

try:
    # 임베딩 기반 라우팅을 위한 백엔드
    from backend.rag.embeddings import embed_query_cached
except Exception:
    embed_query_cached = None


_LOCAL_KWS = [
    "맛집",
    "카페",
    "식당",
    "레스토랑",
    "호텔",
    "병원",
    "약국",
    "근처",
    "주변",
    "가까운",
    "주소",
    "위치",
    "영업시간",
    "운영시간",
    "전화",
    "리뷰",
    "후기",
]

_NEWS_KWS = [
    "주가",
    "환율",
    "실적",
    "공시",
    "뉴스",
    "속보",
    "브리핑",
    "증시",
    "가격",
]


def pick_endpoint(query: str) -> str:
    t = re.sub(r"\s+", " ", (query or "").strip())
    if any(k in t for k in _LOCAL_KWS):
        return "local"
    if any(k in t for k in _NEWS_KWS):
        return "news"
    return "webkr"


def pick_endpoints(
    query: str, last_loc: str | None = None, max_k: int = 2
) -> List[str]:
    """
    임베딩 기반 서비스 라우팅:
    - seed 센트로이드 + 예시 top-k 평균 대신 간단 센트로이드 유사도 근사(문구 대표) 사용
    - last_loc가 있으면 'local'은 항상 포함
    - 임베딩 백엔드가 없으면 키워드 기반으로 폴백
    반환: 우선순위 순서의 endpoint 리스트(길이 1~2)
    """
    try:
        if embed_query_cached is None:
            raise RuntimeError("no embedding backend")
        seeds = {
            "news": ["속보 뉴스 시황 시세 공시 증시 경제 브리핑"],
            "book": ["도서 책 출판 서적 신간"],
            "encyc": ["백과사전 사전 정의 설명 위키"],
            "cafearticle": ["카페 글 후기 추천 모임 리뷰"],
            "kin": ["질문 답변 QnA 지식인"],
            "webkr": ["웹문서 사이트 페이지 정보 검색 결과"],
            "local": ["맛집 카페 식당 근처 주변 주소 위치 영업시간 전화 지도"],
        }
        qv = embed_query_cached(query)
        import numpy as np  # type: ignore

        sims: list[tuple[float, str]] = []
        qn = float(np.linalg.norm(qv) or 1.0)
        for ep, texts in seeds.items():
            sv = embed_query_cached(" ".join(texts))
            sn = float(np.linalg.norm(sv) or 1.0)
            s = float(np.dot(qv, sv) / (qn * sn))
            sims.append((s, ep))
        sims.sort(key=lambda x: -x[0])
        out = [ep for _, ep in sims[:max_k]]
        if last_loc and "local" not in out:
            if len(out) >= max_k:
                out[-1] = "local"
            else:
                out.append("local")
        # 중복 제거 유지
        seen = set()
        uniq = []
        for ep in out:
            if ep not in seen:
                uniq.append(ep)
                seen.add(ep)
        return uniq or [pick_endpoint(query)]
    except Exception:
        # 폴백: 기존 키워드 기반
        ep = pick_endpoint(query)
        out = [ep]
        if last_loc and ep != "local":
            out.append("local")
        return out[:max_k]
