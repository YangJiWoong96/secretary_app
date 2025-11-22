import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("router")

try:
    # 임베딩 기반 라우팅을 위한 백엔드
    from backend.rag.embeddings import embed_query_openai as embed_query_cached
except Exception:
    embed_query_cached = None

# 계산된 시드 벡터를 저장할 캐시
ENDPOINT_SEED_VECTORS: Dict[str, List[List[float]]] = {}


def _seeds_signature() -> str:
    payload = json.dumps(ENDPOINT_SEEDS, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def initialize_seed_vectors():
    """
    애플리케이션 시작 시 시드 문장 임베딩을 파일(npz)로 캐싱합니다.
    - 프로젝트 루트의 .cache/router_seed_vectors.npz 사용
    - 시그니처(sha256) 불일치 시에만 재계산하여 API 호출 폭주를 방지합니다.
    """
    if not embed_query_cached:
        raise RuntimeError("임베딩 백엔드 없음")

    sig = _seeds_signature()
    cache_dir = Path(__file__).resolve().parents[2] / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "router_seed_vectors.npz"
    sig_path = cache_dir / "router_seed_vectors.sig"

    try:
        if cache_path.exists() and sig_path.exists():
            saved_sig = sig_path.read_text(encoding="utf-8").strip()
            if saved_sig == sig:
                import numpy as _np

                data = _np.load(cache_path, allow_pickle=True)
                keys = list(ENDPOINT_SEEDS.keys())
                for ep in keys:
                    ENDPOINT_SEED_VECTORS[ep] = data[ep].tolist()
                logger.info(
                    f"[router:semantic] 시드 벡터 캐시 로드 완료: {list(ENDPOINT_SEED_VECTORS.keys())}"
                )
                return
    except Exception as e:
        logger.warning(f"[router:semantic] 캐시 로드 실패, 재계산 진행: {e}")

    logger.info("[router:semantic] 시드 벡터 초기화를 계산합니다...")
    try:
        import numpy as _np

        for endpoint, seeds in ENDPOINT_SEEDS.items():
            seed_vectors = [embed_query_cached(seed) for seed in seeds]
            ENDPOINT_SEED_VECTORS[endpoint] = seed_vectors
        # 저장
        arrs = {ep: _np.array(vecs) for ep, vecs in ENDPOINT_SEED_VECTORS.items()}
        _np.savez_compressed(cache_path, **arrs)
        sig_path.write_text(sig, encoding="utf-8")
        logger.info(
            f"[router:semantic] 시드 벡터 초기화 완료 및 캐시 저장: {list(ENDPOINT_SEED_VECTORS.keys())}"
        )
    except Exception as e:
        logger.warning(f"[router:semantic] 시드 벡터 계산 중 오류: {e}")


# 엔드포인트별 시드 패턴 (각 20개 이상) — 일반화된 추상 패턴 유지
ENDPOINT_SEEDS = {
    # 📍 지역/장소 탐색
    "local": [
        "<지역명> <카페/맛집/식당> 추천",
        "<지하철역/동네명> <커피/술집/디저트> 맛집",
        "<지역명> <호텔/숙소> 예약",
        "<지역명> <병원/약국> 영업시간",
        "<지역명> <편의점/마트> 위치",
        "<지역명> <헬스장/수영장> 추천",
        "<지역명> <영화관/공연장> 위치",
        "<지역명> <주차장/주유소> 안내",
        "<지역명> <쇼핑몰/시장> 위치",
        "<지역명> <반려동물 병원/카페> 추천",
        "<지역명> <어린이 시설/키즈카페>",
        "<현재 위치> 주변 <카테고리> 검색",
        "<지역명> <은행/ATM> 위치",
    ],
    # 📰 뉴스/이슈
    "news": [
        "<분야> 최신 뉴스 요약",
        "<기업/정치/경제> 속보",
        "<국가/지역> 사회 이슈",
        "<스포츠/연예> 뉴스",
        "<테크 기업> 행사/발표",
        "<날씨/재해> 소식",
    ],
    # 🌐 일반 웹 정보/가이드
    "webkr": [
        "<프로그래밍 언어> 튜토리얼",
        "<프레임워크/라이브러리> 사용법",
        "<법률/의학/요리> 정보",
        "<재테크/투자> 가이드",
        "<여행지> 정보",
        "<교육/학습법> 문서",
        "<IT/디자인/마케팅> 참고자료",
    ],
    # 🛍️ 쇼핑/상품 검색
    "shop": [
        "<상품명> 가격 비교",
        "<제품 카테고리> 최저가",
        "<브랜드> 신제품 정보",
        "<패션/가전/생활용품> 쇼핑",
        "<쿠폰/프로모션> 검색",
    ],
    # ✍️ 블로그 후기/리뷰
    "blog": [
        "<주제> 블로그 후기",
        "<맛집/여행> 블로그 글",
        "<일상/취미> 블로그",
        "<리뷰> 블로그 찾기",
        "<패션/뷰티/운동> 후기",
        "<창업/재테크> 경험담",
        "<요리/육아> 후기",
    ],
    # 💬 네이버 카페·커뮤니티 글
    "cafearticle": [
        "<맘카페> 정보",
        "<동호회/스터디> 카페 글",
        "<부동산/재테크> 커뮤니티",
        "<자격증/입시> 카페 게시글",
        "<중고 거래> 카페",
        "<반려동물> 커뮤니티 글",
        "<창업/취업> 카페 글",
        "<게임> 공략 카페 글",
        "<공지/세미나> 게시글",
        "<질문/답변> 스레드",
    ],
    # 🖼️ 이미지/비주얼 콘텐츠
    "image": [
        "<음식> 사진",
        "<풍경/여행지> 이미지",
        "<패션/제품> 이미지",
        "<인테리어/건축> 사진",
        "<일러스트/배경화면> 이미지",
    ],
    # ❓ 질문/지식인류
    "kin": [
        "<주제> 질문",
        "<법률/의료> 상담 질문",
        "<프로그래밍/영어> 질문",
        "<생활 정보> 질문",
        "<게임/취미> 질문",
    ],
    # 📚 도서/학습 콘텐츠
    "book": [
        "<주제> 입문서",
        "<분야> 자기계발/경영",
        "<장르> 소설/에세이",
        "<분야> 과학/철학/역사",
        "<테마> 추천 도서",
    ],
    # 🧠 백과사전성 정보
    "encyc": [
        "<학문/역사/문화> 백과사전",
        "<동물/식물> 백과사전",
        "<지리/인물> 백과사전",
        "<IT/과학/경제> 백과사전",
    ],
    # 📖 학술 논문/연구 자료
    "academic": [
        "<분야> 논문",
        "<주제> 연구",
        "<학회/저널> 논문",
        "<실험/이론> 방법론",
        "<산업 적용> 연구 사례",
    ],
}

# ENDPOINT_SEEDS = {
#     "local": [
#         "<장소> 근처 <시설 종류> 추천",
#         "현재 위치 주변 <음식 종류> 맛집",
#         "<지역명> <가게명> 주소 알려줘",
#         "<지역명> <호텔/숙소> 추천",
#         "<지역명> <병원/약국> 영업시간",
#         "<지역명> <편의점/마트> 위치",
#         "<지역명> <주차장> 있는 곳",
#         "<지하철역/버스터미널> 근처 <카페/식당>",
#         "<동네명> <미용실/세탁소> 찾기",
#         "<지역명> <헬스장/수영장> 추천",
#         "<지역명> <영화관/공연장> 위치",
#         "<지역명> <베이커리/디저트> 맛집",
#         "<지역명> <술집/바> 추천",
#         "<관광지/상권> <레스토랑> 예약",
#         "<지역명> <쇼핑몰/시장> 안내",
#         "<고속도로/톨게이트> 인근 <주유소>",
#         "<지역명> 반경 <n>km <카테고리> 검색",
#         "<현재 위치> 주변 <키워드> 검색",
#         "<지역명> <은행/ATM> 위치",
#         "<지역명> <반려동물 병원/카페> 추천",
#         "<지역명> <어린이 시설/키즈카페>",
#     ],
#     "news": [
#         "<분야> 최신 뉴스 요약",
#         "<기업/지수> <주가/실적> 소식",
#         "<국가/지역> 정치 현황",
#         "<이벤트> 속보",
#         "<경제 지표> 동향",
#         "<산업> 동향 요약",
#         "<코인/원자재> 시세",
#         "<선거/정책> 결과",
#         "<스포츠 리그/팀> 경기 결과",
#         "<날씨/기상특보> 관련 뉴스",
#         "<기업> 공시/IR 발표",
#         "<부동산/금융> 시장 브리핑",
#         "<세계/국내> 헤드라인",
#         "<사회 이슈> 정리",
#         "<테크 기업> 발표/행사 소식",
#         "<전염병/보건> 현황",
#         "<환율/금리> 변동",
#         "<주요인물> 발언 요약",
#         "<업계> 리포트 요약",
#         "<분기/연간> 실적 요약",
#     ],
#     "webkr": [
#         "<프로그래밍 언어> 튜토리얼",
#         "<프레임워크> 공식 문서",
#         "<라이브러리> 사용법",
#         "<개념> 설명/정의",
#         "<질병명> 증상 및 치료법",
#         "<요리> 레시피",
#         "<여행지> 가이드",
#         "<법률 주제> 참고문서",
#         "<육아/교육> 정보",
#         "<자동차 정비> 가이드",
#         "<재테크/투자> 블로그",
#         "<학습법> 웹페이지",
#         "<IT 기술> 문서",
#         "<디자인> 참고 자료",
#         "<마케팅> 전략 문서",
#         "<창업> 가이드",
#         "<이력서/면접> 팁",
#         "<프로젝트 관리> 방법론",
#         "<데이터 분석> 튜토리얼",
#         "<클라우드> 서비스 비교",
#         "<검색 키워드> 관련 위키/문서",
#     ],
#     "shop": [
#         "<제품 카테고리> 최저가 비교",
#         "<상품명> 가격",
#         "<브랜드> 신제품 쇼핑 정보",
#         "<카테고리> 할인 정보",
#         "<전자제품> 가격 비교",
#         "<의류/패션> 쇼핑",
#         "<가구/인테리어> 가격",
#         "<도서> 구매",
#         "<생활용품> 쇼핑",
#         "<식품> 최저가",
#         "<건강식품> 구매",
#         "<유아용품> 쇼핑",
#         "<반려동물 용품> 가격",
#         "<자동차 용품> 구매",
#         "<스포츠/등산/캠핑> 용품",
#         "<악기> 구매",
#         "<문구/사무용품> 쇼핑",
#         "<선물/기념품> 추천",
#         "<중고> 가격 비교",
#         "<쿠폰/프로모션> 검색",
#     ],
#     "blog": [
#         "<주제> 블로그 추천",
#         "<맛집/여행> 후기 블로그",
#         "<개발/디자인> 블로그 글",
#         "<일상/취미> 블로그",
#         "<리뷰> 블로그 찾기",
#         "<포트폴리오> 블로그",
#         "<패션/뷰티> 블로그",
#         "<독서/영화/게임> 리뷰",
#         "<생활 정보> 블로그",
#         "<운동/헬스> 일지",
#         "<육아> 블로그",
#         "<여행지> 후기",
#         "<요리> 레시피 블로그",
#         "<사진> 갤러리",
#         "<창업/재테크> 블로그",
#         "<스터디> 블로그",
#         "<교육> 블로그",
#         "<공예/만들기> 블로그",
#         "<반려동물> 블로그",
#         "<가전/제품> 리뷰",
#     ],
#     "cafearticle": [
#         "<주제> 카페 글",
#         "<동호회> 카페 글",
#         "<맘카페> 정보",
#         "<부동산> 카페 글",
#         "<자동차> 동호회",
#         "<여행/요리/취미> 카페",
#         "<스터디> 카페 글",
#         "<운동> 카페 후기",
#         "<게임> 공략 카페",
#         "<IT> 질문 카페",
#         "<반려동물> 카페",
#         "<주식/재테크> 카페",
#         "<창업/취업> 카페",
#         "<시험/자격증> 카페",
#         "<학원> 정보 카페",
#         "<지역> 카페 글",
#         "<중고 거래> 카페",
#         "<강좌/세미나> 공지",
#         "<커뮤니티> 핫게시글",
#         "<질문/답변> 스레드",
#     ],
#     "image": [
#         "<동물> 사진",
#         "<풍경/자연> 이미지",
#         "<음식> 사진",
#         "<인테리어/건축> 이미지",
#         "<패션> 사진",
#         "<로고/아이콘> 이미지",
#         "<일러스트/배경화면> 이미지",
#         "<포스터/그래픽> 디자인",
#         "<여행지/도시 야경> 사진",
#         "<제품/메뉴> 이미지",
#         "<인물/초상> 사진",
#         "<예술 작품> 이미지",
#         "<자동차/기계> 사진",
#         "<스포츠> 사진",
#         "<과학/우주> 이미지",
#         "<해양/바다> 사진",
#         "<항공/드론> 사진",
#         "<흑백/필름> 사진",
#         "<타이포그래피> 이미지",
#         "<패턴/텍스처> 이미지",
#     ],
#     "kin": [
#         "<주제> 질문",
#         "<프로그래밍/영어/수학> 질문",
#         "<컴퓨터 오류> 해결 질문",
#         "<법률/의료> 상담 질문",
#         "<여행/육아> 조언 질문",
#         "<반려동물> 질문",
#         "<부동산/재테크> 질문",
#         "<창업/취업/입시> 질문",
#         "<건강/운동> 질문",
#         "<취미 시작> 질문",
#         "<생활 정보> 질문",
#         "<음식/요리> 질문",
#         "<문화/예술> 질문",
#         "<자동차> 질문",
#         "<가전/제품> 질문",
#         "<게임> 질문",
#         "<교육/학습> 질문",
#         "<소비/금융> 질문",
#         "<과학/기술> 질문",
#         "<커리어> 질문",
#     ],
#     "book": [
#         "<주제> 입문서",
#         "<장르> 소설 추천",
#         "<장르> 에세이",
#         "<분야> 경영/자기계발",
#         "<분야> 역사/과학",
#         "<분야> 철학/심리",
#         "<분야> 요리/여행",
#         "<분야> 외국어 학습",
#         "<분야> 수험서/자격증",
#         "<분야> 아동/청소년",
#         "<장르> 시집",
#         "<분야> 사진/예술",
#         "<분야> IT 기술",
#         "<저자/시리즈> 추천",
#         "<번역/원서> 선택",
#         "<출판사> 신간",
#         "<테마> 큐레이션",
#         "<오디오북/전자책> 추천",
#         "<독서 리스트> 구성",
#         "<학습 로드맵> 도서",
#     ],
#     "encyc": [
#         "<학문/역사/문화> 백과사전",
#         "<동물/식물> 백과사전",
#         "<지리/인물> 백과사전",
#         "<음악/영화/스포츠> 백과사전",
#         "<의학/법률/경제> 백과사전",
#         "<IT/건축/요리/패션> 백과사전",
#         "<철학/종교> 백과사전",
#         "<수학/물리/화학> 백과사전",
#         "<언어학/교육학/사회학> 백과사전",
#         "<환경/에너지> 백과사전",
#         "<항공/우주> 백과사전",
#         "<통계/데이터> 백과사전",
#         "<미술/디자인> 백과사전",
#         "<문학/고전> 백과사전",
#         "<컴퓨팅/보안> 백과사전",
#         "<심리/뇌과학> 백과사전",
#         "<법학/정치학> 백과사전",
#         "<경영/마케팅> 백과사전",
#         "<토목/기계/전자> 백과사전",
#         "<교육/평생학습> 백과사전",
#     ],
#     "academic": [
#         "<분야> 논문",
#         "<주제> 연구",
#         "<학회/저널> 논문",
#         "<연구자/랩> 출판물",
#         "<데이터셋/벤치마크> 논문",
#         "<리뷰/서베이> 논문",
#         "<실험> 방법론",
#         "<결과> 해석",
#         "<이론> 증명",
#         "<응용> 사례",
#         "<재현성> 코드/자료",
#         "<실험 설계> 논의",
#         "<오픈소스> 구현",
#         "<후속 연구> 제안",
#         "<도구/프레임워크> 비교",
#         "<통계 분석> 방법",
#         "<수학적 모델> 소개",
#         "<윤리/보안> 이슈",
#         "<산업 적용> 사례",
#         "<국제 협력> 프로젝트",
#     ],
# }


def pick_endpoint(query: str) -> str:
    """
    의미 기반 엔드포인트 라우팅 (단일 엔드포인트 반환)

    Args:
        query: 사용자 검색 쿼리

    Returns:
        str: 가장 적합한 엔드포인트 ('local', 'news', 'webkr')
    """
    if not query or not query.strip():
        return "webkr"

    try:
        if not ENDPOINT_SEED_VECTORS:
            raise RuntimeError("시드 벡터가 초기화되지 않았습니다.")

        import numpy as np

        # 쿼리 임베딩 API 1회 호출
        qv = embed_query_cached(query)
        qn = float(np.linalg.norm(qv) or 1.0)

        # 각 엔드포인트의 모든 시드와 유사도 계산
        endpoint_scores = {}

        for endpoint, seeds in ENDPOINT_SEEDS.items():
            # 해당 엔드포인트의 미리 계산된 벡터 목록을 캐시에서 가져옵니다.
            precomputed_vectors = ENDPOINT_SEED_VECTORS.get(endpoint, [])
            if not precomputed_vectors:
                continue
            scores = []
            for sv in precomputed_vectors:  # API 호출 대신 캐시된 벡터 사용
                sn = float(np.linalg.norm(sv) or 1.0)
                sim = float(np.dot(qv, sv) / (qn * sn))
                scores.append(sim)

            # Top-5 평균으로 엔드포인트 스코어 계산
            scores.sort(reverse=True)
            avg_score = float(np.mean(scores[:5]))
            endpoint_scores[endpoint] = avg_score

        # 금융 의도 가드레일: 시세/주가/가격/환율 등은 news 대신 webkr 유지(가격 링크/금융 블록으로 대체)
        ql = (query or "").lower()
        finance_hint = any(
            k in ql
            for k in (
                "주가",
                "시세",
                "실시간 가격",
                "현재 가격",
                "가격",
                "환율",
                "종가",
            )
        )

        # 최고 스코어 엔드포인트 선택
        best_endpoint = max(endpoint_scores, key=endpoint_scores.get)
        if finance_hint:
            best_endpoint = "webkr"
        logger.info(
            f"[router:semantic] query='{query[:60]}' scores={endpoint_scores} → {best_endpoint}"
        )

        return best_endpoint

    except Exception as e:
        logger.warning(f"[router:semantic] 실패, webkr 폴백: {e}")
        return "webkr"


def pick_endpoints(
    query: str, last_loc: str | None = None, max_k: int = 2
) -> List[str]:
    """
    의미 기반 엔드포인트 라우팅 (다중 엔드포인트 반환)

    Args:
        query: 사용자 검색 쿼리
        last_loc: 마지막 위치 정보 (있으면 local 우선)
        max_k: 반환할 엔드포인트 개수

    Returns:
        List[str]: 우선순위 순서의 엔드포인트 리스트
    """
    if not query or not query.strip():
        return ["webkr"]

    try:
        if embed_query_cached is None:
            raise RuntimeError("임베딩 백엔드 없음")

        import numpy as np

        # 쿼리 임베딩
        qv = embed_query_cached(query)
        qn = float(np.linalg.norm(qv) or 1.0)

        # 각 엔드포인트의 모든 시드와 유사도 계산 (사전 계산 캐시 사용)
        endpoint_scores = {}

        for endpoint, _seeds in ENDPOINT_SEEDS.items():
            precomputed_vectors = ENDPOINT_SEED_VECTORS.get(endpoint, [])
            if not precomputed_vectors:
                continue
            scores = []
            for sv in precomputed_vectors:
                sn = float(np.linalg.norm(sv) or 1.0)
                sim = float(np.dot(qv, sv) / (qn * sn))
                scores.append(sim)

            # Top-5 평균으로 엔드포인트 스코어 계산
            scores.sort(reverse=True)
            avg_score = float(np.mean(scores[:5]))
            endpoint_scores[endpoint] = avg_score

        # 스코어 기준 정렬
        sorted_endpoints = sorted(endpoint_scores.items(), key=lambda x: -x[1])

        result = [ep for ep, _ in sorted_endpoints[:max_k]]

        # last_loc가 있으면 local을 반드시 포함
        if last_loc and "local" not in result:
            if len(result) >= max_k:
                result[-1] = "local"
            else:
                result.append("local")

        # 중복 제거
        seen = set()
        unique_result = []
        for ep in result:
            if ep not in seen:
                unique_result.append(ep)
                seen.add(ep)

        logger.info(
            f"[router:semantic:multi] query='{query[:60]}' scores={endpoint_scores} → {unique_result}"
        )

        return unique_result or ["webkr"]

    except Exception as e:
        logger.warning(f"[router:semantic:multi] 실패, webkr 폴백: {e}")
        return ["webkr"]
