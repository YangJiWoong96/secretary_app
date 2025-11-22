"""
backend.config.settings - 환경변수 및 설정 관리

모든 환경변수를 Pydantic Settings 클래스로 통합 관리.
싱글톤 패턴으로 전역 접근 제공.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger("config")


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정 클래스

    환경변수(.env 파일 또는 시스템 환경변수)에서 자동으로 값을 로드하며,
    기본값이 지정된 경우 환경변수가 없어도 작동합니다.
    """

    # ===== OpenAI 설정 =====
    OPENAI_API_KEY: str = Field(..., description="OpenAI API 키 (필수)")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="기본 LLM 모델")
    THINKING_MODEL: str = Field(
        default="gpt-5-thinking", description="이성 추론형 모델"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="임베딩 모델"
    )
    # ===== 임베딩 캐시 설정 =====
    EMBEDDING_CACHE_TTL: int = Field(default=86400, description="임베딩 캐시 TTL(초)")
    EMBEDDING_MODEL_VERSION: str = Field(default="v1", description="임베딩 모델 버전")

    # ===== 외부 API 키 (Naver 등) =====
    CLIENT_ID: Optional[str] = Field(default=None, description="Naver API Client ID")
    CLIENT_SECRET: Optional[str] = Field(
        default=None, description="Naver API Client Secret"
    )

    # ===== 데이터베이스 설정 =====
    MILVUS_HOST: str = Field(default="localhost", description="Milvus 호스트")
    MILVUS_PORT: str = Field(default="19530", description="Milvus 포트")
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Redis 연결 URL"
    )
    MILVUS_RESET_PROFILE_CHUNKS: bool = Field(
        default=False, description="profile_chunks 강제 초기화 플래그"
    )

    # ===== Firestore 설정 =====
    FIRESTORE_ENABLE: bool = Field(default=True, description="Firestore 사용 여부")
    FIRESTORE_USERS_COLL: str = Field(
        default="users", description="Firestore 사용자 컬렉션명"
    )
    FIRESTORE_EVENTS_SUB: str = Field(
        default="unified_events", description="Firestore 이벤트 서브컬렉션명"
    )
    GCP_SERVICE_ACCOUNT_PATH: Optional[str] = Field(
        default=None, description="GCP 서비스 계정 키 파일 경로"
    )

    # ===== MCP 서버 =====
    MCP_SERVER_URL: str = Field(default="http://mcp:5000", description="MCP 서버 URL")

    # ===== 타임아웃 설정 =====
    TIMEOUT_MOBILE: float = Field(
        default=1.0, description="모바일 컨텍스트 조회 타임아웃(초)"
    )
    TIMEOUT_RAG: float = Field(default=3.5, description="RAG 검색 타임아웃(초)")
    TIMEOUT_WEB: float = Field(default=3.5, description="웹 검색 타임아웃(초)")
    REWRITE_TIMEOUT_S: float = Field(
        default=1.25, description="쿼리 재작성 타임아웃(초)"
    )
    EXTRACT_TIMEOUT_S: float = Field(default=0.7, description="앵커 추출 타임아웃(초)")
    FOLLOWUP_TIMEOUT_S: float = Field(
        default=0.7, description="팔로업 판별 타임아웃(초)"
    )
    PREVALIDATE_TIMEOUT_S: float = Field(
        default=0.9, description="사전 검증 타임아웃(초)"
    )

    # ===== 토큰 및 모델 설정 =====
    REWRITE_MAX_TOKENS: int = Field(default=128, description="재작성 최대 토큰 수")
    REWRITE_MODEL: str = Field(default="gpt-4o-mini", description="재작성용 모델")

    # ===== 담화 앵커 및 힌트 설정 =====
    TOPIC_TTL_S: int = Field(default=600, description="앵커 TTL(초)")
    HINT_LOOKBACK: int = Field(default=8, description="힌트 추출 시 최근 발화 개수")
    HINT_MAX_ITEMS: int = Field(default=2, description="힌트 최대 라인 수")
    HINT_SIM_THRESHOLD: float = Field(default=0.25, description="코사인 유사도 임계값")

    # ===== 라우팅 임계값 =====
    TAU_RAG: float = Field(default=0.44, description="RAG 라우팅 임계값")
    TAU_WEB: float = Field(default=0.44, description="WEB 라우팅 임계값")
    AMBIGUITY_BAND: float = Field(default=0.03, description="애매 구간 대역폭")

    # ===== 캘리브레이션 설정 =====
    CAL_WEB: Dict[str, float] = Field(
        default={"a": 1.0, "b": 0.0, "T": 1.00},
        description="웹 라우팅 Platt 캘리브레이션 파라미터",
    )
    CAL_RAG: Dict[str, float] = Field(
        default={"a": 1.0, "b": 0.0, "T": 1.00},
        description="RAG 라우팅 Platt 캘리브레이션 파라미터",
    )

    # ===== 프라이어 패턴 (휴리스틱) =====
    WEB_PRIOR_PAT: str = Field(
        default=r"(추천|근처|가까운|영업시간|가격|리뷰|랭킹|뉴스|최신|주소|전화)",
        description="웹 검색 프라이어 정규식 패턴",
    )
    RAG_PRIOR_PAT: str = Field(
        default=r"(내 문서|사내|정책|내 일정|프로젝트|노트|요약했|회의록|RAG|내정보)",
        description="RAG 검색 프라이어 정규식 패턴",
    )

    # ===== 대화 모드 설정 =====
    SINGLE_CALL_CONV: bool = Field(
        default=True, description="순수 대화 모드 단일 호출 여부"
    )
    USE_LLM_ROUTER: bool = Field(default=True, description="LLM 라우터 사용 여부")
    STREAM_ENABLED: bool = Field(default=True, description="스트리밍 응답 사용 여부")
    OPENAI_PREFIX_CACHE_ENABLED: bool = Field(
        default=False, description="OpenAI 프리픽스 캐시 메타 사용 여부"
    )

    # ===== Redis/요약/스냅샷 정책 상수 =====
    MAX_TOKEN_LIMIT: int = Field(
        default=3000, description="Redis 단기 메모리 하드 한도"
    )
    RECENT_RAW_TOKENS_BUDGET: int = Field(
        default=1500, description="최근 원문 보존 예산"
    )
    SUMMARY_TARGET_TOKENS: int = Field(default=500, description="축약 목표 토큰 수")
    SYSTEM_TOOL_BUDGET: int = Field(default=200, description="시스템/툴 메타 여유분")
    EDGE_THRESHOLD: int = Field(default=3000, description="에지 트리거 임계치")
    DEBOUNCE_TURNS: int = Field(default=5, description="배치 최소 턴 수")
    DEBOUNCE_SECONDS: int = Field(default=60, description="배치 최소 시간(초)")
    SNAPSHOT_QUEUE_MAXSIZE: int = Field(default=128, description="스냅샷 작업 큐 크기")
    WORKER_CONCURRENCY: int = Field(default=2, description="동시 스냅샷 처리 워커 수")
    EMBED_CONCURRENCY: int = Field(default=2, description="임베딩 동시 제한")

    # ===== Evidence 채널 설정 =====
    RAG_THR: float = Field(default=0.35, description="RAG 점수 임계값")
    RAG_MMR_LAMBDA: float = Field(default=0.6, description="RAG MMR 다양성 파라미터")
    EVIDENCE_TOKEN_CAP: int = Field(default=1600, description="Evidence 채널 토큰 버짓")
    RAG_SIM_METRIC: str = Field(default="cosine", description="RAG 유사도 메트릭")
    WEB_THR: float = Field(default=0.25, description="웹 검색 점수 임계값")
    EVIDENCE_RATIO: float = Field(default=0.35, description="Evidence 채널 비율")
    EVIDENCE_HEADROOM: int = Field(default=256, description="Evidence 헤드룸 토큰")
    CTX_WINDOW: int = Field(default=8192, description="컨텍스트 윈도우(토큰)")
    RAG_DECAY_LAM_LOG: float = Field(default=0.02, description="로그 채널 감쇠 람다")
    RAG_DECAY_LAM_PROFILE: float = Field(
        default=0.01, description="프로필 채널 감쇠 람다"
    )

    # ===== 중복 억제/신규성 게이트 설정 =====
    SNAPSHOT_EDGE_TOKENS: int = Field(
        default=4500, description="스냅샷 적재 트리거 토큰 수"
    )
    NOVELTY_SIM_THRESHOLD: float = Field(
        default=0.92, description="로그 근사중복 억제 임계값"
    )
    NOVELTY_MIN_PROFILE_DELTA: int = Field(
        default=1, description="프로필 신규 항목 최소 개수"
    )
    SNAPSHOT_LOOKBACK_MONTHS: int = Field(
        default=1, description="근사중복 탐색 기간(월)"
    )

    # ===== MTM/Coordinator 트리거 설정 =====
    STM_TO_MTM_TOKENS: int = Field(default=500, description="STM→MTM 토큰 임계값")
    STM_TO_MTM_TURNS: int = Field(default=5, description="STM→MTM 최소 턴 수")
    MTM_TO_LTM_TOKENS: int = Field(default=4500, description="MTM→LTM 토큰 임계값")
    MTM_TO_LTM_TURNS: int = Field(default=3, description="MTM→LTM 최소 턴 수")
    MTM_TO_LTM_DEBOUNCE_SEC: int = Field(default=120, description="LTM 디바운스(초)")
    MTM_TTL_DAYS: int = Field(default=7, description="MTM 요약 TTL(일)")
    MTM_MAX_SUMMARIES: int = Field(default=60, description="MTM 최대 요약 개수")
    # MTM 주제 중복 제거 임베딩 유사도 임계값
    # 0.85 = 매우 유사한 주제만 중복으로 판단 (권장)
    # 0.75 = 유사한 주제 중복 제거 (더 공격적)
    # 0.95 = 거의 동일한 주제만 중복 제거 (보수적)
    MTM_TOPIC_SIMILARITY_THRESHOLD: float = Field(
        default=0.85, 
        description="MTM 주제 중복 제거 임베딩 유사도 임계값"
    )
    MEMORY_COORDINATOR_ENABLED: bool = Field(
        default=True, description="단일 진입점 코디네이터 사용 여부"
    )

    # ===== 재시도 설정 =====
    MAX_RETRIES_OPENAI: int = Field(
        default=2, description="OpenAI API 최대 재시도 횟수"
    )
    MAX_RETRIES_HTTP: int = Field(default=2, description="HTTP 요청 최대 재시도 횟수")
    RETRY_BASE_DELAY: float = Field(
        default=0.25, description="재시도 기본 지연 시간(초)"
    )

    # ===== 디버그 설정 =====
    DEBUG_META: bool = Field(default=False, description="메타 디버그 정보 출력 여부")
    WS_DEBUG_META: bool = Field(
        default=False, description="WebSocket 디버그 메타 출력 여부"
    )
    LOG_LEVEL: str = Field(default="INFO", description="로깅 레벨")

    # ===== 프로필 스코어링/시간 감쇠 설정 (Session 5) =====
    PROFILE_DECAY_HALF_LIFE_CORE: float = Field(
        default=90.0, description="Core 프로필 시간 반감기(일)"
    )
    PROFILE_DECAY_HALF_LIFE_DYNAMIC: float = Field(
        default=14.0, description="Dynamic 프로필 시간 반감기(일)"
    )
    PROFILE_DECAY_MIN_CORE: float = Field(
        default=0.6, description="Core 프로필 시간 감쇠 최소 계수"
    )
    PROFILE_DECAY_MIN_DYNAMIC: float = Field(
        default=0.3, description="Dynamic 프로필 시간 감쇠 최소 계수"
    )
    SCORE_ALPHA: float = Field(default=1.0, description="최종 스코어 지수 α (sim)")
    SCORE_BETA: float = Field(default=1.0, description="최종 스코어 지수 β (decay)")
    SCORE_GAMMA: float = Field(default=1.0, description="최종 스코어 지수 γ (priority)")
    SCORE_DELTA: float = Field(default=0.7, description="최종 스코어 지수 δ (interest)")
    PRIORITY_CAP: float = Field(default=2.2, description="우선순위 가중치 상한")

    # ===== Bot Profile(정적 기본) =====
    BOT_PERSONA: str = Field(default="한국어 개인 비서", description="봇 기본 페르소나")
    BOT_STYLE: str = Field(default="neutral", description="봇 기본 스타일")
    BOT_ABILITIES: str = Field(default="", description="봇 능력(콤마구분)")

    # ===== Adaptive Budget Caps =====
    BUDGET_EVIDENCE_MIN: int = Field(default=1000, description="Evidence 최소 토큰")
    BUDGET_EVIDENCE_CAP: int = Field(default=1600, description="Evidence 최대 토큰")
    BUDGET_MEMORY_MIN: int = Field(default=400, description="Memory 최소 토큰")
    BUDGET_MEMORY_CAP: int = Field(default=800, description="Memory 최대 토큰")
    BUDGET_PROFILE_MIN: int = Field(default=200, description="Profile 최소 토큰")
    BUDGET_PROFILE_CAP: int = Field(default=400, description="Profile 최대 토큰")

    # ===== Feature Flags =====
    OUTPUT_PRUNING_ENABLED: bool = Field(
        default=True, description="출력 프루닝 사용 여부"
    )
    FEATURE_CACHE_GUARD: bool = Field(default=True, description="Cache Guard 사용 여부")
    FEATURE_BANDIT: bool = Field(default=True, description="밴딧 사용 여부")
    FEATURE_PROACTIVE_SCORING: bool = Field(
        default=True, description="프로액티브 점수 게이트 사용"
    )
    FEATURE_SERENDIPITY: bool = Field(
        default=True, description="세렌디피티 재랭킹 사용"
    )
    FEATURE_DEVICE_STATE: bool = Field(default=True, description="디바이스 상태 사용")
    FEATURE_WHY_TAG: bool = Field(default=True, description="Why-Tag 생성 사용")
    FEATURE_DOMAIN_ROUTING: bool = Field(default=True, description="도메인 라우팅 사용")
    FEATURE_EVIDENCE_CROSSCHECK: bool = Field(
        default=True, description="증거 교차검증 사용"
    )
    # ---- YouTube/멀티에이전트 전용 플래그 ----
    FEATURE_YOUTUBE_TRANSCRIPT: bool = Field(
        default=True, description="YouTube 자막 추출 사용"
    )
    FEATURE_YOUTUBE_STT: bool = Field(
        default=False, description="YouTube 자막 없을 때 STT 폴백 사용"
    )

    # ===== 멀티에이전트 컨텍스트 길이 상한(정확도 우선) =====
    MA_WEB_CONTENT_MAX_CHARS: int = Field(
        default=12000, description="MA 웹 텍스트 상한(문자)"
    )
    MA_YT_CONTENT_MAX_CHARS: int = Field(
        default=20000, description="MA 유튜브 텍스트 상한(문자)"
    )

    # ===== 검색/캐시/로컬 탐색 =====
    SEARCH_TTL_SEC: float = Field(default=120.0, description="검색 결과 TTL(초)")
    CACHE_TTL_SEC: float = Field(
        default=1800.0, description="애플리케이션 캐시 TTL(초)"
    )
    LOCAL_RADIUS_KM: float = Field(default=1.0, description="로컬 검색 반경(KM)")

    # ===== 모델/컨텍스트 창 =====
    MODEL_CONTEXT_WINDOW: int = Field(
        default=16384, description="모델 컨텍스트 창 크기"
    )
    RESPONSE_RESERVE_TOKENS: int = Field(default=2048, description="응답 예약 토큰")

    # ===== 라우터(임계/탑K/온도) =====
    ROUTER_SEED_TOPK: int = Field(default=8, description="라우터 시드 TopK")
    ROUTER_TOPK_WEIGHT: float = Field(default=0.5, description="라우터 TopK 가중")
    ROUTER_TOPK_EXAMPLE_WEIGHTED: bool = Field(
        default=True, description="예시 가중 사용"
    )
    ROUTER_TOPK_WEIGHT_MIN: float = Field(default=0.2, description="예시 가중 최소치")
    ROUTER_USE_ZSCORE: bool = Field(default=True, description="Z-Score 사용")
    ROUTER_TEMP: float = Field(default=0.5, description="라우터 기본 온도")
    ROUTER_ADAPTIVE_TEMP: bool = Field(default=True, description="적응형 온도 사용")
    RAG_RECALL_THR: float = Field(default=0.44, description="RAG 리콜 임계")
    RAG_RECALL_MARGIN: float = Field(default=0.03, description="RAG 리콜 마진")

    # ===== 생성기/밸리데이터 =====
    SHOW_CITATION: bool = Field(default=False, description="출력에 출처 표시")
    # 우선순위: VALIDATOR_TIMEOUT_S > VALIDATION_TIMEOUT_S > default(1.8)
    VALIDATOR_TIMEOUT_S: Optional[float] = Field(
        default=None, description="검증 타임아웃(초, 최우선)"
    )
    VALIDATION_TIMEOUT_S: Optional[float] = Field(
        default=None, description="검증 타임아웃(초)"
    )

    # ===== Proactive/Notification =====
    PROACTIVE_DAILY_LIMIT: int = Field(default=3, description="하루 푸시 상한")
    PROACTIVE_MIN_INTERVAL_MIN: int = Field(
        default=45, description="최소 푸시 간격(분)"
    )
    PROACTIVE_RANKER_VARIANT: str = Field(
        default="gbdt_v1", description="프로액티브 랭커 버전"
    )
    PROACTIVE_VARIANT: str = Field(
        default="baseline_v1", description="알림 생성 파이프라인 버전"
    )
    PROACTIVE_SCORE_THRESHOLD: float = Field(
        default=0.55, description="프로액티브 점수 임계"
    )
    NOTIF_MODEL: str = Field(default="gpt-4o-mini", description="알림 생성 모델")

    # ===== Why-Tag =====
    WHY_MAX_LEN: int = Field(default=120, description="Why 태그 최대 길이")
    WHY_SENS_FILTER: str = Field(default="low", description="민감도 필터")

    # ===== Evidence Ref/Archive =====
    EVIDENCE_REF_ENABLED: bool = Field(
        default=True, description="Evidence 참조 기능 활성화"
    )
    EVIDENCE_MAX_CLAIMS: int = Field(default=6, description="최대 증거 클레임 수")
    EVIDENCE_TTL_TURNS: int = Field(default=4, description="증거 TTL(턴)")
    DELAYED_ARCHIVE_ENABLED: bool = Field(
        default=True, description="지연 아카이브 사용"
    )
    ARCHIVE_EXPIRY_TURNS: int = Field(default=3, description="아카이브 만료 턴")

    # ===== 관측/로깅 =====
    OBSERVABILITY_SAMPLE_SEED: Optional[str] = Field(
        default=None, description="샘플링 시드"
    )
    OBSERVABILITY_ENABLED: bool = Field(default=True, description="관측 활성화")
    LOG_GUARD: bool = Field(default=True, description="로그 가드(민감정보 차단)")
    TRACE_TEXT: bool = Field(default=False, description="긴 텍스트 상세 로깅")

    # ===== Adaptive Budget =====
    ADAPTIVE_BUDGET_TUNING_ENABLED: bool = Field(
        default=True, description="적응형 버짓 튜닝"
    )

    # ===== Compiler / Profile =====
    PROFILE_TIER_ON_DEMAND: bool = Field(
        default=True, description="프로필 온디맨드 로딩"
    )
    CTX_COMPRESS_ENABLED: bool = Field(default=True, description="컨텍스트 압축 사용")
    UNIFIED_COMPILER_ENABLED: bool = Field(
        default=True, description="통합 컴파일러 사용"
    )

    # ===== Behavior Slots =====
    BEHAVIOR_ENABLED: bool = Field(
        default=True, description="Behavior Slot 기능 활성화"
    )
    BEHAVIOR_UPSERT_TO_RAG: bool = Field(
        default=True, description="Behavior를 Milvus RAG에 업서트"
    )
    BEHAVIOR_TOPK_OVERLAY: int = Field(default=3, description="Overlay 표시 TopK")

    # ===== Directives 설정 =====
    DIR_QUEUE_MAX: int = Field(default=128, description="디렉티브 큐 최대 크기")
    DIR_WORKERS: int = Field(default=2, description="디렉티브 워커 수")
    DIR_DEBOUNCE_TURNS: int = Field(default=5, description="큐 디바운스 턴")
    DIR_ENQ_GUARD_S: int = Field(default=10, description="엔큐 가드(초)")
    DIR_USE_REDIS_LOCK: bool = Field(default=False, description="Redis 락 사용")
    DIR_LOCK_EX_S: int = Field(default=10, description="락 TTL(초)")
    DIR_CONF_THRESH: float = Field(default=0.55, description="업데이트 임계")
    DIR_COOLDOWN_S: int = Field(default=3600, description="쿨다운(초)")
    DIR_EMA_ALPHA: float = Field(default=0.35, description="EMA 알파")
    DIR_MAX_CHANGES: int = Field(default=3, description="필드 최대 변경 횟수")
    DIR_EXTRACT_TIMEOUT_S: float = Field(
        default=5.0, description="디렉티브 추출 타임아웃"
    )
    DIR_TTL_SEC: int = Field(default=604800, description="디렉티브 보존 TTL(초)")
    DIR_SCHEDULE_ENABLED: bool = Field(default=True, description="디렉티브 스케줄 사용")
    DIR_SCHEDULE_HOUR_KST: int = Field(default=3, description="스케줄 시간(KST)")
    DIR_BATCH_DELAY_MS: int = Field(default=50, description="배치 딜레이(ms)")

    # ===== STWM/Gazetteer =====
    STWM_TTL_MIN: int = Field(default=60, description="STWM TTL(분)")
    STWM_MAX_SLOTS: int = Field(default=30, description="STWM 슬롯 수")
    STWM_FLUSH_BATCH: int = Field(default=10, description="STWM 플러시 배치")
    STWM_USE_MECAB: bool = Field(default=True, description="MeCab 사용")
    STWM_USE_KONER: bool = Field(default=True, description="KoNER 사용")
    STWM_NER_DEBUG: bool = Field(default=False, description="NER 디버그 출력")
    STWM_CARRY_STRICT: float = Field(default=0.70, description="캐리-Strict 임계")
    STWM_CARRY_MED: float = Field(default=0.50, description="캐리-Medium 임계")
    STWM_RET_TTL_SEC: int = Field(default=600, description="STWM Retrieval TTL(초)")
    STWM_RET_TAU: float = Field(default=0.55, description="STWM Retrieval 임계")
    GAZ_EMB_WEIGHT: float = Field(default=0.8, description="Gazetteer 임베딩 가중")

    # ===== 기타 유틸 =====
    COMPRESS_MAX_CONCURRENCY: int = Field(default=4, description="컴프레서 동시성")
    COMPRESS_TIMEOUT_S: float = Field(default=0.8, description="컴프레서 타임아웃(초)")

    # ===== Bot Profile Dynamic Update =====
    DYN_BP_MIN_EVENTS: int = Field(
        default=10, description="동적 BotProfile 최소 이벤트 수(7d)"
    )
    DYN_BP_MIN_INTERVAL_SEC: int = Field(
        default=86400, description="동적 BotProfile 최소 갱신 간격(초)"
    )

    # ===== 외부 API(선택) =====
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google API Key")
    CSE_CX: Optional[str] = Field(default=None, description="Google CSE CX")
    KAKAO_REST_API_KEY: Optional[str] = Field(
        default=None, description="Kakao REST API Key"
    )
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(
        default=None, description="Alpha Vantage API Key (finance quotes)"
    )
    REDDIT_CLIENT_ID: Optional[str] = Field(
        default=None, description="Reddit Client ID"
    )
    REDDIT_CLIENT_SECRET: Optional[str] = Field(
        default=None, description="Reddit Client Secret"
    )
    REDDIT_USER_AGENT: Optional[str] = Field(
        default=None, description="Reddit User Agent"
    )

    class Config:
        # .env 파일에서 자동으로 환경변수 로드
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 환경변수가 없을 때 대소문자 구분 없이 검색
        case_sensitive = False
        extra = "ignore"

    @validator("FIRESTORE_ENABLE", pre=True)
    def parse_firestore_enable(cls, v):
        """FIRESTORE_ENABLE을 bool로 변환 (문자열 "0", "1" 지원)"""
        if isinstance(v, str):
            return bool(int(v))
        return bool(v)

    @validator(
        "DEBUG_META",
        "WS_DEBUG_META",
        "SINGLE_CALL_CONV",
        "USE_LLM_ROUTER",
        "STREAM_ENABLED",
        "OPENAI_PREFIX_CACHE_ENABLED",
        "MEMORY_COORDINATOR_ENABLED",
        "OUTPUT_PRUNING_ENABLED",
        "FEATURE_CACHE_GUARD",
        "FEATURE_BANDIT",
        "FEATURE_PROACTIVE_SCORING",
        "FEATURE_SERENDIPITY",
        "FEATURE_DEVICE_STATE",
        "FEATURE_WHY_TAG",
        "FEATURE_DOMAIN_ROUTING",
        "FEATURE_EVIDENCE_CROSSCHECK",
        "FEATURE_YOUTUBE_TRANSCRIPT",
        "FEATURE_YOUTUBE_STT",
        "OBSERVABILITY_ENABLED",
        "LOG_GUARD",
        "ADAPTIVE_BUDGET_TUNING_ENABLED",
        "PROFILE_TIER_ON_DEMAND",
        "CTX_COMPRESS_ENABLED",
        "UNIFIED_COMPILER_ENABLED",
        "DIR_USE_REDIS_LOCK",
        "DIR_SCHEDULE_ENABLED",
        "EVIDENCE_REF_ENABLED",
        "DELAYED_ARCHIVE_ENABLED",
        "ROUTER_TOPK_EXAMPLE_WEIGHTED",
        "ROUTER_USE_ZSCORE",
        "ROUTER_ADAPTIVE_TEMP",
        "STWM_USE_MECAB",
        "STWM_USE_KONER",
        "STWM_NER_DEBUG",
        pre=True,
    )
    def parse_bool_flags(cls, v):
        """불린 플래그를 문자열 "0", "1"에서 변환"""
        if isinstance(v, str):
            return bool(int(v))
        return bool(v)

    @validator("VALIDATION_TIMEOUT_S", pre=True)
    def parse_validation_timeout(cls, v):
        """
        VALIDATION_TIMEOUT_S가 미설정이면 호환 키 VALIDATION_TIMEOUT_SEC를 조회.
        """
        if v is not None:
            return float(v)
        alt = os.getenv("VALIDATION_TIMEOUT_SEC")
        return float(alt) if alt is not None else None

    def setup_firestore_credentials(self) -> None:
        """
        Firestore 서비스 계정 키 경로를 자동으로 설정

        우선순위:
        1. 환경변수 GCP_SERVICE_ACCOUNT_PATH
        2. GOOGLE_APPLICATION_CREDENTIALS (유효성 체크)
        3. 프로젝트 루트의 gcp-service-account-key.json
        """
        if not self.FIRESTORE_ENABLE:
            return

        # 기본 경로: backend/gcp-service-account-key.json
        default_key_path = self.GCP_SERVICE_ACCOUNT_PATH
        if not default_key_path:
            # app.py가 있는 backend 디렉토리 기준
            # backend_dir = Path(__file__).resolve().parent.parent
            # default_key_path = str(backend_dir / "gcp-service-account-key.json")
            project_root = Path(__file__).resolve().parent.parent.parent
            default_key_path = str(project_root / "gcp-service-account-key.json")

        current_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # 1) 현재 설정이 유효하지 않으면 교정 시도
        if (
            current_creds
            and not Path(current_creds).exists()
            and Path(default_key_path).exists()
        ):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_key_path
            logger.info(
                f"[fs] GOOGLE_APPLICATION_CREDENTIALS override -> {default_key_path}"
            )

        # 2) 미설정이면 기본 경로로 셋업
        elif (not current_creds) and Path(default_key_path).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_key_path
            logger.info(
                f"[fs] GOOGLE_APPLICATION_CREDENTIALS set -> {default_key_path}"
            )

        # 3) 파일이 없으면 경고
        elif not Path(default_key_path).exists():
            logger.warning(f"[fs] service account key not found at {default_key_path}")

    def log_startup_info(self) -> None:
        """시작 시 주요 설정 정보 로깅"""
        logger.info(
            f"[boot] model={self.LLM_MODEL} embed={self.EMBEDDING_MODEL} "
            f"milvus={self.MILVUS_HOST}:{self.MILVUS_PORT} redis={self.REDIS_URL}"
        )
        logger.info(
            f"[boot] thresholds tau_rag={self.TAU_RAG} tau_web={self.TAU_WEB} "
            f"ambiguity={self.AMBIGUITY_BAND} timeouts web={self.TIMEOUT_WEB}s rag={self.TIMEOUT_RAG}s"
        )

    # ===== 호환/별칭 보조 속성 =====
    @property
    def RAG_THRESHOLD(self) -> float:
        """
        하위 호환: 기존 RAG_THR 또는 RAG_THRESHOLD 개념을 단일 진입점으로 제공.
        """
        try:
            return float(self.RAG_THR)
        except Exception:
            return 0.35


# ===== 싱글톤 인스턴스 =====
_settings_instance: Optional[Settings] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    전역 Settings 싱글톤 인스턴스를 반환

    애플리케이션 전역에서 동일한 설정 인스턴스를 공유하며,
    최초 호출 시 환경변수를 로드하고 Firestore 자격증명을 설정합니다.

    Returns:
        Settings: 전역 설정 인스턴스

    Example:
        >>> from backend.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.LLM_MODEL)
        'gpt-4o-mini'
    """
    global _settings_instance

    if _settings_instance is None:
        # 프로젝트 루트의 .env를 선제 로드하여 작업 디렉터리에 의존하지 않도록 함
        try:
            root_dir = Path(__file__).resolve().parents[2]
            env_path = root_dir / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception:
            pass

        _settings_instance = Settings()
        _settings_instance.setup_firestore_credentials()
        _settings_instance.log_startup_info()

    return _settings_instance
