"""
backend.rag.profile_schema - 프로필 스키마 정의 및 검증

목적:
- key_path 화이트리스트 (Enum 기반)
- LLM이 생성한 key_path를 서버에서 검증
- 환각된 키 차단
- Milvus 컬렉션 스키마 및 인덱스 정의

구조:
- ProfileKeyPath: 사전 정의된 키 경로 Enum
- ProfileCategory: 카테고리 (preferences, traits, constraints, goals)
- ProfileTier: 프로필 계층 (GUARD, CORE, DYNAMIC)
- ProfileItem: 프로필 항목 스키마 (Pydantic)
- ProfileProposal: LLM 출력 스키마 (Pydantic)
- PROFILE_SCHEMA: Milvus profile_chunks 컬렉션 스키마
- PROFILE_INDEXES: Milvus 인덱스 정의
- TIER_TTL: 티어별 캐시 TTL
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pymilvus import CollectionSchema, DataType, FieldSchema

from .config import EMBEDDING_DIM


class ProfileCategory(str, Enum):
    """프로필 카테고리"""

    PREFERENCES = "preferences"
    TRAITS = "traits"
    CONSTRAINTS = "constraints"
    GOALS = "goals"


class ProfileKeyPath(str, Enum):
    """
    사전 정의된 키 경로 (화이트리스트)

    규칙:
    - 계층 구조: {category}.{domain}.{attribute}
    - 예: food.spice.level, music.genre.preferred
    - 최대 100개 이하 유지 (관리 용이성)
    """

    # === Preferences: 선호도 ===

    # Food (음식)
    FOOD_SPICE_LEVEL = "food.spice.level"
    FOOD_PRICE_RANGE = "food.price.range"
    FOOD_CUISINE_TYPE = "food.cuisine.type"
    FOOD_PORTION_SIZE = "food.portion.size"

    # Music (음악)
    MUSIC_GENRE_PREFERRED = "music.genre.preferred"
    MUSIC_TEMPO_PREFERENCE = "music.tempo.preference"
    MUSIC_MOOD = "music.mood"

    # Travel (여행)
    TRAVEL_DESTINATION_TYPE = "travel.destination.type"
    TRAVEL_ACCOMMODATION_TYPE = "travel.accommodation.type"
    TRAVEL_BUDGET_RANGE = "travel.budget.range"

    # Shopping (쇼핑)
    SHOPPING_BRAND_PREFERENCE = "shopping.brand.preference"
    SHOPPING_ONLINE_VS_OFFLINE = "shopping.online_vs_offline"
    SHOPPING_IMPULSE_LEVEL = "shopping.impulse.level"

    # Entertainment (엔터테인먼트)
    ENTERTAINMENT_MOVIE_GENRE = "entertainment.movie.genre"
    ENTERTAINMENT_GAME_TYPE = "entertainment.game.type"
    ENTERTAINMENT_BOOK_GENRE = "entertainment.book.genre"

    # === Traits: 특성 ===

    # Communication (의사소통)
    COMMUNICATION_STYLE = "communication.style"
    COMMUNICATION_FORMALITY = "communication.formality"
    COMMUNICATION_EMOJI_USAGE = "communication.emoji.usage"

    # Response (응답)
    RESPONSE_LENGTH_PREFERRED = "response.length.preferred"
    RESPONSE_DETAIL_LEVEL = "response.detail.level"
    RESPONSE_TONE = "response.tone"

    # Decision Making (의사결정)
    DECISION_SPEED = "decision.speed"
    DECISION_RISK_TOLERANCE = "decision.risk.tolerance"
    DECISION_ANALYSIS_DEPTH = "decision.analysis.depth"

    # Personality (성격)
    PERSONALITY_INTROVERT_EXTROVERT = "personality.introvert_extrovert"
    PERSONALITY_OPTIMISM_LEVEL = "personality.optimism.level"
    PERSONALITY_ANXIETY_LEVEL = "personality.anxiety.level"

    # === Constraints: 제약사항 ===

    # Diet (식이)
    DIET_RESTRICTIONS = "constraints.diet.restrictions"
    DIET_ALLERGIES = "constraints.diet.allergies"
    DIET_VEGAN_VEGETARIAN = "constraints.diet.vegan_vegetarian"

    # Time (시간)
    TIME_AVAILABLE_HOURS = "constraints.time.available_hours"
    TIME_WEEKEND_PREFERENCE = "constraints.time.weekend.preference"
    TIME_MORNING_NIGHT_PREFERENCE = "constraints.time.morning_night"

    # Budget (예산)
    BUDGET_MONTHLY_SPENDING = "constraints.budget.monthly_spending"
    BUDGET_LUXURY_THRESHOLD = "constraints.budget.luxury.threshold"

    # Physical (신체)
    PHYSICAL_MOBILITY = "constraints.physical.mobility"
    PHYSICAL_HEALTH_CONDITIONS = "constraints.physical.health.conditions"

    # === Goals: 목표 ===

    # Short Term (단기)
    GOAL_SHORT_TERM_PRIMARY = "goals.short_term.primary"
    GOAL_SHORT_TERM_SECONDARY = "goals.short_term.secondary"

    # Long Term (장기)
    GOAL_LONG_TERM_CAREER = "goals.long_term.career"
    GOAL_LONG_TERM_PERSONAL = "goals.long_term.personal"
    GOAL_LONG_TERM_FINANCIAL = "goals.long_term.financial"

    # Learning (학습)
    GOAL_LEARNING_TOPICS = "goals.learning.topics"
    GOAL_LEARNING_STYLE = "goals.learning.style"


class ProfileTier(str, Enum):
    """프로필 계층 (우선순위)"""

    GUARD = "guard"  # 절대 무시 불가 (알레르기, 종교적 제약 등)
    CORE = "core"  # 핵심 선호도 (매운맛, 여행 취향 등)
    DYNAMIC = "dynamic"  # 가변 선호도 (최근 관심사 등)


class ProfileItem(BaseModel):
    """
    프로필 항목 스키마 (Strict Typing)

    LLM이 생성한 프로필 항목을 서버에서 검증한다.
    """

    key_path: ProfileKeyPath  # Enum으로 제한 (화이트리스트)
    value: Union[str, int, float, bool, List[str]]
    evidence: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tier: Optional[ProfileTier] = Field(default=ProfileTier.DYNAMIC)

    @validator("value")
    def validate_value(cls, v, values):
        """
        key_path별 값 타입 검증

        예시:
        - food.spice.level: ["순함", "보통", "매움", "아주 매움"]
        - response.length.preferred: ["짧게", "보통", "길게"]
        """
        key = values.get("key_path")

        # 매운맛 레벨 검증
        if key == ProfileKeyPath.FOOD_SPICE_LEVEL:
            allowed = ["순함", "보통", "매움", "아주 매움"]
            if v not in allowed:
                raise ValueError(f"Invalid spice level: {v}. Allowed: {allowed}")

        # 응답 길이 검증
        elif key == ProfileKeyPath.RESPONSE_LENGTH_PREFERRED:
            allowed = ["짧게", "보통", "길게", "매우 길게"]
            if v not in allowed:
                raise ValueError(f"Invalid response length: {v}. Allowed: {allowed}")

        # 추가 검증 규칙...
        # (실제 프로덕션에서는 모든 key_path에 대해 정의)

        return v


class ProfileProposal(BaseModel):
    """
    LLM 출력 스키마 (Strict Mode)

    OpenAI JSON Schema + strict=True로 강제한다.
    """

    preferences: List[ProfileItem] = Field(default_factory=list, max_items=5)
    traits: List[ProfileItem] = Field(default_factory=list, max_items=3)
    constraints: List[ProfileItem] = Field(default_factory=list, max_items=3)
    goals: List[ProfileItem] = Field(default_factory=list, max_items=2)


# === 유틸리티 함수 ===


def validate_key_path(key_path: str) -> bool:
    """
    key_path가 화이트리스트에 있는지 검증

    Args:
        key_path: 검증할 키 경로

    Returns:
        bool: 유효하면 True, 아니면 False
    """
    try:
        ProfileKeyPath(key_path)
        return True
    except ValueError:
        return False


def infer_tier(key_path: str) -> ProfileTier:
    """
    key_path로부터 tier 추론

    규칙:
    - constraints.diet.* → GUARD
    - preferences.food.* → CORE
    - 기타 → DYNAMIC
    """
    if key_path.startswith("constraints.diet"):
        return ProfileTier.GUARD
    elif key_path.startswith("preferences.food"):
        return ProfileTier.CORE
    else:
        return ProfileTier.DYNAMIC


# === Milvus 스키마 정의 ===

PROFILE_SCHEMA = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tier", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="scope", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="key_path", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="norm_key", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="value_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="confidence", dtype=DataType.FLOAT),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="tags", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
        FieldSchema(name="pii_hashed", dtype=DataType.BOOL),
        FieldSchema(name="audit_log_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="version", dtype=DataType.INT64),
        FieldSchema(name="extras", dtype=DataType.JSON),
    ],
    description="Profile Chunks Collection",
)

PROFILE_INDEXES = [
    {
        "field": "embedding",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 200},
    },
]


# === 티어별 캐시 TTL (초 단위) ===

TIER_TTL: Dict[ProfileTier, int] = {
    ProfileTier.GUARD: 3600 * 24 * 30,  # 30일
    ProfileTier.CORE: 3600 * 24 * 7,  # 7일
    ProfileTier.DYNAMIC: 0,  # 캐시 없음
}
