# C:\My_Business\backend\rag\profile_writer.py
"""
backend.rag.profile_writer - 프로필 쓰기 경로 (단계 3)

LLM은 프로필 '제안'만 생성하고, 서버가 검증/정규화/상태 전이를 관리한다.
정책:
- 데이터 주권: explicit > inferred > default
- 충돌 감지: 이산/서열형 우선, 임베딩은 보조
- 패턴 누적: 7일 내 3회 이상 반복 시 active 승격
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from backend.utils.tracing import traceable
from backend.utils.logger import safe_log_event


class ProfileWriter:
    """프로필 업데이트 관리"""

    def __init__(self) -> None:
        from backend.config import get_llm_cold, get_settings
        from backend.memory.summarizer import model_supports_response_format
        from backend.personalization.preference_scoreboard import PreferenceScoreboard

        self.settings = get_settings()
        self.llm_cold = get_llm_cold()
        self.model_supports_response_format = model_supports_response_format
        self.scoreboard = PreferenceScoreboard(self.settings.REDIS_URL)

        # Redis (패턴 누적/캐시)
        try:
            import redis

            self.redis = redis.Redis.from_url(
                self.settings.REDIS_URL, decode_responses=True
            )
        except Exception:
            self.redis = None

        # Milvus 컬렉션 핸들 (profile_chunks)
        from backend.rag.milvus import ensure_profile_collection

        self.profile_coll = ensure_profile_collection()

    @traceable(
        name="Profile: update_from_turn", run_type="chain", tags=["profile", "rag"]
    )
    async def update_from_turn(
        self, user_id: str, session_id: str, turn_summary: Dict
    ) -> None:
        """
        턴 요약에서 프로필 델타 추출 및 업데이트

        전략:
        1) LLM이 프로필 '제안' 생성
        2) 서버가 검증 및 정규화
        3) 패턴 누적 (7일 내 3회 이상)
        4) 신뢰도 기반 상태 전이
        """

        # 봇 프로필 학습 경로 (user_id가 'bot' 또는 'bot:{user_id}' 스코프인 경우)
        if str(user_id).strip().lower().startswith("bot"):
            try:
                # 기존 봇 프로필(Guard/Core) 조회
                existing_profile = {"guard": {}, "core": {}}
                try:
                    from backend.rag.profile_rag import get_profile_rag
                    from backend.rag.profile_schema import ProfileTier

                    rag = get_profile_rag()
                    guard_items = await rag.query_by_tier(
                        user_id=str(user_id), tier=ProfileTier.GUARD
                    )
                    core_items = await rag.query_by_tier(
                        user_id=str(user_id), tier=ProfileTier.CORE
                    )
                    existing_profile["guard"] = {
                        it.get("key_path"): it.get("value")
                        for it in (guard_items or [])
                    }
                    existing_profile["core"] = {
                        it.get("key_path"): it.get("value") for it in (core_items or [])
                    }
                except Exception:
                    existing_profile = {"guard": {}, "core": {}}

                # 대화 텍스트 조립
                ui = str((turn_summary or {}).get("user_input") or "").strip()
                ao = str((turn_summary or {}).get("ai_output") or "").strip()
                if not ui or not ao:
                    return
                conversation_text = f"[User]\n{ui}\n\n[AI]\n{ao}"

                # LLM 기반 봇 프로필 항목 추론
                bot_items = await self.infer_bot_profile(
                    conversation_text=conversation_text,
                    existing_profile=existing_profile,
                )
                if not bot_items:
                    return

                did_update = False
                for item in bot_items:
                    key_path = str(item.get("key_path") or "").strip()
                    value = item.get("value")
                    if not key_path or value is None:
                        continue
                    norm_key = self._normalize_key(key_path)

                    # 행동 기반 점수 업데이트(봇 힌트는 보수적으로 positive 1회로 취급)
                    sb = {}
                    try:
                        if self.scoreboard.available():
                            sb = self.scoreboard.update(
                                user_id, norm_key, {"positive": 1}, intensity=0.0
                            )
                    except Exception:
                        sb = {}

                    # tier 결정: 모델 출력 우선 → 키 기반 보조
                    try:
                        from backend.rag.profile_schema import infer_tier

                        tier = str(item.get("tier") or "").strip().lower()
                        if tier not in ("guard", "core", "dynamic"):
                            # 값 기반 보조 분류
                            tier = self._classify_tier(key_path, str(value))
                        if tier not in ("guard", "core", "dynamic"):
                            tier = str(infer_tier(key_path).value)
                    except Exception:
                        tier = self._classify_tier(key_path, str(value))
                        if tier not in ("guard", "core", "dynamic"):
                            tier = "dynamic"

                    # 상태 결정/업서트(점수기 기반)
                    conflict = await self._check_conflict(user_id, norm_key, value)
                    status = str((sb or {}).get("status") or "pending")
                    confidence = max(0.5, float((sb or {}).get("score") or 0.5))
                    if conflict:
                        status = "contradiction"
                        confidence = min(confidence, 0.5)

                    # 카테고리 보정: guard → bot_guard, 그 외 → bot_hint
                    category_candidate = str(item.get("category") or "bot_hint")
                    if tier == "guard":
                        category_candidate = "bot_guard"

                    await self._upsert_chunk(
                        user_id=user_id,
                        category=category_candidate,
                        key_path=key_path,
                        norm_key=norm_key,
                        value=value,
                        source=str(item.get("source") or "inferred"),
                        status=status,
                        confidence=confidence,
                        evidence_turn_ids=[(turn_summary or {}).get("id")],
                        extras={"existing_profile": existing_profile, "scoreboard": sb},
                        tier=tier,
                        scope="bot",
                    )
                    did_update = True

                if did_update:
                    try:
                        from backend.directives.invalidator import mark_dirty

                        mark_dirty("bot", reason="bot_profile_update")
                    except Exception:
                        pass
                return
            except Exception:
                return

        # 사용자 프로필 학습 기본 경로
        proposals = await self._propose_from_llm(turn_summary)
        if not isinstance(proposals, dict):
            return
        did_update = False
        for category, items in (proposals or {}).items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                key_path = str(item.get("key_path") or "").strip()
                value = item.get("value")
                evidence = str(item.get("evidence") or "").strip()
                if not key_path or value is None:
                    continue

                # 신규: key_path 검증
                from backend.rag.profile_schema import validate_key_path
                import logging

                logger = logging.getLogger("profile_writer")

                if not validate_key_path(key_path):
                    logger.warning(
                        f"[profile_writer] Invalid key_path rejected: {key_path}"
                    )

                    # 메트릭: key_path 거부
                    try:
                        from backend.utils.metrics import profile_key_path_rejected

                        profile_key_path_rejected.labels(key_path=key_path).inc()
                    except Exception:
                        pass

                    continue  # 유효하지 않은 key_path는 스킵

                norm_key = self._normalize_key(key_path)

                # 행동 기반 점수 업데이트(기본 positive 1회). 감정 강도는 turn_summary에서 가능하면 사용
                sb = {}
                try:
                    intensity = 0.0
                    try:
                        intensity = float(
                            ((turn_summary or {}).get("signals") or {}).get(
                                "emotional_intensity", 0.0
                            )
                        )
                    except Exception:
                        intensity = 0.0
                    if self.scoreboard.available():
                        sb = self.scoreboard.update(
                            user_id, norm_key, {"positive": 1}, intensity=intensity
                        )
                except Exception:
                    sb = {}

                # tier 결정(키 기반 보조)
                try:
                    from backend.rag.profile_schema import infer_tier

                    # 사용자 프로필은 스타일/응답/언어 키는 core로 우선 분류, 그 외 키는 보조 함수 후 infer
                    tier = self._classify_tier(key_path, str(value))
                    if tier not in ("guard", "core", "dynamic"):
                        tier = str(infer_tier(key_path).value)
                except Exception:
                    tier = "dynamic"

                # 충돌 감지 및 상태/신뢰도 결정
                conflict = await self._check_conflict(user_id, norm_key, value)
                if conflict:
                    # 부정 이벤트로 점수 업데이트
                    try:
                        if self.scoreboard.available():
                            self.scoreboard.update(
                                user_id, norm_key, {"negative": 1}, intensity=0.0
                            )
                    except Exception:
                        pass
                    await self._upsert_chunk(
                        user_id=user_id,
                        category=category,
                        key_path=key_path,
                        norm_key=norm_key,
                        value=value,
                        source="inferred",
                        status="contradiction",
                        confidence=0.5,
                        evidence_turn_ids=[(turn_summary or {}).get("id")],
                        extras={"evidence": evidence, "conflict": conflict},
                        tier=tier,
                        scope="user",
                    )
                    did_update = True
                    await self._request_user_clarification(
                        user_id, norm_key, conflict, value
                    )
                else:
                    status = str((sb or {}).get("status") or "pending")
                    conf = max(0.5, float((sb or {}).get("score") or 0.5))
                    await self._upsert_chunk(
                        user_id=user_id,
                        category=category,
                        key_path=key_path,
                        norm_key=norm_key,
                        value=value,
                        source="inferred",
                        status=status,
                        confidence=conf,
                        evidence_turn_ids=[(turn_summary or {}).get("id")],
                        extras={"evidence": evidence, "scoreboard": sb},
                        tier=tier,
                        scope="user",
                    )
                    did_update = True

                    # 메트릭: 프로필 업데이트
                    try:
                        from backend.utils.metrics import profile_updates_total

                        profile_updates_total.labels(
                            user_id=user_id, source="inferred", status=status
                        ).inc()
                    except Exception:
                        pass

        if did_update:
            try:
                from backend.directives.invalidator import mark_dirty

                mark_dirty(user_id, reason="profile_update_from_turn")
            except Exception:
                pass

    @traceable(
        name="Profile: upsert_explicit_items", run_type="tool", tags=["profile", "rag"]
    )
    async def upsert_explicit_items(self, user_id: str, items: List[Dict]) -> None:
        """
        사용자 명시적 사실(Explicit Facts)을 즉시 활성 상태로 적재한다.
        - source: "explicit"
        - status: "active"
        - confidence: 0.9 (초기값, 후속 관측으로 상향 가능)
        items 형식: [{"key_path": str, "value": Any, "evidence": str}]
        """
        if not items:
            return
        did_update = False
        for it in items:
            try:
                key_path = str((it or {}).get("key_path") or "").strip()
                if not key_path:
                    continue
                value = (it or {}).get("value")
                evidence = str((it or {}).get("evidence") or "").strip()
                norm_key = self._normalize_key(key_path)
                # 기본 카테고리 추정: key_path의 첫 세그먼트 사용(없으면 preferences)
                category = (key_path.split(".")[0] or "preferences").strip()

                # 행동 점수기 업데이트(명시적 사실은 강한 긍정 증거로 취급 가능)
                sb = {}
                try:
                    if self.scoreboard.available():
                        sb = self.scoreboard.update(user_id, norm_key, {"explicit": 1})
                except Exception:
                    sb = {}

                await self._upsert_chunk(
                    user_id=user_id,
                    category=category,
                    key_path=key_path,
                    norm_key=norm_key,
                    value=value,
                    source="explicit",
                    status="active",
                    confidence=0.9,
                    evidence_turn_ids=[],
                    extras={"evidence": evidence, "scoreboard": sb},
                )
                did_update = True
            except Exception:
                continue
        if did_update:
            try:
                from backend.directives.invalidator import update_tier_version_atomic

                # 명시적 변경은 CORE 티어로 간주 → 버전 갱신 + Dirty Bit
                update_tier_version_atomic(user_id, "core")
            except Exception:
                pass

    @traceable(
        name="Profile: propose_from_llm", run_type="tool", tags=["profile", "llm"]
    )
    async def _propose_from_llm(self, turn_summary: Dict) -> Dict:
        """
        턴 요약에서 프로필 제안 JSON을 추출한다 (Strict Schema).

        신규 동작:
            - OpenAI JSON Schema + strict=True 사용
            - ProfileProposal 스키마 강제
            - key_path 화이트리스트 검증
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        import logging

        logger = logging.getLogger("profile_writer")

        schema = {
            "type": "object",
            "properties": {
                "preferences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key_path": {"type": "string"},
                            "value": {"type": ["string", "number", "boolean", "null"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["key_path", "value"],
                        "additionalProperties": False,
                    },
                },
                "traits": {"type": "array"},
                "constraints": {"type": "array"},
                "goals": {"type": "array"},
            },
            "additionalProperties": False,
        }

        # 프롬프트에 key_path 규칙 추가
        prompt = ChatPromptTemplate.from_template(
            (
                "[턴 요약]\n{turn}\n\n"
                "사용자 프로필에 추가할 새로운 정보를 JSON으로 추출하라.\n"
                "규칙:\n"
                "- 명시적 언급만 추출 (추측 금지)\n"
                "- evidence 필드에 근거 문장 포함\n"
                "- 일시적/일회성 정보 제외 (예: '오늘 배고파')\n"
                "- key_path는 사전 정의된 키만 사용 (예: food.spice.level, response.length.preferred)\n"
                "- 임의의 키 생성 금지\n"
            )
        )

        if self.model_supports_response_format(self.settings.LLM_MODEL):
            try:
                from backend.utils.schema_builder import build_json_schema

                llm = self.llm_cold.bind(
                    response_format=build_json_schema(
                        "ProfileProposal", schema, strict=True
                    )
                )
            except Exception:
                llm = self.llm_cold
        else:
            llm = self.llm_cold

        try:
            chain = prompt | llm | StrOutputParser()
            text = await chain.ainvoke({"turn": str(turn_summary or {})})
            data = json.loads(text)
            if not isinstance(data, dict):
                return {}
            # 키 보정
            for k in ("preferences", "traits", "constraints", "goals"):
                data.setdefault(k, [])
            return data
        except Exception as e:
            logger.warning(f"[profile_writer] LLM proposal failed: {e}")
            return {}

    async def infer_bot_profile(
        self, conversation_text: str, existing_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        대화 텍스트에서 봇 프로필 항목을 추론한다.

        반환 형식 예시:
        [{"key_path": "bot_hint.response_style", "value": "친근하고 간결", "confidence": 0.8, "category": "bot_hint"}]
        """
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            # 모델이 JSON 강제 응답을 지원하면 해당 모드 사용
            if self.model_supports_response_format(self.settings.LLM_MODEL):
                llm_bp = self.llm_cold.bind(response_format={"type": "json_object"})
            else:
                llm_bp = self.llm_cold

            existing_json = json.dumps(existing_profile, ensure_ascii=False, indent=2)

            chain = (
                ChatPromptTemplate.from_template(
                    "[기존 Bot Profile]\n{existing}\n\n[최근 대화]\n{conversation}\n\n"
                    "위 대화에서 드러난 봇의 특성을 추출하라. 다음 JSON 스키마로 반환하라:\n"
                    "{{\n"
                    '  "items": [\n'
                    "    {\n"
                    '      "key_path": "응답 특성 경로 (예: bot_hint.response_style)",\n'
                    '      "value": "추론된 값 (한국어 설명)",\n'
                    '      "confidence": 0.0~1.0 신뢰도,\n'
                    '      "category": "bot_hint" 고정\n'
                    "    }\n"
                    "  ]\n"
                    "}}\n\n"
                    "규칙:\n"
                    "1. 기존 프로필과 중복되지 않는 새로운 특성만 추출\n"
                    "2. 명확히 드러난 특성만 포함 (추측 금지)\n"
                    "3. confidence가 0.6 이상인 항목만 포함\n"
                    "4. 최대 5개 항목으로 제한\n"
                    "5. key_path는 'bot_hint.'로 시작"
                )
                | llm_bp
                | JsonOutputParser()
            )

            result = await chain.ainvoke(
                {"existing": existing_json, "conversation": conversation_text}
            )
            items = result.get("items", []) if isinstance(result, dict) else []

            validated: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                key_path = str(it.get("key_path") or "").strip()
                value = str(it.get("value") or "").strip()
                confidence = float(it.get("confidence") or 0.5)
                if not key_path or not value or confidence < 0.6:
                    continue
                validated.append(
                    {
                        "key_path": key_path,
                        "value": value,
                        "confidence": confidence,
                        "category": "bot_hint",
                        "source": "inferred",
                    }
                )
            return validated
        except Exception:
            return []

    def _classify_tier(self, key_path: str, value: str) -> str:
        """
        간단한 키/값 기반 Tier 분류.
        - Guard: 금지/윤리/법적 제약 관련 키워드 포함 시
        - Core: 스타일/톤/길이/페르소나 관련 표현 포함 시
        - Dynamic: 그 외 기본
        """
        kp = (key_path or "").lower()
        val = (value or "").lower()

        guard_keywords = [
            "taboo",
            "forbidden",
            "금지",
            "제약",
            "ethical",
            "윤리",
            "legal",
            "법적",
            "privacy",
            "개인정보",
            "pii",
        ]
        if any(kw in kp or kw in val for kw in guard_keywords):
            return "guard"

        core_keywords = [
            "style",
            "스타일",
            "tone",
            "톤",
            "formality",
            "격식",
            "length",
            "길이",
            "persona",
            "페르소나",
            "character",
            "캐릭터",
        ]
        if any(kw in kp or kw in val for kw in core_keywords):
            return "core"

        return "dynamic"

    def _normalize_key(self, key_path: str) -> str:
        """
        정규화 키 생성 (한글 보존)

        예:
        - "food.spice.level" → "food_spice_level"
        - "음식 선호도.매운맛" → "음식_선호도_매운맛"
        """
        normalized = key_path.lower()
        normalized = re.sub(r"[^\w가-힣0-9.]", "_", normalized, flags=re.UNICODE)
        normalized = normalized.replace(".", "_")
        normalized = re.sub(r"_{2,}", "_", normalized)
        return normalized.strip("_")

    async def _check_conflict(
        self, user_id: str, norm_key: str, new_value: Any
    ) -> Optional[Dict]:
        """
        충돌 감지 (이산/서열형 우선 → 의미 임베딩 보조)
        """
        # 기존 active 프로필 조회
        existing = self.profile_coll.query(
            expr=(
                f"user_id == '{user_id}' and "
                f"norm_key == '{norm_key}' and "
                f"status == 'active'"
            ),
            output_fields=["value", "source", "confidence", "value_type"],
            limit=1,
        )

        if not existing:
            return None

        # old_value = existing[0].get("value")
        old_raw = existing[0].get("value")
        try:
            old_value = json.loads(old_raw) if isinstance(old_raw, str) else old_raw
        except Exception:
            old_value = old_raw
        value_type = (existing[0].get("value_type") or "string").strip()

        # 1) 이산형/숫자형 정확 비교
        if value_type in ("boolean", "number"):
            if str(old_value) != str(new_value):
                return {
                    "old_value": old_value,
                    "conflict_type": "exact_mismatch",
                }

        # 2) 서열형 간극 규칙
        ordinal_patterns = {
            "spice": ["순함", "보통", "매움", "아주 매움"],
            "price": ["저렴", "보통", "비쌈", "고급"],
        }
        for key_frag, scale in ordinal_patterns.items():
            if key_frag in norm_key.lower():
                try:
                    old_idx = scale.index(old_value) if old_value in scale else -1
                    new_idx = scale.index(new_value) if new_value in scale else -1
                except Exception:
                    old_idx, new_idx = -1, -1
                if old_idx >= 0 and new_idx >= 0 and abs(old_idx - new_idx) >= 2:
                    return {
                        "old_value": old_value,
                        "conflict_type": "ordinal_gap",
                        "gap": abs(old_idx - new_idx),
                    }

        # 3) 의미 임베딩 비교 (key+value 합성)
        try:
            import numpy as np

            from backend.rag.embeddings import embed_query_openai

            old_composite = f"{norm_key}:{old_value}"
            new_composite = f"{norm_key}:{new_value}"
            old_emb = embed_query_openai(old_composite)
            new_emb = embed_query_openai(new_composite)
            old_arr = np.array(old_emb, dtype="float32")
            new_arr = np.array(new_emb, dtype="float32")
            den = float((np.linalg.norm(old_arr) * np.linalg.norm(new_arr)) or 1.0)
            sim = float(np.dot(old_arr, new_arr) / den)
            if sim < 0.3:
                return {
                    "old_value": old_value,
                    "conflict_type": "semantic_mismatch",
                    "similarity": sim,
                }
        except Exception:
            pass

        return None

    # # 싱글톤 접근자
    # _PROFILE_WRITER_INSTANCE: Optional[ProfileWriter] = None

    # def get_profile_writer() -> ProfileWriter:
    #     """프로세스 전역에서 재사용 가능한 ProfileWriter 인스턴스 반환"""
    #     global _PROFILE_WRITER_INSTANCE
    #     if _PROFILE_WRITER_INSTANCE is None:
    #         _PROFILE_WRITER_INSTANCE = ProfileWriter()
    #     return _PROFILE_WRITER_INSTANCE

    @traceable(name="Profile: upsert_chunk", run_type="tool", tags=["profile", "rag"])
    async def _upsert_chunk(
        self,
        *,
        user_id: str,
        category: str,
        key_path: str,
        norm_key: str,
        value: Any,
        source: str,
        status: str,
        confidence: float,
        evidence_turn_ids: List[str],
        extras: Optional[Dict] = None,
        tier: Optional[str] = None,
        scope: str = "user",
    ) -> None:
        """profile_chunks 컬렉션에 업서트한다."""
        try:
            from backend.directives.invalidator import (
                mark_dirty,
                update_tier_version_atomic,
            )
            from backend.policy import sha256
            from backend.rag.embeddings import embed_query_openai

            # 티어 판정 및 무효화 트리거 (업서트 성공 시)
            from backend.rag.profile_schema import ProfileTier, infer_tier

            now_ns = int(time.time_ns())
            entity_id = f"{user_id}:{category}:{norm_key}:1"  # 버전 필드는 value/증거 축적 단계에서 증분

            # 값 직렬화 및 임베딩
            value_json = json.dumps(value, ensure_ascii=False)
            composite_text = f"{category}:{norm_key}\n{value_json}"
            emb = embed_query_openai(composite_text)

            # TAGS 힌트 (간단한 키워드 기반 — 휴리스틱 최소화, 후속 개선 지점)
            tags = [category, norm_key][:10]

            # 최종 tier 확정
            _tier_final = tier or str(infer_tier(key_path).value)

            self.profile_coll.upsert(
                [
                    {
                        "id": entity_id,
                        "user_id": user_id,
                        "category": category,
                        "tier": _tier_final,
                        "scope": scope,
                        "key_path": key_path,
                        "norm_key": norm_key,
                        "value": value_json,
                        "value_type": self._infer_value_type(value),
                        "source": source,
                        "confidence": float(confidence),
                        "status": status,
                        "tags": tags,
                        "embedding": emb,
                        "created_at": now_ns,
                        "updated_at": now_ns,
                        "pii_hashed": False,
                        "audit_log_id": sha256(entity_id)[:12],
                        "version": 1,
                        "extras": extras or {},
                    }
                ]
            )

            # 업서트 완료 후 티어 기반 무효화 트리거
            try:
                _tier_enum = infer_tier(key_path)
                if _tier_enum == ProfileTier.GUARD:
                    update_tier_version_atomic(user_id, "guard")
                elif _tier_enum == ProfileTier.CORE:
                    # inferred는 Dirty만, explicit은 상위 경로에서 처리
                    source_norm = str(source or "").strip().lower()
                    if source_norm == "explicit":
                        update_tier_version_atomic(user_id, "core")
                    else:
                        mark_dirty(user_id, reason=f"core_inferred:{norm_key}")
                else:
                    mark_dirty(user_id, reason=f"dynamic_upsert:{norm_key}")
            except Exception:
                pass

            # 구조화 로깅: profile_chunks 업서트 요약
            try:
                safe_log_event(
                    "rag.profile_upsert",
                    {
                        "user_id": user_id,
                        "collection": "profile_chunks",
                        "text_len": len(value_json),
                        "vector_dim": len(emb or []),
                        "eid_count": len(evidence_turn_ids or []),
                        "reason": source,
                        "tier": _tier_final,
                        "scope": scope,
                        "chunk_count": 1,
                    },
                )
            except Exception:
                pass
        except Exception:
            # 업서트 실패는 상위에서 로깅하도록 위임 (여기서는 조용히 반환)
            return

    def _infer_value_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    async def _request_user_clarification(
        self, user_id: str, norm_key: str, conflict: Dict, new_value: Any
    ) -> None:
        """
        사용자 확인 요청 훅.
        실제 알림/UX는 상위 계층에서 구현. 본 함수는 이벤트 로그/플래그만 남긴다.
        """
        try:
            if self.redis is not None:
                payload = {
                    "norm_key": norm_key,
                    "conflict": conflict,
                    "proposed": new_value,
                    "ts": int(time.time()),
                }
                self.redis.lpush(
                    f"prof:clarify:{user_id}", json.dumps(payload, ensure_ascii=False)
                )
                self.redis.ltrim(f"prof:clarify:{user_id}", 0, 99)
        except Exception:
            pass


# 싱글톤 접근자
_PROFILE_WRITER_INSTANCE: Optional[ProfileWriter] = None


def get_profile_writer() -> ProfileWriter:
    """프로세스 전역에서 재사용 가능한 ProfileWriter 인스턴스 반환"""
    global _PROFILE_WRITER_INSTANCE
    if _PROFILE_WRITER_INSTANCE is None:
        _PROFILE_WRITER_INSTANCE = ProfileWriter()
    return _PROFILE_WRITER_INSTANCE
