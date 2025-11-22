"""
backend.memory.coordinator - MemoryCoordinator 단일 진입점

턴 종료 시점의 메모리 계층 전이를 중앙에서 관리한다.
처리 순서: STWM → STM → (임계 충족 시) MTM → (디바운스/임계 충족 시) LTM

특징:
- Exactly-Once 보장: turn_id 기반 멱등성 키 사용(tx:done:{turn_id})
- Redis Streams 트랜잭션 로그(begin/commit/rollback)
- 임계값/디바운스는 settings에서 관리
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Tuple

import redis
from langchain_core.messages import BaseMessage

from backend.config import get_llm_cold, get_settings
from backend.memory.redis_memory import get_short_term_memory
from backend.memory.redis_mtm import RedisMTM
from backend.memory.stwm import update_stwm
from backend.memory.tx_log import RedisTransactionLog
from backend.rag.snapshot_pipeline import update_long_term_memory as ltm_snapshot
from backend.routing.router_context import user_for_session

logger = logging.getLogger("memory_coordinator")


@dataclass
class TurnResult:
    """턴 처리 결과 요약"""

    turn_id: str
    mtm_created: bool
    ltm_triggered: bool
    mtm_summary: str | None = None
    routing_summary: str | None = None


class MemoryCoordinator:
    """
    메모리 계층 전이 코디네이터

    - on_turn_end(user_id, session_id, user_input, ai_output): 단일 진입점
    - 내부에서 STWM/STM 갱신 후, 임계값에 따라 MTM 생성 및 LTM 스냅샷 트리거
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tx = RedisTransactionLog()
        self.mtm = RedisMTM()
        self._r = redis.Redis.from_url(self.settings.REDIS_URL, decode_responses=True)
        self.llm_cold = get_llm_cold()

    # -----------------------------
    # Public API
    # -----------------------------
    async def on_turn_end(
        self, user_id: str, session_id: str, user_input: str, ai_output: str
    ) -> TurnResult:
        """
        단일 진입점: 한 턴의 사용자/AI 발화를 입력으로 받아 메모리 계층을 갱신한다.
        """
        # 세션→사용자 매핑(존재 시 우선). 영구 스토어는 user_id 기준으로만 운영한다.
        mapped_user_id = user_for_session(session_id) or user_id or session_id

        turn_id = self._make_turn_id(user_id, session_id, user_input)

        # 멱등성: 이미 처리된 턴이면 즉시 반환
        if self.tx.is_turn_done(turn_id):
            return TurnResult(turn_id=turn_id, mtm_created=False, ltm_triggered=False)

        tx_id = self.tx.begin(user_id, session_id, turn_id)

        mtm_created = False
        ltm_triggered = False
        mtm_sum_txt: str | None = None
        routing_sum_txt: str | None = None
        try:
            # 현재 턴 순번(위의 turn_id 생성 과정에서 증가된 값)을 조회
            cur_seq = int(self._r.get(self._key_turn_seq(user_id, session_id)) or 0)

            # 1) STWM 갱신(규칙/NER 기반 휘발성 슬롯)
            update_stwm(session_id, user_input or "", user_id=mapped_user_id)

            # 2) STM 저장(원문 히스토리)
            stm = get_short_term_memory(session_id, user_id=mapped_user_id)
            stm.save_context({"input": user_input or ""}, {"output": ai_output or ""})

            # 2.5) Behavior Slot 분류 및 EWMA 업데이트(+옵션 RAG upsert)
            try:
                if bool(getattr(self.settings, "BEHAVIOR_ENABLED", True)):
                    from backend.behavior.classify_slot import classify_slot  # type: ignore
                    from backend.memory.stwm import get_stwm_snapshot  # type: ignore
                    from backend.behavior.behavior_extractor import update_from_slots  # type: ignore
                    from backend.directives.invalidator import mark_dirty as _mark_dirty  # type: ignore

                    stwm = (
                        get_stwm_snapshot(session_id, user_id=mapped_user_id)
                        if user_input
                        else {}
                    )
                    slots = classify_slot(user_input or "", stwm)
                    if slots:
                        # 감정 강도는 신호가 있을 때 전달(없으면 0.0)
                        intensity = (
                            float((self._r.get(f"sig:emo:{session_id}") or "0.0"))
                            if False
                            else 0.0
                        )
                        await update_from_slots(
                            mapped_user_id, slots, intensity=intensity
                        )
                        # Behavior 업데이트 시 Overlay 캐시 무효화(세션 == user 가정 깨질 수 있으므로 user 기준)
                        try:
                            _mark_dirty(mapped_user_id, reason="behavior_update")
                        except Exception:
                            pass
            except Exception:
                # 실패해도 파이프라인 진행
                pass

            # 3) 요약/스냅샷 트리거 판단
            should_mtm = self._should_create_summary(user_id, session_id)
            if should_mtm:
                mtm_created = True
                mtm_sum, routing_sum = await self._create_dual_summaries(
                    user_id, session_id
                )
                mtm_sum_txt, routing_sum_txt = mtm_sum, routing_sum
                self.mtm.add_summary(user_id, session_id, mtm_sum, routing_sum)
                # 마지막 MTM 턴/시각 갱신
                self._r.set(self._key_last_mtm_seq(user_id, session_id), str(cur_seq))
                self._r.set(
                    self._key_last_mtm_ts(user_id, session_id), str(int(time.time()))
                )

                # 3-1) 명시적 사실(Explicit Facts) 추출 및 LTM 승격 조건 평가
                try:
                    explicit_items = await self._detect_explicit_facts(session_id)
                except Exception:
                    explicit_items = []

                if explicit_items:
                    # LTM 승격: 명시적 사실 존재 시 즉시 스냅샷 트리거
                    try:
                        from backend.rag.profile_writer import get_profile_writer
                        from backend.rag.profile_ids import bot_user_id_for

                        writer = get_profile_writer()
                        # 사용자/봇 분기
                        bot_items: list[dict] = []
                        user_items: list[dict] = []
                        for it in explicit_items:
                            try:
                                kp = (
                                    str((it or {}).get("key_path") or "")
                                    .strip()
                                    .lower()
                                )
                            except Exception:
                                kp = ""
                            # 간단 분류: bot.* / bot_* / '봇 ' 키워드 포함
                            is_bot = (
                                kp.startswith("bot.")
                                or kp.startswith("bot_")
                                or ("봇" in kp)
                            )
                            if is_bot:
                                # 대표 매핑: 이름 지정은 bot_meta.name으로 표준화
                                if ("name" in kp) or ("이름" in kp):
                                    it["key_path"] = "bot_meta.name"
                                bot_items.append(it)
                            else:
                                user_items.append(it)
                        if user_items:
                            await writer.upsert_explicit_items(
                                user_id=mapped_user_id, items=user_items
                            )
                        if bot_items:
                            await writer.upsert_explicit_items(
                                user_id=bot_user_id_for(mapped_user_id), items=bot_items
                            )
                    except Exception:
                        pass

                    try:
                        # 스냅샷은 세션 히스토리를 사용하되, 적재 user_id는 매핑된 사용자 ID를 사용하도록 내부에서 처리한다.
                        ltm_snapshot(session_id)
                        ltm_triggered = True
                        self._r.set(
                            self._key_last_ltm_seq(mapped_user_id, session_id),
                            str(cur_seq),
                        )
                        self._r.set(
                            self._key_last_ltm_ts(mapped_user_id, session_id),
                            str(int(time.time())),
                        )
                    except Exception:
                        pass

            # ===== LTM 승격 판단 (기존 임계 + 신규 SEED 기반) =====
            should_ltm_legacy = self._should_snapshot(mapped_user_id, session_id)

            # 신규 승격 판단기 호출
            try:
                from backend.memory.ltm_promoter import get_ltm_promoter

                promoter = get_ltm_promoter()
                should_promote, category, confidence = promoter.should_promote(
                    user_input=user_input, ai_output=ai_output
                )
            except Exception:
                should_promote, category, confidence = False, "error", 0.0

            should_ltm_final = bool(should_ltm_legacy or should_promote)

            if should_ltm_final:
                ltm_triggered = True
                ltm_snapshot(session_id)
                self._r.set(
                    self._key_last_ltm_seq(mapped_user_id, session_id), str(cur_seq)
                )
                self._r.set(
                    self._key_last_ltm_ts(mapped_user_id, session_id),
                    str(int(time.time())),
                )

                # 승격 사유 로깅(신규 판단기가 트리거에 기여한 경우)
                if should_promote:
                    logger.info(
                        f"[LTM] Promoted by category: user_id={mapped_user_id} "
                        f"category={category} confidence={confidence:.3f}"
                    )

            self.tx.commit(user_id, session_id, turn_id, tx_id)
            return TurnResult(
                turn_id=turn_id,
                mtm_created=mtm_created,
                ltm_triggered=ltm_triggered,
                mtm_summary=mtm_sum_txt,
                routing_summary=routing_sum_txt,
            )
        except Exception as e:
            # 롤백 이벤트 기록(실제 데이터 롤백은 idempotent 설계로 흡수)
            self.tx.rollback(user_id, session_id, turn_id, tx_id, reason=str(e))
            logger.exception("[coord] on_turn_end failed")
            # 실패 시에도 멱등성 키는 마킹하지 않는다(재시도 허용)
            raise

    # -----------------------------
    # Internals
    # -----------------------------
    def _make_turn_id(self, user_id: str, session_id: str, user_input: str) -> str:
        """user/session/순번/입력 요약으로 고유 턴 ID 생성"""
        seq = self._turn_seq(user_id, session_id)
        ts_ms = int(time.time() * 1000)
        digest = hashlib.sha256((user_input or "").encode("utf-8")).hexdigest()[:16]
        raw = f"{user_id}|{session_id}|{seq}|{ts_ms}|{digest}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _turn_seq(self, user_id: str, session_id: str) -> int:
        """턴 순번: Redis 원자적 증가"""
        try:
            return int(self._r.incr(self._key_turn_seq(user_id, session_id)))
        except Exception:
            # 실패 시에도 1 반환하여 진행(드문 케이스)
            return 1

    def _should_create_summary(self, user_id: str, session_id: str) -> bool:
        """
        STM → MTM 요약 생성 필요 여부 판정
        - 토큰 임계 초과 OR 최근 MTM 이후 턴 수 임계 초과
        """
        try:
            from backend.memory.summarizer import get_summarizer

            stm = get_short_term_memory(session_id, user_id=user_id)
            msgs: list[BaseMessage] = stm.chat_memory.messages
            summarizer = get_summarizer()
            total_tokens = summarizer.count_tokens_msgs(msgs)
            if total_tokens >= int(self.settings.STM_TO_MTM_TOKENS):
                return True
        except Exception:
            pass

        # 턴 수 기반
        try:
            last_seq = int(
                self._r.get(self._key_last_mtm_seq(user_id, session_id)) or 0
            )
        except Exception:
            last_seq = 0
        cur_seq = int(self._r.get(self._key_turn_seq(user_id, session_id)) or 0)
        if cur_seq and (cur_seq - last_seq) >= int(self.settings.STM_TO_MTM_TURNS):
            return True
        return False

    def _should_snapshot(self, user_id: str, session_id: str) -> bool:
        """
        MTM → LTM 스냅샷 필요 여부 판정
        - 토큰 임계 초과 OR (턴 수 임계 AND 디바운스 시간 경과)
        """
        # 토큰 기반(최근 STM 기준)
        try:
            from backend.memory.summarizer import get_summarizer

            stm = get_short_term_memory(session_id, user_id=user_id)
            msgs: list[BaseMessage] = stm.chat_memory.messages
            summarizer = get_summarizer()
            total_tokens = summarizer.count_tokens_msgs(msgs)
            if total_tokens >= int(self.settings.MTM_TO_LTM_TOKENS):
                return True
        except Exception:
            pass

        # 턴 + 디바운스
        try:
            last_seq = int(
                self._r.get(self._key_last_ltm_seq(user_id, session_id)) or 0
            )
            last_ts = int(self._r.get(self._key_last_ltm_ts(user_id, session_id)) or 0)
        except Exception:
            last_seq, last_ts = 0, 0
        cur_seq = int(self._r.get(self._key_turn_seq(user_id, session_id)) or 0)
        if cur_seq and (cur_seq - last_seq) >= int(self.settings.MTM_TO_LTM_TURNS):
            if (int(time.time()) - last_ts) >= int(
                self.settings.MTM_TO_LTM_DEBOUNCE_SEC
            ):
                return True
        return False

    async def _create_dual_summaries(
        self, user_id: str, session_id: str
    ) -> Tuple[str, str]:
        """
        2종 요약을 생성한다.
        - mtm_summary: 상세/보존용 요약 (200~350 토큰 목표)
        - routing_summary: 라우팅 전용 초경량 요약(20토큰 내, 프롬프트 주입 금지)
        """
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        # 타입 힌트 유지 목적(외부 의존성 변경 없음)
        from backend.memory.redis_memory import HybridSummaryMemory  # noqa: F401
        from backend.memory.summarizer import get_tokenizer

        # STM 메시지 로드
        stm = get_short_term_memory(session_id, user_id=user_id)
        msgs = getattr(stm, "chat_memory", {}).messages or []

        # 이전 MTM 요약 조회 (증분 요약용) — Cross-Session 범위로 조회
        prev_mtm_summary = ""
        try:
            latest = self.mtm.get_latest(user_id, session_id=None)  # Cross-Session
            if latest:
                prev_mtm_summary = (
                    latest.get("mtm_summary") or latest.get("summary") or ""
                )
        except Exception:
            prev_mtm_summary = ""

        # 델타 턴(최근 N개 메시지) 구성 (역할 태깅 포함)
        N = 8  # user+ai 합쳐 최대 8개 메시지(약 4턴)
        delta_msgs = msgs[-N:] if msgs else []

        def _role_of(m: BaseMessage) -> str:
            t = (getattr(m, "type", "") or "").lower()
            if t == "human":
                return "user"
            if t == "ai":
                return "ai"
            return t or "other"

        delta_turns = "\n".join(
            f"{_role_of(m)}: {(getattr(m, 'content', '') or '').strip()}"
            for m in delta_msgs
        )

        # 상세 요약 프롬프트(증분 갱신 규칙)
        detail_prompt = ChatPromptTemplate.from_template(
            (
                "[기존 요약]\n{prev_mtm_summary}\n\n"
                "[최근 대화 (델타, 역할 태깅 포함)]\n{delta_turns}\n\n"
                "기존 요약을 갱신하라. 규칙:\n"
                "1. 중복 제거 (기존 요약과 겹치는 내용 생략)\n"
                "2. 새로운 사실만 추가\n"
                "3. 200-350 토큰 유지\n"
                "4. 사용자 선호도, 목표, 제약사항 중심으로 요약\n"
                "5. 가능한 한 각 문장 앞에 'user:' 또는 'ai:' 라벨을 명시하여 발화 주체를 구분하라"
            )
        )

        # 라우팅 요약 프롬프트(임베딩 전용)
        routing_prompt = ChatPromptTemplate.from_template(
            (
                "다음 대화를 20토큰 이내로 요약:\n"
                "형식: 유저: [핵심 의도] / AI: [핵심 답변]\n\n"
                "⚠️ 라우팅 판단용이므로 액션(RAG/Web/Conv)만 명확히 드러나게 요약\n"
                "⚠️ 이 요약은 프롬프트에 절대 주입되지 않으며 임베딩 전용임\n\n"
                "{turns}"
            )
        )

        chain_detail = detail_prompt | self.llm_cold | StrOutputParser()
        chain_route = routing_prompt | self.llm_cold | StrOutputParser()

        # LLM 호출
        mtm_summary = chain_detail.invoke(
            {"prev_mtm_summary": prev_mtm_summary, "delta_turns": delta_turns}
        )
        routing_summary = chain_route.invoke({"turns": delta_turns or prev_mtm_summary})

        # 라우팅 요약 토큰 강제 트림 + 태깅
        try:
            tokenizer = get_tokenizer()
            tokens = tokenizer.encode(routing_summary or "")
            if len(tokens) > 25:
                routing_summary = tokenizer.decode(tokens[:25])
        except Exception:
            pass
        routing_summary = (
            f"[ROUTING_ONLY]{(routing_summary or '').strip()}[/ROUTING_ONLY]"
        )

        return (mtm_summary or "").strip(), (routing_summary or "").strip()

    async def _detect_explicit_facts(self, session_id: str) -> list:
        """
        STM 최신 대화에서 '사용자 명시적 사실(Explicit Facts)'을 추출한다.
        - 정의: 사용자가 직접적으로 말한 핵심 정보(거주지, 직업, 선호 등) 또는
          "기억해/기억해줘"와 같은 명시적 고정 요청이 동반된 사실.
        - 결과: [{"key_path": str, "value": Any, "evidence": str}]
        """
        try:
            from backend.utils.retry import openai_chat_with_retry
        except Exception:
            openai_chat_with_retry = None  # type: ignore

        try:
            # session_id에서 user_id 추론 (매핑 우선)
            uid = user_for_session(session_id) or session_id
            stm = get_short_term_memory(session_id, user_id=uid)
            msgs: list[BaseMessage] = getattr(stm, "chat_memory", {}).messages or []
            # 최근 대화 12개(약 6턴)만 분석, 역할 태깅 포함
            window = msgs[-12:] if msgs else []

            def _role_of(m: BaseMessage) -> str:
                t = (getattr(m, "type", "") or "").lower()
                return "user" if t == "human" else ("ai" if t == "ai" else t or "other")

            convo = "\n".join(
                f"{_role_of(m)}: {(getattr(m, 'content', '') or '').strip()}"
                for m in window
            )

            from backend.utils.schema_builder import build_json_schema
            from backend.utils.schema_registry import get_explicit_facts_schema

            schema_inner = get_explicit_facts_schema()

            sys_msg = {
                "role": "system",
                "content": (
                    "너는 대화에서 '사용자 명시적 사실(Explicit Facts)'만 추출하는 분석기다.\n"
                    "명시적 사실의 예: 거주지, 직업, 연락처, 취향/선호('저는 강아지를 좋아해요').\n"
                    "AI의 추론/제안은 제외하고, 사용자 발화가 직접적으로 밝힌 경우만 포함.\n"
                    "'기억해/기억해줘' 같은 표현이 있으면 해당 사실을 우선 포함."
                ),
            }
            user_msg = {
                "role": "user",
                "content": f"[대화]\n{convo}\n\nJSON만 출력",
            }

            if openai_chat_with_retry is None:
                return []

            resp = await openai_chat_with_retry(
                model="gpt-4o-mini",
                messages=[sys_msg, user_msg],
                response_format=build_json_schema(
                    "ExplicitFacts", schema_inner, strict=True
                ),
                temperature=0.0,
                max_tokens=400,
            )
            text = (resp.choices[0].message.content or "").strip()
            import json as _json

            data = _json.loads(text) if text.startswith("{") else {}
            items = data.get("explicit_facts", []) if isinstance(data, dict) else []
            # 간단 검증/정제
            out = []
            for it in items:
                try:
                    kp = str((it or {}).get("key_path") or "").strip()
                    if not kp:
                        continue
                    out.append(
                        {
                            "key_path": kp,
                            "value": (it or {}).get("value"),
                            "evidence": str((it or {}).get("evidence") or "").strip(),
                        }
                    )
                except Exception:
                    continue
            return out
        except Exception:
            return []

    # -----------------------------
    # Redis Keys
    # -----------------------------
    def _key_turn_seq(self, user_id: str, session_id: str) -> str:
        return f"turn_seq:{user_id}:{session_id}"

    def _key_last_mtm_seq(self, user_id: str, session_id: str) -> str:
        return f"mtm:last_seq:{user_id}:{session_id}"

    def _key_last_mtm_ts(self, user_id: str, session_id: str) -> str:
        return f"mtm:last_ts:{user_id}:{session_id}"

    def _key_last_ltm_seq(self, user_id: str, session_id: str) -> str:
        return f"ltm:last_seq:{user_id}:{session_id}"

    def _key_last_ltm_ts(self, user_id: str, session_id: str) -> str:
        return f"ltm:last_ts:{user_id}:{session_id}"


# ===== 싱글톤 제공 =====
_coord_instance: MemoryCoordinator | None = None


def get_memory_coordinator() -> MemoryCoordinator:
    global _coord_instance
    if _coord_instance is None:
        _coord_instance = MemoryCoordinator()
        logger.info("[coord] MemoryCoordinator instance created")
    return _coord_instance
