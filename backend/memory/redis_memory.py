"""
backend.memory.redis_memory - Redis 기반 단기 메모리

LangChain RedisChatMessageHistory를 기반으로 한 커스텀 메모리 클래스를 제공합니다.
토큰 한도 초과 시 자동으로 요약하여 압축합니다.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger("redis_memory")


class HybridSummaryMemory(ConversationSummaryBufferMemory):
    """
    하이브리드 요약 메모리 (Redis + 자동 요약)

    Redis에 대화 히스토리를 저장하며, 토큰 한도(3000) 초과 시
    오래된 부분을 요약(500 토큰)으로 교체하고 최근 원문(1500 토큰)은 유지합니다.

    특징:
    - 3000 토큰 하드 한도
    - 오래된 대화 → 500 토큰 요약
    - 최근 1500 토큰 원문 보존
    - 비동기 요약 생성 (이벤트 루프 블로킹 방지)
    - 재요약 방지 (이미 요약된 블록은 스킵)
    """

    def __init__(self, *args, **kwargs):
        """
        HybridSummaryMemory 초기화

        Args:
            *args, **kwargs: ConversationSummaryBufferMemory의 인자들
                - llm: 요약용 LLM
                - chat_memory: RedisChatMessageHistory
                - max_token_limit: 토큰 한도 (기본: 3000)
                - return_messages: True 권장
                - memory_key: "chat_history"
        """
        super().__init__(*args, **kwargs)

        # 설정에서 max_token_limit 가져오기
        try:
            from backend.config import get_settings

            settings = get_settings()
            self.max_token_limit = settings.MAX_TOKEN_LIMIT
        except Exception:
            self.max_token_limit = kwargs.get("max_token_limit", 3000)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        대화 턴을 메모리에 저장 (오버라이드)

        토큰 한도를 초과하면 자동으로 요약을 생성합니다.

        Args:
            inputs: 사용자 입력 딕셔너리 {"input": str}
            outputs: AI 출력 딕셔너리 {"output": str}
        """
        # 기본 저장 (원문 유지)
        user_msg = inputs.get(self.input_key, inputs.get("input", ""))
        ai_msg = outputs.get(self.output_key, outputs.get("output", ""))

        # 타임스탬프 추가
        from backend.utils.datetime_utils import now_kst

        ts_iso = now_kst().isoformat()

        if user_msg:
            self.chat_memory.add_message(
                HumanMessage(content=user_msg, additional_kwargs={"ts": ts_iso})
            )
        if ai_msg:
            self.chat_memory.add_message(
                AIMessage(content=ai_msg, additional_kwargs={"ts": ts_iso})
            )

        # 토큰 한도 체크
        from backend.memory.summarizer import get_summarizer

        summarizer = get_summarizer()
        msgs = self.chat_memory.messages
        total_tokens = summarizer.count_tokens_msgs(msgs)

        if total_tokens <= self.max_token_limit:
            return

        # 분리: 오래된(old) / 최근(recent)
        old_msgs, recent_msgs = summarizer.split_for_summary(msgs)

        if not old_msgs:
            return  # 최근만으로도 3000 넘는 경우 (드뭄) → 건너뜀

        # 재요약 방지: old가 이미 요약 블록이면 스킵
        try:
            old_text_flat = "\n".join(getattr(m, "content", "") for m in old_msgs)
            if (
                old_text_flat.strip().startswith("[SUMMARIZED@")
                and "[SUMMARY]" in old_text_flat
            ):
                logger.info("[redis] Skip re-summarization (already summarized block)")
                return
        except Exception:
            pass

        # 요약 생성은 백그라운드로 처리 (이벤트 루프 블로킹 방지)
        async def _compact_async(
            session_id_local: str, old_msgs_local, recent_msgs_local
        ):
            """비동기 요약 생성 및 메모리 재구성"""
            combo_local, meta_local = summarizer.build_structured_and_summary(
                session_id_local, old_msgs_local
            )

            try:
                # 메모리 재구성
                self.chat_memory.clear()

                # 요약 블록 추가
                from backend.utils.datetime_utils import now_kst

                stamp = now_kst().isoformat()

                header = (
                    f"[SUMMARIZED@{stamp}] "
                    f"tokens~{summarizer.settings.SUMMARY_TARGET_TOKENS} | "
                    f"{meta_local['summary_version']} | "
                    f"model={meta_local['model']}"
                )

                self.chat_memory.add_message(
                    AIMessage(
                        content=header + "\n\n" + combo_local,
                        additional_kwargs={"ts": stamp},
                    )
                )

                # 최근 원문 복원
                for m in recent_msgs_local:
                    kwargs = getattr(m, "additional_kwargs", {}) or {}
                    if getattr(m, "type", "") == "human":
                        self.chat_memory.add_message(
                            HumanMessage(content=m.content, additional_kwargs=kwargs)
                        )
                    else:
                        self.chat_memory.add_message(
                            AIMessage(content=m.content, additional_kwargs=kwargs)
                        )

                logger.info(
                    f"[redis] Compacted: old→summary(≈{summarizer.settings.SUMMARY_TARGET_TOKENS} tok), "
                    f"kept recent(≈{summarizer.settings.RECENT_RAW_TOKENS_BUDGET} tok)"
                )
            except Exception as e:
                logger.warning(f"[redis] Compact error: {e}")

            # 단일 진입점 코디네이터가 활성화된 경우 직접 스냅샷 예약을 생략한다.
            # (중복 실행 및 순서 불일치 방지) 비활성화 시에만 레거시 경로로 예약.
            try:
                from backend.config import get_settings

                settings = get_settings()
                coordinator_enabled = bool(
                    getattr(settings, "MEMORY_COORDINATOR_ENABLED", True)
                )
                if (
                    (not coordinator_enabled)
                    and session_id_local
                    and isinstance(session_id_local, str)
                ):
                    from backend.policy.snapshot_manager import enqueue_snapshot

                    try:
                        # get_short_term_memory에서 설정한 복합 키 "{user_id}:{session_id}" 활용
                        if (
                            isinstance(session_id_local, str)
                            and ":" in session_id_local
                        ):
                            _uid, _sid = session_id_local.split(":", 1)
                            enqueue_snapshot(_uid, _sid)
                    except Exception:
                        pass
            except Exception:
                pass

        # 비동기 태스크 스케줄링: 실행중인 이벤트 루프가 없을 때도 안전하게 처리
        def _schedule_background(coro):
            """
            백그라운드 스케줄러

            - 현재 스레드에 실행중인 이벤트 루프가 있으면 create_task 사용
            - 없으면 전용 백그라운드 이벤트 루프를 데몬 스레드에 생성하여 안전하게 실행
            - Windows 환경에서도 'no running event loop' 오류 방지
            """
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
                return
            except RuntimeError:
                # 실행중인 루프 없음 → 백그라운드 루프 사용
                pass

            # 전역 백그라운드 루프/스레드 보장
            try:
                global _BG_LOOP, _BG_THREAD
            except NameError:
                _BG_LOOP = None
                _BG_THREAD = None

            if _BG_LOOP is None or (_BG_THREAD and not _BG_THREAD.is_alive()):
                _BG_LOOP = asyncio.new_event_loop()

                def _run_loop():
                    asyncio.set_event_loop(_BG_LOOP)
                    _BG_LOOP.run_forever()

                _BG_THREAD = threading.Thread(
                    target=_run_loop, name="redis_memory_bgloop", daemon=True
                )
                _BG_THREAD.start()

            try:
                asyncio.run_coroutine_threadsafe(coro, _BG_LOOP)
            except Exception as e:
                logger.warning(f"[redis] Background scheduling failed: {e}")

        # 스케줄 실행
        try:
            session_id = getattr(self.chat_memory, "session_id", None) or "unknown"
            _schedule_background(
                _compact_async(session_id, list(old_msgs), list(recent_msgs))
            )
        except Exception as e:
            logger.warning(f"[redis] Schedule compact error: {e}")


def get_short_term_memory(
    session_id: str, user_id: Optional[str] = None
) -> ConversationSummaryBufferMemory:
    """
    세션별 단기 메모리 인스턴스 생성 (팩토리 함수)

    Redis 키 스키마:
        - 신규: message_store:{user_id}:{session_id}
        - 레거시 호환: user_id 미제공 시 session_id를 uid로 사용

    Args:
        session_id: 세션 ID
        user_id: 사용자 ID (선택적, 미제공 시 router_context에서 추론)

    Returns:
        ConversationSummaryBufferMemory: 커스텀 메모리 인스턴스

    Example:
        >>> from backend.memory.redis_memory import get_short_term_memory
        >>> stm = get_short_term_memory("session123", user_id="user456")
        >>> stm.save_context({"input": "안녕"}, {"output": "반가워요"})
        >>> history = stm.chat_memory.messages
    """
    from backend.config import get_llm_cold, get_settings
    from backend.memory.redis_index import register_key
    from backend.routing.router_context import user_for_session

    settings = get_settings()
    llm_cold = get_llm_cold()

    # user_id 추론: 명시적 전달 → 매핑 조회 → session_id 폴백
    uid = user_id or user_for_session(session_id) or session_id

    # 복합 키 생성
    redis_key = f"{uid}:{session_id}"
    redis_hist = RedisChatMessageHistory(session_id=redis_key, url=settings.REDIS_URL)

    # 인덱스 세트 등록 (user_id 기준)
    try:
        register_key(uid, f"message_store:{redis_key}")
    except Exception:
        pass

    # HybridSummaryMemory 생성
    return HybridSummaryMemory(
        llm=llm_cold,
        chat_memory=redis_hist,
        max_token_limit=settings.MAX_TOKEN_LIMIT,
        return_messages=True,
        memory_key="chat_history",
    )
