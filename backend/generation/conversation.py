"""
backend.generation.conversation - 대화 체인

순수 대화 모드 응답 생성을 담당합니다.
"""

import logging
import time

from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger("conversation")


class ConversationGenerator:
    """순수 대화 응답 생성기"""

    def __init__(self):
        self._settings = None

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def generate(
        self, session_id: str, user_input: str, stm: ConversationSummaryBufferMemory
    ) -> str:
        """
        순수 대화 모드 응답 생성

        Args:
            session_id: 세션 ID
            user_input: 사용자 입력
            stm: 단기 메모리

        Returns:
            str: 생성된 응답 (SINGLE_CALL_CONV이면 빈 문자열)
        """
        from backend.directives.store import get_compiled as get_compiled_directives
        from backend.utils.retry import openai_chat_with_retry

        # 사용자 고정 취향 JSON 로드
        slot_sys, _ = get_compiled_directives(session_id)

        # 히스토리 + 현재 입력 프롬프트
        hist = "\n".join(f"{m.type}: {m.content}" for m in stm.chat_memory.messages)
        prompt = (
            "너는 한국어 범용 비서앱이다. 사용자의 지속적 취향(JSON 지시문)이 있다면 우선 적용하라.\n"
            "빈 RAG/Web 컨텍스트를 언급하지 말고, 실시간성/시스템 메타 발언 없이 자연스럽게 답하라.\n"
            "두괄식으로 핵심을 먼저 말하고, 필요하면 최소한으로만 덧붙여라.\n"
            f"[대화 히스토리]\n{hist}\n"
            f"[최신 입력]\n{user_input}"
        )

        # LLM 호출 메시지
        messages = ([{"role": "system", "content": slot_sys}] if slot_sys else []) + [
            {"role": "system", "content": "대화 전개 지침을 따르라."},
            {"role": "user", "content": prompt},
        ]

        # SINGLE_CALL_CONV 모드면 main_response에서 처리
        if self.settings.SINGLE_CALL_CONV:
            return ""

        # LLM 호출
        t0 = time.time()
        resp = await openai_chat_with_retry(
            model=self.settings.LLM_MODEL,
            messages=messages,
            temperature=1.0,
        )
        took = (time.time() - t0) * 1000
        content = (resp.choices[0].message.content or "").strip()

        logger.info(
            f"[conversation] model={resp.model} took_ms={took:.1f} output_len={len(content)}"
        )

        return content


_conv_generator_instance = None


def get_conversation_generator():
    global _conv_generator_instance
    if _conv_generator_instance is None:
        _conv_generator_instance = ConversationGenerator()
    return _conv_generator_instance


async def conversation_chain(session_id, user_input, stm):
    """호환성 래퍼"""
    generator = get_conversation_generator()
    return await generator.generate(session_id, user_input, stm)
