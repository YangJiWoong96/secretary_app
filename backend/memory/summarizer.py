"""
backend.memory.summarizer - 대화 요약 및 구조화

Redis 단기 메모리의 대화 히스토리를 구조화된 정보와 생성적 요약으로 변환합니다.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("summarizer")


class DialogueSummarizer:
    """
    대화 요약 및 구조화 클래스

    Redis 메모리가 토큰 한도를 초과할 때, 오래된 대화를 구조화+요약으로 압축합니다.

    특징:
    - 추출적 구조화 (엔티티, 목표, 태스크 등 JSON)
    - 생성적 요약 (자연어 요약)
    - 핀 고정 사실 보호 (프로필 기반)
    - 토큰 예산 관리
    """

    # ===== 프롬프트 템플릿 =====
    STRUCTURE_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", "대화에서 사실을 추출하라. JSON만 출력. 스키마:\n{schema_json}"),
            (
                "user",
                "[핀 고정(변경 금지)]:\n{pinned_json}\n\n[과거 대화]:\n{text_block}",
            ),
        ]
    )

    SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 대화 요약 시스템이다. 제공된 과거 대화를 사용자 선호도를 중점으로 최대 300~350 토큰내로 압축/요약하라. "
                "사실 추가/변경 금지, 애매하면 생략. 한국어 유지.",
            ),
            (
                "user",
                "[핀 고정(참고, 변경 금지)]:\n{pinned_json}\n\n[과거 대화]:\n{text_block}\n\n규칙: {verify_rule}",
            ),
        ]
    )

    # ===== 구조화 스키마 =====
    STRUCTURE_SCHEMA = {
        "entities": ["이름/장소/제품/조직"],
        "goals": ["사용자 목표"],
        "tasks": ["할일/액션아이템"],
        "deadlines": ["YYYY-MM-DD"],
        "facts": ["중요 사실"],
        "decisions": ["결정사항"],
        "constraints": ["제약/선호"],
        "references": ["링크/식별자"],
    }

    # ===== 검증 규칙 =====
    VERIFY_RULE = "절대 새로운 사실을 추가하지 말고, 원문에 없는 수치는 넣지 마라."

    def __init__(self):
        """DialogueSummarizer 초기화"""
        self._settings = None
        self._llm_cold = None
        self._tokenizer = None

    @property
    def settings(self):
        """설정 인스턴스 (지연 로딩)"""
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def llm_cold(self):
        """메모리/요약 전용 LLM (지연 로딩)"""
        if self._llm_cold is None:
            from backend.config import get_llm_cold

            self._llm_cold = get_llm_cold()
        return self._llm_cold

    @property
    def tokenizer(self):
        """Tiktoken 인코더 (지연 로딩)"""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def count_tokens_text(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산

        Args:
            text: 토큰 수를 계산할 텍스트

        Returns:
            int: 토큰 수
        """
        return len(self.tokenizer.encode(text))

    def count_tokens_msgs(self, messages) -> int:
        """
        메시지 리스트의 총 토큰 수 계산

        Args:
            messages: LangChain 메시지 리스트

        Returns:
            int: 총 토큰 수
        """
        total = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            msg_type = getattr(m, "type", "")
            total += len(self.tokenizer.encode(f"{msg_type}: {content}"))
        return total

    def messages_to_text(self, messages) -> str:
        """
        메시지 리스트를 텍스트로 변환

        Args:
            messages: LangChain 메시지 리스트

        Returns:
            str: "type: content" 형식의 텍스트
        """
        return "\n".join(
            f"{getattr(m, 'type', 'unknown')}: {getattr(m, 'content', '')}"
            for m in messages
        )

    def split_for_summary(
        self, messages, recent_budget: Optional[int] = None
    ) -> Tuple[list, list]:
        """
        메시지를 old/recent로 분리

        뒤에서부터 recent_budget만큼 토큰을 채우고, 나머지는 old로 분류합니다.

        Args:
            messages: LangChain 메시지 리스트
            recent_budget: 최근 원문 보존 토큰 예산 (None이면 설정값 사용)

        Returns:
            Tuple[list, list]: (old_msgs, recent_msgs)
        """
        if recent_budget is None:
            recent_budget = self.settings.RECENT_RAW_TOKENS_BUDGET

        recent, old = [], []
        remain = recent_budget

        for m in reversed(messages):
            content = getattr(m, "content", "") or ""
            tk = len(self.tokenizer.encode(content))

            if remain > 0:
                recent.append(m)
                remain -= tk
            else:
                old.append(m)

        recent.reverse()
        old.reverse()

        return old, recent

    @staticmethod
    def model_supports_response_format(model_name: str) -> bool:
        """
        모델이 response_format을 지원하는지 체크

        Args:
            model_name: 모델 이름

        Returns:
            bool: 지원 여부
        """
        try:
            m = (model_name or "").lower()
            return any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
        except Exception:
            return False

    def build_structured_and_summary(
        self, session_id: str, old_msgs
    ) -> Tuple[str, Dict]:
        """
        구조화 + 생성적 요약 생성

        오래된 메시지들을 구조화된 JSON + 자연어 요약으로 변환합니다.

        Args:
            session_id: 세션 ID (프로필 참조용)
            old_msgs: 요약할 오래된 메시지 리스트

        Returns:
            Tuple[str, Dict]: (결합된 요약 텍스트, 메타데이터)

        Example:
            >>> summarizer = DialogueSummarizer()
            >>> combo, meta = summarizer.build_structured_and_summary("user123", old_msgs)
            >>> print(combo)
            >>> # [STRUCTURED]
            >>> # {"entities": [...], "goals": [...]}
            >>> #
            >>> # [SUMMARY]
            >>> # 사용자는 서울에 거주하며...
        """
        # 고정 사실 추출
        from backend.policy.profile_utils import get_pinned_facts

        pinned = get_pinned_facts(session_id)
        old_text = self.messages_to_text(old_msgs)

        schema_str = json.dumps(self.STRUCTURE_SCHEMA, ensure_ascii=False)
        pinned_str = json.dumps(pinned, ensure_ascii=False)
        text_block = old_text

        # 1) 추출적 구조화 (JSON)
        t0 = time.time()
        try:
            # response_format 지원 모델이면 JSON 강제
            if self.model_supports_response_format(self.settings.LLM_MODEL):
                llm_struct = self.llm_cold.bind(response_format={"type": "json_object"})
            else:
                llm_struct = self.llm_cold

            struct = (self.STRUCTURE_PROMPT | llm_struct | StrOutputParser()).invoke(
                {
                    "schema_json": schema_str,
                    "pinned_json": pinned_str,
                    "text_block": text_block,
                }
            )
        except Exception as e:
            logger.warning(f"[summarizer] Structure LLM error: {e}")
            struct = "{}"

        t1 = (time.time() - t0) * 1000
        logger.info(f"[summarizer] Structure extraction took {t1:.1f}ms")

        # JSON 파싱
        try:
            struct_json = json.loads(struct)
        except Exception:
            struct_json = {
                k: []
                for k in [
                    "entities",
                    "goals",
                    "tasks",
                    "deadlines",
                    "facts",
                    "decisions",
                    "constraints",
                    "references",
                ]
            }

        # 2) 생성적 요약
        t0 = time.time()
        try:
            summ = (self.SUMMARY_PROMPT | self.llm_cold | StrOutputParser()).invoke(
                {
                    "pinned_json": pinned_str,
                    "text_block": text_block,
                    "verify_rule": self.VERIFY_RULE,
                }
            )
        except Exception as e:
            logger.warning(f"[summarizer] Generative summary error: {e}")
            summ = ""

        t1 = (time.time() - t0) * 1000
        logger.info(f"[summarizer] Generative summary took {t1:.1f}ms")

        # 결합
        combo = (
            "[STRUCTURED]\n"
            + json.dumps(struct_json, ensure_ascii=False)
            + "\n\n[SUMMARY]\n"
            + (summ or "").strip()
        )

        # 토큰 예산 제한
        tk = self.count_tokens_text(combo)
        target = self.settings.SUMMARY_TARGET_TOKENS

        if tk > target + 100:
            # 토큰 수 초과 시 자르기
            combo = self.tokenizer.decode(self.tokenizer.encode(combo)[:target])

        # 메타데이터
        meta = {
            "summary_version": "v1_struct+gen",
            "model": self.settings.LLM_MODEL,
            "pinned_count": len(pinned),
        }

        return combo, meta


# ===== 싱글톤 인스턴스 =====
_summarizer_instance: Optional["DialogueSummarizer"] = None


def get_summarizer() -> DialogueSummarizer:
    """
    전역 DialogueSummarizer 싱글톤 인스턴스 반환

    Returns:
        DialogueSummarizer: 전역 요약기 인스턴스
    """
    global _summarizer_instance

    if _summarizer_instance is None:
        _summarizer_instance = DialogueSummarizer()
        logger.info("[summarizer] DialogueSummarizer instance created")

    return _summarizer_instance


# ===== 호환성을 위한 함수형 인터페이스 =====


def count_tokens_text(text: str) -> int:
    """텍스트 토큰 수 (호환성 래퍼)"""
    summarizer = get_summarizer()
    return summarizer.count_tokens_text(text)


def count_tokens_msgs(messages) -> int:
    """메시지 리스트 토큰 수 (호환성 래퍼)"""
    summarizer = get_summarizer()
    return summarizer.count_tokens_msgs(messages)


def messages_to_text(messages) -> str:
    """메시지를 텍스트로 변환 (호환성 래퍼)"""
    summarizer = get_summarizer()
    return summarizer.messages_to_text(messages)


def model_supports_response_format(model_name: str) -> bool:
    """모델의 response_format 지원 여부 (호환성 래퍼)"""
    return DialogueSummarizer.model_supports_response_format(model_name)


def split_for_summary(
    messages, recent_budget: Optional[int] = None
) -> Tuple[list, list]:
    """메시지 분리 old/recent (호환성 래퍼)"""
    summarizer = get_summarizer()
    return summarizer.split_for_summary(messages, recent_budget)


def build_structured_and_summary(session_id: str, old_msgs) -> Tuple[str, Dict]:
    """구조화 + 요약 생성 (호환성 래퍼)"""
    summarizer = get_summarizer()
    return summarizer.build_structured_and_summary(session_id, old_msgs)


# ===== 전역 상수 export (호환성) =====
def get_structure_schema() -> Dict:
    """구조화 스키마 반환"""
    return DialogueSummarizer.STRUCTURE_SCHEMA


def get_verify_rule() -> str:
    """검증 규칙 반환"""
    return DialogueSummarizer.VERIFY_RULE


# Tiktoken 인코더 전역 참조 (호환성)
_enc_instance = None


def get_tokenizer():
    """Tiktoken 인코더 인스턴스 반환"""
    global _enc_instance
    if _enc_instance is None:
        _enc_instance = tiktoken.get_encoding("cl100k_base")
    return _enc_instance
