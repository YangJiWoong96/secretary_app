"""
backend.generation.filters - 컨텍스트 필터링

RAG/WEB 컨텍스트와 사용자 질의 간 의미 불일치를 감지하여 필터링합니다.
"""

import asyncio
import json
import logging
import re

logger = logging.getLogger("filters")


class ContextFilter:
    """컨텍스트 필터 클래스"""

    def __init__(self):
        self._settings = None

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def filter_rag_semantic_mismatch(self, user_input: str, rag_ctx: str) -> str:
        """RAG 컨텍스트 의미 불일치 필터링"""
        if not rag_ctx or len(rag_ctx) < 60:
            return rag_ctx

        from backend.utils.retry import openai_chat_with_retry
        from backend.utils.schema_registry import get_rag_filter_schema

        schema_inner = get_rag_filter_schema()

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 RAG 컨텍스트 필터다. 사용자 질문과 무관하거나 주제적으로 상충하는 컨텍스트는 제거한다. "
                    "특히 장소/업종/카테고리가 다르면(예: 바 vs 한식집) 제거하라."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[질문]\n{user_input}\n\n[컨텍스트]\n{rag_ctx}\n\n"
                    "규칙: 1) 질문과 무관/상충 부분은 삭제한다. 2) 관련된 부분만 남긴다. 3) 결과는 JSON만.\n"
                    "스키마: {keep:boolean, filtered:string}"
                ),
            },
        ]

        try:
            from backend.memory import model_supports_response_format
            from backend.utils.schema_builder import build_json_schema

            kwargs = {
                "model": self.settings.LLM_MODEL,
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": 220,
            }
            if model_supports_response_format(self.settings.LLM_MODEL):
                kwargs["response_format"] = build_json_schema(
                    "RagFilter", schema_inner, strict=True
                )

            resp = await asyncio.wait_for(
                openai_chat_with_retry(**kwargs),
                timeout=min(0.9, self.settings.TIMEOUT_RAG),
            )

            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content) if content.startswith("{") else {}
            keep = bool(data.get("keep", False))

            if not keep:
                # 완전 제거는 검색 응답 품질을 급락시킴 → 최소 안전 폴백(형식 보존) 반환
                try:
                    from backend.search_engine.formatter_guard import ensure_block_shape

                    safe_min = ensure_block_shape(rag_ctx)
                    return safe_min or (rag_ctx or "")
                except Exception:
                    return rag_ctx or ""

            filtered = (data.get("filtered") or "").strip()
            return filtered or rag_ctx
        except Exception:
            return rag_ctx

    async def filter_web_context(self, user_input: str, web_ctx: str) -> str:
        """WEB 컨텍스트 필터링"""
        if not web_ctx or len(web_ctx) < 30:
            return web_ctx

        # 소규모 인사/잡담이면 제거
        if self._is_small_talk(user_input):
            return ""

        from backend.utils.retry import openai_chat_with_retry
        from backend.utils.schema_registry import get_web_filter_schema

        schema_inner = get_web_filter_schema()

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 WEB 컨텍스트 필터다. 사용자 질문과 무관하거나 주제적으로 상충하는 웹 결과 블록은 제거한다. "
                    "블록 형식(이름/간단 설명/링크)과 링크는 유지하라."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[질문]\n{user_input}\n\n[WEB 컨텍스트]\n{web_ctx}\n\n"
                    "규칙: 1) 질문과 무관/상충 블록은 삭제. 2) 관련 블록만 유지. 3) 결과는 JSON만.\n"
                    "스키마: {keep:boolean, filtered:string}"
                ),
            },
        ]

        try:
            from backend.memory import model_supports_response_format
            from backend.utils.schema_builder import build_json_schema

            kwargs = {
                "model": self.settings.LLM_MODEL,
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": 220,
            }
            if model_supports_response_format(self.settings.LLM_MODEL):
                kwargs["response_format"] = build_json_schema(
                    "WebFilter", schema_inner, strict=True
                )

            resp = await asyncio.wait_for(
                openai_chat_with_retry(**kwargs),
                timeout=min(0.9, self.settings.TIMEOUT_WEB),
            )

            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content) if content.startswith("{") else {}
            keep = bool(data.get("keep", False))

            if not keep:
                return ""

            filtered = (data.get("filtered") or "").strip()

            # 블록 형식 보존
            try:
                from backend.search_engine.formatter_guard import ensure_block_shape

                safe = ensure_block_shape(filtered or web_ctx)
                return safe or (filtered or web_ctx)
            except Exception:
                return filtered or web_ctx
        except Exception:
            # 폴백: 토큰 기반 간단 필터
            try:
                # 금융 블록(시세)은 필터링을 건너뛰기 위해 title 라벨을 탐지하여 유지
                if "실시간 가격:" in (web_ctx or ""):
                    return web_ctx
                toks = [t for t in re.split(r"\W+", (user_input or "")) if len(t) >= 2][
                    :6
                ]
                if not toks:
                    return ""
                blocks = (web_ctx or "").split("\n\n")
                kept = [b for b in blocks if any(t in b for t in toks)]
                return "\n\n".join(kept) if kept else ""
            except Exception:
                return ""

    @staticmethod
    def _is_small_talk(text: str) -> bool:
        """소규모 인사/잡담 감지"""
        t = re.sub(r"\s+", " ", text or "").strip()
        if not t:
            return True

        small_talk_pat = re.compile(
            r"^(안녕|하이|헬로|hello|hi|반가워|고마워|감사|ㅎㅎ|ㅋㅋ)\b", re.IGNORECASE
        )

        if small_talk_pat.search(t):
            return True

        generic_pat = re.compile(r"(뭐하니|뭐해|어때)\b")
        if len(t) <= 20 and generic_pat.search(t):
            return True

        return False


_filter_instance = None


def get_context_filter():
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = ContextFilter()
    return _filter_instance


async def filter_semantic_mismatch(user_input, rag_ctx):
    filter_inst = get_context_filter()
    return await filter_inst.filter_rag_semantic_mismatch(user_input, rag_ctx)


async def filter_web_ctx(user_input, web_ctx):
    filter_inst = get_context_filter()
    return await filter_inst.filter_web_context(user_input, web_ctx)
