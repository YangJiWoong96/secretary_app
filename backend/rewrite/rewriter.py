"""
backend.rewrite.rewriter - 쿼리 재작성

RAG/WEB 검색을 위한 쿼리 재작성 로직을 제공합니다.
"""

import logging
import re
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger("rewriter")

from backend.utils.tracing import traceable

# 재작성 시스템 프롬프트
RAG_REWRITE_SYS = (
    "너는 Retrieval을 위한 쿼리 리라이팅 전문가다. 사용자의 의도를 파악하여 입력 문장에서 핵심 주제나 사실적 요청만 남기고, 임의로 관련없는 내용을 추가/추측하지 마라. "
    "모호한 지시어나 상대시제는 realtime_ctx 기준으로 명확하게 해석하여 이해하되, 최종 출력 문장에는 날짜 표현을 넣지마라."
    "쿼리에 대한 RAG DB에 가장 효율적인 검색 문장을 만들어라. 출력은 핵심 키워드 또는 짧은 문장 형태로 하며, 구두점은 제거하라. "
)

WEB_REWRITE_SYS = (
    "너는 웹검색 쿼리 리라이팅 전문가다. 사용자의 현재 질문뿐 아니라 최근 대화 맥락을 고려하여,"
    " 실제 검색엔진에서 인간이 쓸 수 있는 자연스러운 질의로 재작성하라."
    " 사용자 이름/감탄사/지시어(예: 알아봐줘, 해줘, 검색해봐)는 제거하되, 핵심 엔티티(특히 지명/장소)는 반드시 보존하라."
    " 현재 발화에 지명이 명시되지 않았더라도, 직전 대화에서 장소가 정해진 상태라면 그 장소를 유지(carry-over)하라."
    " 단, 근거가 불충분한 추측은 금지하고, 최근 맥락으로 정당화되는 경우에만 포함하라."
    " realtime_ctx는 현재 시점을 이해하는 데만 사용하라. 불필요한 구두점을 제거하고, 핵심 키워드 2~6개로 출력하라."
    " 지명이 존재하면 사람이 검색엔진에 입력하는 자연스러운 키워드 형태(예: '분위기 좋은 용산 카페')로 재구성하라."
)


class QueryRewriter:
    """쿼리 재작성 클래스"""

    def __init__(self):
        self._settings = None

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    def last_user_utterance(self, hist: str) -> str:
        """히스토리에서 마지막 사용자 발화 추출 (휴리스틱 최소화)"""
        for line in reversed((hist or "").splitlines()):
            if line.lower().startswith(("human:", "user:")):
                return line.split(":", 1)[1].strip()
        return ""

    def _hist_tail(self, hist: str, max_lines: int = 12) -> str:
        """대화 히스토리의 최근 max_lines만 추출"""
        lines = [ln for ln in (hist or "").splitlines() if ln.strip()]
        return "\n".join(lines[-max_lines:])

    @traceable(name="Rewrite: rewrite_query", run_type="chain", tags=["rewrite"])
    async def rewrite_query(
        self,
        task: str,
        user_input: str,
        hist: str,
        anchor_hint: Optional[str] = None,
        session_id: Optional[str] = None,
        preview_ctx: Optional[str] = None,
        realtime_ctx: Optional[str] = None,
    ) -> Dict:
        """
        쿼리 재작성 (RAG 또는 WEB)

        Args:
            task: "rag" 또는 "web"
            user_input: 사용자 입력
            hist: 대화 히스토리
            anchor_hint: 앵커 힌트 (선택)
            session_id: 세션 ID (선택)

        Returns:
            Dict: RAG 시 {"query_text": str, "date_filter": tuple}, WEB 시 {"web_query": str}
        """
        from backend.utils.datetime_utils import (
            extract_date_range_for_rag,
            month_tokens_for_web,
        )
        from backend.utils.retry import rewrite_with_retries

        # 1) 재작성 대상은 항상 "마지막 사용자 발화"만으로 한정한다.
        #    preview_ctx/realtime_ctx는 참고 단서로만 제공한다. (출력 토큰 주입 금지)
        base = (user_input or "").strip()
        preview_ctx_text = (preview_ctx or "").strip()
        realtime_ctx_text = (realtime_ctx or "").strip()

        if task == "rag":
            date_range = extract_date_range_for_rag(base)
            q_rules = base

            # 힌트 생성 (컨텍스트에서)
            hint_text = (
                await self._extract_hints(session_id, user_input) if session_id else ""
            )

            t0 = time.perf_counter()
            # 시스템 가드: 마지막 발화의 명시 키워드 보존(특히 지명/인물/제품)
            sys_guard = (
                "[규칙 - 반드시 준수]\n"
                "- 재작성을 '마지막 사용자 질문'을 기준으로 진행하라!\n"
                "- preview/요약/라우팅 메타 및 이전 맥락은 현 사용자 질문을 이해하기 위해 반드시 참고하되, 재작성에 출력 하지마라.\n"
                "- 마지막 사용자 질문의 도메인/카테고리 키워드를 반드시 보존하라.\n"
            )

            msgs = [
                {
                    "role": "system",
                    "content": RAG_REWRITE_SYS
                    + " 참고 단서는 모호성 해소용이며, 출력에 외부 단어를 주입하지 말라.",
                },
                {"role": "system", "content": sys_guard},
                {"role": "user", "content": f"재작성 대상:\n{q_rules}"},
            ]
            if preview_ctx_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"[참고 단서(출력에 주입 금지)]\n{preview_ctx_text}",
                    }
                )
            if realtime_ctx_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": (
                            "[현재 시각]\n"
                            + realtime_ctx_text
                            + "\n상대시제(올해/내년/작년/이번달/다음달 등)는 현재 시각을 기준으로 절대 날짜로 해석하되, 출력 쿼리에는 날짜 토큰을 직접 넣지 말라."
                        ),
                    }
                )
            if hint_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"[참고 단서(출력에 주입 금지)]\n{hint_text}",
                    }
                )

            out = await rewrite_with_retries(
                msgs,
                base_timeout_s=max(self.settings.REWRITE_TIMEOUT_S, 1.8),
                attempts=2,
                delta_s=0.8,
                max_tokens=self.settings.REWRITE_MAX_TOKENS,
            )
            query_text = out or q_rules

            took_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[rewriter:RAG] base='{q_rules[:80]}' → out='{query_text[:80]}' took_ms={took_ms:.1f}"
            )

            return {"query_text": query_text, "date_filter": date_range}

        # WEB 케이스
        else:
            q_rules = base
            web_query = q_rules

            # 힌트 생성
            hint_text = (
                await self._extract_hints(session_id, base) if session_id else ""
            )

            # JSON 스키마 강제 (표준 빌더 사용)
            from backend.utils.schema_builder import build_json_schema
            from backend.utils.schema_registry import get_keyword_query_schema

            schema = build_json_schema(
                "KeywordQuery", get_keyword_query_schema(), strict=True
            )

            t0 = time.perf_counter()
            sys_guard = (
                "[규칙 - 반드시 준수]\n"
                "- 재작성의 기준은 '마지막 사용자 질문'이지만, 질문이 모호하거나 지명이 빠졌으면 최근 대화 맥락에서 핵심 엔티티(지명/인물/제품 등)를 보완하여 포함하라.\n"
                "- 특히 직전 턴에서 명시된 위치(예: 강남역, 이태원 등)는 현재 질문에 위치가 없으면 반드시 유지해라(carry-over).\n"
                "- 새로운 사실/지명을 발명하지 마라. 최근 맥락으로 정당화되는 경우에만 포함하라.\n"
                "- 날짜/연도/월 등의 시점 토큰을 임의로 추가하지 마라.\n"
                "- 지명이 존재하면 '지명 카테고리 수식' 순서의 2~6개 키워드로 간결히 표현하라.\n"
            )

            hist_tail = self._hist_tail(hist, max_lines=12)

            msgs = [
                {
                    "role": "system",
                    "content": WEB_REWRITE_SYS
                    + " 참고 단서는 오탈자/모호성 해소용. 외부 단어 주입 금지.",
                },
                {"role": "system", "content": sys_guard},
                {
                    "role": "user",
                    "content": f"입력: {q_rules}\n출력: 2~6개 키워드만 공백으로 구분.",
                },
            ]
            if hist_tail:
                msgs.append(
                    {"role": "system", "content": f"[최근 대화 맥락]\n{hist_tail}"}
                )
            if preview_ctx_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"[맥락 단서(필요시 출력에 반영)]\n{preview_ctx_text}",
                    }
                )
            if hint_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": f"[맥락 단서(필요시 출력에 반영)]\n{hint_text}",
                    }
                )
            if realtime_ctx_text:
                msgs.append(
                    {
                        "role": "system",
                        "content": (
                            "[현재 시각]\n"
                            + realtime_ctx_text
                            + "\n상대시제를 현재 시각으로 이해하되, 최종 출력 키워드에는 날짜/연도/월/일 등 시점 토큰을 포함하지 마라."
                        ),
                    }
                )

            import json

            out_json = await rewrite_with_retries(
                msgs,
                base_timeout_s=min(
                    max(
                        self.settings.REWRITE_TIMEOUT_S, self.settings.TIMEOUT_WEB * 0.6
                    ),
                    2.2,
                ),
                attempts=2,
                delta_s=0.8,
                max_tokens=self.settings.REWRITE_MAX_TOKENS,
                response_format=schema,
            )

            if out_json:
                # 1) JSON 파싱 시도
                try:
                    data = json.loads(out_json)
                    cand = (data.get("q") or "").strip()
                    if 2 <= len(cand.split()) <= 6 and not re.search(r"[.,!?]", cand):
                        web_query = cand
                    else:
                        # 2) 프리폼 토큰 추출 2~6개 허용
                        toks = re.findall(r"[\w가-힣]+", out_json)
                        if 2 <= len(toks) <= 6:
                            web_query = " ".join(toks)
                        else:
                            web_query = (out_json or q_rules).strip()
                except Exception:
                    # JSON 파싱 실패 폴백: 'q' 값을 정규식으로 추출 시도 → 실패 시 키워드만 추출
                    try:
                        m = re.search(r'"q"\s*:\s*"([^"]+)"', out_json)
                        if m:
                            cand = m.group(1).strip()
                            if 2 <= len(cand.split()) <= 6 and not re.search(
                                r"[.,!?]", cand
                            ):
                                web_query = cand
                            else:
                                toks = re.findall(r"[\w가-힣]+", cand)
                                web_query = (
                                    " ".join(toks[:6])
                                    if toks
                                    else (q_rules or "").strip()
                                )
                        else:
                            toks = re.findall(r"[\w가-힣]+", out_json)
                            if 2 <= len(toks) <= 6:
                                web_query = " ".join(toks)
                            else:
                                web_query = (q_rules or "").strip()
                    except Exception:
                        web_query = (q_rules or "").strip()
            else:
                # 타임아웃/오류 폴백: 마지막 발화만 사용
                web_query = q_rules
                logger.warning(
                    "[rewriter:WEB] fallback=user_only (timeout/error), using last utterance only"
                )

            web_query = re.sub(r"\s+", " ", web_query).strip()

            took_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[rewriter:WEB] base='{q_rules[:80]}' → out='{web_query[:80]}' took_ms={took_ms:.1f}"
            )

            return {"web_query": web_query}

    async def _extract_hints(self, session_id: str, user_input: str) -> str:
        """컨텍스트에서 힌트 추출"""
        try:
            from backend.context.unified import UnifiedContextualChatbot

            unified = UnifiedContextualChatbot()
            focused = await unified.query_processor.extract_query_focused_context(
                session_id, user_input
            )

            hint_bits = []
            for key in ("stwm", "redis_recent", "summaries"):
                for it in (focused.get(key) or [])[:2]:
                    c = str(it.get("content") or "").strip()
                    if c:
                        toks = [t for t in re.split(r"\W+", c) if len(t) >= 2][:5]
                        if toks:
                            hint_bits.append(" ".join(toks[:3]))

            return ("\n".join(hint_bits))[:200]
        except Exception:
            return ""


_rewriter_instance = None


def get_rewriter():
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter()
    return _rewriter_instance


@traceable(name="Rewrite: rewrite_query_wrapper", run_type="chain", tags=["rewrite"])
async def rewrite_query(
    task,
    user_input,
    hist,
    anchor_hint=None,
    session_id=None,
    preview_ctx=None,
    realtime_ctx=None,
):
    """
    호환성 래퍼

    - 외부 API 표면/반환 포맷을 변경하지 않는다.
    - 내부 QueryRewriter 구현에 대한 thin wrapper이며, 호출 시그니처와 키는 기존과 동일하다.
    """
    rewriter = get_rewriter()
    return await rewriter.rewrite_query(
        task,
        user_input,
        hist,
        anchor_hint,
        session_id,
        preview_ctx,
        realtime_ctx,
    )
