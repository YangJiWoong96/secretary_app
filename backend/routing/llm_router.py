"""
backend.routing.llm_router - LLM 기반 라우터

임베딩/소분류기로 확정하지 못한 애매한 케이스에 대해 LLM으로 라우팅합니다.
"""

import json
import logging

logger = logging.getLogger("llm_router")


class LLMRouter:
    """LLM 기반 라우터"""

    def __init__(self):
        self._settings = None

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def route_one_call(self, user_input: str, hist: str) -> dict:
        """
        단일 호출 라우팅: conv|rag|web + 쿼리/명확화

        Args:
            user_input: 사용자 입력
            hist: 대화 히스토리

        Returns:
            dict: {"route": str, "rag_query": str, "web_query": str, "clarify": str}

        주의(리팩토링 원칙):
        - 외부 시그니처/모델/토큰/response_format 사용 조건 및 반환 키는 절대 변경하지 않는다.
        - 내부 구현 가독성 개선을 위한 주석 추가만 수행한다.
        """
        from backend.utils.retry import openai_chat_with_retry

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 라우팅 에이전트다. 사용자 입력과 최근 히스토리를 보고, conv|rag|web|weather 중 하나만 고르라. "
                    "필요시 rag_query 또는 web_query를 생성하고, 불확실하면 clarify에 짧은 질문 1개를 넣어라. JSON만 출력."
                ),
            },
            {
                "role": "user",
                "content": f"[hist]\n{hist[-1500:]}\n\n[input]\n{user_input}",
            },
        ]

        # 1차: json_object 시도
        try:
            resp1 = await openai_chat_with_retry(
                model=self.settings.LLM_MODEL,
                messages=msgs,
                response_format={"type": "json_object"},
                max_tokens=self.settings.REWRITE_MAX_TOKENS,
            )
            content1 = (resp1.choices[0].message.content or "").strip()
            data1 = json.loads(content1)
            if isinstance(data1, dict) and data1.get("route"):
                return data1
        except Exception:
            pass

        # 2차: 프리폼 시도
        try:
            resp2 = await openai_chat_with_retry(
                model=self.settings.LLM_MODEL,
                messages=msgs,
                max_tokens=self.settings.REWRITE_MAX_TOKENS,
            )
            content2 = (resp2.choices[0].message.content or "").strip()

            # 안전 JSON 추출
            start = content2.find("{")
            end = content2.rfind("}")
            if start != -1 and end != -1 and end > start:
                data2 = json.loads(content2[start : end + 1])
                if isinstance(data2, dict) and data2.get("route"):
                    return data2
        except Exception:
            pass

        return {"route": "conv"}

    async def route_guard(self, user_input: str, hist: str) -> dict:
        """
        최종 라우팅 가드: conv/rag/web 강제 선택 또는 명확화

        Args:
            user_input: 사용자 입력
            hist: 대화 히스토리

        Returns:
            dict: {"route": str, "clarify": str}

        주의(리팩토링 원칙):
        - 모델/response_format 적용 조건, max_tokens는 기존과 동일하다.
        - 반환 딕셔너리의 키/값 제약은 변경하지 않는다.
        """
        from backend.memory import model_supports_response_format
        from backend.utils.retry import openai_chat_with_retry

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 최종 라우팅 가드다. 현재 입력이 인삿말/소통이면 'conv'를 반환하라."
                    "개인 과거 회상/지난 대화 내용이면 'rag', 외부 정보 탐색(로컬/시황/웹문서)이면 'web',"
                    "날씨/기상/기온/강수/미세먼지/우산/체감온도/예보/오늘/지명 등은 'weather'를 반환하라."
                    "불확실하면 clarify에 짧은 질문 1개만. JSON만: {route, clarify}."
                ),
            },
            {
                "role": "user",
                "content": f"[hist]\n{hist[-1500:]}\n\n[input]\n{user_input}",
            },
        ]

        try:
            kwargs = {
                "model": self.settings.LLM_MODEL,
                "messages": msgs,
                "max_tokens": 80,
            }
            if model_supports_response_format(self.settings.LLM_MODEL):
                kwargs["response_format"] = {"type": "json_object"}

            resp = await openai_chat_with_retry(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content) if content.startswith("{") else {}
            route = (data.get("route") or "").strip()
            clarify = (data.get("clarify") or "").strip()

            if route in ("conv", "rag", "web", "weather") or clarify:
                return {"route": route, "clarify": clarify}
        except Exception:
            pass

        return {"route": "", "clarify": ""}


_llm_router_instance = None


def get_llm_router():
    global _llm_router_instance
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter()
    return _llm_router_instance


async def router_one_call(user_input, hist):
    """호환성 래퍼"""
    router = get_llm_router()
    return await router.route_one_call(user_input, hist)


async def route_guard(user_input, hist):
    """호환성 래퍼"""
    router = get_llm_router()
    return await router.route_guard(user_input, hist)


async def llm_decider(user_input: str) -> str:
    """
    LLM 단일 경로 결정자. 임계값 미만일 때 호출하여 conv|rag|web 중 하나를 강제 선택한다.

    - 모델: gpt-4o-mini
    - temperature: 0.0 (결정적)
    - 출력: 'web' | 'rag' | 'conv' 중 하나의 단어

    주의(리팩토링 원칙):
    - 입력/출력/모델/온도/토큰 수 등은 변경하지 않는다.
    """
    from backend.utils.retry import openai_chat_with_retry

    system_prompt = (
        "너는 사용자 쿼리를 분석하여 'web'(웹 검색), 'rag'(내부 DB 검색), "
        "'conv'(단순 대화) 중 가장 적절한 단 하나의 경로를 결정하는 라우팅 전문가이다. "
        "반드시 세 단어 중 하나로만 답변해야 한다."
    )

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input or ""},
    ]

    try:
        resp = await openai_chat_with_retry(
            model="gpt-4o-mini", messages=msgs, temperature=0.0, max_tokens=4
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        # 관용적 표현/노이즈 제거 후 안전 파싱
        for token in ("web", "rag", "conv"):
            if token in raw:
                return token
        # 단어 경계 기준으로 첫 토큰 검사
        import re as _re

        m = _re.search(r"\b(web|rag|conv)\b", raw)
        if m:
            return m.group(1)
    except Exception:
        pass

    return "conv"
