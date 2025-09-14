import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI


THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")


class VState(dict):
    """검증 상태(간단 TypedDict 대체). keys: draft, contexts, policy, report."""


def _node_screen(state: VState) -> VState:
    """안전/보안/정책 위반 스크리닝 및 최소 보정"""
    llm = ChatOpenAI(model=THINKING_MODEL, temperature=0.1)
    draft = state.get("draft", {}) or {}
    contexts = state.get("contexts", {}) or {}
    sys = (
        "넌 지시문/시스템 프롬프트 검증자다. 보안/윤리/개인정보/허위/과장/정책 위반을 점검하고, "
        "사용자 취향의 일관성을 해치지 않는 선에서 최소 보정만 하라. 과격한 톤 전환 금지."
    )
    user = (
        f"[초안]\n{draft}\n\n[맥락]\n{contexts}\n\n"
        "출력: JSON {approved:boolean, reasons:[string], fixed:object}"
        "(approved=false면 fixed는 보정안)."
    )
    schema = {
        "name": "DirGuard",
        "schema": {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "fixed": {"type": "object"},
            },
            "required": ["approved"],
            "additionalProperties": False,
        },
    }
    try:
        resp = llm.invoke(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        )
        content = getattr(resp, "content", "") or ""
        import json

        data = json.loads(content) if content.startswith("{") else {}
        state["report"] = data
        if data.get("approved", True):
            return state
        fixed = data.get("fixed") or {}
        # 최소 보정 반영
        state["draft"] = fixed or draft
        return state
    except Exception:
        return state


def build_validator_graph():
    builder = StateGraph(VState)
    builder.add_node("screen", _node_screen)
    builder.set_entry_point("screen")
    builder.add_edge("screen", END)
    return builder.compile(checkpointer=MemorySaver())


GRAPH = build_validator_graph()


def validate_directives(
    draft: Dict[str, Any], contexts: Dict[str, Any]
) -> Dict[str, Any]:
    state: VState = {"draft": draft, "contexts": contexts}
    out: VState = GRAPH.invoke(state)
    return out.get("draft", draft)
