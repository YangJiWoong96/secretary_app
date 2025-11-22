# backend/proactive/planner.py
# LangChain + LangGraph 기반 프로액티브 질문 플래너 (타이밍 계산 없음)
# 요구: langchain>=0.2, langchain-openai>=0.1, langgraph>=0.2, google-cloud-firestore, pymilvus

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, TypedDict

from google.cloud import firestore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pymilvus import Collection, connections

from backend.rag import ensure_collections
from backend.rag.embeddings import embed_query_cached

# ======================
# 환경 변수 / 상수
# ======================
FIRESTORE_USERS_COLL = "users"
UNIFIED_EVENTS_SUBCOLL = "unified_events"  # ingest/main.py 저장 경로
PUSHLOG_SUBCOLL = "proactive_push_logs"  # (선택) 과거 푸시/반응 로그

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5")
THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# app.py와 동일한 규칙으로 임베딩 차원 및 컬렉션 이름 정렬(v3)
_EMBED_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM", _EMBED_DIM_MAP.get(EMBEDDING_MODEL, 1536))
)
PROFILE_COLLECTION_NAME = f"user_profiles_v3_{EMBEDDING_DIM}d"  # Milvus 컬렉션 (프로필)
LOG_COLLECTION_NAME = (
    f"conversation_logs_v3_{EMBEDDING_DIM}d"  # Milvus 컬렉션 (요약 로그)
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


# ======================
# 클라이언트 초기화 (지연)
# ======================
# Firestore 클라이언트 (공용 싱글톤 사용)
def _fs():
    """Firestore 클라이언트 반환 (backend.config.clients 공용 싱글톤)"""
    from backend.config import get_firestore_client

    return get_firestore_client()


# Milvus
if not connections.has_connection("default"):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# LLM/Embeddings
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=THINKING_MODEL,
)
# 임베딩은 backend.rag.embeddings의 추상화(자동 백엔드 선택)만 사용


# ======================
# LangGraph State 정의
# ======================
class PlannerState(TypedDict, total=False):
    # 입력
    user_id: str
    lookback_days: int
    top_n: int

    # 중간 산출물
    recent: Dict[str, List[Dict]]  # {"events":[...], "push_logs":[...]}
    recent_summary: str  # Firestore 이벤트 요약 문자열
    rag_queries: List[str]  # Milvus 검색용 질의어
    rag_profile: str  # RAG 프로필 컨텍스트
    rag_logs: str  # RAG 대화 로그 컨텍스트
    candidates: List[Dict]  # LLM 후보 질문 [{"q":..., "tags":[...]}]
    scored: List[Tuple[float, Dict]]  # [(score, candidate), ...]

    # 최종 출력
    output: Dict[str, any]


# ======================
# 유틸
# ======================
def _uniq_trim(xs: List[str], max_len: int) -> List[str]:
    seen, acc, cur = set(), [], 0
    for s in xs:
        if not s or s in seen:
            continue
        seen.add(s)
        cur += len(s)
        if cur > max_len:
            break
        acc.append(s)
    return acc


def _summarize_events(events: List[Dict], limit: int = 15) -> str:
    out = []
    for e in events[:limit]:
        t = e.get("dataType")
        p = e.get("payload", {})
        if t == "LOCATION":
            addr = p.get("address")
            if addr:
                out.append(f"loc:{addr}")
            else:
                out.append(f"loc:({p.get('latitude')},{p.get('longitude')})")
        elif t == "CALENDAR_UPDATE":
            out.append(f"cal:{len(p.get('events', []))} events")
        elif t == "PREFERENCE_UPDATE":
            out.append(f"pref:{list(p.keys())[:3]}")
    return "; ".join(out)


def _extract_json(text: str) -> Dict:
    # LLM이 텍스트 앞뒤에 코멘트를 붙이는 실수를 대비
    try:
        return json.loads(text)
    except Exception:
        # 가장 바깥 { ... } 추출 시도 (간단)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}


# ======================
# 노드 구현
# ======================
def node_fetch_recent(state: PlannerState) -> PlannerState:
    """Firestore에서 lookback_days 기간의 이벤트/푸시 로그 조회 + 요약"""
    user_id = state["user_id"]
    lookback_days = state.get("lookback_days", 7)
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    db = _fs()
    events: List[Dict] = []
    push_logs: List[Dict] = []
    if db is not None:
        try:
            ev_q = (
                db.collection(FIRESTORE_USERS_COLL)
                .document(user_id)
                .collection(UNIFIED_EVENTS_SUBCOLL)
                .where("recordTimestamp", ">=", since)
                .order_by("recordTimestamp", direction=firestore.Query.DESCENDING)
                .limit(200)
            )
            events = [d.to_dict() for d in ev_q.stream()]
        except Exception:
            events = []
        try:
            pl_q = (
                db.collection(FIRESTORE_USERS_COLL)
                .document(user_id)
                .collection(PUSHLOG_SUBCOLL)
                .where("timestamp", ">=", since)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(200)
            )
            push_logs = [d.to_dict() for d in pl_q.stream()]
        except Exception:
            push_logs = []

    recent_summary = _summarize_events(events) if events else ""

    return {
        **state,
        "recent": {"events": events, "push_logs": push_logs},
        "recent_summary": recent_summary,
    }


def node_build_queries(state: PlannerState) -> PlannerState:
    """Milvus 검색용 질의어 구성"""
    queries = ["사용자 장기 취향/습관 핵심 요약"]
    recent = state.get("recent", {})
    if recent.get("events"):
        queries += [
            "최근 사용자 위치/동선 관련 관심사",
            "다가오는 일정/준비물/리마인드",
        ]
    return {**state, "rag_queries": queries}


def node_search_rag(state: PlannerState) -> PlannerState:
    """Milvus에서 개인 프로필/로그 검색해 컨텍스트 구성"""
    user_id = state["user_id"]
    queries = state.get("rag_queries", []) or ["사용자 장기 취향/습관 핵심 요약"]

    prof, logs = ensure_collections()

    prof_acc, logs_acc = [], []
    sp = {"metric_type": "COSINE", "params": {"ef": 64}}

    for q in queries:
        qv = embed_query_cached(q)

        pr = prof.search(
            [qv],
            "embedding",
            sp,
            limit=1,
            expr=f"user_id == '{user_id}'",
            output_fields=["text"],
        )
        lr = logs.search(
            [qv],
            "embedding",
            sp,
            limit=5,
            expr=f"user_id == '{user_id}'",
            output_fields=["text"],
        )

        if pr and pr[0]:
            prof_acc += [h.entity.get("text") for h in pr[0] if h.entity.get("text")]
        if lr and lr[0]:
            logs_acc += [h.entity.get("text") for h in lr[0] if h.entity.get("text")]

    profile_ctx = "\n".join(_uniq_trim(prof_acc, 2000))
    logs_ctx = "\n".join(_uniq_trim(logs_acc, 4000))

    return {**state, "rag_profile": profile_ctx, "rag_logs": logs_ctx}


def node_generate_candidates(state: PlannerState) -> PlannerState:
    """LLM으로 후보 질문 8~12개 생성(JSON)"""
    profile_ctx = state.get("rag_profile", "")
    logs_ctx = state.get("rag_logs", "")
    recent_ctx = state.get("recent_summary", "")

    prompt = f"""너는 사용자 맞춤 ‘선(先)상호작용)’ 질문을 고안하는 플래너다.
[개인 프로필]
{profile_ctx}

[최근 대화/행동 요약]
{logs_ctx}

[최근 위치·캘린더·선호 요약]
{recent_ctx}

규칙:
1) 위 컨텍스트 ‘안에서만’ 질문을 만든다(추측 금지).
2) 8~18자 한국어 간결 질문 8개 생성.
3) 각 질문에 목적 태그를 붙인다 (예: #건강 #일정 #습관 #관계 #할일 #회고).
4) 질문은 서로 중복/의미충돌 금지.
결과는 JSON만:
{{"candidates":[{{"q":"...", "tags":["#..."]}}, ...]}}
"""
    messages = [
        SystemMessage(content="컨텍스트 밖 추측 금지. 반드시 JSON만 반환."),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    text = (resp.content or "").strip()
    data = _extract_json(text)
    cands = data.get("candidates", []) if isinstance(data, dict) else []
    return {**state, "candidates": cands[:12]}


def node_score_candidates(state: PlannerState) -> PlannerState:
    """정책 스코어링(개인화/신선도/피로도) → Top-N 및 최종 출력"""
    top_n = state.get("top_n", 3)
    cands = state.get("candidates", []) or []
    recent = state.get("recent", {}) or {}
    pushes = recent.get("push_logs", [])

    recent_q = set(p.get("q") for p in pushes[:50] if p.get("q"))
    last24 = [
        p
        for p in pushes
        if p.get("timestamp")
        and (datetime.now(timezone.utc) - p["timestamp"]).total_seconds() < 86400
    ]

    scored: List[Tuple[float, Dict]] = []
    for c in cands:
        q = (c.get("q") or "").strip()
        tags = c.get("tags", [])
        if not q:
            continue

        # 개인화 가중(간단 태그 기반; 필요 시 키워드-프로필 매칭으로 고도화)
        personal = (
            1.0 if any(t in ["#습관", "#일정", "#건강", "#할일"] for t in tags) else 0.7
        )

        # 신선도: 최근 동일 질문 회피
        novelty = 0.2 if q in recent_q else 1.0

        # 피로도: 최근 24h 발송량에 따른 패널티
        fatigue_pen = 1.0 / (1.0 + 0.3 * max(len(last24), 0))

        score = personal * novelty * fatigue_pen
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [
        {"score": round(s, 3), "q": c["q"], "tags": c.get("tags", [])}
        for s, c in scored[:top_n]
    ]

    output = {
        "user_id": state["user_id"],
        "top_questions": picks,
        "debug": {
            "recent_summary": state.get("recent_summary", ""),
            "queries": state.get("rag_queries", []),
            "profile_len": len(state.get("rag_profile", "")),
            "logs_len": len(state.get("rag_logs", "")),
            "cand_count": len(cands),
        },
    }
    return {**state, "scored": scored, "output": output}


# ======================
# 그래프 정의
# ======================
builder = StateGraph(PlannerState)

builder.add_node("fetch_recent", node_fetch_recent)
builder.add_node("build_queries", node_build_queries)
builder.add_node("search_rag", node_search_rag)
builder.add_node("generate_candidates", node_generate_candidates)
builder.add_node("score_candidates", node_score_candidates)

builder.set_entry_point("fetch_recent")
builder.add_edge("fetch_recent", "build_queries")
builder.add_edge("build_queries", "search_rag")
builder.add_edge("search_rag", "generate_candidates")
builder.add_edge("generate_candidates", "score_candidates")
builder.add_edge("score_candidates", END)

# 체크포인트 메모리 활성화(경량)로 상태/재시도 기록 가능
_checkpoint = MemorySaver()
GRAPH = builder.compile(checkpointer=_checkpoint)


# ======================
# 외부 호출 함수
# ======================
def plan_proactive_questions(
    user_id: str, lookback_days: int = 7, top_n: int = 3
) -> Dict:
    """
    LangGraph 파이프라인을 통해 개인화 질문 Top-N을 산출.
    동기 실행 버전(서버 사이드에서 바로 호출 가능).
    """
    initial: PlannerState = {
        "user_id": user_id,
        "lookback_days": lookback_days,
        "top_n": top_n,
    }
    final_state: PlannerState = GRAPH.invoke(initial)
    return final_state.get("output", {"user_id": user_id, "top_questions": []})


# (선택) 간단한 로컬 테스트
if __name__ == "__main__":
    uid = os.getenv("TEST_USER_ID", "demo-user")
    out = plan_proactive_questions(uid, lookback_days=7, top_n=3)
    print(json.dumps(out, ensure_ascii=False, indent=2))
