import os
import json
import uuid
import asyncio
import requests
from dotenv import load_dotenv
import tiktoken
import openai
import numpy as np
from openai import AsyncOpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# ----------------------------------------------------------------------
# 셀 1~3: 환경 변수 및 라이브러리 불러오기
# ----------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

EMBEDDING_DIM = 384 if EMBEDDING_MODEL.endswith("3-small") else 1536
PROFILE_COLLECTION_NAME = "user_profiles_v2"
LOG_COLLECTION_NAME = "conversation_logs_v2"

# openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

llm = OpenAI(openai_api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.7)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

PROFILE_DB = {}

# ----------------------------------------------------------------------
# Embedding-based Intent Router 설정
# ----------------------------------------------------------------------
INTENT_EXAMPLES = {
    "rag": [
        "내가 설정한 목표 다시 알려줘.",
        "우리 지난주에 무슨 얘기까지 했지?",
        "내 프로젝트 이름 기억나?",
    ],
    "web": [
        "오늘 서울 날씨 어때?",
        "가장 가까운 스타벅스 어디야?",
        "엔비디아의 최신 GPU 모델 이름이 뭐야?",
    ],
    "conv": [
        "고마워!",
        "ㅋㅋㅋㅋㅋ",
        "재밌는 농담 하나 해줘.",
        "대한민국의 수도는 어디야?",
    ],
}

# 각 intent별 대표 embedding을 평균으로 계산
INTENT_EMBEDDINGS = {}
for label, texts in INTENT_EXAMPLES.items():
    vecs = [embeddings.embed_query(t) for t in texts]
    avg = [sum(x) / len(x) for x in zip(*vecs)]
    INTENT_EMBEDDINGS[label] = np.array(avg)


def embedding_router(query: str, threshold: float = 0.8) -> str | None:
    q_emb = np.array(embeddings.embed_query(query))
    sims = {}
    for label, emb in INTENT_EMBEDDINGS.items():
        sims[label] = float(
            np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        )
    best = max(sims, key=sims.get)
    return best if sims[best] >= threshold else None


# ----------------------------------------------------------------------
# 셀 3: Milvus DB 헬퍼 함수 (이전과 동일)
# ----------------------------------------------------------------------
def get_milvus_connection():
    alias = "default"
    if not connections.has_connection(alias):
        connections.connect(alias, host=MILVUS_HOST, port=MILVUS_PORT)
    return connections.get_connection(alias)


def create_milvus_collection(name: str, desc: str):
    if utility.has_collection(name):
        return Collection(name)
    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("user_id", DataType.VARCHAR, max_length=256),
        FieldSchema("type", DataType.VARCHAR, max_length=50),
        FieldSchema("created_at", DataType.INT64),
    ]
    schema = CollectionSchema(fields, desc)
    coll = Collection(name, schema)
    coll.create_index(
        "embedding",
        {"index_type": "IVF_PQ", "metric_type": "L2", "params": {"nlist": 128, "m": 8}},
    )
    return coll


# ----------------------------------------------------------------------
# 셀 4: 장기 기억(RAG) 업데이트 (이전과 동일)
# ----------------------------------------------------------------------
def update_long_term_memory(session_id: str):
    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    if not history.messages:
        return
    conv_all = "\n".join(f"{m.type}: {m.content}" for m in history.messages)
    summary_prompt = PromptTemplate.from_template(
        "다음 대화에서 인사말 등 불필요한 잡담을 모두 제거하고, "
        "사용자 프로필에 유의미한 핵심 정보만 요약해라.\n{conversation}"
    )
    summary_text = LLMChain(llm=llm, prompt=summary_prompt).run(conversation=conv_all)
    old_prof = json.dumps(PROFILE_DB.get(session_id, {}), ensure_ascii=False)
    profile_update_tpl = PromptTemplate.from_template(
        "[기존 프로필]\n{old}\n[요약된 최신 대화]\n{sum}\n"
        "위 내용을 반영하여 사용자 개인회를 위한 JSON 프로필로 반환해줘."
    )
    new_prof_str = LLMChain(llm=llm, prompt=profile_update_tpl).run(
        old=old_prof, sum=summary_text
    )
    try:
        new_prof = json.loads(new_prof_str)
        PROFILE_DB[session_id] = new_prof
    except json.JSONDecodeError:
        return
    get_milvus_connection()
    prof_coll = create_milvus_collection(PROFILE_COLLECTION_NAME, "User Profiles")
    log_coll = create_milvus_collection(LOG_COLLECTION_NAME, "Conversation Logs")
    prof_emb = embeddings.embed_query(json.dumps(new_prof, ensure_ascii=False))
    prof_coll.upsert(
        [
            {
                "id": session_id,
                "embedding": prof_emb,
                "text": json.dumps(new_prof, ensure_ascii=False),
                "user_id": session_id,
                "type": "profile",
                "created_at": int(os.times().user),
            }
        ]
    )
    log_emb = embeddings.embed_query(summary_text)
    log_coll.insert(
        [
            {
                "id": str(uuid.uuid4()),
                "embedding": log_emb,
                "text": summary_text,
                "user_id": session_id,
                "type": "log",
                "created_at": int(os.times().user),
            }
        ]
    )


# ----------------------------------------------------------------------
# 셀 5: RAG 검색 및 단기 기억 설정 (이전과 동일)
# ----------------------------------------------------------------------
def retrieve_from_rag(session_id: str, query: str, top_k: int = 2) -> str:
    try:
        get_milvus_connection()
        prof_coll = Collection(PROFILE_COLLECTION_NAME)
        log_coll = Collection(LOG_COLLECTION_NAME)
        prof_coll.load()
        log_coll.load()
        query_emb = embeddings.embed_query(query)
        params = {"metric_type": "L2", "params": {"nprobe": 10}}
        prof_res = prof_coll.search(
            [query_emb], "embedding", params, limit=1, expr=f"user_id == '{session_id}'"
        )
        log_res = log_coll.search(
            [query_emb],
            "embedding",
            params,
            limit=top_k,
            expr=f"user_id == '{session_id}'",
        )
        context = ""
        if prof_res and prof_res[0]:
            context += f"[RAG 프로필]\n{prof_res[0][0].entity.get('text')}\n"
        if log_res and log_res[0]:
            for hit in log_res[0]:
                context += f"[RAG 로그]\n{hit.entity.get('text')}\n"
        return context or "RAG 결과 없음"
    except:
        return "RAG 에러"


def get_short_term_memory(session_id: str) -> ConversationSummaryBufferMemory:
    redis_hist = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    return ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=redis_hist,
        max_token_limit=3000,
        return_messages=True,
        memory_key="chat_history",
    )


# ----------------------------------------------------------------------
# 셀 7: Naver Search Chain 구현 (이전과 동일)
# ----------------------------------------------------------------------
def naver_search(query: str, display: int = 5) -> dict:
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    params = {"query": query, "display": display}
    res = requests.get(url, headers=headers, params=params, timeout=5)
    return res.json() if res.status_code == 200 else {}


def search_web(query: str) -> str:
    data = naver_search(query)
    items = data.get("items", [])
    if not items:
        return "검색 결과가 없습니다."
    snippets = []
    for item in items:
        title = item.get("title", "").replace("<b>", "").replace("</b>", "")
        address = item.get("roadAddress", item.get("address", ""))
        snippets.append(f"{title} — {address}")
    return "\n".join(snippets[:5])


# ----------------------------------------------------------------------
# 셀 8: Conversation Chain 구현 (이전과 동일)
# ----------------------------------------------------------------------
async def conversation_chain(
    session_id: str, user_input: str, stm: ConversationSummaryBufferMemory
) -> str:
    hist = "\n".join(f"{m.type}: {m.content}" for m in stm.chat_memory.messages)
    prompt = (
        "너는 소통 전문가다. 사용자의 감정과 상황에 기반하여 질문에 답변하라.\n"
        f"[대화 히스토리]\n{hist}\n"
        f"[최신 입력]\n{user_input}"
    )
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------------------------------------------------
# 셀 9: Function Calling 기반 Router 구현
# ----------------------------------------------------------------------
ROUTER_FUNCTION = {
    "name": "route_tools",
    "description": "Decide which experts (RAG, WebSearch, Conversation) to invoke",
    "parameters": {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["RAG", "WebSearch", "Conversation"],
                },
                "description": "List of tools to apply",
            }
        },
        "required": ["tools"],
    },
}


async def call_router(session_id: str, user_input: str) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "너는 비서실장(라우터)이다. 아래 세 전문가 중 어떤 전문가가 필요할지 결정하라:\n"
                "1) 기억 전문가 (RAG)\n"
                "2) 정보 분석가 (WebSearch)\n"
                "3) 소통 전문가 (Conversation)\n"
                "반드시 함수 호출 형식으로 응답하라."
            ),
        },
        {"role": "user", "content": user_input},
    ]
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        functions=[ROUTER_FUNCTION],
        function_call={"name": "route_tools"},
        temperature=0.0,
    )
    msg = resp.choices[0].message
    # Pydantic 모델이므로 .function_call 속성으로 직접 확인
    if hasattr(msg, "function_call") and msg.function_call is not None:
        args = json.loads(msg.function_call.arguments)
        return args.get("tools", [])
    # fallback simple parse
    text = msg.content or ""
    teams = []
    if "RAG" in text:
        teams.append("RAG")
    if "WebSearch" in text or "검색" in text:
        teams.append("WebSearch")
    if "Conversation" in text or "반응" in text:
        teams.append("Conversation")
    return teams or ["Conversation"]


# ----------------------------------------------------------------------
# 셀 10: Main LLM 최종 응답 템플릿 (이전과 동일)
# ----------------------------------------------------------------------
FINAL_PROMPT = PromptTemplate(
    input_variables=["rag_ctx", "web_ctx", "conv_ctx", "question"],
    template=(
        "너는 전문적이면서도 친근한 개인 비서 AI이다.\n\n"
        "[RAG 결과]\n{rag_ctx}\n\n"
        "[Web 검색 결과]\n{web_ctx}\n\n"
        "[소통 체인 결과]\n{conv_ctx}\n\n"
        "사용자 질문: {question}\n"
        "→ 위 모든 정보를 참고하여 완전한 답변을 제공하라."
    ),
)


# ----------------------------------------------------------------------
# 셀 11: FastAPI + WebSocket 서버
# ----------------------------------------------------------------------
app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


async def background_rag_update(session_id: str):
    await asyncio.to_thread(update_long_term_memory, session_id)


async def main_response(
    session_id: str,
    user_input: str,
    websocket: WebSocket,
    rag_ctx: str,
    web_ctx: str,
    conv_ctx: str,
) -> str:
    prompt = FINAL_PROMPT.format(
        rag_ctx=rag_ctx, web_ctx=web_ctx, conv_ctx=conv_ctx, question=user_input
    )
    messages = [
        {"role": "system", "content": "개인 비서 AI이며, 아래 지침에 따라 답하라."},
        {"role": "user", "content": prompt},
    ]
    resp = await client.chat.completions.create(
        model=LLM_MODEL, messages=messages, stream=True, temperature=0.7
    )

    full_answer = ""
    async for chunk in resp:
        # Pydantic 모델인 ChoiceDelta에서 .content 속성 사용
        delta = chunk.choices[0].delta
        token = delta.content or ""
        if token:
            full_answer += token
            await websocket.send_text(token)

    return full_answer


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    enc = tiktoken.get_encoding("cl100k_base")

    try:
        while True:
            user_input = await websocket.receive_text()
            stm = get_short_term_memory(session_id)
            hist = "\n".join(f"{m.type}: {m.content}" for m in stm.chat_memory.messages)
            if len(enc.encode(hist)) >= 3000:
                asyncio.create_task(background_rag_update(session_id))

            # 1차 관문: Embedding 기반 간단 Router
            intent = embedding_router(user_input)
            if intent == "conv":
                # simple conversation만 수행
                answer = await conversation_chain(session_id, user_input, stm)
                await websocket.send_text(answer)
                stm.save_context({"input": user_input}, {"output": answer})
                continue

            # 2차 관문: Function Calling LLM 기반 Router
            teams = await call_router(session_id, user_input)

            # 전문가 팀 병렬 실행
            tasks = {}
            if "RAG" in teams:
                tasks["rag"] = asyncio.to_thread(
                    retrieve_from_rag, session_id, user_input
                )
            if "WebSearch" in teams:
                tasks["web"] = asyncio.to_thread(search_web, user_input)
            if "Conversation" in teams:
                tasks["conv"] = asyncio.create_task(
                    conversation_chain(session_id, user_input, stm)
                )

            results = await asyncio.gather(*tasks.values())
            rag_ctx = results[list(tasks).index("rag")] if "rag" in tasks else ""
            web_ctx = results[list(tasks).index("web")] if "web" in tasks else ""
            conv_ctx = results[list(tasks).index("conv")] if "conv" in tasks else ""

            # 최종 메인 LLM 스트리밍 응답 및 full_answer 수집
            full_answer = await main_response(
                session_id, user_input, websocket, rag_ctx, web_ctx, conv_ctx
            )

            # 대화 저장
            stm.save_context({"input": user_input}, {"output": full_answer})

    except WebSocketDisconnect:
        pass


# 백엔드 실행
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 웹소캣
# wscat -c ws://localhost:8000/ws/my-session
