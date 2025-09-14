import os
import json
import uuid
import time
import asyncio
import logging
import torch
import requests
import tiktoken
import numpy as np
import re
import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from threading import Lock
import httpx
import random

import openai
from openai import AsyncOpenAI

from dotenv import load_dotenv
from backend.search_engine import build_web_context
from backend.memory.stwm import update_stwm, get_stwm_snapshot
from backend.memory.turns import (
    add_turn as tb_add_turn,
    maybe_summarize as tb_maybe_summarize,
    get_summaries as tb_get_summaries,
)
from backend.memory.selector import select_summaries
from backend.evidence.builder import build_evidence
from backend.generation.composer import (
    compose_fact_answer,
    apply_style_wrapper,
    ComposeInput,
)
from backend.generation.wrapper import (
    wrap_web_reply,
    wrap_generic_reply,
    wrap_greeting_reply,
)
from backend.rewrite.log import add_rewrite, RewriteRecord
from backend.planner.logger import log_planner, PlannerLog
from backend.policy.state import redact_text

from backend.directives.pipeline import (
    ensure_directive_workers,
    schedule_directive_update,
)
from backend.directives.store import get_compiled as get_compiled_directives

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend.rag import (
    retrieve_from_rag,
    ensure_collections,
    embed_query_cached,
    METRIC,
)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    Partition,
)

from transformers import (
    AutoTokenizer as HFTokenizer,
    AutoConfig as HFConfig,
    AutoModelForSequenceClassification as HFForSeq,
)

try:
    # Firestore는 선택적 의존성. 환경/권한 없을 경우 None 처리
    from google.cloud import firestore as gcf  # type: ignore
except Exception:
    gcf = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ----------------------------------------------------------------------
# 로깅 설정
# ----------------------------------------------------------------------
LOG_LEVEL = os.getenv("ROUTER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("router")

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
FIRESTORE_ENABLE = bool(int(os.getenv("FIRESTORE_ENABLE", "1")))
TIMEOUT_MOBILE = float(os.getenv("TIMEOUT_MOBILE", "1.0"))
FIRESTORE_USERS_COLL = os.getenv("FIRESTORE_USERS_COLL", "users")
FIRESTORE_EVENTS_SUB = os.getenv("FIRESTORE_EVENTS_SUB", "unified_events")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp:5000")
DEBUG_META = bool(int(os.getenv("DEBUG_META", "0")))
WS_DEBUG_META = bool(int(os.getenv("WS_DEBUG_META", "0")))

# Firestore 서비스 계정 키 경로 자동 설정 (기본: 프로젝트 루트/gcp-service-account-key.json)
try:
    if FIRESTORE_ENABLE:
        default_key_path = os.getenv(
            "GCP_SERVICE_ACCOUNT_PATH",
            str(Path(__file__).resolve().parent / "gcp-service-account-key.json"),
        )
        cur = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        # 1) 현재 설정이 유효하지 않으면 교정 시도
        if cur and not Path(cur).exists() and Path(default_key_path).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_key_path
            logger.info(
                "[fs] GOOGLE_APPLICATION_CREDENTIALS override -> %s", default_key_path
            )
        # 2) 미설정이면 기본 경로로 셋업
        elif (not cur) and Path(default_key_path).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_key_path
            logger.info("[fs] GOOGLE_APPLICATIONS set -> %s", default_key_path)
        elif not Path(default_key_path).exists():
            logger.warning("[fs] service account key not found at %s", default_key_path)
except Exception as _fs_e:
    logger.warning("[fs] failed to set GOOGLE_APPLICATION_CREDENTIALS: %r", _fs_e)

# 이성 추론형 모델(에이전트/그래프용)
THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")

# ===== 추가: 재작성 타임아웃/토큰/모델 =====
REWRITE_TIMEOUT_S = float(os.getenv("REWRITE_TIMEOUT_S", "1.25"))
REWRITE_MAX_TOKENS = int(os.getenv("REWRITE_MAX_TOKENS", "128"))
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")

# ===== 담화 앵커/팔로업/임베딩 힌트 설정 =====
TOPIC_TTL_S = int(os.getenv("TOPIC_TTL_S", "600"))  # 앵커 TTL(초)
HINT_LOOKBACK = int(
    os.getenv("HINT_LOOKBACK", "8")
)  # 힌트 추출 시 최근 발화 개수 (기본 8로 축소)
HINT_MAX_ITEMS = int(os.getenv("HINT_MAX_ITEMS", "2"))  # 힌트 최대 라인 수
HINT_SIM_THRESHOLD = float(
    os.getenv("HINT_SIM_THRESHOLD", "0.25")
)  # 코사인 유사도 임계
EXTRACT_TIMEOUT_S = float(os.getenv("EXTRACT_TIMEOUT_S", "0.7"))  # 앵커 추출 타임아웃
FOLLOWUP_TIMEOUT_S = float(
    os.getenv("FOLLOWUP_TIMEOUT_S", "0.7")
)  # 팔로업 판별 타임아웃

# ===== (기존 상단 설정부 아래에 추가) =====
EMBEDDING_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,  # 레거시 사용 시
}
EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM", EMBEDDING_DIM_MAP.get(EMBEDDING_MODEL, 1536))
)

# 차원 섞임 방지: 컬렉션 이름 v3 (날짜 메타 추가)
PROFILE_COLLECTION_NAME = f"user_profiles_v3_{EMBEDDING_DIM}d"
LOG_COLLECTION_NAME = f"conversation_logs_v3_{EMBEDDING_DIM}d"

# 라우팅 스레숄드 / 애매 구간 / 타임아웃
TAU_RAG = float(os.getenv("TAU_RAG", 0.55))
TAU_WEB = float(os.getenv("TAU_WEB", 0.55))
AMBIGUITY_BAND = float(os.getenv("AMBIGUITY_BAND", "0.03"))  # 🔧 기본값 0.03로 하향
TIMEOUT_RAG = float(os.getenv("TIMEOUT_RAG", 2.5))
TIMEOUT_WEB = float(os.getenv("TIMEOUT_WEB", 2.5))

# ===== 소분류기 보정(Platt/TS) 및 프라이어 가점 =====
CAL_WEB = {"a": 1.0, "b": 0.0, "T": 1.00}
CAL_RAG = {"a": 1.0, "b": 0.0, "T": 1.00}

WEB_PRIOR_PAT = r"(추천|근처|가까운|영업시간|가격|리뷰|랭킹|뉴스|최신|주소|전화)"
RAG_PRIOR_PAT = r"(내 문서|사내|정책|내 일정|프로젝트|노트|요약했|회의록|RAG|내정보)"


def _apply_calibration(logit: float, cal: dict) -> float:
    # Platt (기본). TS를 쓰려면: return 1.0/(1.0+np.exp(-(logit/cal["T"])))
    return 1.0 / (1.0 + np.exp(-(cal.get("a", 1.0) * logit + cal.get("b", 0.0))))


def _heuristic_priors(txt: str) -> tuple[float, float]:
    import re as _re

    web_boost = 0.0
    rag_boost = 0.0
    if _re.search(WEB_PRIOR_PAT, txt):
        web_boost += 0.25
    if _re.search(r"(오늘|내일|이번주|지금|월|화|수|목|금|토|일)", txt):
        web_boost += 0.10
    if _re.search(r"(역|동|구|시|군|도|가|로)\b", txt):
        web_boost += 0.05  # 지명 힌트
    if _re.search(RAG_PRIOR_PAT, txt):
        rag_boost += 0.30
    web_boost = min(web_boost, 0.35)
    rag_boost = min(rag_boost, 0.35)
    return web_boost, rag_boost


# 순수 대화 모드에서 단일 스트리밍 호출로 단순화
SINGLE_CALL_CONV = bool(int(os.getenv("SINGLE_CALL_CONV", "1")))

# LLM 라우터 사용 설정(휴리스틱 우회)
USE_LLM_ROUTER = bool(int(os.getenv("USE_LLM_ROUTER", "1")))

# 사전 검증 타임아웃
PREVALIDATE_TIMEOUT_S = float(os.getenv("PREVALIDATE_TIMEOUT_S", "0.9"))

# ------------------------------
# 고정 시스템 정체성 프롬프트(3~4줄)
# ------------------------------
IDENTITY_PROMPT = (
    "나는 한국어 개인 비서다. 사용자의 질문 의도에 따라 대화/장기기억(RAG)/웹검색을 스스로 선택해 답한다.\n"
    "증거(RAG/웹)가 있을 때는 그 범위 안에서만 정확히 인용하고, 없으면 자연스럽게 설명하거나 필요 시 되묻는다.\n"
    "안전/개인정보/허위는 금지하며, 사용자의 취향 설정(동적 지시문)을 존중해 말투와 형식을 맞춘다."
)

# ===== Redis/요약/스냅샷 정책 상수 =====
MAX_TOKEN_LIMIT = 3000  # Redis 단기 메모리 하드 한도
RECENT_RAW_TOKENS_BUDGET = 1500  # 최근 원문 보존 예산
SUMMARY_TARGET_TOKENS = 500  # 축약 목표(구조화+생성 합산)
SYSTEM_TOOL_BUDGET = 200  # 시스템/툴 메타 여유분
EDGE_THRESHOLD = MAX_TOKEN_LIMIT  # 에지 트리거 임계치 (3000)
DEBOUNCE_TURNS = 5  # 배치: 최소 턴
DEBOUNCE_SECONDS = 60  # 배치: 최소 시간
SNAPSHOT_QUEUE_MAXSIZE = 128  # 작업 큐
WORKER_CONCURRENCY = 2  # 동시 스냅샷 처리
EMBED_CONCURRENCY = 2  # 임베딩 동시 제한

# 스트리밍 사용 설정(조직 권한 없을 시 런타임 비활성화)
STREAM_ENABLED = bool(int(os.getenv("STREAM_ENABLED", "1")))
_STREAM_RUNTIME_DISABLED = False


def _stream_allowed() -> bool:
    return STREAM_ENABLED and (not _STREAM_RUNTIME_DISABLED)


# ===== 중복 억제/신규성 게이트용 설정 (상단 설정부에 추가) =====
SNAPSHOT_EDGE_TOKENS = int(
    os.getenv("SNAPSHOT_EDGE_TOKENS", "4500")
)  # 스냅샷 적재 트리거(메모리 3000과 분리)
NOVELTY_SIM_THRESHOLD = float(
    os.getenv("NOVELTY_SIM_THRESHOLD", "0.92")
)  # 로그 근사중복 억제 임계
NOVELTY_MIN_PROFILE_DELTA = int(
    os.getenv("NOVELTY_MIN_PROFILE_DELTA", "1")
)  # 프로필 신규 항목 최소 개수
SNAPSHOT_LOOKBACK_MONTHS = int(
    os.getenv("SNAPSHOT_LOOKBACK_MONTHS", "1")
)  # 근사중복 탐색 기간

# ===== 외부 호출 재시도 공통 설정 =====
MAX_RETRIES_OPENAI = int(os.getenv("MAX_RETRIES_OPENAI", "2"))
MAX_RETRIES_HTTP = int(os.getenv("MAX_RETRIES_HTTP", "2"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.25"))

logger.info(
    f"[boot] model={LLM_MODEL} embed={EMBEDDING_MODEL} milvus={MILVUS_HOST}:{MILVUS_PORT} redis={REDIS_URL}"
)
logger.info(
    f"[boot] thresholds tau_rag={TAU_RAG} tau_web={TAU_WEB} ambiguity={AMBIGUITY_BAND} timeouts web={TIMEOUT_WEB}s rag={TIMEOUT_RAG}s"
)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Firestore 클라이언트 (지연 초기화)
_FS_DB = None


def _ensure_fs_db():
    global _FS_DB
    if _FS_DB is not None:
        return _FS_DB
    if not FIRESTORE_ENABLE or gcf is None:
        return None
    try:
        _FS_DB = gcf.Client()
        logger.info("[fs] firestore client initialized")
        return _FS_DB
    except Exception as e:
        logger.warning("[fs] init failed: %r", e)
        return None


# ----------------------------------------------------------------------
# 공통: OpenAI/HTTP 재시도 래퍼 (지수 백오프 + 지터)
# ----------------------------------------------------------------------


async def _backoff_sleep(attempt: int):
    delay = RETRY_BASE_DELAY * (2**attempt) * (0.5 + random.random())
    await asyncio.sleep(delay)


async def openai_chat_with_retry(**create_kwargs):
    last_err = None
    # 사전 매핑/정리: Chat Completions는 일관되게 max_tokens를 사용
    # - max_completion_tokens가 들어오면 max_tokens로 변환
    # - 구형 gpt-4-* 모델에서 response_format은 지원되지 않으므로 선제 제거
    try:
        model_name = str(create_kwargs.get("model", LLM_MODEL) or "")
        # 1) max_completion_tokens → max_tokens 정규화
        if "max_completion_tokens" in create_kwargs:
            try:
                mt = int(create_kwargs.pop("max_completion_tokens"))
                create_kwargs["max_tokens"] = mt
            except Exception:
                create_kwargs.pop("max_completion_tokens", None)

        # 2) response_format 호환성 체크 (gpt-4o/4.1/o3 등 일부 모델만 안정 지원)
        rf = create_kwargs.get("response_format")
        if isinstance(rf, dict):
            rf_type = str(rf.get("type", "")).lower()
            # 지원 모델 힌트 키워드
            supports = any(
                k in model_name for k in ("gpt-4o", "gpt-4.1", "o3", "o4", "4o")
            )
            if not supports:
                # 구형 모델(gpt-4-0613 등)에서는 400을 유발하므로 선제 제거
                create_kwargs.pop("response_format", None)
            elif rf_type not in ("json_object", "json_schema"):
                # 알 수 없는 타입은 제거
                create_kwargs.pop("response_format", None)
    except Exception:
        pass
    for attempt in range(MAX_RETRIES_OPENAI + 1):
        try:
            return await client.chat.completions.create(**create_kwargs)
        except Exception as e:
            last_err = e
            # 400 파라미터 호환 이슈에 대한 자동 교정 후 재시도
            if attempt < MAX_RETRIES_OPENAI:
                msg = str(e)
                try:
                    # 남아있는 비호환 매개변수 정리
                    if "max_completion_tokens" in create_kwargs:
                        mt = int(create_kwargs.pop("max_completion_tokens"))
                        create_kwargs["max_tokens"] = mt
                    if isinstance(create_kwargs.get("response_format"), dict):
                        create_kwargs.pop("response_format", None)
                    if "stream" in msg and create_kwargs.get("stream") is True:
                        create_kwargs["stream"] = False
                except Exception:
                    pass
                await _backoff_sleep(attempt)
                continue
            break
    raise last_err


async def _rewrite_with_retries(
    messages: list[dict],
    base_timeout_s: float,
    attempts: int = 1,
    delta_s: float = 1.0,
    max_tokens: int | None = None,
    response_format: dict | None = None,
):
    """
    재작성 호출에 대해 타임아웃 발생 시 최대 attempts-1회까지 1초씩 타임아웃을 늘려 재시도.
    - 성공 시 content 문자열을 반환, 모든 시도가 실패하면 None 반환.
    """
    for i in range(attempts):
        tout = base_timeout_s + i * delta_s
        try:
            resp = await asyncio.wait_for(
                openai_chat_with_retry(
                    model=REWRITE_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=(
                        max_tokens if max_tokens is not None else REWRITE_MAX_TOKENS
                    ),
                    **({"response_format": response_format} if response_format else {}),
                ),
                timeout=tout,
            )
            return (resp.choices[0].message.content or "").strip()
        except asyncio.TimeoutError:
            if i + 1 >= attempts:
                break
            logger.warning("[rewrite] timeout(%.2fs) -> retry %d", tout, i + 1)
        except Exception as e:
            logger.warning("[rewrite] error=%r", e)
            break
    return None


async def http_get_with_retry(
    url: str, headers: dict | None, params: dict | None, timeout: httpx.Timeout | None
):
    last_err = None
    for attempt in range(MAX_RETRIES_HTTP + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client_http:
                return await client_http.get(url, headers=headers, params=params)
        except Exception as e:
            last_err = e
            if attempt >= MAX_RETRIES_HTTP:
                break
            await _backoff_sleep(attempt)
    raise last_err


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=LLM_MODEL,
)

# 메모리/요약용: 진짜 LLM 인스턴스(= BaseLanguageModel)
llm_cold = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=LLM_MODEL,
)


# 임베딩 통합: backend.rag.embeddings의 백엔드 추상화 사용으로 교체
from backend.rag.embeddings import get_embeddings, embed_documents as embed_docs

embeddings = get_embeddings()

PROFILE_DB = {}

# ----------------------------------------------------------------------
# KST & 날짜 유틸 (상대시제 처리)
# ----------------------------------------------------------------------
KST = timezone(timedelta(hours=9))


def _now_kst() -> datetime:
    return datetime.now(KST)


def _ym(dt: datetime) -> int:
    return dt.year * 100 + dt.month  # 202508


def _ymd(dt: datetime) -> int:
    return dt.year * 10000 + dt.month * 100 + dt.day  # 20250817


def _week_range(base: datetime, offset_weeks=0) -> Tuple[datetime, datetime]:
    d0 = base + timedelta(weeks=offset_weeks)
    start = d0 - timedelta(days=d0.weekday())  # 월요일
    end = start + timedelta(days=6)
    return start.replace(tzinfo=KST), end.replace(tzinfo=KST)


def _month_range(base: datetime, offset_months=0) -> Tuple[datetime, datetime]:
    y, m = base.year, base.month + offset_months
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    start = datetime(y, m, 1, tzinfo=KST)
    y2, m2 = y + (m // 12), (m % 12) + 1
    next_first = datetime(y2, m2, 1, tzinfo=KST)
    end = next_first - timedelta(days=1)
    return start, end


def _ym_minus_months(base: datetime, months: int) -> int:
    y, m = base.year, base.month - months
    while m <= 0:
        y -= 1
        m += 12
    return y * 100 + m


async def _post_verify_answer(
    user_input: str, rag_ctx: str, web_ctx: str, answer: str, websocket: WebSocket
):
    """
    RAG/WEB 컨텍스트가 존재할 때 답변의 적합성을 경량 검토. 실시간성 위해 비차단 전송.
    - 규칙: 질문과 무관/상충/과장/추측 여부를 점검. 문제시 1줄 경고와 재검토 제안.
    """
    if not (rag_ctx.strip() or web_ctx.strip()):
        return
    # f-string 표현식 내부에는 백슬래시(예: '\n')가 들어갈 수 없으므로 사전 계산
    ctx_cut = (f"{rag_ctx}\n{web_ctx}")[:1200]
    ans_cut = (answer or "")[:800]

    msgs = [
        {
            "role": "system",
            "content": (
                "너는 답변 검토자다. [컨텍스트]와 [질문] 대비 [답변]의 적합성을 확인하고,"
                " 무관/상충/추측/과장을 감지하면 경고 메시지 1줄을 한국어로 출력하라."
                " 문제가 없으면 빈 문자열만 돌려라."
            ),
        },
        {
            "role": "user",
            "content": f"[질문]\n{user_input}\n\n[컨텍스트]\n{ctx_cut}\n\n[답변]\n{ans_cut}",
        },
    ]
    try:
        resp = await asyncio.wait_for(
            openai_chat_with_retry(
                model=LLM_MODEL, messages=msgs, temperature=0.0, max_tokens=80
            ),
            timeout=0.9,
        )
        tip = (resp.choices[0].message.content or "").strip()
        if tip:
            try:
                await websocket.send_text(f"\n[검토] {tip}")
            except Exception:
                pass
    except Exception:
        return


async def _validate_final_answer(
    user_input: str, rag_ctx: str, web_ctx: str, answer: str
) -> bool:
    """
    최종 답변 적합성 사전 검증: 질문/컨텍스트 대비 부적합하면 False.
    - 웹 래퍼 경로에서도 사용하여 부적합시 LLM 스트리밍으로 폴백.
    """
    try:
        if not answer:
            return False
        schema = {
            "name": "AnswerFit",
            "schema": {
                "type": "object",
                "properties": {"keep": {"type": "boolean"}},
                "required": ["keep"],
                "additionalProperties": False,
            },
        }
        ctx_short = (rag_ctx or "") + "\n" + (web_ctx or "")
        ctx_short = ctx_short[:1200]
        ans_short = (answer or "")[:800]
        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 최종 검증자다. [질문]과 [컨텍스트] 대비 [답변]이 의도에 적합하면 keep=true,"
                    " 부적합/무관/노이즈 과다면 keep=false. JSON만."
                ),
            },
            {
                "role": "user",
                "content": f"[질문]\n{user_input}\n\n[컨텍스트]\n{ctx_short}\n\n[답변]\n{ans_short}",
            },
        ]
        kwargs = {
            "model": LLM_MODEL,
            "messages": msgs,
            "max_tokens": 20,
            "temperature": 0.0,
        }
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        resp = await asyncio.wait_for(openai_chat_with_retry(**kwargs), timeout=0.9)
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt) if txt.startswith("{") else {}
        return bool(data.get("keep", False))
    except Exception:
        return True  # 검증 실패 시 차단하지 않음


def _flatten_profile_items(p: dict) -> set[str]:
    # 프로필 JSON에서 핵심 필드만 평탄화하여 비교(소문자/공백정규화)
    keys = ("facts", "goals", "tasks", "decisions", "constraints")
    out = set()
    for k in keys:
        v = p.get(k)
        if isinstance(v, list):
            for x in v:
                s = str(x).strip().lower()
                if s:
                    out.add(s)
        elif isinstance(v, str):
            s = v.strip().lower()
            if s:
                out.add(s)
    return out


def _near_duplicate_log(
    session_id: str, log_emb: List[float], ym_min: int
) -> tuple[bool, float]:
    # 최근 N개월 범위에서 가장 유사한 로그 1개를 조회해 근사중복 여부를 반환
    prof_coll, log_coll = ensure_milvus_collections()
    search_params = {"metric_type": METRIC, "params": {"ef": 32}}  # 비용 낮게
    expr = f"user_id == '{session_id}' and date_ym >= {ym_min}"
    try:
        res = log_coll.search(
            data=[log_emb],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=expr,
            output_fields=["text", "date_ym"],
        )
        if res and res[0]:
            sim = _hit_similarity(res[0][0])
            return (sim >= NOVELTY_SIM_THRESHOLD, sim)
    except Exception as e:
        logger.warning("[novelty] near-dup search error: %r", e)
    return (False, 0.0)


# ----------------------------------------------------------------------
# 상대시제 → 절대 날짜 토크나이저 (WEB/RAG)
# ----------------------------------------------------------------------
RELATIVE_PATTERNS_DAY = [
    (r"\b오늘\b", lambda now: (now, now)),
    (r"\b내일\b", lambda now: (now + timedelta(days=1), now + timedelta(days=1))),
    (r"\b모레\b", lambda now: (now + timedelta(days=2), now + timedelta(days=2))),
    (r"\b글피\b", lambda now: (now + timedelta(days=3), now + timedelta(days=3))),
    (r"\b내글피\b", lambda now: (now + timedelta(days=4), now + timedelta(days=4))),
    (r"\b어제\b", lambda now: (now - timedelta(days=1), now - timedelta(days=1))),
    (
        r"\b그제\b|\b그저께\b|\b엊그제\b",
        lambda now: (now - timedelta(days=2), now - timedelta(days=2)),
    ),
    (r"\b그끄제\b", lambda now: (now - timedelta(days=3), now - timedelta(days=3))),
]
RELATIVE_PATTERNS_WEEK = [
    (r"\b이번\s*주말\b", lambda now: _week_range(now, 0)),
    (r"\b지난\s*주말\b", lambda now: _week_range(now, -1)),
    (r"\b다음\s*주말\b", lambda now: _week_range(now, 1)),
    (r"\b이번\s*주\b", lambda now: _week_range(now, 0)),
    (r"\b지난\s*주\b", lambda now: _week_range(now, -1)),
    (r"\b다음\s*주\b", lambda now: _week_range(now, 1)),
]
RELATIVE_PATTERNS_MONTH_YEAR = [
    (r"\b이번\s*달\b|\b이달\b", lambda now: _month_range(now, 0)),
    (r"\b지난\s*달\b|\b저번\s*달\b", lambda now: _month_range(now, -1)),
    (r"\b다음\s*달\b", lambda now: _month_range(now, 1)),
    (
        r"\b올해\b",
        lambda now: (
            datetime(now.year, 1, 1, tzinfo=KST),
            datetime(now.year, 12, 31, tzinfo=KST),
        ),
    ),
    (
        r"\b작년\b",
        lambda now: (
            datetime(now.year - 1, 1, 1, tzinfo=KST),
            datetime(now.year - 1, 12, 31, tzinfo=KST),
        ),
    ),
    (
        r"\b재작년\b",
        lambda now: (
            datetime(now.year - 2, 1, 1, tzinfo=KST),
            datetime(now.year - 2, 12, 31, tzinfo=KST),
        ),
    ),
    (
        r"\b내년\b",
        lambda now: (
            datetime(now.year + 1, 1, 1, tzinfo=KST),
            datetime(now.year + 1, 12, 31, tzinfo=KST),
        ),
    ),
    (
        r"\b내후년\b",
        lambda now: (
            datetime(now.year + 2, 1, 1, tzinfo=KST),
            datetime(now.year + 2, 12, 31, tzinfo=KST),
        ),
    ),
]


def _extract_date_range_for_rag(
    text: str, now: Optional[datetime] = None
) -> Optional[Tuple[int, int]]:
    now = now or _now_kst()
    start_dt, end_dt = None, None

    def _apply(pats):
        nonlocal start_dt, end_dt
        for pat, fn in pats:
            m = re.search(pat, text)
            if m:
                s, e = fn(now)
                start_dt = s if start_dt is None else min(start_dt, s)
                end_dt = e if end_dt is None else max(end_dt, e)

    _apply(RELATIVE_PATTERNS_DAY)
    _apply(RELATIVE_PATTERNS_WEEK)
    _apply(RELATIVE_PATTERNS_MONTH_YEAR)
    if start_dt and end_dt:
        return (_ymd(start_dt), _ymd(end_dt))
    return None


def _month_tokens_for_web(
    text: str, now: Optional[datetime] = None
) -> Optional[Tuple[int, int]]:
    now = now or _now_kst()
    start_dt, end_dt = None, None

    def _apply(pats):
        nonlocal start_dt, end_dt
        for pat, fn in pats:
            m = re.search(pat, text)
            if m:
                s, e = fn(now)
                start_dt = s if start_dt is None else min(start_dt, s)
                end_dt = e if end_dt is None else max(end_dt, e)

    _apply(RELATIVE_PATTERNS_DAY)
    _apply(RELATIVE_PATTERNS_WEEK)
    _apply(RELATIVE_PATTERNS_MONTH_YEAR)
    if start_dt and end_dt:
        return (_ym(start_dt), _ym(end_dt))
    return None


# ----------------------------------------------------------------------
# Embedding-based Intent Router 설정 (보조)
# ----------------------------------------------------------------------
INTENT_EXAMPLES = {
    "rag": [
        "내가 저번에 설정한 목표 다시 알려줘.",
        "내가 너에게 알려준 내 이름이 뭐라고 했지?"
        "너가 어제 추천해줬던 브렌드가 뭐였지?"
        "우리 지난주에 무슨 얘기까지 했지?",
        "진행 중인 프로젝트의 진행 상황 요약해봐.",
        "내가 엊그제 너한테 말했던 고민 기억해?",
        "오늘 대화를 바탕으로 내 최종 목표를 업데이트해서 정리해줘.",
        "지난 대화에서 OKR 정리해줘.",
    ],
    "web": [
        "오늘 서울 날씨 어때?",
        "가장 가까운 스타벅스 어디야?",
        "엔비디아의 최신 GPU 모델 이름이 뭐야?",
        "한국의 현 정치 상황",
        "저녁 먹으려는데 강남역 맛집 추천해줄래?",
    ],
    "conv": [
        "고마워! 잘 이해했어.",
        "양자역학은 왜 이렇게 어려울까?",
        "심심해. 재밌는 농담 하나 해줘.",
        "대한민국의 수도는 어디야?",
        "만나서 반가워.",
    ],
}

# 파일 기반 seed 확장(옵션): My_Business/data의 conv/web/rag 데이터로 증분
try:
    DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
    seed_files = {
        "conv": os.path.join(DATA_DIR, "conv_data.txt"),
        "web": os.path.join(DATA_DIR, "web_data.txt"),
        "rag": os.path.join(DATA_DIR, "rag_data.txt"),
    }
except Exception:
    seed_files = {}

_INTENT_READY = False
_INTENT_LOCK = Lock()
INTENT_EMBEDDINGS: dict[str, np.ndarray] = {}
INTENT_EMBEDDINGS_LIST: dict[str, list[np.ndarray]] = {}


def _ensure_intent_embeddings():
    global _INTENT_READY
    if _INTENT_READY:
        return
    with _INTENT_LOCK:
        if _INTENT_READY:
            return
        logger.info("[boot] preparing intent embeddings (lazy) ...")
        # 파일 시드 병합
        labels = list(INTENT_EXAMPLES.keys())
        groups = []
        for l in labels:
            texts = list(INTENT_EXAMPLES.get(l, []))
            try:
                p = seed_files.get(l)
                if p and Path(p).exists():
                    with open(p, "r", encoding="utf-8") as f:
                        extra = [ln.strip() for ln in f.readlines() if ln.strip()]
                        # 과도한 노이즈 방지: 최대 1000 라인까지 사용
                        texts.extend(extra[:1000])
            except Exception:
                pass
            groups.append(texts)
        flat_texts = [t for texts in groups for t in texts]
        # 통일 추상화(embed_docs)
        flat_vecs = embed_docs(flat_texts)
        idx = 0
        for label, texts in zip(labels, groups):
            n = len(texts)
            vecs = flat_vecs[idx : idx + n]
            idx += n
            arr = [np.array(v, dtype=np.float32) for v in vecs]
            avg = np.mean(np.stack(arr, axis=0), axis=0)
            INTENT_EMBEDDINGS[label] = avg
            INTENT_EMBEDDINGS_LIST[label] = arr
        _INTENT_READY = True
        logger.info(
            "[boot] intent embeddings ready: %s", list(INTENT_EMBEDDINGS.keys())
        )


def embedding_router(query: str, threshold: float = 0.7) -> str | None:
    _ensure_intent_embeddings()
    start = time.time()
    q_emb = np.array(embed_query_cached(query))
    sims = {}
    topk = int(os.getenv("ROUTER_SEED_TOPK", "8"))
    qn = float(np.linalg.norm(q_emb) or 1.0)
    for label, emb in INTENT_EMBEDDINGS.items():
        # 1) 센트로이드
        s_cent = float(np.dot(q_emb, emb) / (qn * (float(np.linalg.norm(emb)) or 1.0)))
        # 2) 예시별 top-k 평균
        lst = INTENT_EMBEDDINGS_LIST.get(label, [])
        if lst:
            per = []
            for v in lst:
                den = float(np.linalg.norm(v) or 1.0) * qn
                per.append(float(np.dot(q_emb, v) / (den or 1.0)))
            per.sort(reverse=True)
            s_top = float(np.mean(per[: max(1, min(topk, len(per)))]))
            sims[label] = 0.5 * s_cent + 0.5 * s_top
        else:
            sims[label] = s_cent
    best = max(sims, key=sims.get)
    took = (time.time() - start) * 1000
    logger.info(
        f"[router:embedding] sims={sims} best={best} thr={threshold} took_ms={took:.1f}"
    )
    return best if sims[best] >= threshold else None


def _prefer_rag_when_recall(user_input: str) -> bool:
    return bool(
        re.search(r"지난|저번|엊그제|그때|우리 대화|어제 추천|지난번 추천", user_input)
    )


# ----------------------------------------------------------------------
# 셀 2.5) 소분류기 로드 (멀티라벨 need_rag/need_web)
# ----------------------------------------------------------------------
# OUTPUT_DIR = "./router_kor_electra_small"
# OUTPUT_DIR = str(Path(__file__).resolve().parent / "router_kor_electra_small")
OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent / "models" / "router_kor_electra_small"
)
USE_ONNX = bool(int(os.getenv("ROUTER_USE_ONNX", "0")))
ONNX_PATH = os.getenv(
    "ROUTER_ELECTRA_ONNX",
    str(Path(OUTPUT_DIR) / "model.onnx"),
)

MAX_LEN = 192

loaded_tok = HFTokenizer.from_pretrained(OUTPUT_DIR)
loaded_cfg = HFConfig.from_pretrained(OUTPUT_DIR)
_onnx_sess = None
loaded_model = None
if USE_ONNX:
    try:
        import onnxruntime as ort  # type: ignore

        _onnx_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        logger.info(
            f"[boot] loaded small classifier(ONNX) from {ONNX_PATH} (multi-label 2 heads)"
        )
    except Exception as _e_onx:
        logger.warning(
            f"[boot] onnx load failed: {repr(_e_onx)} -> fallback to torch weights at {OUTPUT_DIR}"
        )
        loaded_model = HFForSeq.from_pretrained(OUTPUT_DIR, config=loaded_cfg)
        loaded_model.eval()
        logger.info(
            f"[boot] loaded small classifier(Torch) from {OUTPUT_DIR} (multi-label 2 heads)"
        )
else:
    loaded_model = HFForSeq.from_pretrained(OUTPUT_DIR, config=loaded_cfg)
    loaded_model.eval()
    logger.info(
        f"[boot] loaded small classifier(Torch) from {OUTPUT_DIR} (multi-label 2 heads)"
    )


def predict_need_flags(query: str, tau_rag: float = TAU_RAG, tau_web: float = TAU_WEB):
    import numpy as _np

    logits = None
    # ONNX 경로
    if _onnx_sess is not None:
        try:
            enc_np = loaded_tok(
                query,
                return_tensors="np",
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
            )
            inputs = {
                "input_ids": enc_np["input_ids"].astype(_np.int64),
                "attention_mask": enc_np["attention_mask"].astype(_np.int64),
            }
            if "token_type_ids" in enc_np:
                inputs["token_type_ids"] = enc_np["token_type_ids"].astype(_np.int64)
            outs = _onnx_sess.run(None, inputs)
            logits = _np.array(outs[0])[0]
        except Exception as _e_onnx:
            logger.warning(
                f"[clf] onnx inference failed: {repr(_e_onnx)} -> fallback torch"
            )
    # Torch 경로
    if logits is None:
        with torch.no_grad():
            enc = loaded_tok(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
            )
            out = loaded_model(**enc)
            logits = out.logits[0].detach().cpu().numpy()  # shape (2,) rag, web
        # 캘리브레이션(Platt) 적용 후 휴리스틱 프라이어 가점
        p_rag = _apply_calibration(float(logits[0]), CAL_RAG)
        p_web = _apply_calibration(float(logits[1]), CAL_WEB)
        w_boost, r_boost = _heuristic_priors(query)
        need_rag_prob = float(np.clip(p_rag + r_boost, 0.0, 1.0))
        need_web_prob = float(np.clip(p_web + w_boost, 0.0, 1.0))
        need_rag = int(need_rag_prob >= tau_rag)
        need_web = int(need_web_prob >= tau_web)
        logger.info(
            f"[clf] q='{query[:80]}' cal_p(rag)={p_rag:.3f}+{r_boost:.2f}->{need_rag_prob:.3f} "
            f"cal_p(web)={p_web:.3f}+{w_boost:.2f}->{need_web_prob:.3f} "
            f"tau_rag={tau_rag} tau_web={tau_web} -> need_rag={need_rag} need_web={need_web}"
        )
        return {
            "need_rag_prob": need_rag_prob,
            "need_web_prob": need_web_prob,
            "need_rag": need_rag,
            "need_web": need_web,
            "p_rag_cal": float(p_rag),
            "p_web_cal": float(p_web),
            "r_boost": float(r_boost),
            "w_boost": float(w_boost),
        }


# ----------------------------------------------------------------------
# LRU 임베딩 캐시: rag.embeddings.embed_query_cached로 일원화 (중복 정의 제거)
# ----------------------------------------------------------------------
# 주의: 기존 동일 함수명이 존재했으나 이제는 backend.rag.embeddings 모듈의 구현을 사용한다.


# ----------------------------------------------------------------------
# 셀 3: Milvus DB 헬퍼 함수 (HNSW + 날짜 메타 필드 + 파티션)
# ----------------------------------------------------------------------
def ensure_milvus():
    alias = "default"
    if not connections.has_connection(alias):
        logger.info(
            "[milvus] connecting alias=%s host=%s port=%s",
            alias,
            MILVUS_HOST,
            MILVUS_PORT,
        )
        connections.connect(alias, host=MILVUS_HOST, port=MILVUS_PORT)


def _ensure_partition(coll: Collection, ym: int):
    part_name = f"ym_{ym}"
    try:
        if part_name not in [p.name for p in coll.partitions]:
            coll.create_partition(partition_name=part_name, description=f"YYYYMM={ym}")
            logger.info("[milvus] created partition %s in %s", part_name, coll.name)
    except Exception as e:
        logger.warning("[milvus] ensure partition error: %r", e)
    return part_name


def create_milvus_collection(name: str, desc: str):
    if utility.has_collection(name):
        coll = Collection(name)
        for f in coll.schema.fields:
            if f.name == "embedding":
                existing_dim = f.params.get("dim")
                if existing_dim != EMBEDDING_DIM:
                    logger.error(
                        f"[milvus] dim mismatch for {name}: existing={existing_dim} expected={EMBEDDING_DIM}"
                    )
                    raise RuntimeError(
                        "Milvus collection dim mismatch. Create a new collection with correct dim."
                    )
        have_dates = set(x.name for x in coll.schema.fields)
        expected = {"date_start", "date_end", "date_ym"}
        missing = expected - have_dates
        if missing:
            logger.warning(
                f"[milvus] collection {name} missing date fields {missing}. Consider migrating to v3."
            )
        logger.info("[milvus] reuse collection=%s", name)
        return coll

    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("user_id", DataType.VARCHAR, max_length=256),
        FieldSchema("type", DataType.VARCHAR, max_length=50),
        FieldSchema("created_at", DataType.INT64),
        FieldSchema("date_start", DataType.INT64),  # YYYYMMDD
        FieldSchema("date_end", DataType.INT64),  # YYYYMMDD
        FieldSchema("date_ym", DataType.INT64),  # YYYYMM
    ]
    schema = CollectionSchema(fields, desc)
    coll = Collection(name, schema)
    coll.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    logger.info(
        "[milvus] created collection=%s index=HNSW(COSINE) M=16 efC=200 dim=%d",
        name,
        EMBEDDING_DIM,
    )
    return coll


# ----------------------------------------------------------------------
# 셀 4: 장기 기억(RAG) 업데이트 (업데이트: 구조화+요약 스냅샷/멱등 업서트)
# ----------------------------------------------------------------------
_prof_coll = None
_log_coll = None
_milvus_ready = False
_coll_lock = Lock()


def ensure_milvus_collections():
    # 유지: 외부 호출자 호환. 내부는 rag.ensure_collections 사용
    return ensure_collections()


# ===== 안전한 전역 템플릿 =====
STRUCTURE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "대화에서 사실을 추출하라. JSON만 출력. 스키마:\n{schema_json}"),
        ("user", "[핀 고정(변경 금지)]:\n{pinned_json}\n\n[과거 대화]:\n{text_block}"),
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
VERIFY_RULE = "절대 새로운 사실을 추가하지 말고, 원문에 없는 수치는 넣지 마라."

# ===== 토큰 유틸 =====
enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens_text(txt: str) -> int:
    return len(enc.encode(txt))


def _count_tokens_msgs(msgs) -> int:
    total = 0
    for m in msgs:
        total += len(enc.encode((m.type + ": " + (m.content or ""))))
    return total


def _messages_to_text(msgs) -> str:
    return "\n".join(f"{m.type}: {m.content}" for m in msgs)


def _model_supports_response_format(model_name: str) -> bool:
    try:
        m = (model_name or "").lower()
        return any(k in m for k in ("gpt-4o", "gpt-4.1", "4o", "o3", "o4", "gpt-5"))
    except Exception:
        return False


def _split_for_summary(msgs, recent_budget=RECENT_RAW_TOKENS_BUDGET):
    # 뒤에서부터 recent_budget 채우고, 나머지는 old로
    recent, old = [], []
    remain = recent_budget
    for m in reversed(msgs):
        tk = len(enc.encode(m.content or ""))
        if remain > 0:
            recent.append(m)
            remain -= tk
        else:
            old.append(m)
    recent.reverse()
    old.reverse()
    return old, recent  # 오래된것들, 최근원문


def _pinned_facts_of(session_id: str) -> List[str]:
    pf = PROFILE_DB.get(session_id, {})
    # 구조 예측: 주요 키가 goals/constraints/facts 등일 것
    out = []
    for k in ("goals", "constraints", "facts", "preferences"):
        v = pf.get(k)
        if isinstance(v, list):
            out.extend([str(x) for x in v])
        elif isinstance(v, str):
            out.append(v)
    return out[:50]


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ===== 멱등키 캐시/세션 상태 =====
from typing import Any  # ensure Pydantic-friendly typing

SESSION_STATE: Dict[str, Dict[str, Any]] = {}
IDEMPOTENCY_CACHE: Dict[str, str] = {}  # session_id -> last_hash
SNAPSHOT_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=SNAPSHOT_QUEUE_MAXSIZE)
EMBED_SEM = asyncio.Semaphore(EMBED_CONCURRENCY)


async def _ensure_workers():
    # 이미 실행 중이면 무시
    if getattr(_ensure_workers, "_started", False):
        return
    _ensure_workers._started = True
    for i in range(WORKER_CONCURRENCY):
        asyncio.create_task(_snapshot_worker(i))


async def _snapshot_worker(worker_id: int):
    while True:
        session_id = await SNAPSHOT_QUEUE.get()
        t0 = time.time()
        try:
            await asyncio.to_thread(update_long_term_memory, session_id)
            took = (time.time() - t0) * 1000
            logger.info(
                "[snapshot:worker-%d] done session=%s took_ms=%.1f",
                worker_id,
                session_id,
                took,
            )
        except Exception as e:
            logger.warning(
                "[snapshot:worker-%d] error session=%s err=%r", worker_id, session_id, e
            )
        finally:
            SNAPSHOT_QUEUE.task_done()


def _enqueue_snapshot(session_id: str):
    try:
        SNAPSHOT_QUEUE.put_nowait(session_id)
        logger.info(
            "[snapshot:q] enqueued session=%s qsize=%d",
            session_id,
            SNAPSHOT_QUEUE.qsize(),
        )
    except asyncio.QueueFull:
        logger.warning("[snapshot:q] queue full -> drop session=%s", session_id)


def _edge_and_debounce(session_id: str, tokens_prev: int, tokens_now: int):
    # 세션 상태가 anchor 등 다른 키로만 초기화되었을 수 있으므로 개별 키를 안전하게 보정
    st = SESSION_STATE.setdefault(session_id, {})
    if "last_flush_at" not in st:
        st["last_flush_at"] = 0.0
    if "turns_since_last" not in st:
        st["turns_since_last"] = 0
    if "prev_tokens" not in st:
        st["prev_tokens"] = 0
    now = time.time()
    # 메모리 컴팩션 임계(3000)는 HybridSummaryMemory에서 처리. 여기서는 스냅샷 적재 임계만 본다.
    edge = tokens_prev < SNAPSHOT_EDGE_TOKENS and tokens_now >= SNAPSHOT_EDGE_TOKENS
    elapsed = now - st["last_flush_at"]
    turns_ok = st["turns_since_last"] >= DEBOUNCE_TURNS
    time_ok = elapsed >= DEBOUNCE_SECONDS

    if edge and time_ok and turns_ok:
        _enqueue_snapshot(session_id)
        schedule_directive_update(session_id)
        st["last_flush_at"] = now
        st["turns_since_last"] = 0
    else:
        st["turns_since_last"] += 1
    st["prev_tokens"] = tokens_now


# ====== 핵심: 구조화+요약 생성 ======
def _build_structured_and_summary(session_id: str, old_msgs) -> Tuple[str, dict]:
    pinned = _pinned_facts_of(session_id)
    old_text = _messages_to_text(old_msgs)

    schema_str = json.dumps(STRUCTURE_SCHEMA, ensure_ascii=False)
    pinned_str = json.dumps(pinned, ensure_ascii=False)
    text_block = old_text

    # 1) 추출적 구조화(JSON)
    t0 = time.time()
    try:
        llm_struct = (
            llm_cold.bind(response_format={"type": "json_object"})
            if _model_supports_response_format(LLM_MODEL)
            else llm_cold
        )
        struct = (STRUCTURE_PROMPT | llm_struct | StrOutputParser()).invoke(
            {
                "schema_json": schema_str,
                "pinned_json": pinned_str,
                "text_block": text_block,
            }
        )
    except Exception as e:
        logger.warning("[summary] structure LLM error: %r", e)
        struct = "{}"
    t1 = (time.time() - t0) * 1000
    logger.info("[summary] structure took_ms=%.1f", t1)

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
        summ = (SUMMARY_PROMPT | llm_cold | StrOutputParser()).invoke(
            {
                "pinned_json": pinned_str,
                "text_block": text_block,
                "verify_rule": VERIFY_RULE,
            }
        )
    except Exception as e:
        logger.warning("[summary] generative LLM error: %r", e)
        summ = ""
    t1 = (time.time() - t0) * 1000
    logger.info("[summary] generative took_ms=%.1f", t1)

    combo = (
        "[STRUCTURED]\n"
        + json.dumps(struct_json, ensure_ascii=False)
        + "\n\n[SUMMARY]\n"
        + (summ or "").strip()
    )
    tk = _count_tokens_text(combo)
    if tk > SUMMARY_TARGET_TOKENS + 100:
        combo = enc.decode(enc.encode(combo)[:SUMMARY_TARGET_TOKENS])

    meta = {
        "summary_version": "v1_struct+gen",
        "model": LLM_MODEL,
        "pinned_count": len(pinned),
    }
    return combo, meta


# ====== 커스텀 메모리: 3000 초과 시 2000→500 요약 교체 ======
class HybridSummaryMemory(ConversationSummaryBufferMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # llm, chat_memory 등 그대로
        self.max_token_limit = MAX_TOKEN_LIMIT

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # 기본 저장 (원문 유지)
        user_msg = inputs.get(self.input_key, inputs.get("input", ""))
        ai_msg = outputs.get(self.output_key, outputs.get("output", ""))

        ts_iso = _now_kst().isoformat()
        if user_msg:
            self.chat_memory.add_message(
                HumanMessage(content=user_msg, additional_kwargs={"ts": ts_iso})
            )
        if ai_msg:
            self.chat_memory.add_message(
                AIMessage(content=ai_msg, additional_kwargs={"ts": ts_iso})
            )

        # 3000 초과 시: 오래된 구간을 500 토큰으로 요약하여 교체(append 아님), 최근 원문은 유지
        msgs = self.chat_memory.messages
        total_tokens = _count_tokens_msgs(msgs)
        if total_tokens <= self.max_token_limit:
            return

        # 분리: 오래된(old) / 최근(recent)
        old_msgs, recent_msgs = _split_for_summary(msgs, RECENT_RAW_TOKENS_BUDGET)
        if not old_msgs:
            return  # 최근만으로도 3000 넘는 경우(드뭄) -> 건너뜀

        # 재요약 방지: old가 단일 요약 블록([SUMMARIZED@...])만 포함하면 스킵
        try:
            old_text_flat = "\n".join(getattr(m, "content", "") for m in old_msgs)
            if (
                old_text_flat.strip().startswith("[SUMMARIZED@")
                and "[SUMMARY]" in old_text_flat
            ):
                logger.info("[redis] skip re-summarization (already summarized block)")
                return
        except Exception:
            pass

        # 요약 생성/재구성은 이벤트 루프 블로킹을 피하기 위해 백그라운드로 처리
        async def _compact_async(
            session_id_local: str, old_msgs_local, recent_msgs_local
        ):
            combo_local, meta_local = _build_structured_and_summary(
                session_id_local, old_msgs_local
            )
            try:
                self.chat_memory.clear()
                stamp = _now_kst().isoformat()
                header = f"[SUMMARIZED@{stamp}] tokens~{SUMMARY_TARGET_TOKENS} | {meta_local['summary_version']} | model={meta_local['model']}"
                self.chat_memory.add_message(
                    AIMessage(
                        content=header + "\n\n" + combo_local,
                        additional_kwargs={"ts": stamp},
                    )
                )
                for m in recent_msgs_local:
                    kwargs = getattr(m, "additional_kwargs", {}) or {}
                    if m.type == "human":
                        self.chat_memory.add_message(
                            HumanMessage(content=m.content, additional_kwargs=kwargs)
                        )
                    else:
                        self.chat_memory.add_message(
                            AIMessage(content=m.content, additional_kwargs=kwargs)
                        )
                logger.info(
                    "[redis] compacted(async): old->summary(≈%d tok), kept recent(raw≈%d tok)",
                    SUMMARY_TARGET_TOKENS,
                    RECENT_RAW_TOKENS_BUDGET,
                )
            except Exception as e:
                logger.warning("[redis] compact error(async): %r", e)

            # 스냅샷 파이프라인에 즉시 분기(동일 멱등/중복 게이트는 update_long_term_memory에서 수행)
            try:
                if session_id_local and isinstance(session_id_local, str):
                    _enqueue_snapshot(session_id_local)
            except Exception:
                pass

        try:
            session_id = getattr(self.chat_memory, "session_id", None) or "unknown"
            asyncio.create_task(
                _compact_async(session_id, list(old_msgs), list(recent_msgs))
            )
        except Exception as e:
            logger.warning("[redis] schedule compact error: %r", e)


# 기존 함수명 유지: 내부 구현만 커스텀 메모리로 교체


def get_short_term_memory(session_id: str) -> ConversationSummaryBufferMemory:
    redis_hist = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    # pinned facts 보호: 요약 LLM은 llm_cold 사용, max_token_limit=3000
    return HybridSummaryMemory(
        llm=llm_cold,
        chat_memory=redis_hist,
        max_token_limit=MAX_TOKEN_LIMIT,
        return_messages=True,
        memory_key="chat_history",
    )


# ----------------------------------------------------------------------
# 셀 5: RAG 검색 및 단기 기억 설정(그대로)
# ----------------------------------------------------------------------

METRIC = os.getenv("MILVUS_METRIC", "COSINE").upper()  # "COSINE" 권장


def _msg_ts_dt(m) -> Optional[datetime]:
    try:
        ts = getattr(m, "additional_kwargs", {}).get("ts")
        if not ts:
            return None
        # ISO 문자열 → datetime (tz 포함)
        dt = datetime.fromisoformat(ts)
        # KST로 정규화
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        else:
            dt = dt.astimezone(KST)
        return dt
    except Exception:
        return None


def _extract_ts_bounds(
    msgs, fallback_now: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    fallback_now = fallback_now or _now_kst()
    dts = [_msg_ts_dt(m) for m in msgs]
    dts = [d for d in dts if d is not None]
    if not dts:
        # 과거 메시지에 ts가 없을 수 있으므로 안전한 폴백
        return fallback_now, fallback_now
    return min(dts), max(dts)


def _hit_similarity(hit) -> float:
    # backend.rag.utils.hit_similarity를 직접 사용(호환 유지용 래퍼)
    try:
        return __import__(
            "backend.rag.utils", fromlist=["hit_similarity"]
        ).hit_similarity(hit)
    except Exception:
        d = getattr(hit, "distance", None)
        s = getattr(hit, "score", None)
        if METRIC == "IP":
            return float(d if d is not None else s)
        elif METRIC == "COSINE":
            dist = float(d if d is not None else 1.0)
            return 1.0 - dist
        else:
            return -float(d if d is not None else 1e9)


def _milvus_hits_to_ctx(hits, score_min: float = 0.45, top_k: int = 3) -> str:
    if not hits:
        return ""
    picked = []
    for hit in hits[:top_k]:
        sim = _hit_similarity(hit)
        logger.info(
            f"[rag] id={getattr(hit,'id',None)} sim={sim:.3f} dist={getattr(hit,'distance',None)}"
        )
        if sim >= score_min:
            picked.append(hit.entity.get("text"))
    return "\n".join(p for p in picked if p)


def retrieve_from_rag(
    session_id: str,
    query: str,
    top_k: int = 2,
    date_filter: Optional[Tuple[int, int]] = None,
) -> str:
    # 호환: rag 패키지 함수로 위임
    return __import__("backend.rag", fromlist=["retrieve_from_rag"]).retrieve_from_rag(
        session_id, query, top_k, date_filter
    )


# ----------------------------------------------------------------------
# 셀 6.5: 규칙 기반 코어프 + 재작성 (WEB/RAG)
# ----------------------------------------------------------------------
def _last_user_utterance(hist: str) -> str:
    for line in reversed(hist.splitlines()):
        if line.lower().startswith(("human:", "user:")):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_topic_np(text: str) -> str:
    t = re.sub(r"[\(\)\[\]{}]", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    chunks = re.split(
        r"(?:에 대해서|에 관해|에 대한|였던|이었던|관련|이야기한|대해서|에|에서|으로|에게|와|과|의|은|는|이|가|을|를)",
        t,
    )
    cands = [c.strip() for c in chunks if c and len(c.strip()) >= 2]
    cands = [
        c
        for c in cands
        if not re.fullmatch(r"[0-9\W_]+", c)
        and c not in {"그거", "이거", "저번", "지난번", "그날", "그때"}
    ]

    def score(s):
        return (len(re.findall(r"[가-힣]", s)), len(s))

    return max(cands, key=score, default="")


def _shallow_coref(txt: str, hist: str) -> str:
    out = txt
    last = _last_user_utterance(hist)
    topic = _extract_topic_np(last) if last else ""
    if topic:
        out = re.sub(r"\b(그거|이거|그날|그때|저번에|지난번에)\b", topic, out)
        if re.search(r"\b우리\s*대화\b", out):
            out = out.replace("우리 대화", f"우리 '{topic}' 대화")
    return re.sub(r"\s+", " ", out).strip()


RAG_REWRITE_SYS = (
    "너는 쿼리 리라이팅 도우미다. 쿼리에서 임의로 내용을 추가/추측하지 마라. "
    "모호한 지시어나 상대시제는 명확한 절대 날짜 범위(YYYYMMDD~YYYYMMDD)로 해석하되, "
    "최종 출력 문장에는 날짜 표현을 넣지 말고 RAG DB 검색에 효과적인 핵심 주제/키워드만 남겨라. "
)

WEB_REWRITE_SYS = (
    "너는 웹검색 쿼리 리라이팅 도우미다. 쿼리에서 임의로 내용을 추가/추측하지 마라."
    "사용자의 쿼리를 보고 뉴스/웹문서/로컬검색에 가장 적합한 핵심 키워드만 남겨라."
    "마침표/쉼표 등 구두점 금지."
)


def _rewrite_prompt(task: str, q_rules: str, hints: str | None = None) -> list[dict]:
    sys = RAG_REWRITE_SYS if task == "rag" else WEB_REWRITE_SYS
    hint_block = f"\n참고(추가 금지): {hints}" if hints else ""
    return [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": f"입력: {q_rules}{hint_block}\n출력: 입력의 의미를 보존하며 재작성된 한 문장만 출력하라.",
        },
    ]


# ---- 휴리스틱: 소규모 인사/잡담 감지 및 웹검색 키워드 유효성 검사 ----
_SMALL_TALK_PAT = re.compile(
    r"^(안녕|하이|하이요|헬로|hello|hi|반가워|반갑|고마워|고맙다|감사|땡큐|ㅎㅎ|ㅋㅋ|ㅎㅇ|응|웅|야|헬로우)\b",
    re.IGNORECASE,
)
_GENERIC_CHITCHAT_PAT = re.compile(r"(뭐하니|뭐해|머해|뭐함|뭐하고|뭐할|어때|어떠니)\b")


def _is_small_talk(text: str) -> bool:
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return True
    if _SMALL_TALK_PAT.search(t) is not None:
        return True
    # 짧은 일반 대화문의 경우(질문형이지만 목적 불명확)도 소통으로 간주
    if len(t) <= 20 and _GENERIC_CHITCHAT_PAT.search(t):
        return True
    return False


def _is_valid_web_query(q: str) -> bool:
    if not q:
        return False
    # 구두점/물음표 포함 또는 단어 수 2 미만이면 검색성 낮다고 판단
    if re.search(r"[\.,!?]", q):
        return False
    if len(q.split()) < 2:
        return False
    return True


def _is_local_search_intent(text: str) -> bool:
    # 키워드 기반 휴리스틱은 사용하지 않음(프로덕션 비권장).
    # 웹/로컬 탐색 의도는 소분류기 및 LLM 가드로 판단.
    return False


def _looks_like_web_intent(text: str) -> bool:
    # 키워드/정규식 휴리스틱 없이 라우터/소분류기/LLM으로 판단
    return False


# ----------------------------------------------------------------------
# 셀 6.6: 담화 앵커(장소/주제) · 팔로업 탐지 · 임베딩 힌트 선택
# ----------------------------------------------------------------------


def _get_anchor_state(session_id: str) -> dict:
    st = SESSION_STATE.setdefault(session_id, {})
    anchor = st.setdefault("anchor", {"place": "", "topic": "", "ts": 0.0})
    # TTL 적용
    now_ts = time.time()
    try:
        last_ts = float(anchor.get("ts", 0.0))
    except Exception:
        last_ts = 0.0
    if last_ts and (now_ts - last_ts) > TOPIC_TTL_S:
        anchor = {"place": "", "topic": "", "ts": 0.0}
        st["anchor"] = anchor
    return anchor


def _update_anchor_state(session_id: str, place: str | None, topic: str | None):
    try:
        anchor = _get_anchor_state(session_id)
        changed = False
        if isinstance(place, str) and place.strip():
            anchor["place"] = place.strip()
            changed = True
        if isinstance(topic, str) and topic.strip():
            anchor["topic"] = topic.strip()
            changed = True
        if changed:
            anchor["ts"] = time.time()
    except Exception:
        pass


async def _extract_anchors(user_input: str, hist_tail: str) -> dict:
    """LLM으로 현재 입력/최근 맥락에서 장소/주제 앵커를 경량 추출한다."""
    schema = {
        "type": "object",
        "properties": {
            "place": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }
    msgs = [
        {
            "role": "system",
            "content": (
                "너는 담화 앵커 추출기다. 입력과 최근 맥락에서 장소명(예: 상봉역)과 핵심 주제(예: 디저트 카페)를 추출하라."
                " 모호하면 빈 문자열로 돌려라. JSON만: {place?:string, topic?:string}."
            ),
        },
        {"role": "user", "content": f"[입력]\n{user_input}\n\n[최근]\n{hist_tail}"},
    ]
    try:
        kwargs = {
            "model": LLM_MODEL,
            "messages": msgs,
            "max_tokens": 80,
            "temperature": 0.0,
        }
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_object"}
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=EXTRACT_TIMEOUT_S
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        place = (data.get("place") or "").strip()
        topic = (data.get("topic") or "").strip()
        return {"place": place, "topic": topic}
    except Exception:
        return {"place": "", "topic": ""}


def _build_hints_by_embedding(
    session_id: str,
    hist_msgs,
    query_text: str,
    lookback: int = HINT_LOOKBACK,
    max_items: int = HINT_MAX_ITEMS,
    sim_thr: float = HINT_SIM_THRESHOLD,
) -> str:
    """최근 사용자 발화에서 임베딩 유사도가 높은 문장 1~2개만 선택하여 힌트로 사용한다.
    - API 호출 최적화: OpenAIEmbeddings.embed_documents로 배치 임베딩
    - 세션 단위 캐시: SESSION_STATE[session_id]["hint_vecs"] (텍스트 해시→벡터)
    """
    try:
        # 최근에서 사용자(human) 발화만 수집
        lines = []
        for m in reversed(hist_msgs):
            if getattr(m, "type", "") == "human":
                txt = (getattr(m, "content", "") or "").strip()
                if txt:
                    lines.append(txt)
            if len(lines) >= lookback:
                break
        lines = list(reversed(lines))
        if not lines:
            return ""

        # 세션 캐시 준비
        st = SESSION_STATE.setdefault(session_id, {})
        hint_cache: dict = st.setdefault("hint_vecs", {})

        # 쿼리 임베딩 (단일, 캐시 사용)
        qv = embed_query_cached(query_text)

        # 캐시에 없는 줄만 배치 임베딩 시도
        missing = []
        idx_map = []
        for i, ln in enumerate(lines):
            key = _sha256(ln)
            if key not in hint_cache:
                missing.append(ln)
                idx_map.append((i, key))

        if missing:
            try:
                vecs = embed_docs(missing)
                for (i, key), vec in zip(idx_map, vecs):
                    hint_cache[key] = np.array(vec, dtype=np.float32)
            except Exception as _e:
                # 배치 실패 시 라인 단위 폴백 (임베딩 캐시 재사용)
                for i, key in idx_map:
                    hint_cache[key] = embed_query_cached(lines[i])

        # 최신 lookback 윈도우 내 키만 유지 (메모리 통제)
        keep_keys = set(_sha256(ln) for ln in lines)
        if len(hint_cache) > len(keep_keys):
            for k in list(hint_cache.keys()):
                if k not in keep_keys:
                    hint_cache.pop(k, None)

        # 유사도 스코어링
        scored: list[tuple[float, str]] = []
        for ln in lines:
            lv = hint_cache.get(_sha256(ln))
            if lv is None:
                lv = embed_query_cached(ln)
            denom = float(np.linalg.norm(qv) * np.linalg.norm(lv)) or 1.0
            sim = float(np.dot(qv, lv) / denom)
            if sim >= sim_thr:
                scored.append((sim, ln))

        if not scored:
            return ""
        scored.sort(key=lambda x: -x[0])
        picked = [s for _, s in scored[:max_items]]
        hint = " ".join(picked)
        return hint[:200]
    except Exception:
        return ""


async def _clarify_for_anchors(
    session_id: str, user_input: str, hist_tail: str, anchors: dict
) -> str:
    """장소/주제 앵커가 비거나 불명확할 때 LLM으로 짧은 명확화 질문을 생성한다.
    - 둘 다 충분하면 빈 문자열 반환
    - JSON 스키마 강제
    """
    place = (anchors.get("place") or "").strip()
    topic = (anchors.get("topic") or "").strip()
    if place or topic:
        return ""
    # 세션 상태 기반 힌트 구성: 저장 앵커 + STWM 스냅샷(구조화된 상태만)
    try:
        persisted = _get_anchor_state(session_id)
    except Exception:
        persisted = {"place": "", "topic": "", "ts": 0.0}
    try:
        stwm_now = get_stwm_snapshot(session_id)
    except Exception:
        stwm_now = {
            "last_loc": "",
            "last_topic": "",
        }
    hint_place = (persisted.get("place") or stwm_now.get("last_loc") or "").strip()
    hint_topic = (persisted.get("topic") or stwm_now.get("last_topic") or "").strip()
    hint = (f"장소={hint_place} | 주제={hint_topic}").strip(" | ")

    msgs = [
        {
            "role": "system",
            "content": (
                "너는 명확화 에이전트다. 아래 힌트가 있으면 그 맥락을 유지한 구체 질문 1문장만 한국어로 생성하고,"
                " 힌트가 충분하거나 모호하지 않다면 빈 문자열을 반환하라."
                " 질문은 한 문장, 공손체, 메타/광고/링크 금지. JSON만: {clarify:string}."
                f" [힌트] {hint}"
            ),
        },
        {
            "role": "user",
            "content": f"[입력]\n{user_input}\n\n[최근]\n{hist_tail}",
        },
    ]
    try:
        kwargs = {
            "model": LLM_MODEL,
            "messages": msgs,
            "max_tokens": 60,
            "temperature": 0.0,
        }
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_object"}
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=FOLLOWUP_TIMEOUT_S
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        clarify = (data.get("clarify") or "").strip()
        return clarify
    except Exception:
        return ""


async def rewrite_query(
    task: str,
    user_input: str,
    hist: str,
    anchor_hint: str | None = None,
    session_id: str | None = None,
) -> Dict[str, str | Tuple[int, int] | None]:
    """
    반환:
      - RAG: {"query_text": <날짜 제거 텍스트>, "date_filter": (start_ymd,end_ymd)}
      - WEB: {"web_query": <키워드 나열(2~6개) + (옵션)YYYY년 M월>}
    """
    base = _shallow_coref(user_input, hist)

    if task == "rag":
        date_range = _extract_date_range_for_rag(base)
        q_rules = base
        # 힌트: 최근 대화 중 해당 주제 관련 핵심 라인 일부만 요약해 사용 (오염 최소화)
        hints = ""
        try:
            msgs = hist.splitlines()[-20:]
            topic = _extract_topic_np(base)
            hints = " ".join([m for m in msgs if topic and topic in m])[:200]
        except Exception:
            hints = ""

        t0 = time.time()
        out = await _rewrite_with_retries(
            _rewrite_prompt("rag", q_rules, hints if hints else None),
            base_timeout_s=REWRITE_TIMEOUT_S,
            attempts=1,
            delta_s=1.0,
            max_tokens=REWRITE_MAX_TOKENS,
        )
        query_text = out or q_rules
        logger.info(
            "[rewrite:RAG] base='%s' -> out='%s' took_ms=%.1f",
            q_rules[:80],
            query_text[:80],
            (time.time() - t0) * 1000,
        )
        return {"query_text": query_text, "date_filter": date_range}

    # ---- WEB 케이스: LLM만으로 키워드 강제 ----
    ym_range = _month_tokens_for_web(base)
    q_rules = base
    web_query = q_rules

    # 동적 재작성 타임아웃(웹 타임아웃에 비례)
    RW_TIMEOUT = min(max(REWRITE_TIMEOUT_S, TIMEOUT_WEB * 0.6), 1.8)

    # 1차 시도: JSON 스키마로 형식 강제
    schema = {
        "name": "KeywordQuery",
        "schema": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "공백으로 구분된 2~6개의 한국어 핵심 키워드 (문장/구두점 금지)",
                }
            },
            "required": ["q"],
            "additionalProperties": False,
        },
    }
    # 힌트: 최근 대화에서 임베딩 기반으로 최소 라인만 선택(오염 방지) + 앵커 힌트 병합
    hints = ""
    try:
        # hist는 텍스트이므로 임시 메시지 컨테이너 구성
        class _Msg:
            def __init__(self, t, c):
                self.type = t
                self.content = c

        hist_msgs = []
        for ln in hist.splitlines()[-HINT_LOOKBACK:]:
            if ln.startswith("human:") or ln.startswith("Human:"):
                hist_msgs.append(_Msg("human", ln.split(":", 1)[1].strip()))
        # 세션 캐시 최적화를 위해 전달받은 session_id 사용 (없으면 default)
        try:
            emb_hints = _build_hints_by_embedding(
                session_id or "default", hist_msgs, base
            )
        except Exception:
            emb_hints = _build_hints_by_embedding("default", hist_msgs, base)
        hints = ((anchor_hint or "") + " " + (emb_hints or "")).strip()
    except Exception:
        hints = ""

    # 1) JSON 스키마 방식 재작성 + 타임아웃 재시도 (response_format 강제)
    t0 = time.time()
    hint_suffix = ("\n참고(추가 금지): " + hints) if hints else ""
    user_content_json = f"입력: {q_rules}{hint_suffix}\n출력: 문장 금지, 2~6개 키워드만 공백으로 구분해 한 줄로."
    out_json = await _rewrite_with_retries(
        [
            {"role": "system", "content": WEB_REWRITE_SYS},
            {"role": "user", "content": user_content_json},
        ],
        base_timeout_s=RW_TIMEOUT,
        attempts=1,
        delta_s=1.0,
        max_tokens=REWRITE_MAX_TOKENS,
        response_format={"type": "json_schema", "json_schema": schema},
    )
    if out_json:
        try:
            data = json.loads(out_json)
            cand = (data.get("q") or "").strip()
            if 2 <= len(cand.split()) <= 6 and not re.search(r"[.,!?]", cand):
                web_query = cand
        except Exception:
            pass
    if not web_query:
        # 2) 프리폼 백업 + 재시도
        user_content_free = f"입력: {q_rules}{hint_suffix}\n출력: 문장 금지, 2~6개 키워드만 공백으로 구분해 한 줄로."
        out2 = await _rewrite_with_retries(
            [
                {"role": "system", "content": WEB_REWRITE_SYS},
                {"role": "user", "content": user_content_free},
            ],
            base_timeout_s=min(TIMEOUT_WEB * 0.8, 2.2),
            attempts=1,
            delta_s=1.0,
            max_tokens=REWRITE_MAX_TOKENS,
        )
        if out2 and 2 <= len(out2.split()) <= 6 and not re.search(r"[.,!?]", out2):
            web_query = out2
    logger.info(
        "[rewrite:WEB] base='%s' -> out='%s' took_ms=%.1f",
        q_rules[:80],
        (web_query or q_rules)[:80],
        (time.time() - t0) * 1000,
    )

    # 연-월 토큰 주입(있으면) — 로컬검색/뉴스 혼용을 고려하여 유지
    # 로컬 검색은 연-월 토큰을 붙이면 검색 품질이 나빠질 수 있어 기본적으로 비활성화
    # 필요 시 뉴스/시황성 질의에 한해 별도 규칙으로 재도입 가능

    # 후처리: 다중 공백 정리
    web_query = re.sub(r"\s+", " ", web_query).strip()
    return {"web_query": web_query}


# ----------------------------------------------------------------------
# 셀 7: Naver Search Chain
# ----------------------------------------------------------------------
async def naver_search(query: str, display: int = 5) -> dict:
    """(호환) MCP 서버를 통해 네이버 검색 호출. service 모듈 사용 권장."""
    kind, ctx = await build_web_context(MCP_SERVER_URL, query, display, TIMEOUT_WEB)
    return {"kind": kind, "data": {"items": []}, "ctx": ctx}


async def search_web(query: str) -> str:
    kind, ctx = await build_web_context(MCP_SERVER_URL, query, 2, TIMEOUT_WEB)
    logger.info(f"[web:ctx] kind={kind} ctx_len={len(ctx)}")
    return ctx


# ----------------------------------------------------------------------
# 셀 6.8: 모바일 컨텍스트(Firestore) 조회 및 요약
# ----------------------------------------------------------------------
def _kst_day_bounds(now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    now = now or _now_kst()
    start = datetime(now.year, now.month, now.day, tzinfo=KST)
    end = start + timedelta(days=1)
    return start, end


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(timezone.utc)


def _safe_parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    s = dt_str.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _fmt_hm_kst(dt: datetime) -> str:
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_kst = dt.astimezone(KST)
        return dt_kst.strftime("%H:%M")
    except Exception:
        return ""


def _build_mobile_ctx_sync(user_id: str) -> str:
    db = _ensure_fs_db()
    if not db:
        return ""
    try:
        kst_start, kst_end = _kst_day_bounds()
        start_utc = _to_utc(kst_start)
        end_utc = _to_utc(kst_end)

        q = (
            db.collection(FIRESTORE_USERS_COLL)
            .document(user_id)
            .collection(FIRESTORE_EVENTS_SUB)
            .where("recordTimestamp", ">=", start_utc)
            .where("recordTimestamp", "<", end_utc)
            .order_by("recordTimestamp", direction=gcf.Query.DESCENDING)
            .limit(200)
        )
        docs = list(q.stream())
        if not docs:
            return ""
        events = [d.to_dict() for d in docs]

        # 최근 위치 1건
        latest_loc = None
        for e in events:
            if (e.get("dataType") or "").upper() == "LOCATION":
                latest_loc = e
                break

        # 오늘 일정 추출
        today_events: list[dict] = []
        for e in events:
            if (e.get("dataType") or "").upper() != "CALENDAR_UPDATE":
                continue
            payload = e.get("payload", {}) or {}
            for ev in payload.get("events", []) or []:
                st_raw = ev.get("startTime")
                if not st_raw:
                    continue
                st_dt = _safe_parse_iso(st_raw)
                if not st_dt:
                    continue
                # KST 기준 오늘 범위 내 여부
                st_kst = st_dt if st_dt.tzinfo else st_dt.replace(tzinfo=timezone.utc)
                st_kst = st_kst.astimezone(KST)
                if kst_start <= st_kst < kst_end:
                    today_events.append(ev)

        # 일정 포맷 (최대 5개)
        lines_cal = []
        if today_events:
            # 가장 이른 시작시간 순 정렬
            def _key(ev):
                dt = _safe_parse_iso(ev.get("startTime") or "") or _now_kst()
                return dt

            today_events.sort(key=_key)
            for ev in today_events[:5]:
                st = _safe_parse_iso(ev.get("startTime") or "")
                hm = _fmt_hm_kst(st) if st else ""
                title = (ev.get("title") or "").strip() or "(제목 없음)"
                loc = (ev.get("location") or "").strip()
                if loc:
                    lines_cal.append(f"- {hm} {title} @ {loc}")
                else:
                    lines_cal.append(f"- {hm} {title}")

        # 위치 포맷
        line_loc = ""
        if latest_loc:
            p = latest_loc.get("payload", {}) or {}
            addr = (p.get("address") or "").strip()
            if addr:
                line_loc = f"현재 위치: {addr}"
            else:
                lat = p.get("latitude")
                lng = p.get("longitude")
                if lat is not None and lng is not None:
                    line_loc = f"현재 위치: ({lat:.5f}, {lng:.5f})"

        blocks = []
        if lines_cal:
            blocks.append("[오늘 일정]\n" + "\n".join(lines_cal))
        if line_loc:
            blocks.append("[현재 위치]\n" + line_loc)
        return "\n\n".join(blocks)
    except Exception as e:
        logger.warning("[mobile] fetch error: %r", e)
        return ""


async def build_mobile_ctx(user_id: str) -> str:
    # Firestore SDK는 동기 클라이언트이므로 스레드로 오프로딩
    return await asyncio.to_thread(_build_mobile_ctx_sync, user_id)


# ----------------------------------------------------------------------
# 셀 6.9: RAG 의미 불일치(semantic mismatch) 필터
# ----------------------------------------------------------------------
async def filter_semantic_mismatch(user_input: str, rag_ctx: str) -> str:
    """사용자 질의와 RAG 컨텍스트 간 의미 불일치를 LLM으로 빠르게 감지해 필터링한다.
    - rag_ctx가 비거나 매우 짧으면 그대로 반환
    - JSON 스키마: {"keep": bool, "filtered": string}
    - 타임아웃 짧게(≤0.9s) 운영
    """
    if not rag_ctx or len(rag_ctx) < 60:
        return rag_ctx
    schema = {
        "name": "RagFilter",
        "schema": {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "filtered": {"type": "string"},
            },
            "required": ["keep"],
            "additionalProperties": False,
        },
    }
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
        kwargs = {
            "model": LLM_MODEL,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 220,
        }
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=min(0.9, TIMEOUT_RAG)
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        keep = bool(data.get("keep", False))
        if not keep:
            return ""
        filtered = (data.get("filtered") or "").strip()
        return filtered or rag_ctx
    except Exception:
        return rag_ctx


# ----------------------------------------------------------------------
# 셀 6.95: WEB 컨텍스트 필터 (현 사용자 쿼리 기준)
# ----------------------------------------------------------------------
async def filter_web_ctx(user_input: str, web_ctx: str) -> str:
    """사용자 질의와 WEB 컨텍스트 간 의미 불일치를 LLM으로 감지하여 관련 없는 블록을 제거한다.
    - web_ctx가 비거나 매우 짧으면 그대로 반환
    - 소규모 인사/잡담이면 무조건 제거(웹검색 불필요)
    - JSON 스키마: {keep:boolean, filtered:string}
    - 타임아웃 짧게(≤0.9s) 운영
    """
    if not web_ctx or len(web_ctx) < 30:
        return web_ctx
    if _is_small_talk(user_input):
        return ""
    schema = {
        "name": "WebFilter",
        "schema": {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "filtered": {"type": "string"},
            },
            "required": ["keep"],
            "additionalProperties": False,
        },
    }
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
        kwargs = {
            "model": LLM_MODEL,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 220,
        }
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        resp = await asyncio.wait_for(
            openai_chat_with_retry(**kwargs), timeout=min(0.9, TIMEOUT_WEB)
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        keep = bool(data.get("keep", False))
        if not keep:
            return ""
        filtered = (data.get("filtered") or "").strip()
        return filtered or web_ctx
    except Exception:
        return web_ctx


# ----------------------------------------------------------------------
# 셀 8: Conversation Chain
# ----------------------------------------------------------------------
async def conversation_chain(
    session_id: str, user_input: str, stm: ConversationSummaryBufferMemory
) -> str:
    # 로컬 임포트로 의존성 최소화 (상단 import 수정 불필요)
    from backend.directives.store import get_compiled as get_compiled_directives

    # 1) 사용자 고정 취향 JSON이 컴파일된 system 프롬프트(캐시) 로드
    slot_sys, _ = get_compiled_directives(session_id)

    # 2) 히스토리 + 현재 입력으로 프롬프트 구성
    hist = "\n".join(f"{m.type}: {m.content}" for m in stm.chat_memory.messages)
    prompt = (
        "너는 한국어 비서앱이다. 사용자의 지속적 취향(JSON 지시문)이 있다면 우선 적용하라.\n"
        "빈 RAG/Web 컨텍스트를 언급하지 말고, 실시간성/시스템 메타 발언 없이 자연스럽게 답하라.\n"
        "두괄식으로 핵심을 먼저 말하고, 필요하면 최소한으로만 덧붙여라.\n"
        f"[대화 히스토리]\n{hist}\n"
        f"[최신 입력]\n{user_input}"
    )

    # 3) LLM 호출 (지시문이 있으면 system 최전단에 주입)
    messages = ([{"role": "system", "content": slot_sys}] if slot_sys else []) + [
        {"role": "system", "content": "대화 전개 지침을 따르라."},
        {"role": "user", "content": prompt},
    ]

    # 순수 대화 모드는 main_response에서 단일 스트리밍 호출로 대체하므로,
    # 여기서는 품질 보완이 필요한 경우에만 사용. 기본은 경량 요약 수준만 반환.
    if SINGLE_CALL_CONV:
        return ""
    t0 = time.time()
    resp = await openai_chat_with_retry(
        model=LLM_MODEL,
        messages=messages,
        temperature=1.0,
    )
    took = (time.time() - t0) * 1000
    content = (resp.choices[0].message.content or "").strip()
    logger.info(
        f"[conv] model_used={resp.model} took_ms={took:.1f} out_len={len(content)}"
    )
    return content


# ----------------------------------------------------------------------
# 셀 10: Main LLM 최종 응답 템플릿
# ----------------------------------------------------------------------
FINAL_PROMPT = PromptTemplate(
    input_variables=[
        "rag_ctx",
        "web_ctx",
        "mobile_ctx",
        "conv_ctx",
        "aux_ctx",
        "question",
        "web_summary",
    ],
    template=(
        "너는 전문적이면서도 친근한 개인 비서 AI이다.\n\n"
        "{rag_ctx}\n\n"
        "{web_summary}"
        "{web_ctx}\n\n"
        "[모바일 컨텍스트]\n{mobile_ctx}\n\n"
        "[소통 체인 결과]\n{conv_ctx}\n\n"
        "[맥락 보조]\n{aux_ctx}\n\n"
        "사용자 질문: {question}\n"
        "규칙: 1) web_ctx가 비어 있지 않다면 web_ctx의 각 블록을 그대로 나열하라. 각 블록은 3줄(이름, 간단한 설명, 링크)이며 링크는 반드시 유지한다. 필요하면 맨 위에 한 줄 요약만 추가할 수 있다. 그 한 줄 요약에는 mobile_ctx를 참고한 맥락을 포함해도 된다. "
        "2) rag_ctx/web_ctx에 '있는 내용만' 인용·요약하고, 블록의 원문 형식은 보존한다. "
        "3) rag_ctx/web_ctx가 존재하면 conv_ctx는 무시. 단, mobile_ctx/aux_ctx는 톤/맥락 shaping 보조로만 사용하며 사실 인용 금지. "
        "4) 주소/링크는 그대로. "
        "5) mobile_ctx는 권한/정확도 이슈로 불완전할 수 있으니 사용자의 최신 질문을 최우선으로 하고, 필요할 때만 조심스레 보조로 반영하라."
    ),
)


# ----------------------------------------------------------------------
# 업데이트: 스냅샷/프로필 업서트 파이프라인 (기존 함수명 유지)
# ----------------------------------------------------------------------
def update_long_term_memory(session_id: str):
    """
    - 과거 대화(old) 구조화+요약(≈500tok) 스냅샷 생성
    - 멱등 키(sha256)로 로그 업서트 방지
    - 프로필 갱신 후 업서트
    """
    logger.info("[rag:update] start session_id=%s", session_id)
    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    messages = history.messages
    if not messages:
        logger.info("[rag:update] no messages -> skip")
        return

    # 분리
    old_msgs, recent_msgs = _split_for_summary(messages, RECENT_RAW_TOKENS_BUDGET)
    snap_start_dt, snap_end_dt = _extract_ts_bounds(old_msgs, _now_kst())
    ymd_start = _ymd(snap_start_dt)
    ymd_end = _ymd(snap_end_dt)
    ym_end = _ym(snap_end_dt)
    logger.info(
        "[rag:update] ts_bounds start=%s end=%s (ymd_start=%d ymd_end=%d ym_end=%d)",
        snap_start_dt.isoformat(),
        snap_end_dt.isoformat(),
        ymd_start,
        ymd_end,
        ym_end,
    )

    if not old_msgs:
        logger.info("[rag:update] nothing to summarize(old empty) -> skip")
        return
    old_text = _messages_to_text(old_msgs)
    snap_text, meta = _build_structured_and_summary(session_id, old_msgs)
    snap_hash = _sha256(old_text)

    # 멱등성 체크
    if IDEMPOTENCY_CACHE.get(session_id) == snap_hash:
        logger.info("[rag:update] idempotent skip (same hash)")
        return
    IDEMPOTENCY_CACHE[session_id] = snap_hash

    # 프로필 요약/통합
    conv_all = _messages_to_text(messages)
    summary_chain = (
        ChatPromptTemplate.from_template(
            "다음 대화에서 사용자의 특징과 관계없는 인사말 등 불필요한 잡담과 내용은 모두 제거하고, "
            "사용자 프로필에 유의미한 핵심 정보만 요약해라.\n{conversation}"
        )
        | llm_cold
        | StrOutputParser()
    )
    summary_text = summary_chain.invoke({"conversation": conv_all})
    logger.info("[rag:update] profile_summary_len=%d", len(summary_text or ""))

    old_prof = json.dumps(PROFILE_DB.get(session_id, {}), ensure_ascii=False)
    llm_profile = (
        llm_cold.bind(response_format={"type": "json_object"})
        if _model_supports_response_format(LLM_MODEL)
        else llm_cold
    )
    profile_chain = (
        ChatPromptTemplate.from_template(
            "[기존 프로필]\n{old}\n[요약된 최신 대화]\n{sum}\n"
            "위 내용을 반영하여 사용자 핵심 정보를 기반으로 개인화를 위한 JSON 프로필로 반환해줘."
        )
        | llm_profile
        | StrOutputParser()
    )
    new_prof_str = profile_chain.invoke({"old": old_prof, "sum": summary_text})

    try:
        new_prof = json.loads(new_prof_str)
        PROFILE_DB[session_id] = new_prof
        logger.info("[rag:update] profile_json_ok keys=%s", list(new_prof.keys()))
    except json.JSONDecodeError:
        logger.warning("[rag:update] profile json decode failed -> skip profile update")
        new_prof = PROFILE_DB.get(session_id, {})

    # RAG 모듈의 보장 함수로 단일화 (dim/스키마 일치 보장)
    try:
        from backend.rag import ensure_collections as _ensure_cols

        prof_coll, log_coll = _ensure_cols()
    except Exception as _ec_e:
        logger.warning("[rag:update] ensure_collections error: %r", _ec_e)
        # 호환 폴백: 기존 경로 유지 시도
        ensure_milvus()
        prof_coll = create_milvus_collection(PROFILE_COLLECTION_NAME, "User Profiles")
        log_coll = create_milvus_collection(LOG_COLLECTION_NAME, "Conversation Logs")

    now = _now_kst()
    ym = ym_end
    ymd = ymd_end
    try:
        part_prof = _ensure_partition(prof_coll, ym)
    except Exception:
        part_prof = None
    try:
        part_log = _ensure_partition(log_coll, ym)
    except Exception:
        part_log = None

    # 업서트: 프로필
    prof_emb = embed_query_cached(json.dumps(new_prof, ensure_ascii=False))
    try:
        prof_coll.upsert(
            [
                {
                    "id": session_id,
                    "embedding": prof_emb,
                    "text": json.dumps(new_prof, ensure_ascii=False),
                    "user_id": session_id,
                    "type": "profile",
                    "created_at": int(time.time_ns()),
                    "date_start": ymd_start,  # ⬅️ old 범위 시작
                    "date_end": ymd_end,  # ⬅️ old 범위 끝(최신)
                    "date_ym": ym,  # ⬅️ 끝월
                }
            ],
            partition_name=part_prof if part_prof else None,
        )
        logger.info("[rag:update] upsert profile session_id=%s", session_id)
    except Exception as e:
        logger.warning("[rag:update] profile upsert error: %r", e)

    # ===== 신규성 평가: 직전 프로필 vs 신규 프로필의 핵심 항목 차이 =====
    try:
        old_prof_obj = json.loads(old_prof) if old_prof else {}
    except Exception:
        old_prof_obj = {}
    new_items = _flatten_profile_items(new_prof if isinstance(new_prof, dict) else {})
    old_items = _flatten_profile_items(
        old_prof_obj if isinstance(old_prof_obj, dict) else {}
    )
    profile_delta_cnt = len(new_items - old_items)
    logger.info(
        "[novelty] profile_delta_cnt=%d (min=%d)",
        profile_delta_cnt,
        NOVELTY_MIN_PROFILE_DELTA,
    )

    # ===== 스냅샷 텍스트 정규화 해시/SimHash/근사중복 체크 =====
    # 1) 정규화 텍스트 해시 (완전중복 즉시 배제)
    try:
        _norm = " ".join((snap_text or "").strip().lower().split())
        norm_hash = _sha256(_norm)
    except Exception:
        norm_hash = _sha256(snap_text or "")

    # 2) SimHash 서명(경량): 중복 후보 빠른 배제에 사용 (Redis 저장)
    try:
        simhash64 = __import__("backend.rag.refs", fromlist=["_simhash64"])._simhash64
        sig64 = simhash64(snap_text or "")
        # Redis에 최근 서명 저장 후 근접 서명 존재 시 빠른 배제 힌트로 활용
        try:
            import redis as _redis

            _r = _redis.Redis.from_url(REDIS_URL, decode_responses=True)
            sigkey = f"snap:sig:{session_id}"
            # 근접 탐색은 간단히 동일 블록 서명만 확인(확장 여지)
            if _r.sismember(sigkey, str(sig64)):
                logger.info("[novelty] simhash immediate-hit -> likely duplicate")
        except Exception:
            pass
    except Exception:
        sig64 = 0

    # 회차 요약의 엔티티/키프레이즈로 TAGS 헤더 주입(검색 가중치용)
    try:
        struct_try = json.loads(
            (STRUCTURE_PROMPT | llm_cold | StrOutputParser()).invoke(
                {
                    "schema_json": json.dumps(STRUCTURE_SCHEMA, ensure_ascii=False),
                    "pinned_json": json.dumps(
                        _pinned_facts_of(session_id), ensure_ascii=False
                    ),
                    "text_block": old_text,
                }
            )
        )
    except Exception:
        struct_try = {}
    tags_line = ""
    try:
        ents = struct_try.get("entities", []) if isinstance(struct_try, dict) else []
        kps = struct_try.get("facts", []) if isinstance(struct_try, dict) else []
        tags = list(dict.fromkeys([str(x) for x in (ents + kps) if x]))[:10]
        if tags:
            tags_line = "[TAGS] " + ",".join(tags) + "\n"
    except Exception:
        pass

    text_blob = (
        f"[SNAPSHOT meta:turn_range=?, token_count=?, ver={meta['summary_version']}, model={meta['model']}] \n"
        + tags_line
        + snap_text
    )
    log_emb = embed_query_cached(text_blob)

    ym_min = _ym_minus_months(_now_kst(), SNAPSHOT_LOOKBACK_MONTHS)
    is_dup, dup_sim = _near_duplicate_log(session_id, log_emb, ym_min)
    logger.info(
        "[novelty] near_dup=%s sim=%.3f thr=%.2f ym_min=%d",
        is_dup,
        dup_sim,
        NOVELTY_SIM_THRESHOLD,
        ym_min,
    )

    # ===== 게이트: 근사중복이거나, 프로필 신규 항목이 거의 없으면 로그 적재 스킵 =====
    # 완전중복(정규화 해시) 및 근사중복/신규성 부족 컷
    prev_hash = SESSION_STATE.get(session_id, {}).get("last_norm_hash")
    if prev_hash == norm_hash:
        is_dup = True
    if is_dup or profile_delta_cnt < NOVELTY_MIN_PROFILE_DELTA:
        logger.info(
            "[rag:update] skip log upsert due to %s",
            "near-duplicate" if is_dup else "low-novelty(profile)",
        )
    else:
        # (기존) log_coll.upsert(...) 그대로 실행
        try:
            log_coll.upsert(
                [
                    {
                        "id": f"{session_id}:{snap_hash}",
                        "embedding": log_emb,
                        "text": text_blob,
                        "user_id": session_id,
                        "type": "log",
                        "created_at": int(time.time_ns()),
                        "date_start": ymd_start,
                        "date_end": ymd_end,
                        "date_ym": ym,
                    }
                ],
                partition_name=part_log if part_log else None,
            )
            logger.info("[rag:update] upsert log id=%s", f"{session_id}:{snap_hash}")
        except Exception as e:
            logger.warning("[rag:update] log upsert error: %r", e)

    # 서명/정규화 해시 상태 업데이트
    try:
        st = SESSION_STATE.setdefault(session_id, {})
        st["last_norm_hash"] = norm_hash
        # Redis에 simhash 집합 업데이트
        try:
            import redis as _redis

            _r = _redis.Redis.from_url(REDIS_URL, decode_responses=True)
            sigkey = f"snap:sig:{session_id}"
            _r.sadd(sigkey, str(sig64))
            _r.expire(sigkey, int(os.getenv("SNAP_SIG_TTL_SEC", "15552000")))  # 180d
        except Exception:
            pass
    except Exception:
        pass


# ----------------------------------------------------------------------
# 셀 11: FastAPI + WebSocket 서버
# ----------------------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
async def _on_startup():
    await _ensure_workers()
    await ensure_directive_workers()
    # 일일 03:00(KST) 배치 스케줄러 기동
    try:
        from backend.directives.scheduler import ensure_daily_scheduler

        await ensure_daily_scheduler()
    except Exception as e:
        logger.warning("[startup] directive scheduler init error: %r", e)
    await asyncio.to_thread(_ensure_intent_embeddings)  # ← 추가
    try:
        ensure_milvus_collections()
    except Exception as e:
        logger.warning("[startup] milvus warm error : %r", e)
    logger.info("[startup] workers ready")

    # 프로액티브 스케줄러 기동
    try:
        from backend.proactive.scheduler import ensure_proactive_scheduler

        await ensure_proactive_scheduler()
        logger.info("[startup] proactive scheduler ready")
    except Exception as e:
        logger.warning("[startup] proactive scheduler init error: %r", e)


@app.get("/")
def health():
    logger.info("[health] ok")
    return {"status": "ok"}


# ----------------------------------------------------------------------
# 내부 엔드포인트 (MCP 프록시용)
# ----------------------------------------------------------------------


@app.post("/internal/rag/retrieve")
async def internal_rag_retrieve(payload: Dict[str, Any]):
    """
    입력: {"session_id": str, "query": str, "top_k": int|None, "date_filter": [int,int]|null}
    출력: {"blocks": string}
    """
    try:
        sid = str(payload.get("session_id") or "").strip()
        q = str(payload.get("query") or "").strip()
        top_k = int(payload.get("top_k") or 2)
        df = payload.get("date_filter")
        date_filter = (
            (int(df[0]), int(df[1]))
            if isinstance(df, (list, tuple)) and len(df) == 2
            else None
        )
        blocks = retrieve_from_rag(sid, q, top_k=top_k, date_filter=date_filter)
        return {"blocks": blocks or ""}
    except Exception as e:
        logger.warning("[/internal/rag/retrieve] error: %r", e)
        return {"blocks": ""}


@app.post("/internal/mobile/context")
async def internal_mobile_context(payload: Dict[str, Any]):
    """
    입력: {"session_id": str}
    출력: {"blocks": string}
    """
    try:
        sid = str(payload.get("session_id") or "").strip()
        blocks = await build_mobile_ctx(sid)
        return {"blocks": blocks or ""}
    except Exception as e:
        logger.warning("[/internal/mobile/context] error: %r", e)
        return {"blocks": ""}


@app.post("/internal/evidence/bundle")
async def internal_evidence_bundle(payload: Dict[str, Any]):
    """
    입력: {"session_id": str, "query": str, "web_on": bool, "rag_on": bool, "timeout_s": float|None}
    출력: {"web": {"blocks": string}, "rag": {"blocks": string}}
    """
    try:
        from backend.evidence.builder import build_evidence as _build_evidence

        sid = str(payload.get("session_id") or "").strip()
        q2 = str(payload.get("query") or "").strip()
        web_on = bool(payload.get("web_on", True))
        rag_on = bool(payload.get("rag_on", True))
        timeout_s = float(payload.get("timeout_s") or max(TIMEOUT_WEB, TIMEOUT_RAG))
        mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp:5000")
        _, web_ctx, rag_ctx = await _build_evidence(
            mcp_url, sid, q2, web_on, rag_on, timeout_s
        )
        return {"web": {"blocks": web_ctx or ""}, "rag": {"blocks": rag_ctx or ""}}
    except Exception as e:
        logger.warning("[/internal/evidence/bundle] error: %r", e)
        return {"web": {"blocks": ""}, "rag": {"blocks": ""}}


@app.get("/internal/directives/{session_id}/compiled")
async def internal_directives_compiled(session_id: str):
    try:
        from backend.directives.store import get_compiled as _get_compiled

        prompt, ver = _get_compiled(session_id)
        return {"prompt": prompt or "", "version": ver or ""}
    except Exception as e:
        logger.warning("[/internal/directives/compiled] error: %r", e)
        return {"prompt": "", "version": ""}


async def background_rag_update(session_id: str):
    logger.info("[bg] schedule rag update session=%s", session_id)
    _enqueue_snapshot(session_id)


# ---- app.py: main_response (그대로) ----


async def main_response(
    session_id: str,
    user_input: str,
    websocket: WebSocket,
    mobile_ctx: str,
    rag_ctx: str,
    web_ctx: str,
    conv_ctx: str,
) -> str:
    # 로컬 임포트로 의존성 최소화
    from backend.directives.store import get_compiled as get_compiled_directives

    # web_ctx가 장황할 경우에만 한 줄 요약 프리앰블 구성
    web_summary = ""
    if web_ctx and len(web_ctx) > 400:
        web_summary = "[요약] 아래 검색 결과를 한 문장으로 요약: "

    # aux_ctx는 톤/맥락 shaping 전용: STWM/회차요약을 별도 슬롯으로 전달
    aux_ctx = SESSION_STATE.get(session_id, {}).get("aux_ctx", "")
    prompt = FINAL_PROMPT.format(
        rag_ctx=rag_ctx,
        web_ctx=web_ctx,
        mobile_ctx=mobile_ctx,
        conv_ctx=conv_ctx,
        aux_ctx=aux_ctx,
        question=user_input,
        web_summary=web_summary,
    )
    logger.info(
        f"[final] prompt_sizes rag={len(rag_ctx)} web={len(web_ctx)} conv={len(conv_ctx)} q_len={len(user_input)}"
    )

    # 1) 고정 취향 JSON 지시문(배치 에이전트 산출물)을 캐시에서 로드
    slot_sys, _ = get_compiled_directives(session_id)

    evidence_mode = bool(rag_ctx.strip() or web_ctx.strip())

    if evidence_mode:
        sys_rule = (
            "개인 비서 AI이다. 다음 규칙을 반드시 지켜서 사용자 친화적으로 답하라. "
            "1) rag_ctx/web_ctx가 존재하면 해당 범위 내 정보만 사용하고, 빈 섹션은 언급하지 말라. "
            "2) web_ctx가 있으면 각 블록(이름/간단 설명/링크)을 재질문 없이 그대로 나열하되, 맨 위에 한 줄 요약만 덧붙일 수 있다. "
            "3) 추가 질문(clarify)이나 선호 탐색 질문을 생성하지 말라. "
            "4) mobile_ctx/aux_ctx는 톤/연결감을 위한 보조로만 활용하고, 사실 인용은 금지한다. "
            "5) 불필요한 사족 없이 간결하게."
        )
    else:
        sys_rule = (
            "개인 비서 AI이다. rag_ctx/web_ctx가 비어 있으므로 conv_ctx와 aux_ctx(특히 STWM 앵커)를 우선 적용해 답하라. "
            "재질문을 생성하지 말고, 사용자가 제공한 앵커(장소/주제/시간 등)를 기준으로 즉시 실행 가능한 제안을 간결히 제시하라. "
            "사실 단정이나 추측은 금지하며, 필요한 경우 마지막 한 줄에만 선택지를 제안하라(예: '루프탑/조용/칵테일 중심 중 골라주세요'). "
            "mobile_ctx는 톤과 연결감 보조로만 사용하고, 사실 인용은 금지한다."
        )

    # 고정 정체성 프롬프트 + 동적 사용자 지시문(slot_sys) 모두 주입
    messages = (
        ([{"role": "system", "content": IDENTITY_PROMPT}])
        + ([{"role": "system", "content": slot_sys}] if slot_sys else [])
        + [
            {"role": "system", "content": sys_rule},
            {"role": "user", "content": prompt},
        ]
    )

    # Evidence 모드에서도 LLM 단계를 반드시 거친다(직접 출력 금지).
    # 필요 시 래퍼 결과를 힌트로만 사용하여 포맷 안정성을 높인다.
    evidence_hint = ""
    try:
        if evidence_mode and web_ctx.strip():
            draft = wrap_web_reply(user_input, web_ctx, "")
            ok = await _validate_final_answer(user_input, rag_ctx, web_ctx, draft)
            evidence_hint = draft if ok else ""
    except Exception:
        evidence_hint = ""

    # 4) 스트리밍 응답(LLM)
    t0 = time.time()
    full_answer = ""
    model_logged = False
    try:
        # 순수 대화 모드면(conv_ctx만 사용) conv 체인 없이 단일 스트리밍 호출
        # 웹/RAG 컨텍스트가 있을 때는 포맷 안정성과 응답 일관성을 위해 더 낮은 temperature 사용
        if _stream_allowed():
            try:
                create_kwargs = {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                }
                # 증거 모드에서는 창의성 억제(질문/일탈 방지)
                if evidence_mode:
                    create_kwargs["temperature"] = 0.0
                stream = await openai_chat_with_retry(**create_kwargs)
            except Exception as e_stream_flag:
                # 조직 스트리밍 권한 미보유 시 비활성화하고 논스트리밍으로 폴백
                _STREAM_RUNTIME_DISABLED = True
                raise e_stream_flag
        else:
            raise RuntimeError("stream_disabled")

        async for chunk in stream:
            if not model_logged:
                mid = getattr(chunk, "model", None)
                if mid:
                    logger.info(f"[final] model_used={mid}")
                    model_logged = True
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if token:
                full_answer += token
                await websocket.send_text(token)
    except Exception as e:
        logger.warning(f"[final] stream error: {repr(e)}")
        # 폴백: 논스트리밍 (권한 불가/네트워크 오류 포함)
        create_kwargs = {"model": LLM_MODEL, "messages": messages, "stream": False}
        if evidence_mode:
            create_kwargs["temperature"] = 0.0
        resp = await openai_chat_with_retry(**create_kwargs)
        logger.info(f"[final] model_used={resp.model}")
        text = (resp.choices[0].message.content or "").strip()
        full_answer = text
        if text:
            await websocket.send_text(text)

    took = (time.time() - t0) * 1000
    logger.info(f"[final] streamed out_len={len(full_answer)} took_ms={took:.1f}")
    # 경량 사후 검토(증거 모드일 때만). 실시간성 위해 비차단으로 수행
    try:
        if evidence_mode and (rag_ctx.strip() or web_ctx.strip()):
            asyncio.create_task(
                _post_verify_answer(
                    user_input, rag_ctx, web_ctx, full_answer, websocket
                )
            )
    except Exception:
        pass
    return full_answer


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("[ws] accepted session=%s", session_id)

    try:
        while True:
            user_input = await websocket.receive_text()
            turn_id = str(uuid.uuid4())[:8]
            logger.info(
                f"[turn:{turn_id}] recv q_len={len(user_input)} q='{user_input}'"
            )

            # 세션 ID를 세션 상태에 최근 사용 세션으로 기록 (임베딩 힌트 캐시 키로 활용)
            try:
                SESSION_STATE["__last_session__"] = session_id
            except Exception:
                pass
            stm = get_short_term_memory(session_id)
            hist_msgs = stm.chat_memory.messages
            hist = "\n".join(f"{m.type}: {m.content}" for m in hist_msgs)

            # ---- STWM 업데이트 + 턴 버퍼 기록(유저) ----
            try:
                stwm_snap = update_stwm(session_id, user_input)
                try:
                    logger.info(
                        f"[turn:{turn_id}] stwm latest extras=%s entities=%d",
                        (stwm_snap.extras or {}),
                        len(stwm_snap.entities or []),
                    )
                except Exception:
                    pass
            except Exception:
                stwm_snap = None
            try:
                tb_add_turn(session_id, "user", user_input)
            except Exception:
                pass

            # ---- 담화 앵커 추출 및 갱신(LLM 기반) ----
            try:
                hist_tail = "\n".join(
                    f"{m.type}: {m.content}" for m in hist_msgs[-HINT_LOOKBACK:]
                )
                anchors = await _extract_anchors(user_input, hist_tail)
                _update_anchor_state(
                    session_id, anchors.get("place"), anchors.get("topic")
                )
            except Exception as _anc_e:
                logger.warning(f"[turn:{turn_id}] anchor extract error: {repr(_anc_e)}")

            tokens_prev = SESSION_STATE.get(session_id, {}).get("prev_tokens", 0)
            tokens_now = len(enc.encode(hist))
            logger.info(f"[turn:{turn_id}] hist_tokens={tokens_now}")

            # 에지-트리거 + 디바운스: RAG 스냅샷 예약
            _edge_and_debounce(session_id, tokens_prev, tokens_now)

            # 1) 라우팅: 환경변수에 따라 임베딩 우선 또는 소분류기 우선
            ROUTE_EMBEDDING_FIRST = bool(int(os.getenv("ROUTE_EMBEDDING_FIRST", "1")))
            need_rag_prob = 0.0
            need_web_prob = 0.0
            need_rag = False
            need_web = False
            # 슬롯 충족 브릿징: 직전 턴에서 지역 명확화를 요청했고, 이번 턴에 장소가 입력되면 웹으로 승격
            try:
                st = SESSION_STATE.get(session_id, {})
                awaiting = bool(st.get("await_place"))
                place_now = bool((anchors.get("place") or "").strip())
                if awaiting and place_now:
                    need_web = True
                    need_web_prob = max(need_web_prob, max(TAU_WEB, 0.6))
                    # 일회성으로 해제
                    st["await_place"] = False
            except Exception:
                pass
            if ROUTE_EMBEDDING_FIRST:
                # (임베딩 우선) 모든 턴에서 임베딩 라우터 1차 → 애매하면 LLM one-call 보조
                aux_label = await asyncio.to_thread(embedding_router, user_input, 0.4)
                if aux_label == "rag":
                    need_rag = True
                    need_rag_prob = max(need_rag_prob, TAU_RAG + 0.1)
                elif aux_label == "web":
                    need_web = True
                    need_web_prob = max(need_web_prob, TAU_WEB + 0.1)
                if not (need_rag or need_web):
                    try:
                        data = await router_one_call(user_input, hist)
                        if (data.get("route") or "") == "rag":
                            need_rag = True
                        elif (data.get("route") or "") == "web":
                            need_web = True
                    except Exception:
                        pass
            else:
                # (레거시) 소분류기 우선 → 애매구간에서만 임베딩 라우터 보조
                pred = predict_need_flags(user_input, tau_rag=TAU_RAG, tau_web=TAU_WEB)
                need_rag_prob = float(pred["need_rag_prob"])
                need_web_prob = float(pred["need_web_prob"])
                need_rag = bool(pred["need_rag"])  # 1/0 → bool
                need_web = bool(pred["need_web"])  # 1/0 → bool

            logger.info(
                f"[turn:{turn_id}] route rag_prob={need_rag_prob:.3f} web_prob={need_web_prob:.3f} -> need_rag={need_rag} need_web={need_web}"
            )

            # 2) 저신뢰 컷 + 애매밴드 처리 + 타임박스 평행 계획
            rag_timeout = TIMEOUT_RAG
            web_timeout = TIMEOUT_WEB
            LOW_CONF_MARGIN = 0.14
            TIMEBOX_SEC = 1.2

            max_prob = max(need_rag_prob, need_web_prob)
            if max_prob < max(TAU_RAG, TAU_WEB) - LOW_CONF_MARGIN:
                # 비용절감: 검색/RAG 스킵, 보강질문 1문으로 회피
                allow_conv = True
                try:
                    # 앵커/상태가 이미 충분하다면 clarify 생략
                    persisted = _get_anchor_state(session_id)
                    stwm_now = get_stwm_snapshot(session_id)
                    has_any_anchor = any(
                        [
                            str(anchors.get("place") or "").strip(),
                            str(anchors.get("topic") or "").strip(),
                            str(persisted.get("place") or "").strip(),
                            str(persisted.get("topic") or "").strip(),
                            str(stwm_now.get("last_loc") or "").strip(),
                            str(stwm_now.get("last_topic") or "").strip(),
                        ]
                    )
                    clarify_q = ""
                    if not has_any_anchor:
                        clarify_q = await _clarify_for_anchors(
                            session_id, user_input, hist_tail, anchors
                        )
                except Exception:
                    clarify_q = ""
                # 컨텍스트-민감 단서: 이전 턴이 음식/추천 맥락이면 단일 명사는 세부조건으로 해석하여 웹 승격 시도
                try:
                    stwm_q = get_stwm_snapshot(session_id)
                    topic_hint = (stwm_q.get("last_topic") or "") if stwm_q else ""
                except Exception:
                    topic_hint = ""
                single_noun = bool(
                    re.fullmatch(r"[가-힣A-Za-z]{2,10}", user_input.strip())
                )
                food_context = bool(
                    re.search(r"맛집|카페|메뉴|식당|레스토랑|음식|먹|추천", topic_hint)
                )
                if single_noun and food_context:
                    need_web = True
                    allow_conv = False
                    logger.info(
                        f"[turn:{turn_id}] single-noun in food context -> escalate web"
                    )
                elif clarify_q:
                    await websocket.send_text(clarify_q)
                    continue
                else:
                    logger.info(
                        f"[turn:{turn_id}] low_conf no-clarify -> allow conv only"
                    )
            else:
                rag_delta = abs(need_rag_prob - TAU_RAG)
                web_delta = abs(need_web_prob - TAU_WEB)
                rag_amb = rag_delta <= AMBIGUITY_BAND
                web_amb = web_delta <= AMBIGUITY_BAND

                # 상태 기반 라우트 힌트(최근 라우트/앵커/STWM): 모호하고 아직 결정 안 났을 때만 사용
                try:
                    st_now = SESSION_STATE.get(session_id, {})
                    last_route = (st_now.get("last_route") or "").strip()
                    last_route_at = float(st_now.get("last_route_at") or 0.0)
                except Exception:
                    last_route, last_route_at = "", 0.0
                try:
                    persisted = _get_anchor_state(session_id)
                    stwm_now = get_stwm_snapshot(session_id)
                    has_any_anchor = any(
                        [
                            str(persisted.get("place") or "").strip(),
                            str(persisted.get("topic") or "").strip(),
                            str(stwm_now.get("last_loc") or "").strip(),
                            str(stwm_now.get("last_topic") or "").strip(),
                        ]
                    )
                except Exception:
                    has_any_anchor = False
                within_ttl = False
                try:
                    within_ttl = (time.time() - last_route_at) <= max(
                        300.0, float(globals().get("TOPIC_TTL_S", 600.0))
                    )
                except Exception:
                    within_ttl = False

                if (
                    (rag_amb or web_amb)
                    and (not need_rag and not need_web)
                    and has_any_anchor
                    and within_ttl
                ):
                    if last_route == "web":
                        need_web = True
                        logger.info(
                            f"[turn:{turn_id}] state-bridge -> prefer web (last_route ttl+anchors)"
                        )
                    elif last_route == "rag":
                        need_rag = True
                        logger.info(
                            f"[turn:{turn_id}] state-bridge -> prefer rag (last_route ttl+anchors)"
                        )

                if rag_amb or web_amb:
                    aux_label = await asyncio.to_thread(
                        embedding_router, user_input, 0.4
                    )
                    logger.info(f"[turn:{turn_id}] embedding_router label={aux_label}")
                    if aux_label == "rag":
                        need_rag = True
                    elif aux_label == "web":
                        need_web = True
                    elif aux_label == "both":
                        need_rag, need_web = True, True

                # 최후수단: 여전히 애매 → 저비용 병렬(타임박스)
                if not need_rag and not need_web and (rag_amb or web_amb):
                    need_rag, need_web = True, True
                    rag_timeout = TIMEBOX_SEC
                    web_timeout = TIMEBOX_SEC
                    logger.info(
                        f"[turn:{turn_id}] fallback -> cheap parallel(web2,rag2,{TIMEBOX_SEC}s)"
                    )

            # 결정 로그

            logger.info(
                f"[turn:{turn_id}] decision need_rag={need_rag} need_web={need_web}"
            )

            # 최근 라우트 저장(상태 기반 라우팅 힌트를 위해)
            try:
                st = SESSION_STATE.setdefault(session_id, {})
                st["last_route_at"] = time.time()
                st["last_route"] = (
                    "web" if need_web else ("rag" if need_rag else "conv")
                )
            except Exception:
                pass

            # === 하드 게이트: 외부 근거 필요 시 conv 비활성화 ===
            # 초기 판단값으로 설정하되, 웹/RAG 스케줄 결정 이후 최종 재계산한다.
            allow_conv = not (need_rag or need_web)
            logger.info(f"[turn:{turn_id}] allow_conv(initial)={allow_conv}")

            tasks = {}

            # 2.5) 재작성 태스크 예약
            rew_rag_task = None
            rew_web_task = None

            # 🔧 confident fast-path: 확신 크면 리라이트 생략
            fastpath_margin = 0.15
            prob_max = max(need_rag_prob, need_web_prob)
            tau_max = max(TAU_RAG, TAU_WEB)
            fastpath = prob_max >= (tau_max + fastpath_margin)
            logger.info(
                f"[turn:{turn_id}] fastpath={fastpath} prob_max={prob_max:.3f} tau_max={tau_max:.3f}"
            )

            # pronoun/time 지시어 감지: 리라이트 트리거 조건
            def detect_pronoun_or_time(t: str) -> bool:
                return bool(
                    re.search(r"(그거|이거|거기|그때|오늘|내일|이번주|지난주)", t)
                )

            pronoun_hit = detect_pronoun_or_time(user_input)

            # conv 게이트: 외부 근거가 하나라도 있으면 conv는 '스타일/연결'만 (사실 생성 금지)
            allow_conv = not (need_rag or need_web)
            conv_mode = "style_only" if not allow_conv else "full"

            # 3) 전문가 팀 병렬 실행
            if allow_conv and not SINGLE_CALL_CONV:
                tasks["conv"] = asyncio.create_task(
                    conversation_chain(session_id, user_input, stm)
                )
                logger.info(f"[turn:{turn_id}] schedule conv=on (2-call)")
            else:
                logger.info(
                    f"[turn:{turn_id}] schedule conv=off (single-call mode or blocked)"
                )

            # 3.0) 모바일 컨텍스트 준비 (항상 시도, 실패 시 빈 값)
            tasks["mobile"] = asyncio.create_task(build_mobile_ctx(session_id))

            # 3.1) RAG
            rag_query_text = None
            rag_date_filter = None
            if need_rag:
                if fastpath:
                    # 🔧 리라이트 생략: 얕은 코리퍼런스 + 날짜필터만
                    rag_query_text = _shallow_coref(user_input, hist)
                    rag_date_filter = _extract_date_range_for_rag(rag_query_text)
                    asyncio.create_task(
                        asyncio.to_thread(embed_query_cached, rag_query_text)
                    )  # 🔧 프리워밍
                    logger.info(
                        f"[turn:{turn_id}] schedule rag=FAST q='{rag_query_text[:80]}' date_filter={rag_date_filter}"
                    )
                    tasks["rag"] = asyncio.wait_for(
                        asyncio.to_thread(
                            retrieve_from_rag,
                            session_id,
                            rag_query_text,
                            2,
                            rag_date_filter,
                        ),
                        timeout=rag_timeout,
                    )
                    # 리라이트 기록
                    try:
                        add_rewrite(
                            session_id,
                            RewriteRecord(
                                raw_query=user_input,
                                query_rewritten=rag_query_text,
                                applied_slots=["coref", "date"],
                            ),
                        )
                    except Exception:
                        pass
                else:
                    # 앵커 힌트를 구성(있으면)
                    anchor = SESSION_STATE.get(session_id, {}).get("anchor", {})
                    anchor_hint = (
                        " ".join(
                            s
                            for s in [anchor.get("place", ""), anchor.get("topic", "")]
                            if s
                        ).strip()
                        or None
                    )
                    # 리라이트: 대명사/시점/지명 감지 시에만 실행(또는 fastpath=False일 때만)
                    if pronoun_hit or not fastpath:
                        rew_rag_task = asyncio.create_task(
                            rewrite_query(
                                "rag", user_input, hist, anchor_hint, session_id
                            )
                        )
                    try:
                        if rew_rag_task is not None:
                            rag_rw = await asyncio.wait_for(
                                rew_rag_task, timeout=REWRITE_TIMEOUT_S
                            )
                        else:
                            rag_rw = {"query_text": user_input, "date_filter": None}
                    except Exception as e:
                        logger.warning(
                            f"[turn:{turn_id}] rewrite RAG exception={repr(e)} -> use user_input"
                        )
                        rag_rw = {"query_text": user_input, "date_filter": None}
                    rag_query_text = rag_rw.get("query_text") or user_input
                    rag_date_filter = rag_rw.get("date_filter", None)
                    asyncio.create_task(
                        asyncio.to_thread(embed_query_cached, rag_query_text)
                    )  # 🔧 프리워밍
                    logger.info(
                        f"[turn:{turn_id}] schedule rag=on timeout={rag_timeout}s q='{rag_query_text[:80]}' date_filter={rag_date_filter}"
                    )
                    tasks["rag"] = asyncio.wait_for(
                        asyncio.to_thread(
                            retrieve_from_rag,
                            session_id,
                            rag_query_text,
                            2,
                            rag_date_filter,
                        ),
                        timeout=rag_timeout,
                    )
                    try:
                        add_rewrite(
                            session_id,
                            RewriteRecord(
                                raw_query=user_input,
                                query_rewritten=rag_query_text,
                                applied_slots=["rewrite_llm"],
                            ),
                        )
                    except Exception:
                        pass
            # 사용자가 '검색하지 말고 지난 회의 내용' 등 보정 신호를 줄 경우 강제 RAG 선회
            if not need_rag and re.search(
                r"검색.*하지 말|검색.*말고|회의 내용|지난 대화|엊그제", user_input
            ):
                logger.info(f"[turn:{turn_id}] user correction -> force RAG retry")
                need_rag = True
                rq = _shallow_coref(user_input, hist)
                df = _extract_date_range_for_rag(rq)
                tasks["rag"] = asyncio.wait_for(
                    asyncio.to_thread(retrieve_from_rag, session_id, rq, 2, df),
                    timeout=rag_timeout,
                )
                try:
                    add_rewrite(
                        session_id,
                        RewriteRecord(
                            raw_query=user_input,
                            query_rewritten=rq,
                            applied_slots=["coref", "date", "user_correction"],
                        ),
                    )
                except Exception:
                    pass
            else:
                logger.info(f"[turn:{turn_id}] schedule rag=off")

            # 3.2) WEB
            web_query = None
            if need_web:
                # 1) 반드시 LLM 리라이팅 → 검색 순서 강제
                anchor = SESSION_STATE.get(session_id, {}).get("anchor", {})
                anchor_hint = (
                    " ".join(
                        s
                        for s in [anchor.get("place", ""), anchor.get("topic", "")]
                        if s
                    ).strip()
                    or None
                )
                # composite 쿼리(질의+STWM 앵커)
                stwm_q = get_stwm_snapshot(session_id)
                bits = [user_input]
                for k in (
                    "last_loc",
                    "last_time",
                    "last_act",
                    "last_target",
                    "last_topic",
                ):
                    v = str(stwm_q.get(k) or "").strip() if stwm_q else ""
                    if v:
                        bits.append(v)
                composite = " ".join(bits)
                try:
                    rew_web_task = asyncio.create_task(
                        rewrite_query("web", composite, hist, anchor_hint, session_id)
                    )
                    web_rw = await asyncio.wait_for(
                        rew_web_task, timeout=min(REWRITE_TIMEOUT_S, 0.8)
                    )
                    web_query = (web_rw.get("web_query") or composite).strip()
                except Exception as e:
                    logger.warning(
                        f"[turn:{turn_id}] rewrite WEB exception={repr(e)} -> embedding fallback"
                    )
                    # 임베딩 폴백(간단 압축): 토큰 상위 8개
                    try:
                        toks = [t for t in re.split(r"\W+", composite) if len(t) >= 2]
                        web_query = " ".join(toks[:8]).lower()
                    except Exception:
                        web_query = composite.lower()
                logger.info(
                    f"[turn:{turn_id}] schedule web=on timeout={web_timeout}s q='{web_query[:80]}'"
                )
                # 웹 탐색 의도가 뚜렷하지 않으면 차단 (사용자 쿼리 중심)
                # 의도 판정은 소분류기 및 라우터/가드에 위임. 휴리스틱 체크 제거.
                # 2) 검색 실행

                if need_web:
                    # 엔드포인트 선택 전달(현재 search_web 시그니처가 지원할 때 반영)
                    tasks["web"] = asyncio.wait_for(
                        search_web(web_query), timeout=web_timeout
                    )
                try:
                    add_rewrite(
                        session_id,
                        RewriteRecord(
                            raw_query=user_input,
                            query_rewritten=web_query,
                            applied_slots=["rewrite_llm"],
                        ),
                    )
                except Exception:
                    pass
            # 사용자 강제 지시 휴리스틱 제거: 라우터/가드 결과에 따름
            else:
                logger.info(f"[turn:{turn_id}] schedule web=off")

            # === 라우팅 최종 확정 후 conv 허용 여부 재계산 ===
            allow_conv = not (need_rag or need_web)
            logger.info(f"[turn:{turn_id}] allow_conv(final)={allow_conv}")
            # 안전 분기: 임베딩 우선 경로에서는 pred가 없을 수 있으므로 기본값 사용
            p_rag_cal = float(locals().get("pred", {}).get("p_rag_cal", need_rag_prob))
            p_web_cal = float(locals().get("pred", {}).get("p_web_cal", need_web_prob))
            r_boost = float(locals().get("pred", {}).get("r_boost", 0.0))
            w_boost = float(locals().get("pred", {}).get("w_boost", 0.0))
            logger.info(
                f"[turn:{turn_id}] cal_p(rag)={p_rag_cal:.3f}+{r_boost:.2f}->{need_rag_prob:.3f} "
                f"cal_p(web)={p_web_cal:.3f}+{w_boost:.2f}->{need_web_prob:.3f} "
                f"fastpath={fastpath} allow_conv={allow_conv}"
            )

            # 플래너 로그 (관측용)
            try:
                aux_label_str = locals().get("aux_label", None)
            except Exception:
                aux_label_str = None
            try:
                # pred 미정의 시 현재 결정값을 사용
                raw_decision = {
                    "need_rag": bool(
                        locals().get("pred", {}).get("need_rag", need_rag)
                    ),
                    "need_web": bool(
                        locals().get("pred", {}).get("need_web", need_web)
                    ),
                }
                amb = {
                    "rag": bool(locals().get("rag_amb", False)),
                    "web": bool(locals().get("web_amb", False)),
                }
                reason = "rules_hit"
                if max_prob < max(TAU_RAG, TAU_WEB) - LOW_CONF_MARGIN:
                    reason = "low_conf"
                elif not (need_rag or need_web) and (
                    locals().get("rag_amb", False) or locals().get("web_amb", False)
                ):
                    reason = "cheap_parallel"
                elif fastpath:
                    reason = "fastpath"
                pl = PlannerLog(
                    p_rag_calib=float(p_rag_cal),
                    p_web_calib=float(p_web_cal),
                    prior_rag=float(r_boost),
                    prior_web=float(w_boost),
                    tau=float(max(TAU_RAG, TAU_WEB)),
                    delta=float(AMBIGUITY_BAND),
                    low_conf=float(max(TAU_RAG, TAU_WEB) - LOW_CONF_MARGIN),
                    fast_margin=float(fastpath_margin),
                    raw_decision=raw_decision,
                    amb=amb,
                    aux_router=str(aux_label_str or "none"),
                    final_decision={
                        "need_rag": bool(need_rag),
                        "need_web": bool(need_web),
                        "allow_conv": bool(allow_conv),
                    },
                    reason=reason,
                    time_budget_ms=int(max(rag_timeout, web_timeout) * 1000),
                )
                log_planner(pl)
            except Exception as _pl_e:
                logger.warning(f"[turn:{turn_id}] planner log error={repr(_pl_e)}")

            key_list = list(tasks.keys())
            t1 = time.time()
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            gather_ms = (time.time() - t1) * 1000

            out_map = {}
            for k, r in zip(key_list, results):
                if isinstance(r, Exception):
                    logger.warning(f"[turn:{turn_id}] task={k} exception={repr(r)}")
                    out_map[k] = ""
                else:
                    out_map[k] = r or ""
                logger.info(f"[turn:{turn_id}] task={k} ctx_len={len(out_map[k])}")

            rag_ctx = out_map.get("rag", "")
            web_ctx = out_map.get("web", "")
            mobile_ctx = out_map.get("mobile", "")
            conv_ctx = out_map.get("conv", "") if allow_conv else ""

            # 증거 번들 수집(3분 캐시): 필요 시 웹/라그 컨텍스트 보강
            try:
                if need_rag or need_web:
                    q2 = (
                        locals().get("rag_query_text")
                        or locals().get("web_query")
                        or user_input
                    ) or user_input
                    timeout_bw = max(rag_timeout, web_timeout)
                    bundle, web_ctx2, rag_ctx2 = await asyncio.wait_for(
                        build_evidence(
                            MCP_SERVER_URL,
                            session_id,
                            q2,
                            need_web,
                            need_rag,
                            timeout_bw,
                        ),
                        timeout=min(timeout_bw + 0.3, 3.0),
                    )
                    if web_ctx2:
                        web_ctx = web_ctx2
                    if rag_ctx2:
                        rag_ctx = rag_ctx2
            except Exception as _ev_e:
                logger.warning(f"[turn:{turn_id}] evidence build error={repr(_ev_e)}")
            # SINGLE_CALL_CONV 모드에서는 conv 체인을 비활성화하므로, 최종 응답의 맥락 손실을 막기 위해
            # conv_ctx가 비어 있고 순수 대화 모드(allow_conv=True)라면 최근 히스토리를 그대로 전달한다.
            if allow_conv and SINGLE_CALL_CONV and not conv_ctx:
                # 과도한 길이를 방지하기 위해 최근 히스토리만 사용
                conv_ctx = hist[-2000:]
            logger.info(
                f"[turn:{turn_id}] gather_ms={gather_ms:.1f} rag_len={len(rag_ctx)} web_len={len(web_ctx)} mobile_len={len(mobile_ctx)} conv_len={len(conv_ctx)}"
            )

            # RAG 의미 불일치 필터 적용 (짧은 타임아웃)
            if rag_ctx:
                try:
                    _rag_ctx_prev_len = len(rag_ctx)
                    rag_ctx = await filter_semantic_mismatch(user_input, rag_ctx)
                    logger.info(
                        f"[turn:{turn_id}] rag_filter len_in={_rag_ctx_prev_len} len_out={len(rag_ctx)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[turn:{turn_id}] rag_mismatch_filter error={repr(e)}"
                    )

            # WEB 컨텍스트 필터 적용 (현 사용자 쿼리 기준)
            if web_ctx:
                try:
                    _web_ctx_prev_len = len(web_ctx)
                    web_ctx = await filter_web_ctx(user_input, web_ctx)
                    logger.info(
                        f"[turn:{turn_id}] web_filter len_in={_web_ctx_prev_len} len_out={len(web_ctx)}"
                    )
                except Exception as e:
                    logger.warning(f"[turn:{turn_id}] web_filter error={repr(e)}")

            # 🔧 warm-up stream: 외부 근거 모드인데 컨텍스트가 비었을 때 즉시 프리앰블 1~2문장
            # ‘탐색 중' 프리앰블은 엄격한 웹 의도에서만, 그리고 실제 검색을 요청했으나 컨텍스트가 비었을 때만 노출
            if (need_web and not need_rag) and (not rag_ctx and not web_ctx) and True:
                try:
                    await websocket.send_text(
                        "[탐색 중] 관련 정보를 확인하는 중입니다. 근거 확보 후 답변을 이어가겠습니다.\n"
                    )
                    logger.info(f"[turn:{turn_id}] warmup sent")
                except Exception as e:
                    logger.warning(f"[turn:{turn_id}] warmup send error={repr(e)}")

            # 4) 사전 검증: 질문-컨텍스트 적합성. 문제시 재시도/명확화
            # 소규모 인사/잡담이면 명시적으로 컨텍스트 제거 후 인사 래퍼 적용
            if _is_small_talk(user_input):
                rag_ctx = ""
                web_ctx = ""
                conv_ctx = ""
                try:
                    greet = wrap_greeting_reply(user_input)
                    await websocket.send_text(greet)
                    continue
                except Exception:
                    pass

            if USE_LLM_ROUTER:
                try:
                    pre_msgs = [
                        {
                            "role": "system",
                            "content": (
                                "너는 사전 검증기다. 질문과 컨텍스트를 보고 무관/상충/추측이면 'clarify'에 짧은 질문,"
                                " 아니면 빈 문자열. JSON만: {clarify:string}."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"[질문]\n{user_input}\n\n[RAG]\n{rag_ctx[:800]}\n\n[WEB]\n{web_ctx[:800]}",
                        },
                    ]
                    kwargs_pv = {
                        "model": LLM_MODEL,
                        "messages": pre_msgs,
                        "max_tokens": 60,
                    }
                    if _model_supports_response_format(LLM_MODEL):
                        kwargs_pv["response_format"] = {"type": "json_object"}
                    pv = await asyncio.wait_for(
                        openai_chat_with_retry(**kwargs_pv),
                        timeout=PREVALIDATE_TIMEOUT_S,
                    )
                    pv_text = (pv.choices[0].message.content or "").strip()
                    try:
                        pv_json = json.loads(pv_text)
                        clarify_q = (pv_json.get("clarify") or "").strip()
                    except Exception:
                        clarify_q = ""
                    if clarify_q:
                        await websocket.send_text(clarify_q)
                        continue  # 사용자의 추가 정보 수집 후 다음 턴에서 재시도
                except Exception:
                    pass

            # 선택 컨텍스트 구성: TurnSummary Top-3 + STWM 최신 스냅샷 → aux_ctx로 분리
            try:
                q_for_select = (rag_query_text or web_query or user_input) or user_input
                sums_all = tb_get_summaries(session_id)
                picked = select_summaries(
                    q_for_select, sums_all, topk_bm25=10, topk_final=3
                )
                picked_texts = [s.answer_summary for s in picked]
            except Exception:
                picked_texts = []
            sum_block = (
                ("[회차 요약]\n" + "\n".join(f"- {t}" for t in picked_texts))
                if picked_texts
                else ""
            )
            # STWM 최신 스냅샷 병합(정형 슬롯만): 개인화 회상 강화
            try:
                stwm_dict = get_stwm_snapshot(session_id)
                stwm_lines = []
                if stwm_dict:
                    stwm_lines.append(
                        "이름: " + str(stwm_dict.get("last_person") or "")
                    )
                    stwm_lines.append("장소: " + str(stwm_dict.get("last_loc") or ""))
                    stwm_lines.append("시간: " + str(stwm_dict.get("last_time") or ""))
                    stwm_lines.append("행위: " + str(stwm_dict.get("last_act") or ""))
                    stwm_lines.append(
                        "대상: " + str(stwm_dict.get("last_target") or "")
                    )
                    stwm_lines.append(
                        "감정: " + str(stwm_dict.get("last_emotion") or "")
                    )
                    stwm_lines.append("주제: " + str(stwm_dict.get("last_topic") or ""))
                    stwm_lines.append(
                        "아이템: " + str(stwm_dict.get("last_item") or "")
                    )
                stwm_block = "[STWM]\n" + "\n".join(stwm_lines)
            except Exception:
                stwm_block = ""
            aug_blocks = "\n\n".join(b for b in [stwm_block, sum_block] if b)
            if aug_blocks:
                # aux_ctx 슬롯에 저장하여 FINAL_PROMPT에서 톤/맥락 shaping 전용으로 사용
                st = SESSION_STATE.setdefault(session_id, {})
                st["aux_ctx"] = aug_blocks

            # conv-only 경로: 전체 히스토리 대신 임베딩 유사도 Top-k 선택 문장만 conv_ctx에 투입
            try:
                if allow_conv and not (rag_ctx.strip() or web_ctx.strip()):
                    # 질의+앵커 합성 쿼리로 유사도 계산 → 위치/시간/행위/대상 문장 탈락 방지
                    stwm_dict_for_q = get_stwm_snapshot(session_id)
                    anchor_bits = []
                    for k in (
                        "last_loc",
                        "last_time",
                        "last_act",
                        "last_target",
                        "last_topic",
                    ):
                        v = (
                            str(stwm_dict_for_q.get(k) or "").strip()
                            if stwm_dict_for_q
                            else ""
                        )
                        if v:
                            anchor_bits.append(v)
                    composite_q = (user_input + " " + " ".join(anchor_bits)).strip()
                    conv_sel = _build_hints_by_embedding(
                        session_id, hist_msgs, composite_q
                    )
                    if conv_sel:
                        conv_ctx = "[선택 대화 컨텍스트]\n" + conv_sel
            except Exception:
                pass

            # 컨텍스트가 비면 대화 허용(회상 경로 보장)
            if not allow_conv and not (rag_ctx.strip() or web_ctx.strip()):
                allow_conv = True
                conv_ctx = hist[-2000:]
                logger.info(
                    f"[turn:{turn_id}] conv_rescue enabled (no external context)"
                )

            # 5) 최종 메인 LLM 스트리밍 응답 (증거 모드에서는 래퍼 우선)
            full_answer = await main_response(
                session_id,
                user_input,
                websocket,
                mobile_ctx,
                rag_ctx,
                web_ctx,
                conv_ctx,
            )

            # 대화 저장 (스트리밍 누적본) - 저장 전 레드액션 적용
            try:
                safe_user = redact_text(user_input)
                safe_ai = redact_text(full_answer)
            except Exception:
                safe_user, safe_ai = user_input, full_answer
            stm.save_context({"input": safe_user}, {"output": safe_ai})
            logger.info(f"[turn:{turn_id}] saved to STM out_len={len(full_answer)}")

            # Evidence 포인터 저장(웹/RAG 컨텍스트 존재 시)
            try:
                if web_ctx.strip() or rag_ctx.strip():
                    __store_refs = __import__(
                        "backend.rag.refs", fromlist=["store_refs_from_contexts"]
                    ).store_refs_from_contexts
                    refs = __store_refs(session_id, web_ctx, rag_ctx)
                    logger.info(
                        f"[turn:{turn_id}] stored evidence refs count={len(refs)}"
                    )
            except Exception as _ref_e:
                logger.warning(
                    f"[turn:{turn_id}] evidence refs store error={repr(_ref_e)}"
                )

            # 턴 버퍼 및 요약 트리거
            try:
                tb_add_turn(session_id, "assistant", full_answer)
            except Exception:
                pass
            try:
                await tb_maybe_summarize(session_id, stwm_snap)
            except Exception as _sum_e:
                logger.warning(f"[turn:{turn_id}] turn summary error={repr(_sum_e)}")

            # 디버그 메타(옵션): JSON은 UI 비노출, 로그/옵션 탭용
            if WS_DEBUG_META:
                try:
                    meta = {
                        "planner": {
                            "rag": bool(need_rag),
                            "web": bool(need_web),
                            "prob": {
                                "rag": need_rag_prob,
                                "web": need_web_prob,
                            },
                        },
                        "stwm": get_stwm_snapshot(session_id),
                    }
                    await websocket.send_text(
                        "\n[debug_meta] " + json.dumps(meta, ensure_ascii=False)
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info(
            "[ws] disconnected session=%s, scheduling final rag update.", session_id
        )
        _enqueue_snapshot(session_id)
        schedule_directive_update(session_id, force=True)  # directive
        pass


# 애매할 때만 호출: 라벨+재작성+명확화 질문을 한 번에 받는다.
async def router_one_call(user_input: str, hist: str) -> dict:
    # 프리폼 JSON 지침으로 단일 호출 라우팅: conv|rag|web + 필요한 쿼리/명확화
    msgs = [
        {
            "role": "system",
            "content": (
                "너는 라우팅 에이전트다. 사용자 입력과 최근 히스토리를 보고, conv|rag|web 중 하나만 고르라. "
                "필요시 rag_query 또는 web_query를 생성하고, 불확실하면 clarify에 짧은 질문 1개를 넣어라. JSON만 출력."
            ),
        },
        {"role": "user", "content": f"[hist]\n{hist[-1500:]}\n\n[input]\n{user_input}"},
    ]
    tmpl = '{"route":"conv|rag|web","rag_query":"","web_query":"","clarify":""}'
    # 1차: json_object 시도 → 실패 시 프리폼 재시도 → 마지막으로 conv 폴백
    try:
        resp1 = await openai_chat_with_retry(
            model=LLM_MODEL,
            messages=msgs,
            response_format={"type": "json_object"},
            max_tokens=REWRITE_MAX_TOKENS,
        )
        content1 = (resp1.choices[0].message.content or "").strip()
        data1 = json.loads(content1)
        if isinstance(data1, dict) and data1.get("route"):
            return data1
    except Exception:
        pass
    try:
        resp2 = await openai_chat_with_retry(
            model=LLM_MODEL, messages=msgs, max_tokens=REWRITE_MAX_TOKENS
        )
        content2 = (resp2.choices[0].message.content or "").strip()
        # 안전 JSON 추출(첫 {...} 블록)
        start = content2.find("{")
        end = content2.rfind("}")
        if start != -1 and end != -1 and end > start:
            data2 = json.loads(content2[start : end + 1])
            if isinstance(data2, dict) and data2.get("route"):
                return data2
    except Exception:
        pass
    return {"route": "conv"}


async def route_guard(user_input: str, hist: str) -> dict:
    """
    최종 라우팅 가드: 현재 입력과 최근 히스토리를 보고 conv/rag/web 중 1개만 강제 선택하거나,
    불확실하면 clarify 질문을 반환한다. JSON만 반환.
    """
    msgs = [
        {
            "role": "system",
            "content": (
                "너는 최종 라우팅 가드다. 현재 입력이 인삿말/소통이면 'conv'를 반환한다. "
                "개인 과거 회상/지난 대화 내용이면 'rag', 외부 정보 탐색(로컬/시황/웹문서)이면 'web'을 반환한다. "
                "불확실하면 clarify에 짧은 질문 1개만. JSON만: {route, clarify}."
            ),
        },
        {"role": "user", "content": f"[hist]\n{hist[-1500:]}\n\n[input]\n{user_input}"},
    ]
    try:
        kwargs = {"model": LLM_MODEL, "messages": msgs, "max_tokens": 80}
        if _model_supports_response_format(LLM_MODEL):
            kwargs["response_format"] = {"type": "json_object"}
        resp = await openai_chat_with_retry(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content.startswith("{") else {}
        route = (data.get("route") or "").strip()
        clarify = (data.get("clarify") or "").strip()
        if route in ("conv", "rag", "web") or clarify:
            return {"route": route, "clarify": clarify}
    except Exception:
        pass
    return {"route": "", "clarify": ""}


def rag_answerable(hits, metric="COSINE") -> bool:
    if not hits or len(hits[0]) == 0:
        return False
    sims = []
    for h in hits[0][:3]:
        d = getattr(h, "distance", None)
        s = getattr(h, "score", None)
        if metric == "COSINE":
            sim = 1.0 - float(d if d is not None else 1.0)
        else:
            sim = float(s if s is not None else 0.0)
        sims.append(sim)
    sims.sort(reverse=True)
    max_sim = sims[0]
    mean_sim = sum(sims) / len(sims)
    gap = sims[0] - (sims[1] if len(sims) > 1 else 0.0)
    # 예시 기준: 충분히 높은 상한 + 격차로 오판 방지
    return (max_sim >= 0.62) or (max_sim >= 0.55 and gap >= 0.08 and mean_sim >= 0.48)


async def async_search_web(query: str, display: int = 5) -> str:
    # 기본 timeout 하나만 지정하여 ValueError 방지
    timeout = httpx.Timeout(min(TIMEOUT_WEB, 2.2))
    headers = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
    params = {"query": query, "display": display}
    async with httpx.AsyncClient(timeout=timeout) as client_http:
        try:
            r = await client_http.get(
                "https://openapi.naver.com/v1/search/local.json",
                headers=headers,
                params=params,
            )
            data = r.json() if r.status_code == 200 else {}
        except Exception:
            data = {}
    items = data.get("items", []) or []
    return "\n".join(
        f"{i.get('title','').replace('<b>','').replace('</b>','')} — {i.get('roadAddress', i.get('address',''))}"
        for i in items[:5]
    )


# 백엔드 실행
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 웹소캣
# wscat -c ws://localhost:8000/ws/my-session


# ------------------------------
# 수동 트리거(테스트용)
# ------------------------------
@app.post("/proactive/trigger/{user_id}")
async def trigger_proactive(user_id: str):
    try:
        from backend.proactive.agent import select_and_send

        sent = await select_and_send(user_id, max_send=1)
        return {"status": "ok", "sent": len(sent)}
    except Exception as e:
        return {"status": "error", "error": repr(e)}
