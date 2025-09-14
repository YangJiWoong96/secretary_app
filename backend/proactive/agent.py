from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from google.cloud import firestore
from langchain_openai import ChatOpenAI

from .planner import plan_proactive_questions
from backend.rag import retrieve_from_rag
from . import notifier


KST = timezone(timedelta(hours=9))


def _now_kst() -> datetime:
    return datetime.now(KST)


def _ymd(dt: datetime) -> int:
    return dt.year * 10000 + dt.month * 100 + dt.day


def _kst_day_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or _now_kst()
    start = datetime(now.year, now.month, now.day, tzinfo=KST)
    end = start + timedelta(days=1)
    return start, end


def _is_quiet_time(now: datetime | None = None) -> bool:
    now = now or _now_kst()
    h = now.hour + now.minute / 60.0
    # 22:30 ~ 07:30 조용 시간
    return (h >= 22.5) or (h < 7.5)


def _extract_web_intent(text: str) -> bool:
    t = (text or "").strip()
    kws = ["날씨", "뉴스", "주가", "환율", "리뷰", "맛집", "근처", "위치"]
    return any(k in t for k in kws)


def _extract_local_intent(text: str) -> bool:
    t = (text or "").strip()
    kws = ["근처", "주변", "맛집", "카페", "영업시간", "주소", "위치"]
    return any(k in t for k in kws)


# Firestore 클라이언트 (지연 초기화)
_FS: firestore.Client | None = None


def _fs() -> firestore.Client | None:
    global _FS
    if _FS is not None:
        return _FS
    try:
        _FS = firestore.Client()
        return _FS
    except Exception:
        return None


async def _build_contexts(user_id: str, seed: str) -> Dict[str, str]:
    # app 모듈의 헬퍼를 재사용 (모바일/RAG/web)
    from backend.app import build_mobile_ctx
    from backend.search_engine import build_web_context

    # 모바일/RAG/Web을 가능한 범위에서 병렬화하여 지연을 줄인다
    base_url = os.getenv("MCP_SERVER_URL", "http://mcp:5000")
    want_web = _extract_web_intent(seed)
    use_mcp_ev = os.getenv("PROACTIVE_USE_MCP", "0") in ("1", "true", "True")

    mobile_task = asyncio.create_task(build_mobile_ctx(user_id))

    if use_mcp_ev and want_web:
        # MCP Evidence를 한 번 호출하여 rag/web을 동시에 확보(캐시 포함)
        async def _call_evidence():
            try:
                from backend.evidence.builder import build_evidence as _build_evidence

                _, web_ctx2, rag_ctx2 = await _build_evidence(
                    base_url, user_id, seed, True, True, 2.2
                )
                return web_ctx2 or "", rag_ctx2 or ""
            except Exception:
                return "", ""

        ev_task = asyncio.create_task(_call_evidence())
        m, ev = await asyncio.gather(mobile_task, ev_task, return_exceptions=True)
        mobile_ctx = "" if isinstance(m, Exception) else (m or "")
        if isinstance(ev, Exception):
            web_ctx, rag_ctx = "", ""
        else:
            web_ctx, rag_ctx = ev
    else:
        rag_task = asyncio.create_task(
            asyncio.to_thread(retrieve_from_rag, user_id, seed, 2, None)
        )
        web_task = (
            asyncio.create_task(
                build_web_context(base_url, seed, display=5, timeout_s=2.2)
            )
            if want_web
            else None
        )

        if web_task is not None:
            m, r, w = await asyncio.gather(
                mobile_task, rag_task, web_task, return_exceptions=True
            )
            mobile_ctx = "" if isinstance(m, Exception) else (m or "")
            rag_ctx = "" if isinstance(r, Exception) else (r or "")
            if isinstance(w, Exception):
                kind, web_ctx = "error", ""
            else:
                kind, web_ctx = w
                web_ctx = web_ctx or ""
        else:
            m, r = await asyncio.gather(mobile_task, rag_task, return_exceptions=True)
            mobile_ctx = "" if isinstance(m, Exception) else (m or "")
            rag_ctx = "" if isinstance(r, Exception) else (r or "")
            web_ctx = ""

    return {
        "mobile_ctx": mobile_ctx or "",
        "rag_ctx": rag_ctx or "",
        "web_ctx": web_ctx or "",
    }


def _short_title_from(seed: str) -> str:
    t = (seed or "").strip()
    return (t[:28] + "…") if len(t) > 28 else t


async def _draft_push(seed: str, ctx: Dict[str, str]) -> Dict[str, str]:
    """LLM으로 한국어 푸시 본문을 생성한다.

    입력: seed(주제/질문), ctx(rag/web/mobile)
    출력: {"title","body"}
    """
    THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")
    llm = ChatOpenAI(model=THINKING_MODEL)
    sys = (
        "너는 사용자의 일정/위치/대화 맥락을 바탕으로 선제적으로 도움을 제안하는 한국어 비서다. "
        "푸시 알림용 문구를 2~4문장으로 작성하되, 과장 없이 구체적인 행동을 제안하라."
    )
    rag = ctx.get("rag_ctx", "")
    web = ctx.get("web_ctx", "")
    mob = ctx.get("mobile_ctx", "")
    user = (
        f"[주제]\n{seed}\n\n[RAG]\n{rag}\n\n[WEB]\n{web}\n\n[모바일]\n{mob}\n\n"
        "규칙: 1) 맥락 밖 정보 추가 금지. 2) 사용자가 바로 행동할 수 있는 CTA를 한 문장 포함. 3) 존댓말."
    )
    res = llm.invoke(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]
    )
    body = (getattr(res, "content", "") or "").strip()
    title = _short_title_from(seed)
    return {"title": title or "알림", "body": body}


# ---------------------------------
# 푸시 본문 검증/보정 (이성 모델)
# ---------------------------------
async def _validate_and_correct_push(
    seed: str, draft: Dict[str, str], ctx: Dict[str, str]
) -> Dict[str, str]:
    """
    - 보안/안전: 공격적 표현, 민감정보 유도, 허위 사실 방지
    - 취향 적합성: 존댓말/톤, 스타일 과격 변화 방지
    - 시간/맥락: 조용 시간, 오늘 일정/위치 맥락 고려
    실패 시 보정된 본문을 반환.
    """
    THINKING_MODEL = os.getenv("THINKING_MODEL", "gpt-5-thinking")
    llm = ChatOpenAI(model=THINKING_MODEL)

    schema = {
        "name": "PushGuard",
        "schema": {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["approved", "title", "body"],
            "additionalProperties": False,
        },
    }

    sys = (
        "넌 푸시 메시지 검토자다. 안전/보안 위배, 과장/추측, 과격한 스타일 변경을 금지하고, "
        "사용자 취향(존댓말/간결함)과 현재 컨텍스트(모바일/RAG/웹 요약)를 반영해 필요시 최소 수정만 하라."
    )
    user = (
        f"[주제]\n{seed}\n\n[초안]\n제목: {draft.get('title','')}\n본문: {draft.get('body','')}\n\n"
        f"[RAG]\n{ctx.get('rag_ctx','')}\n\n[WEB]\n{ctx.get('web_ctx','')}\n\n[모바일]\n{ctx.get('mobile_ctx','')}\n\n"
        "규칙: 1) 보안/윤리/허위 금지, 2) 존댓말 유지, 3) 과장/빈말 배제, 4) 긴급 알림 아님, 5) 필요시 간결 보정."
    )
    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
        )
        content = getattr(resp, "content", "") or ""
        data = json.loads(content) if content.startswith("{") else {}
        if not data:
            return draft
        out = {
            "title": (data.get("title") or draft.get("title") or "알림").strip(),
            "body": (data.get("body") or draft.get("body") or "").strip(),
        }
        return out
    except Exception:
        return draft


def _fcm_tokens_of(user_id: str) -> List[str]:
    db = _fs()
    if not db:
        return []
    tokens: List[str] = []
    try:
        # 우선순위 1) users/{uid}/fcm_tokens/* {token}
        col = db.collection("users").document(user_id).collection("fcm_tokens")
        for d in col.stream():
            t = (d.to_dict() or {}).get("token")
            if t:
                tokens.append(t)
    except Exception:
        pass
    try:
        # 우선순위 2) users/{uid}/devices/* {fcmToken}
        col2 = db.collection("users").document(user_id).collection("devices")
        for d in col2.stream():
            t = (d.to_dict() or {}).get("fcmToken")
            if t:
                tokens.append(t)
    except Exception:
        pass
    try:
        # 우선순위 3) users/{uid} 문서 필드 {fcmToken | fcm_token}
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            m = doc.to_dict() or {}
            for k in ("fcmToken", "fcm_token"):
                t = m.get(k)
                if t:
                    tokens.append(t)
    except Exception:
        pass
    # 중복 제거
    uniq = []
    seen = set()
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _count_pushes_today(user_id: str) -> int:
    db = _fs()
    if not db:
        return 0
    try:
        start, end = _kst_day_bounds()
        q = (
            db.collection("users")
            .document(user_id)
            .collection("proactive_push_logs")
            .where("timestamp", ">=", start.astimezone(timezone.utc))
            .where("timestamp", "<", end.astimezone(timezone.utc))
        )
        return sum(1 for _ in q.stream())
    except Exception:
        return 0


def _log_push(user_id: str, payload: Dict[str, Any]):
    db = _fs()
    if not db:
        return
    try:
        ref = (
            db.collection("users")
            .document(user_id)
            .collection("proactive_push_logs")
            .document()
        )
        ref.set(payload)
    except Exception:
        pass


async def propose_pushes(user_id: str, max_candidates: int = 3) -> List[Dict[str, Any]]:
    """LangGraph 플래너 출력(seed 질문)을 바탕으로 컨텍스트를 보강해 푸시 후보를 생성한다."""
    plan = await asyncio.to_thread(plan_proactive_questions, user_id, 7, max_candidates)
    seeds = [x.get("q", "").strip() for x in plan.get("top_questions", [])]
    seeds = [s for s in seeds if s]
    if not seeds:
        return []

    ctx_tasks = [asyncio.create_task(_build_contexts(user_id, s)) for s in seeds]
    ctx_list = await asyncio.gather(*ctx_tasks, return_exceptions=True)

    out: List[Dict[str, Any]] = []
    for seed, ctx in zip(seeds, ctx_list):
        if isinstance(ctx, Exception):
            continue
        draft = await _draft_push(seed, ctx)
        # 최종 검증/보정
        safe = await _validate_and_correct_push(seed, draft, ctx)
        out.append(
            {
                "seed": seed,
                "title": safe.get("title", draft.get("title", "알림")),
                "body": safe.get("body", draft.get("body", "")),
                "contexts": ctx,
            }
        )
    return out


async def select_and_send(user_id: str, max_send: int = 1) -> List[Dict[str, Any]]:
    """사용자에 대해 지금 보내기 적절한 푸시를 선별해 전송한다."""
    # 조용 시간/일일 레이트리밋
    if _is_quiet_time():
        return []
    if _count_pushes_today(user_id) >= int(os.getenv("PROACTIVE_DAILY_LIMIT", "3")):
        return []

    tokens = _fcm_tokens_of(user_id)
    if not tokens:
        return []

    cands = await propose_pushes(user_id, max_candidates=3)
    if not cands:
        return []

    # 타이밍 게이트: 캘린더 근접/프라임 타임 우선
    def _timing_score(now: datetime, ctx: Dict[str, str]) -> float:
        # 모바일 컨텍스트에 오늘 일정이 있으면 가중
        mob = (ctx.get("mobile_ctx") or "").lower()
        score = 0.0
        if "오늘 일정" in mob or "@" in mob:
            score += 0.6
        # 프라임 타임 (점심 11:30~13:30, 저녁 19:00~21:30)
        h = now.hour + now.minute / 60.0
        if (11.5 <= h <= 13.5) or (19.0 <= h <= 21.5):
            score += 0.4
        return score

    now = _now_kst()

    # 간단 선별: (타이밍 점수 + RAG/WEB 컨텍스트 길이) 기준 상위 max_send개
    ranked = sorted(
        cands,
        key=lambda x: (
            _timing_score(now, x.get("contexts", {}))
            + 0.001
            * (
                len(x.get("contexts", {}).get("rag_ctx", ""))
                + len(x.get("contexts", {}).get("web_ctx", ""))
            )
        ),
        reverse=True,
    )
    picks = ranked[:max_send]

    sent_payloads: List[Dict[str, Any]] = []
    for p in picks:
        data = {
            "session_id": user_id,
            "question": p.get("seed", ""),
            "kind": "proactive",
        }
        ttl_sec = 3600
        for t in tokens:
            try:
                resp = notifier.send_push(
                    t,
                    p.get("title", "알림"),
                    p.get("body", ""),
                    data=data,
                    ttl_sec=ttl_sec,
                )
                log = {
                    "timestamp": _now_kst().astimezone(timezone.utc),
                    "q": p.get("seed", ""),
                    "title": p.get("title", "알림"),
                    "body": p.get("body", ""),
                    "token": t,
                    "fcm": resp,
                    "contexts": {
                        k: (v[:800] if isinstance(v, str) else v)
                        for k, v in (p.get("contexts") or {}).items()
                    },
                }
                _log_push(user_id, log)
                sent_payloads.append(log)
            except Exception as e:
                _log_push(
                    user_id,
                    {
                        "timestamp": _now_kst().astimezone(timezone.utc),
                        "q": p.get("seed", ""),
                        "title": p.get("title", "알림"),
                        "body": p.get("body", ""),
                        "token": t,
                        "error": repr(e),
                    },
                )
                continue
    return sent_payloads
