"""
backend.proactive.agent - 프로액티브 멀티에이전트 오케스트레이션 레이어

- 역할: 사용자 프로필/로그/RAG/외부 소스를 결합해 선제적 알림/질문을 생성하고 전송한다.
- 주의: 본 모듈은 API 레이어와 분리되어 독립 워커/스케줄러에서 동작한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import firestore
from langchain_openai import ChatOpenAI

from backend.proactive.multi_agent_graph import run_proactive_pipeline
from backend.rag import retrieve_from_rag
from backend.routing.ml_classifier import classify_intent
from backend.utils.datetime_utils import kst_day_bounds, now_kst, ymd
from backend.utils.logger import log_event

from . import notifier
from .planner import plan_proactive_questions

logger = logging.getLogger("proactive")


# 스파이크(바이럴) 검출
try:
    from backend.analysis.trend import detect_viral_from_snapshots
except Exception:
    detect_viral_from_snapshots = None  # type: ignore


def _is_quiet_time(now: datetime | None = None) -> bool:
    """기본 조용 시간(22:30~07:30) 게이트. 사용자별 설정 전역 이전 기본값."""
    now = now or now_kst()
    h = now.hour + now.minute / 60.0
    # 22:30 ~ 07:30 조용 시간
    return (h >= 22.5) or (h < 7.5)


def _parse_hhmm(s: str) -> Optional[Tuple[int, int]]:
    try:
        s = (s or "").strip()
        if not s:
            return None
        hh, mm = s.split(":", 1)
        return int(hh), int(mm)
    except Exception:
        return None


# Firestore 기반 사용자 설정 로드
def _user_settings(uid: str) -> Dict[str, Any]:
    db = _fs()
    if not db:
        return {}
    try:
        doc = (
            db.collection("users")
            .document(uid)
            .collection("settings")
            .document("proactive")
            .get()
        )
        return doc.to_dict() or {}
    except Exception:
        return {}


def _is_quiet_time_user(uid: str, now: datetime | None = None) -> bool:
    """사용자 설정(quiet_hours) 기반 조용 시간 판별. 설정 부재 시 기본 로직 사용."""
    s = _user_settings(uid) or {}
    qh = s.get("quiet_hours") or {}
    start = _parse_hhmm(qh.get("start", "22:30"))
    end = _parse_hhmm(qh.get("end", "07:30"))
    if not start or not end:
        return _is_quiet_time(now)
    now = now or now_kst()
    cur = now.hour + now.minute / 60.0
    s_h, s_m = start
    e_h, e_m = end
    s_val = s_h + s_m / 60.0
    e_val = e_h + e_m / 60.0
    if s_val <= e_val:
        return (cur >= s_val) and (cur < e_val)
    # 야간 래핑(예: 22:30~07:30)
    return (cur >= s_val) or (cur < e_val)


def _last_push_time(user_id: str) -> Optional[datetime]:
    """사용자 마지막 푸시 전송 시각 조회(없으면 None)."""
    db = _fs()
    if not db:
        return None
    try:
        q = (
            db.collection("users")
            .document(user_id)
            .collection("proactive_push_logs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        docs = list(q.stream())
        if not docs:
            return None
        m = docs[0].to_dict() or {}
        ts = m.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        return None
    except Exception:
        return None


def _min_interval_ok(user_id: str, min_interval_min: int) -> bool:
    """최소 전송 간격(min_interval_min) 충족 여부."""
    last = _last_push_time(user_id)
    if not last:
        return True
    now = datetime.now(timezone.utc)
    delta_min = (now - last).total_seconds() / 60.0
    return delta_min >= float(min_interval_min)


def _build_context_summary_for_user(user_id: str) -> Dict[str, str]:
    """
    점수 게이팅용 경량 컨텍스트 요약을 구성한다.
    - device_state: 야간/조용시간 여부를 NIGHT로 반영
    - calendar_state/activity_state/location_state: 현 단계 최소 스텁(추후 확장)
    """
    device_state = "NIGHT" if _is_quiet_time_user(user_id) else "NORMAL"
    # 추후: 회의중/근시각 감지, 활동/위치 상태 연동
    return {
        "device_state": device_state,
        "calendar_state": "",
        "activity_state": "",
        "location_state": "",
    }


def _load_user_profile(user_id: str) -> Dict[str, Any]:
    """
    세렌디피티 재랭킹에 사용할 경량 사용자 프로필 로더.
    - known_entities: 사용자가 자주 언급/방문한 엔터티 식별자 목록(간단 문자열)
    - recent_domains: 최근 소비 도메인(중복 제거)
    실패/부재 시 빈 사전 반환.
    """
    db = _fs()
    if not db:
        return {}
    try:
        doc = (
            db.collection("users")
            .document(user_id)
            .collection("profiles")
            .document("content")
            .get()
        )
        data = doc.to_dict() or {}
        known = list(set((data.get("known_entities") or []) or []))
        domains = list(set((data.get("recent_domains") or []) or []))
        return {"known_entities": known, "recent_domains": domains}
    except Exception:
        return {}


# Firestore 클라이언트 (공용 싱글톤 사용)
def _fs():
    """Firestore 클라이언트 반환 (backend.config.clients 공용 싱글톤)"""
    from backend.config import get_firestore_client

    return get_firestore_client()


async def _build_contexts(user_id: str, seed: str) -> Dict[str, str]:
    """
    세션2: ContextAnalyzerAgent를 호출하여 내부 컨텍스트를 수집/분석한다.

    신규: Behavior Scoreboard Top-5 선호도를 behavior_ctx로 추가한다.
    """

    from backend.proactive.context_analyzer import ContextAnalyzerAgent
    from backend.proactive.data_contracts import ContextAnalyzerInput
    from backend.personalization.preference_scoreboard import PreferenceScoreboard
    from backend.config import get_settings

    agent = ContextAnalyzerAgent()
    inp: ContextAnalyzerInput = {
        "user_id": user_id,
        "session_id": user_id,  # 현재는 session==user
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = await agent.analyze(inp)
    ctx = out.get("internal_contexts", {})

    # 세션3: 웹 리서치 요약을 병합(간단 본문 합치기)
    try:
        from backend.proactive.web_researcher import WebResearchAgent

        web_agent = WebResearchAgent()
        web_out = await web_agent.research(
            query=seed, context_hints=[], max_sources=6, timeout_sec=2.0
        )
        # 상위 3개 결과의 제목/발췌를 요약형으로 결합
        best = list(web_out.get("results", []) or [])[:3]
        web_ctx_text = "\n".join(
            [
                f"- {b.get('title','').strip()} ({b.get('domain','')})"
                for b in best
                if b.get("title")
            ]
        )
    except Exception:
        web_ctx_text = ""

    # 신규: Behavior 신호 추가 (Top-5 선호도)
    behavior_summary = ""
    try:
        settings = get_settings()
        sb = PreferenceScoreboard(settings.REDIS_URL)

        if sb.available():
            top_prefs = sb.get_top_n(user_id, n=5)

            # 텍스트 형식으로 변환
            if top_prefs:
                behavior_lines = [
                    f"- {item['norm_key']}: 점수={item['score']:.2f}, 안정도={item.get('stability', 0.5):.2f}, 상태={item.get('status', 'pending')}"
                    for item in top_prefs
                ]
                behavior_summary = "\n".join(behavior_lines)
    except Exception as e:
        logger.warning(f"[proactive] Behavior context failed: {e}")
        behavior_summary = ""

    # 기존 호출부와 호환되는 키로 재래핑
    return {
        "session_id": user_id,
        "mobile_ctx": ctx.get("mobile_ctx", ""),
        "rag_ctx": ctx.get("rag_ctx", ""),
        "web_ctx": web_ctx_text,
        "behavior_ctx": behavior_summary,  # 신규
    }


def _short_title_from(seed: str) -> str:
    t = (seed or "").strip()
    return (t[:28] + "…") if len(t) > 28 else t


async def _draft_push(seed: str, ctx: Dict[str, str]) -> Dict[str, str]:
    """LLM으로 한국어 푸시 본문을 생성한다.

    입력: seed(주제/질문), ctx(rag/web/mobile/behavior)
    출력: {"title","body"}

    신규: behavior_ctx를 프롬프트에 포함하여 개인화 강화
    """
    from backend.config import get_settings as _gs

    THINKING_MODEL = _gs().THINKING_MODEL
    llm = ChatOpenAI(model=THINKING_MODEL)

    # 정서 톤 반영(부정 감정일 때 첫 문장 공감)
    last_emotion = ""
    try:
        from backend.memory.stwm import get_stwm_snapshot  # 지연 임포트

        sid = ctx.get("session_id") or ""
        if sid:
            snap = get_stwm_snapshot(sid)
            last_emotion = (snap or {}).get("last_emotion") or ""
    except Exception:
        pass

    sys = (
        "당신은 사용자의 맥락을 이해하고 적절한 순간에 유용한 정보를 제공하는 "
        "한국어 푸시 알림 작성 전문가입니다. "
        "사용자의 행동 선호도를 반영하여 초개인화된 메시지를 작성하세요."
    )

    rag = ctx.get("rag_ctx", "")
    web = ctx.get("web_ctx", "")
    mob = ctx.get("mobile_ctx", "")
    behavior = ctx.get("behavior_ctx", "")

    user = (
        f"[주제]\n{seed}\n\n"
        f"[RAG]\n{rag}\n\n"
        f"[WEB]\n{web}\n\n"
        f"[모바일]\n{mob}\n\n"
        f"[행동 선호도]\n{behavior}\n\n"
        f"[정서]\n{last_emotion}\n\n"
        "규칙:\n"
        "1) 행동 선호도를 반영하여 개인화된 제안 생성.\n"
        "2) 맥락 밖 정보 금지.\n"
        "3) CTA 포함 (예: '더 알아보기', '지금 확인').\n"
        "4) 존댓말 사용.\n"
        "5) 부정 감정이면 공감 톤으로 시작.\n"
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
    from backend.config import get_settings as _gs

    THINKING_MODEL = _gs().THINKING_MODEL
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

    # 프로필/민감도 반영
    persona = {}
    try:
        from backend.directives.store import load_persona as _load_persona
        from backend.directives.store import load_signals as _load_signals

        sid = ctx.get("session_id") or ""
        if sid:
            persona = _load_persona(sid) or {}
            signals = _load_signals(sid) or {}
            # 커뮤니케이션/공감 신호를 참고 메시지에 포함
            persona.setdefault("signals", {}).update(signals or {})
    except Exception:
        persona = {}

    sys = (
        "넌 푸시 메시지 검토자다. 안전/보안 위배, 과장/추측, 민감정보 언급을 금지하고, "
        "사용자 프로필/취향(존댓말/간결함/톤)과 현재 컨텍스트(모바일/RAG/웹 요약)를 반영해 필요시 최소 수정만 하라. "
        "프로필 정합성을 우선하고, 내용 사실성을 훼손하지 말 것."
    )
    user = (
        f"[주제]\n{seed}\n\n[초안]\n제목: {draft.get('title','')}\n본문: {draft.get('body','')}\n\n"
        f"[RAG]\n{ctx.get('rag_ctx','')}\n\n[WEB]\n{ctx.get('web_ctx','')}\n\n[모바일]\n{ctx.get('mobile_ctx','')}\n\n"
        f"[프로필]\n{json.dumps(persona, ensure_ascii=False)}\n\n"
        "규칙: 1) 보안/윤리/허위 금지, 2) 존댓말 유지, 3) 과장/빈말 배제, 4) 긴급 알림 아님, 5) 필요시 간결 보정, 6) 민감 데이터(정확 위치/개인식별) 노출 금지."
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
    """사용자의 FCM 토큰 목록 수집(여러 저장 위치를 순회, 중복 제거)."""
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
    """KST 기준 금일 푸시 전송 횟수 조회."""
    db = _fs()
    if not db:
        return 0
    try:
        start, end = kst_day_bounds()
        try:
            from google.cloud.firestore_v1 import FieldFilter  # type: ignore
        except Exception:
            FieldFilter = None  # type: ignore

        coll = (
            db.collection("users").document(user_id).collection("proactive_push_logs")
        )
        if FieldFilter is not None:
            q = coll.where(filter=FieldFilter("timestamp", ">=", start)).where(
                filter=FieldFilter("timestamp", "<", end)
            )
        else:
            q = coll.where("timestamp", ">=", start).where("timestamp", "<", end)
        return sum(1 for _ in q.stream())
    except Exception:
        return 0


def _log_push(user_id: str, payload: Dict[str, Any]):
    """푸시 전송 결과/메타를 Firestore에 기록(개인정보 최소화)."""
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
    except Exception as e:
        logger.warning(
            f"[proactive] Firestore push-log write failed for user={user_id}: {e}"
        )


async def propose_pushes(
    user_id: str, max_candidates: int = 3, topics_allow: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """LangGraph 플래너 출력(seed 질문)을 바탕으로 컨텍스트를 보강해 푸시 후보를 생성한다."""
    # 1) (세션1) 멀티에이전트 그래프 호출로 info_needs(우선순위 카테고리) 산출
    try:
        initial_state = {
            "user_id": user_id,
            "session_id": user_id,
            "trigger_type": "manual",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        s1_state = await asyncio.to_thread(run_proactive_pipeline, initial_state)  # type: ignore
        priorities = list(s1_state.get("info_needs", []) or [])
    except Exception:
        priorities = []

    # 2) 기존 단일 플래너를 통해 질문 후보 생성(점진적 통합 단계)
    plan = await asyncio.to_thread(plan_proactive_questions, user_id, 7, max_candidates)
    tq = plan.get("top_questions", []) or []
    if topics_allow:
        allow = set(topics_allow)
        tq = [c for c in tq if any(t in allow for t in (c.get("tags") or []))]

    # 3) 세션1 우선순위를 반영하여 태그 기반 재정렬
    def _tag_rank(tags: List[str]) -> int:
        if not priorities:
            return 999
        best = 999
        for t in tags or []:
            name = t[1:] if t.startswith("#") else t
            try:
                idx = priorities.index(name)
                if idx < best:
                    best = idx
            except ValueError:
                continue
        return best

    tq.sort(key=lambda c: _tag_rank(list(c.get("tags") or [])))

    pairs = [(x.get("q", "").strip(), list(x.get("tags") or [])) for x in tq]
    pairs = [(s, tags) for (s, tags) in pairs if s]
    if not pairs:
        return []

    ctx_tasks = [asyncio.create_task(_build_contexts(user_id, s)) for s, _ in pairs]
    ctx_list = await asyncio.gather(*ctx_tasks, return_exceptions=True)

    out: List[Dict[str, Any]] = []
    for (seed, tags), ctx in zip(pairs, ctx_list):
        if isinstance(ctx, Exception):
            continue
        # 태그를 컨텍스트에 포함하여 랭커가 활용
        if isinstance(ctx, dict):
            ctx["tags"] = tags
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
    # 사용자 설정 게이트
    settings = _user_settings(user_id)
    if not settings.get("enabled", True):
        return []
    # 조용 시간/일일 레이트리밋
    if _is_quiet_time_user(user_id):
        return []
    max_daily_user = settings.get("max_daily")
    if max_daily_user is not None:
        if _count_pushes_today(user_id) >= int(max_daily_user):
            return []
    else:
        from backend.config import get_settings as _gs

        if _count_pushes_today(user_id) >= int(_gs().PROACTIVE_DAILY_LIMIT):
            return []
    # 최소 간격 게이트
    from backend.config import get_settings as _gs2

    min_interval_min = int(
        settings.get("min_interval_min", _gs2().PROACTIVE_MIN_INTERVAL_MIN)
    )
    if not _min_interval_ok(user_id, min_interval_min):
        return []

    # (신규) 컨텍스트 기반 필요성 점수화 게이트
    from backend.config import get_settings as _gs3

    if bool(_gs3().FEATURE_PROACTIVE_SCORING):
        try:
            from backend.proactive.scoring import ProactiveScoringEngine

            engine = ProactiveScoringEngine()
            ctx = _build_context_summary_for_user(user_id)
            res = engine.score(ctx)
            try:
                log_event(
                    "proactive.gate", {"score": res.score, "reasons": res.reasons}
                )
            except Exception:
                pass
            if res.score < engine.threshold:
                return []
        except Exception:
            # 보수적 폴백: 예외 시 전송 억제
            return []

    tokens = _fcm_tokens_of(user_id)
    if not tokens:
        return []

    topics_allow = settings.get("topics_allow") or []
    # 0) 스파이크(바이럴) 알림 후보 병합(선행)
    cands: List[Dict[str, Any]] = []
    try:
        if detect_viral_from_snapshots is not None:
            viral_list, today_map, yesterday_map = detect_viral_from_snapshots(
                kinds=["news", "webkr", "blog"], threshold=2.0
            )
            # 상위 3개만 후보화
            for title in viral_list[:3]:
                cands.append(
                    {
                        "seed": f"viral:{title}",
                        "title": "급상승 이슈 감지",
                        "body": f"다음 이슈가 전일 대비 급증했습니다: {title}",
                        "contexts": {
                            "rag_ctx": "",
                            "web_ctx": "",
                            "mobile_ctx": "",
                            "meta": {
                                "today": today_map.get(title, 0),
                                "yesterday": yesterday_map.get(title, 0),
                            },
                        },
                        "variant": "viral",
                        "why_tag": "spike",
                    }
                )
    except Exception:
        pass

    # 1) 멀티에이전트 그래프 기반 생성(세션4 경로)
    more = await _propose_via_graph(user_id)
    if more:
        cands.extend(more)
    # 2) 폴백: 기존 경로 유지
    if not cands:
        cands = await propose_pushes(
            user_id, max_candidates=3, topics_allow=topics_allow
        )
    if not cands:
        return []

    # (신규) 세렌디피티 재랭킹: 이미 아는 것 제외/의외성 보너스 기반 재정렬
    from backend.config import get_settings as _gs4

    if bool(_gs4().FEATURE_SERENDIPITY):
        try:
            from backend.proactive.serendipity import rerank_with_serendipity

            profile = _load_user_profile(user_id)
            cands = rerank_with_serendipity(cands, profile)
        except Exception:
            pass

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

    now = now_kst()

    # 밴디트/랭커 기반 선별
    try:
        from .bandit import score as bandit_score  # 지연 임포트

        ranked = sorted(
            cands,
            key=lambda x: bandit_score(
                user_id,
                now,
                x.get("contexts", {}),
                seed=x.get("seed", ""),
            ),
            reverse=True,
        )
        from backend.config import get_settings as _gs5

        variant = _gs5().PROACTIVE_RANKER_VARIANT
    except Exception:
        # 폴백: 기존 타이밍 + 컨텍스트 길이
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
        from backend.config import get_settings as _gs6

        variant = _gs6().PROACTIVE_VARIANT
    picks = ranked[:max_send]

    sent_payloads: List[Dict[str, Any]] = []
    for p in picks:
        push_id = str(uuid.uuid4())
        # 그래프 경로에서 전달된 변이/Why-Tag가 있으면 우선 사용
        p_variant = p.get("variant") or ""
        data = {
            "session_id": user_id,
            "question": p.get("seed", ""),
            "kind": "proactive",
            "variant": (p_variant or variant),
            "push_id": push_id,
        }
        # Why-Tag 전달(프론트에서 사용 가능)
        if p.get("why_tag"):
            try:
                data["why_tag"] = p.get("why_tag")
            except Exception:
                pass
        ttl_sec = 3600
        for t in tokens:
            error_msg = ""
            try:
                resp = notifier.send_push(
                    t,
                    p.get("title", "알림"),
                    p.get("body", ""),
                    data=data,
                    ttl_sec=ttl_sec,
                )
                # 개인정보 최소화 로그: 원문 대신 해시/요약 메타만 저장
                import hashlib

                q = p.get("seed", "")
                body = p.get("body", "")
                q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
                body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                ctx = p.get("contexts") or {}
                ctx_meta = {
                    "rag_len": len(ctx.get("rag_ctx", "")),
                    "web_len": len(ctx.get("web_ctx", "")),
                    "mob_len": len(ctx.get("mobile_ctx", "")),
                }
                log = {
                    "timestamp": now_kst().astimezone(timezone.utc),
                    "q_hash": q_hash or "",
                    "title": p.get("title", "알림"),
                    "body_hash": body_hash or "",
                    "error": error_msg,
                    "token": t,
                    "fcm": resp,
                    "ctx_meta": ctx_meta,
                    "push_id": push_id,
                    "variant": variant,
                }
                _log_push(user_id, log)
                sent_payloads.append(log)
            except Exception as e:
                _log_push(
                    user_id,
                    {
                        "timestamp": now_kst().astimezone(timezone.utc),
                        "q_hash": "",
                        "title": p.get("title", "알림"),
                        "body_hash": "",
                        "token": t,
                        "error": error_msg,
                        "push_id": push_id,
                        "variant": variant,
                    },
                )
                continue
    return sent_payloads


async def _propose_via_graph(user_id: str) -> List[Dict[str, Any]]:
    """세션1~4 멀티에이전트 그래프를 통해 최종 알림을 생성한다.

    - confidence < 0.7 이거나 최종 알림 부재 시 빈 리스트 반환
    - 기존 인터페이스와 호환되도록 {title, body, contexts} 형태로 래핑
    """
    try:
        initial_state = {
            "user_id": user_id,
            "session_id": user_id,
            "trigger_type": "manual",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state = await asyncio.to_thread(run_proactive_pipeline, initial_state)  # type: ignore
        final = state.get("final_notification")
        conf = float(state.get("confidence_score", 0.0) or 0.0)
        if not final or conf < 0.7:
            return []
        meta = final.get("contexts_meta") if isinstance(final, dict) else {}
        variant = final.get("variant") if isinstance(final, dict) else None
        why_tag = final.get("why_tag") if isinstance(final, dict) else None
        return [
            {
                "seed": "graph_pipeline",
                "title": (final.get("title") if isinstance(final, dict) else "알림"),
                "body": (final.get("body") if isinstance(final, dict) else ""),
                "contexts": {
                    "rag_ctx": "",
                    "web_ctx": "",
                    "mobile_ctx": "",
                    "meta": meta or {},
                },
                "variant": variant or "",
                "why_tag": why_tag or None,
            }
        ]
    except Exception:
        return []
