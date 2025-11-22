# # 큐/워커 + app.py 훅용 API

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # C:\My_Business\.env 자동 탐색/로드

import asyncio
import os
import time
from typing import Tuple

from langchain_community.chat_message_histories.redis import RedisChatMessageHistory

from backend.rag import ensure_collections

from .agent import extract_report
from .bot_profile import BotProfile, load_static_bot_profile, merge_bot_profiles
from .compiler import compile_prompt_from_json
from .policy import apply_update_policy, conservative_persona_merge, ema_merge_signals
from .signals_ingest import summarize_ingest_profile
from .store import (
    get_compiled,
    load_directives,
    load_persona,
    load_signals,
    save_directives,
    save_persona,
    save_signals,
    set_compiled,
    version_of,
)
from .validator import validate_directives
from backend.routing.router_context import user_for_session
from backend.directives.store_user_ext import (
    load_persona_user,
    save_persona_user,
)
from backend.directives.store_user_ext import save_directives_user
from backend.personalization.trait_aggregator import TraitAggregator

from backend.utils.tracing import traceable

# (옵션) 멀티프로세스 락용 redis
try:
    import redis as _redis
except Exception:
    _redis = None

# ------------------------------------------------------------------------------
# 환경 변수
# ------------------------------------------------------------------------------
from backend.config import get_settings

_s = get_settings()
REDIS_URL = _s.REDIS_URL

QUEUE_MAX = int(getattr(_s, "DIR_QUEUE_MAX", 128))
WORKERS = int(getattr(_s, "DIR_WORKERS", 2))

# 배치 디바운스: 턴이 몇 개 누적되어야 큐잉할지
DIR_DEBOUNCE_TURNS = int(getattr(_s, "DIR_DEBOUNCE_TURNS", 5))

# 인메모리 가드(중복 큐잉 억제)
ENQ_GUARD_S = int(getattr(_s, "DIR_ENQ_GUARD_S", 10))

# Redis 락 사용 여부(멀티프로세스/멀티인스턴스)
USE_REDIS_LOCK = bool(getattr(_s, "DIR_USE_REDIS_LOCK", False))
DIR_LOCK_EX_S = int(getattr(_s, "DIR_LOCK_EX_S", 10))  # 락 TTL

# ------------------------------------------------------------------------------
# 내부 상태
# ------------------------------------------------------------------------------
_started = False
_Q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAX)

# 세션별 누적 턴 카운터(디바운스용)
_TURNS_SINCE: dict[str, int] = {}

# 인메모리 최근 큐잉 시각(프로세스 단위 가드)
_ENQ_AT: dict[str, int] = {}

# Redis 클라이언트(락 전용) - 지연 초기화
_lock_r = None


def _get_lock_r():
    """멀티프로세스 락용 Redis 클라이언트 싱글톤을 지연 생성한다."""
    global _lock_r
    if not USE_REDIS_LOCK:
        return None
    try:
        if _redis is None:
            return None
        if _lock_r is None:
            _lock_r = _redis.Redis.from_url(REDIS_URL)
        return _lock_r
    except Exception:
        return None


# ------------------------------------------------------------------------------
# 보조 유틸
# ------------------------------------------------------------------------------
from hashlib import sha256


def _hist_hash(msgs) -> str:
    try:
        txt = "\n".join(
            f"{getattr(m, 'type', '')}:{getattr(m, 'content', '')}"
            for m in (msgs or [])
        )
        return sha256((txt or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _ewma(prev: float, cur: float, alpha: float = 0.5) -> float:
    try:
        return float(alpha) * float(cur) + (1.0 - float(alpha)) * float(prev)
    except Exception:
        return cur


# ------------------------------------------------------------------------------
# 워커
# ------------------------------------------------------------------------------
@traceable(name="Directives: worker", run_type="chain", tags=["directives", "worker"])
async def _worker(i: int):
    while True:
        sid = await _Q.get()
        t0 = time.time()
        try:
            # 세션→사용자 매핑(없으면 세션=사용자)
            uid = user_for_session(sid) or sid

            hist = RedisChatMessageHistory(session_id=sid, url=REDIS_URL)
            msgs = hist.messages
            if not msgs:
                continue

            # 히스토리 해시 기반 중복 처리 방지 및 메타 선로딩
            prev_dirs, meta = load_directives(sid)
            prev_sig = load_signals(sid)
            # Persona는 사용자 범위로 관리
            prev_per = load_persona_user(uid) or {}
            meta = meta or {}
            h = _hist_hash(msgs)
            last_h = str(meta.get("last_hist_hash") or "")
            if last_h and last_h == h:
                # 동일 히스토리로 재큐잉된 작업은 스킵
                continue

            # 1) 리포트 추출: directives + signals + persona
            report = await extract_report(msgs)

            # 1-1) 신뢰도 EWMA 및 급락 롤백 트리거
            try:
                conf_now = float(report.get("confidence", 0.0))
            except Exception:
                conf_now = 0.0
            ewma_prev = float(meta.get("ewma_confidence", 0.0) or 0.0)
            alpha_conf = float(getattr(_s, "DIR_CONF_EWMA_ALPHA", 0.5))
            ewma_new = _ewma(ewma_prev, conf_now, alpha=alpha_conf)
            meta["ewma_confidence"] = ewma_new
            meta["last_hist_hash"] = h

            rollback_delta = float(getattr(_s, "DIR_CONF_ROLLBACK_DELTA", 0.15))
            do_rollback = (ewma_prev > 0.0) and (conf_now + rollback_delta < ewma_prev)

            merged_dirs, meta2 = apply_update_policy(prev_dirs, report, meta)

            # 최종 검증(보안/취향/시간/맥락) — 과격한 변경 방지용 최소 보정
            try:
                contexts_for_validation = {
                    "signals": prev_sig,
                    "persona": prev_per,
                }
                merged_dirs = validate_directives(merged_dirs, contexts_for_validation)
            except Exception:
                pass

            # Firestore에서 모바일 요약 결합
            ingest_mobile = summarize_ingest_profile(sid) or {}
            cand_sig = dict(report.get("signals") or {})
            if ingest_mobile:
                cand_sig["mobile"] = ingest_mobile
            merged_sig = ema_merge_signals(prev_sig, cand_sig)

            # Persona 병합(사용자 범위)
            cand_persona = report.get("persona") or {}
            conf_persona = float(report.get("confidence", 0.0))
            merged_per = conservative_persona_merge(
                prev_per, cand_persona, conf_persona
            )

            # 사용자 트레이트 누적(EMA/EWVar) + MBTI 관측/확정
            try:
                agg = TraitAggregator(REDIS_URL)
                if agg.available():
                    bf = (cand_persona or {}).get("bigfive") or {}
                    if bf:
                        agg.update_bigfive(uid, bf, conf_persona)
                    mbti = (cand_persona or {}).get("mbti")
                    if mbti:
                        agg.record_mbti_observation(uid, mbti)
                        final_mbti = agg.maybe_finalize_mbti(uid)
                        if final_mbti:
                            merged_per["mbti"] = final_mbti
            except Exception:
                pass

            # 신뢰 급락 시 롤백: 지시문은 이전값 유지, 신호/페르소나는 보수 병합 결과만 저장
            if do_rollback:
                try:
                    save_directives(
                        sid,
                        prev_dirs,
                        meta2,
                        reasons=(report.get("reasons") or [])
                        + ["rollback_by_conf_drop"],
                    )
                except Exception:
                    pass
                try:
                    save_signals(sid, merged_sig)
                except Exception:
                    pass
                try:
                    # 세션 범위 저장은 호환 유지, 사용자 범위가 기준
                    save_persona_user(uid, merged_per)
                except Exception:
                    pass
                try:
                    # 사용자 범위 지시문 저장(신뢰 급락 시에도 사용자 룰북이 유지되도록)
                    save_directives_user(uid, prev_dirs, meta2)
                except Exception:
                    pass
                # 롤백 경로에서는 이후 재컴파일 판단만 수행
                ver_obj = {
                    "d": prev_dirs,
                    "s": {
                        k: merged_sig.get(k)
                        for k in (
                            "language",
                            "topics",
                            "style",
                            "meta",
                            "affect",
                            "mobile",
                        )
                        if k in merged_sig
                    },
                    "p": {"bigfive": (merged_per.get("bigfive") or {})},
                }
                new_ver = version_of(ver_obj)
                cur_prompt, cur_ver = get_compiled(sid)
                if cur_ver != new_ver:
                    from backend.directives.capabilities import build_capabilities_card

                    base = compile_prompt_from_json(prev_dirs, merged_sig, merged_per)
                    static_bp = load_static_bot_profile()
                    dynamic_bp = BotProfile()  # 롤백 시에는 동적 변화 없음
                    merged_bp = merge_bot_profiles(static_bp, dynamic_bp)
                    bp_text = (
                        "[BotProfile:Static+Dynamic]\n"
                        + f"persona: {merged_bp.persona}\nstyle: {merged_bp.style}\n"
                        + (
                            "abilities: " + ", ".join(merged_bp.abilities) + "\n"
                            if merged_bp.abilities
                            else ""
                        )
                        + (
                            "constraints: "
                            + ", ".join(
                                [f"{k}:{v}" for k, v in merged_bp.constraints.items()]
                            )
                            if merged_bp.constraints
                            else ""
                        )
                    )
                    compiled = (
                        bp_text + "\n\n" + base + "\n\n" + build_capabilities_card()
                    )
                    set_compiled(sid, compiled, new_ver)
                continue

            # 정적(불변) 선호가 prev_dirs에 존재한다고 가정할 때, 동적 변경과 충돌 시 정적 우선
            # - prev_dirs의 값은 고정으로 유지, report에서 새로 제안된 키는 'suggested_*' 네임스페이스로 보존
            dyn_dirs = report.get("directives") or {}
            if isinstance(dyn_dirs, dict):
                suggested: dict = {}
                for k, v in list(dyn_dirs.items()):
                    if k in (prev_dirs or {}):
                        # 충돌 → 정적 유지, 제안은 suggested로 이동
                        suggested[f"suggested_{k}"] = v
                        dyn_dirs.pop(k, None)
                if suggested:
                    # 메타에 제안 기록(감사 추적 목적)
                    meta.setdefault("last_changed", {})
                    meta["last_suggested"] = {
                        **(meta.get("last_suggested") or {}),
                        **suggested,
                    }

            # 3) 저장 및 컴파일된 시스템 프롬프트 갱신
            save_directives(sid, merged_dirs, meta2, reasons=report.get("reasons"))
            save_signals(sid, merged_sig)
            # 사용자 범위(persona) 저장
            save_persona_user(uid, merged_per)
            # 사용자 범위(directives) 저장: 새 세션에도 룰북이 이어지도록
            try:
                save_directives_user(uid, merged_dirs, meta2)
            except Exception:
                pass

            # 버전은 directives+signals+persona 묶음으로 관리하여 불필요한 재컴파일 방지
            ver_obj = {
                "d": merged_dirs,
                "s": {
                    k: merged_sig.get(k)
                    for k in ("language", "topics", "style", "meta", "affect", "mobile")
                    if k in merged_sig
                },
                "p": {"bigfive": (merged_per.get("bigfive") or {})},
            }
            new_ver = version_of(ver_obj)
            cur_prompt, cur_ver = get_compiled(sid)
            if cur_ver != new_ver:
                from backend.directives.capabilities import build_capabilities_card

                base = compile_prompt_from_json(merged_dirs, merged_sig, merged_per)
                # Static Bot Profile 로드 + Dynamic Persona 게이팅(일 1회 또는 이벤트 최소치)
                static_bp = load_static_bot_profile()
                try:
                    import time as _time

                    now_s = int(_time.time())
                    last_s = int(_ENQ_AT.get(f"bp:{sid}", 0))
                    # 최근 7일 bot_event 건수(간략 추정)
                    try:
                        prof_coll, log_coll = ensure_collections()
                        now_ns = int(_time.time_ns())
                        ago_ns = now_ns - int(7 * 86400 * 1e9)
                        expr = f"user_id == '{sid}' and type == 'bot_event' and created_at >= {ago_ns}"
                        rows = log_coll.query(
                            expr=expr, output_fields=["id"], limit=10000
                        )
                        evt_cnt = len(rows or [])
                    except Exception:
                        evt_cnt = 0
                    from backend.config import get_settings as _gs3

                    _s3 = _gs3()
                    min_events = int(getattr(_s3, "DYN_BP_MIN_EVENTS", 10))
                    min_interval = int(getattr(_s3, "DYN_BP_MIN_INTERVAL_SEC", 86400))
                    use_dynamic = ((now_s - last_s) >= min_interval) or (
                        evt_cnt >= min_events
                    )
                except Exception:
                    use_dynamic = False

                dynamic_bp = BotProfile()
                if use_dynamic:
                    # 스키마 정합: LLM Signals는 communication_style/emotional_intensity,
                    # Directives는 formality/verbosity 등을 사용하므로, 우선순위로 병합하여 반영
                    style_str = str(
                        merged_dirs.get("formality")
                        or merged_sig.get("communication_style")
                        or ""
                    )
                    v = merged_dirs.get("verbosity", None)
                    constraints = {"verbosity": v} if v is not None else {}
                    dynamic_bp = BotProfile(
                        persona="",
                        style=style_str,
                        abilities=[],
                        constraints=constraints,
                    )
                    _ENQ_AT[f"bp:{sid}"] = now_s
                merged_bp = merge_bot_profiles(static_bp, dynamic_bp)
                bp_text = (
                    "[BotProfile:Static+Dynamic]\n"
                    + f"persona: {merged_bp.persona}\nstyle: {merged_bp.style}\n"
                    + (
                        "abilities: " + ", ".join(merged_bp.abilities) + "\n"
                        if merged_bp.abilities
                        else ""
                    )
                    + (
                        "constraints: "
                        + ", ".join(
                            f"{k}:{v}" for k, v in merged_bp.constraints.items()
                        )
                        if merged_bp.constraints
                        else ""
                    )
                )
                # 지시문 최상단에 Bot Profile을 고정 주입(최종 컴파일 캐시)
                compiled = bp_text + "\n\n" + base + "\n\n" + build_capabilities_card()
                set_compiled(sid, compiled, new_ver)

        except Exception as e:
            print(f"[dir:worker-{i}] error session={sid} err={repr(e)}")
        finally:
            _Q.task_done()
            took = (time.time() - t0) * 1000.0
            print(f"[dir:worker-{i}] done session={sid} took_ms={took:.1f}")


# ------------------------------------------------------------------------------
# 퍼블릭 API
# ------------------------------------------------------------------------------
async def ensure_directive_workers():
    """워커 기동(한 번만)."""
    global _started
    if _started:
        return
    _started = True
    for i in range(WORKERS):
        asyncio.create_task(_worker(i))


def schedule_directive_update(session_id: str, force: bool = False):
    """
    배치 업데이트 예약.
    - 턴 디바운스(DIR_DEBOUNCE_TURNS)
    - 인메모리 가드(ENQ_GUARD_S) 또는 Redis 락(DIR_USE_REDIS_LOCK)
    """
    # 1) 턴 디바운스: N턴 누적 전이면 스킵
    turns = _TURNS_SINCE.get(session_id, 0) + 1
    _TURNS_SINCE[session_id] = turns
    if not force and turns < DIR_DEBOUNCE_TURNS:
        return
    _TURNS_SINCE[session_id] = 0

    # 2) 가드: 멀티프로세스 락 또는 인메모리 쿨다운
    if USE_REDIS_LOCK and _get_lock_r() is not None:
        # Redis 분산 락 (NX, TTL)
        if not force and not _get_lock_r().set(
            f"dir:lock:{session_id}", "1", nx=True, ex=DIR_LOCK_EX_S
        ):
            return
    else:
        # 인메모리 가드(프로세스 단위)
        now = int(time.time())
        last = _ENQ_AT.get(session_id, 0)
        if not force and now - last < ENQ_GUARD_S:
            return
        _ENQ_AT[session_id] = now

    # 3) 큐에 넣기
    try:
        _Q.put_nowait(session_id)
        print(f"[dir:q] enqueued session={session_id} qsize={_Q.qsize()}")
    except asyncio.QueueFull:
        print(f"[dir:q] queue full -> drop {session_id}")
