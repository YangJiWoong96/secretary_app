# # 큐/워커 + app.py 훅용 API

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))  # C:\My_Business\.env 자동 탐색/로드

import os
import time
import asyncio
from typing import Tuple

from langchain_community.chat_message_histories.redis import RedisChatMessageHistory

from .agent import extract_report
from .policy import apply_update_policy, ema_merge_signals, conservative_persona_merge
from .store import (
    load_directives,
    save_directives,
    save_signals,
    load_signals,
    save_persona,
    load_persona,
    set_compiled,
    get_compiled,
    version_of,
)
from .compiler import compile_prompt_from_json
from .signals_ingest import summarize_ingest_profile
from .validator import validate_directives

# (옵션) 멀티프로세스 락용 redis
try:
    import redis as _redis
except Exception:
    _redis = None

# ------------------------------------------------------------------------------
# 환경 변수
# ------------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

QUEUE_MAX = int(os.getenv("DIR_QUEUE_MAX", "128"))
WORKERS = int(os.getenv("DIR_WORKERS", "2"))

# 배치 디바운스: 턴이 몇 개 누적되어야 큐잉할지
DIR_DEBOUNCE_TURNS = int(os.getenv("DIR_DEBOUNCE_TURNS", "5"))

# 인메모리 가드(중복 큐잉 억제)
ENQ_GUARD_S = int(os.getenv("DIR_ENQ_GUARD_S", "10"))

# Redis 락 사용 여부(멀티프로세스/멀티인스턴스)
USE_REDIS_LOCK = os.getenv("DIR_USE_REDIS_LOCK", "0") == "1"
DIR_LOCK_EX_S = int(os.getenv("DIR_LOCK_EX_S", "10"))  # 락 TTL

# ------------------------------------------------------------------------------
# 내부 상태
# ------------------------------------------------------------------------------
_started = False
_Q: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAX)

# 세션별 누적 턴 카운터(디바운스용)
_TURNS_SINCE: dict[str, int] = {}

# 인메모리 최근 큐잉 시각(프로세스 단위 가드)
_ENQ_AT: dict[str, int] = {}

# Redis 클라이언트(락 전용)
_r = _redis.Redis.from_url(REDIS_URL) if (USE_REDIS_LOCK and _redis) else None


# ------------------------------------------------------------------------------
# 워커
# ------------------------------------------------------------------------------
async def _worker(i: int):
    while True:
        sid = await _Q.get()
        t0 = time.time()
        try:
            hist = RedisChatMessageHistory(session_id=sid, url=REDIS_URL)
            msgs = hist.messages
            if not msgs:
                continue

            # 1) 리포트 추출: directives + signals + persona
            report = await extract_report(msgs)

            # 2) 기존 지시문/신호/페르소나 로드 및 병합
            prev_dirs, meta = load_directives(sid)
            prev_sig = load_signals(sid)
            prev_per = load_persona(sid)

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

            merged_per = conservative_persona_merge(
                prev_per,
                report.get("persona") or {},
                float(report.get("confidence", 0.0)),
            )

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
            save_persona(sid, merged_per)

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
                compiled = compile_prompt_from_json(merged_dirs, merged_sig, merged_per)
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
    if USE_REDIS_LOCK and _r is not None:
        # Redis 분산 락 (NX, TTL)
        if not force and not _r.set(
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
