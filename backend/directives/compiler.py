# # JSON → system 주입용 미니 프롬프트
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from backend.config import get_settings
from backend.utils.logger import log_event

from .schema import Directives

# 시스템 프롬프트는 "JSON 지시문 + 짧은 고정 헤더"만 넣습니다.
HEADER = (
    "너는 한국어 사용자 전용 비서다. 아래 JSON 지시문을 모든 주제에서 일관되게 준수하라. "
    "사용자가 명시적으로 변경을 요구하면 1회 확인 후 해당 턴에 한해 임시로 조정하되, 기본 지시문은 유지하라. "
    "안전/정책(Guard) 규칙은 항상 모든 선호보다 우선한다. 추측이나 환각을 하지 말고, 불명확하면 간결히 되물어 확인하라."
)


def _compact_signals(sig: dict) -> dict:
    """
    시스템 프롬프트에 넣을 만큼만 축약. 토큰 사용을 최소화한다.
    - language: positive/negative/jondaemal만 유지(소수점 1~2자리)
    - topics: 상위 3개
    - style/meta/affect: 핵심 1~2개만
    - mobile: prime_time/avg_calendar_events_per_day만
    """
    if not sig:
        return {}
    out = {}
    lang = sig.get("language") or {}
    if lang:
        out["language"] = {
            "positive": round(float(lang.get("positive_ratio", 0.0)), 2),
            "negative": round(float(lang.get("negative_ratio", 0.0)), 2),
            "jondaemal": round(float(lang.get("jondaemal_ratio", 0.0)), 2),
        }
    topics = sig.get("topics") or []
    if topics:
        out["topics"] = topics[:3]
    style = sig.get("style") or {}
    if style:
        out["style"] = {
            "prefers_short": style.get("prefers_short", 0.0),
            "emotional_intensity": style.get("emotional_intensity", 0.0),
        }
    meta = sig.get("meta") or {}
    if meta:
        out["meta"] = {"repeat_topic_ratio": meta.get("repeat_topic_ratio", 0.0)}
    affect = sig.get("affect") or {}
    if affect:
        out["affect"] = {
            "positive": affect.get("positive", 0.0),
            "negative": affect.get("negative", 0.0),
        }
    mobile = sig.get("mobile") or {}
    if mobile:
        out["mobile"] = {
            "prime_time": mobile.get("prime_time"),
            "avg_calendar_events_per_day": mobile.get("avg_calendar_events_per_day"),
        }
    return out


def compile_prompt_from_json(
    d: Directives, signals: dict | None = None, persona: dict | None = None
) -> str:
    # 꼭 필요한 키만 유지(토큰 절약)
    allow = [
        "tone",
        "formality",
        "emotion",
        "style",
        "verbosity",
        "emojis",
        "markdown",
        "language",
        "taboo_phrases",
        "do",
        "dont",
    ]
    slim = {
        k: v for k, v in (d or {}).items() if k in allow and v not in (None, [], "")
    }
    body = {"directives": slim}
    sig_comp = _compact_signals(signals or {})
    if sig_comp:
        body["signals"] = sig_comp
    if persona:
        # persona는 프롬프트 오염 방지를 위해 bigfive만 축약 반영
        bf = (persona.get("bigfive") or {}) if isinstance(persona, dict) else {}
        if bf:
            body["persona"] = {"bigfive": bf}
    return HEADER + "\n\n" + json.dumps(body, ensure_ascii=False, separators=(",", ":"))


# ─────────────────────────────────────────────────────────────
# 세션2: Directives-RAG 통합 컴파일러
# ─────────────────────────────────────────────────────────────


@dataclass
class ProfilePriority:
    explicit: int = 10
    directives: int = 7
    inferred: int = 5
    default: int = 1


# ─────────────────────────────────────────────────────────────
# 우선순위 Tie-break 규칙 및 Directives ↔ RAG 키 정규화
# ─────────────────────────────────────────────────────────────

# 우선순위 가중치 표
PRIO: dict[str, int] = {"explicit": 10, "directives": 7, "inferred": 5, "default": 1}

# 정책적 우선순위(동률 시) 키별 소스 우선순위 고정
_POLICY_BY_KEY: dict[str, list[str]] = {
    # 예: 존댓말/반말 관련 우선순위는 explicit > directives > inferred > default
    "communication_formality": ["explicit", "directives", "inferred", "default"],
}


def _better(a: dict, b: dict) -> dict:
    """
    동률 해소 로직:
    1) PRIO 높은 소스 우선
    2) ts 최신
    3) confidence 큰 값
    4) 정책 맵(있으면)에서 더 앞선 소스
    """
    asrc = str(a.get("source") or "inferred")
    bsrc = str(b.get("source") or "inferred")
    if PRIO.get(asrc, 0) != PRIO.get(bsrc, 0):
        return a if PRIO.get(asrc, 0) > PRIO.get(bsrc, 0) else b
    ats = int(a.get("ts", 0) or 0)
    bts = int(b.get("ts", 0) or 0)
    if ats != bts:
        return a if ats > bts else b
    ac = float(a.get("confidence", 0) or 0)
    bc = float(b.get("confidence", 0) or 0)
    if ac != bc:
        return a if ac > bc else b
    key = str(a.get("norm_key") or b.get("norm_key") or "").strip()
    if key:
        order = {
            s: i
            for i, s in enumerate(
                _POLICY_BY_KEY.get(
                    key, ["explicit", "directives", "inferred", "default"]
                )
            )
        }
        if order.get(asrc, 99) != order.get(bsrc, 99):
            return a if order.get(asrc, 99) < order.get(bsrc, 99) else b
    # 완전 동률이면 a 유지
    return a


# Directives 키 → 정규화 키 매핑 테이블
NORM_MAP: dict[str, str] = {
    "verbosity": "response_length",
    "formality": "communication_formality",
    "style": "communication_style",
}


def norm_key_from_directives(key: str) -> str:
    k = (key or "").strip()
    mapped = NORM_MAP.get(k, k)
    return mapped.lower().replace(".", "_")


def _merge_with_priority(
    rag_preferences: List[Dict[str, Any]],
    rag_traits: List[Dict[str, Any]],
    directives: Dict[str, Any],
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    """
    우선순위 기반 사용자 프로필 병합

    - explicit > directives > inferred > default
    - 동일 norm_key 충돌 시 높은 우선순위 채택
    """
    pr = ProfilePriority()
    items_map: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    # 1) RAG preferences
    for pref in rag_preferences or []:
        nk = str(pref.get("norm_key") or "").strip()
        if not nk:
            continue
        source = str(pref.get("source") or "inferred").strip()
        prio = pr.explicit if source == "explicit" else pr.inferred
        if nk not in items_map:
            items_map[nk] = (prio, pref)
        else:
            cur_prio, cur_item = items_map[nk]
            if prio > cur_prio:
                items_map[nk] = (prio, pref)
            elif prio == cur_prio:
                items_map[nk] = (prio, _better(pref, cur_item))

    # 2) Directives (중간 우선순위)
    for key, value in (directives or {}).items():
        nk = norm_key_from_directives(key)
        prio = pr.directives
        cand = {
            "key_path": key,
            "norm_key": nk,
            "value": value,
            "source": "directives",
            "confidence": 0.8,
        }
        if nk not in items_map:
            items_map[nk] = (prio, cand)
        else:
            cur_prio, cur_item = items_map[nk]
            if prio > cur_prio:
                items_map[nk] = (prio, cand)
            elif prio == cur_prio:
                items_map[nk] = (prio, _better(cand, cur_item))

    # 결과 정렬
    sorted_items = sorted(items_map.values(), key=lambda x: -x[0])
    preferences = [item for _, item in sorted_items]
    traits = rag_traits or []
    return {"preferences": preferences, "traits": traits}


def _apply_token_budget(
    *,
    base_parts: list[str],
    overlay_sections: dict[str, str],
    budget_tokens: int = 300,
) -> tuple[list[str], dict[str, str]]:
    """
    토큰 예산 관리. 대략 문자수/4를 토큰으로 가정하여 예산을 초과하면 순서대로 드랍.
    1) hints 제거 → 2) preferences(낮은 confidence/비-explicit) 축소 → 3) topics 상위 2개만
    """

    # 간이 토큰 계산
    def _tok(s: str) -> int:
        return max(1, len(s) // 4)

    def _joined_len(parts: list[str], ov: dict[str, str]) -> int:
        total = "\n\n".join(parts + [v for v in ov.values() if v])
        return _tok(total)

    # 1) hints 우선 제거
    if _joined_len(base_parts, overlay_sections) > budget_tokens:
        overlay_sections["hints"] = ""

    # 2) preferences 줄이기: base_parts에서 [User Preferences] 섹션을 줄여 본다
    if _joined_len(base_parts, overlay_sections) > budget_tokens:
        new_base: list[str] = []
        for block in base_parts:
            if block.startswith("[User Preferences]"):
                lines = block.splitlines()
                header, rest = lines[0], lines[1:]
                # explicit 우선 유지, 나머지 최대 6개로 제한
                explicit_lines = [ln for ln in rest if ln.startswith("★-")]
                non_explicit = [ln for ln in rest if not ln.startswith("★-")]
                trimmed = (
                    explicit_lines + non_explicit[: max(0, 6 - len(explicit_lines))]
                )
                new_block = "\n".join([header] + trimmed)
                new_base.append(new_block)
            else:
                new_base.append(block)
        base_parts = new_base

    # 3) topics를 상위 2개만 유지
    if _joined_len(
        base_parts, overlay_sections
    ) > budget_tokens and overlay_sections.get("style"):
        st_lines = overlay_sections["style"].splitlines()
        new_lines: list[str] = []
        for ln in st_lines:
            if ln.startswith("- Recent Topics:"):
                # 포맷: "- Recent Topics: A(0.12), B(0.11), C(0.08)"
                try:
                    head, tail = ln.split(":", 1)
                    items = [x.strip() for x in tail.split(",") if x.strip()]
                    items = items[:2]
                    new_lines.append(f"- Recent Topics: {', '.join(items)}")
                except Exception:
                    new_lines.append(ln)
            else:
                new_lines.append(ln)
        overlay_sections["style"] = "\n".join(new_lines)

    return base_parts, overlay_sections


async def compile_unified_prompt_split(
    user_id: str,
    session_id: str,
    user_query: str,
    top_k: int = 5,
    has_evidence: bool | None = None,
) -> tuple[str, str, str]:
    """
    통합 시스템 프롬프트 컴파일러(분리 버전)
    - base: Header + BotGuard + Preferences + Persona (버전 캐시 대상)
    - overlay: Communication Style + Hints (쿼리 민감, 쿼리 해시 캐시)
    Returns: (base_prompt, overlay_prompt, base_version)
    """
    from backend.directives.store import (
        load_directives,
        load_signals,
        version_of,
    )
    from backend.directives.store_user_ext import (
        load_persona_user,
        load_directives_user,
    )
    from backend.rag.profile_rag import get_profile_rag

    rag = get_profile_rag()

    # 조건부 로드 플래그: 환경변수와 호출자 신호(has_evidence) 모두 만족해야 활성화
    on_demand_enabled = (
        str(os.getenv("PROFILE_TIER_ON_DEMAND", "1")).lower() in ("1", "true", "yes")
        and has_evidence is not None
    )

    # 1) 프로필 계층 조회 (Guard/Core/Dynamic) — 턴 캐시 적용
    try:
        from backend.directives.profile_cache_manager import get_profile_items_cached
        from backend.rag.profile_ids import bot_user_id_for

        # 사용자 스코프
        guard_items = await get_profile_items_cached(user_id, "guard")
        # Core: 증거 없을 때만 로드 (온디맨드가 활성화된 경우)
        core_items = []
        if not on_demand_enabled or not has_evidence:
            core_items = await get_profile_items_cached(user_id, "core")
        # Dynamic: 증거 있을 때만 로드 (온디맨드가 활성화된 경우)
        dynamic_items_user = []
        if not on_demand_enabled or has_evidence:
            dynamic_items_user = await get_profile_items_cached(
                user_id, "dynamic", user_query, top_k=5
            )

        # 봇 스코프(전역)
        _bot_uid = bot_user_id_for(user_id)
        bot_guard_items = await get_profile_items_cached(_bot_uid, "guard")
        bot_dynamic_items = []
        if not on_demand_enabled or has_evidence:
            bot_dynamic_items = await get_profile_items_cached(
                _bot_uid, "dynamic", user_query, top_k=5
            )

        # Guard: 봇 전역 가드룰 우선 포함
        guard_items = (bot_guard_items or []) + (guard_items or [])
        # Dynamic: 사용자/봇 힌트 병합
        dynamic_items = (dynamic_items_user or []) + (bot_dynamic_items or [])
    except Exception:
        # 폴백: 기존 2계층 경로 유지
        user_prof = await rag.query_relevant_profile(
            user_id=user_id, user_input=user_query, top_k=top_k
        )
        bot_prof = await rag.query_bot_profile(user_id=user_id, user_input=user_query)
        guard_items = [
            {"key_path": k, "value": v, "source": "explicit", "tier": "guard"}
            for k, v in (bot_prof.get("guard") or {}).items()
        ]
        # Core/Dynamic 조건부 반영
        core_items = []
        dynamic_items = []
        if not on_demand_enabled or not has_evidence:
            core_items = user_prof.get("preferences", [])
        if not on_demand_enabled or has_evidence:
            dynamic_items = bot_prof.get("hints", [])

    # 2) Directives/Signals/Persona 로드
    # 사용자 범위 룰북을 우선 사용하고, 세션 범위 변경이 있으면 해당 세션에서만 얕게 오버레이
    dirs_user, _meta_user = load_directives_user(user_id)
    dirs_sess, _meta_sess = load_directives(session_id)
    directives = {}
    try:
        directives = {**(dirs_user or {}), **(dirs_sess or {})}
    except Exception:
        directives = dirs_sess or dirs_user or {}
    signals = load_signals(session_id)
    # Persona는 사용자 범위로 관리
    persona = load_persona_user(user_id) or {}

    # 3) 병합: Core(장기 선호) + Directives → Preferences
    merged = _merge_with_priority(
        rag_preferences=core_items or [],
        rag_traits=[],
        directives=directives or {},
        signals=signals or {},
    )

    # 4) base/overlay 구성 (세션3: Guard/Core/Dynamic)
    base_parts: list[str] = []
    base_parts.append(
        "You are an AI assistant for a Korean user. Follow these preferences consistently unless the user explicitly asks to change."
    )
    # BotProfile: 정적 프로필을 상단에 고정 주입
    try:
        from .bot_profile import load_static_bot_profile

        _bp = load_static_bot_profile()
        _bp_text = "[BotProfile]\n" + f"persona: {_bp.persona}\nstyle: {_bp.style}"
        base_parts.insert(1, _bp_text)
    except Exception:
        pass
    # Guard 우선 원칙
    base_parts.append(
        "[Policy] Guard rules override user preferences. Never violate Guard."
    )

    # Tier 1: Guard (base, 불변 규칙)
    if guard_items:
        glines = ["[Tier 1: Guard - Immutable Rules]"]
        for it in guard_items:
            glines.append(f"⛔ {it.get('key_path')}: {it.get('value')}")
        base_parts.append("\n".join(glines))

    # Tier 2: Core (base, 장기 선호)
    prefs = merged.get("preferences") or []
    if prefs:
        plines = ["[Tier 2: Core - Long-term Preferences]"]
        for it in prefs:
            tag = "★" if (str(it.get("source")) == "explicit") else ""
            plines.append(f"{tag}- {it.get('key_path')}: {it.get('value')}")
        base_parts.append("\n".join(plines))

    # Persona (base)
    bf = (persona.get("bigfive") or {}) if isinstance(persona, dict) else {}
    if bf:
        base_parts.append(
            f"[Persona] Openness: {float(bf.get('openness',0.5)):.2f}, Conscientiousness: {float(bf.get('conscientiousness',0.5)):.2f}, Extraversion: {float(bf.get('extraversion',0.5)):.2f}"
        )

    # overlay: Communication Style + Tier 3 Dynamic (현재 맥락)
    overlay_sections: dict[str, str] = {"style": "", "hints": ""}
    if signals:
        try:
            topics = signals.get("topics") or []
            top3 = ", ".join(
                f"{t.get('label')}({float(t.get('weight') or 0):.2f})"
                for t in topics[:3]
            )
        except Exception:
            top3 = ""
        style_line = signals.get("communication_style") or "mixed"
        emo_int = float(signals.get("emotional_intensity", 0.0) or 0.0)
        slines = ["[Communication Style]"]
        slines.append(f"- Style: {style_line}")
        if top3:
            slines.append(f"- Recent Topics: {top3}")
        slines.append(f"- Emotional Intensity: {emo_int:.2f}")
        overlay_sections["style"] = "\n".join(slines)

    if dynamic_items:
        hlines = ["[Tier 3: Dynamic - Current Context]"]
        for it in dynamic_items[:5]:
            val = it.get("value", "")
            hlines.append(f"🔄 {it.get('key_path')}: {val}")
        overlay_sections["hints"] = "\n".join(hlines)

    # (신규) 행동 신호 힌트: PreferenceScoreboard 상위 항목 2~3개만 표시
    try:
        from backend.personalization.preference_scoreboard import (
            PreferenceScoreboard as _SB,
        )

        sb = _SB(get_settings().REDIS_URL)
        top_items_all = sb.top(user_id, top_n=8, include_pending=True)
        # behavior.* 네임스페이스만 필터링
        top_items = [
            (nk, e)
            for nk, e in (top_items_all or [])
            if str(nk).startswith("behavior.")
        ]
        if top_items:
            # Behavior Slots 섹션(문서 요구사항 반영)
            blines = ["[Behavior Slots]"]
            for nk, entry in top_items[:3]:
                sc = float((entry or {}).get("score", 0.0))
                st = str((entry or {}).get("status", ""))
                blines.append(f"- {nk} (status={st}, score={sc:.2f})")
            # overlay에 병합
            prev = overlay_sections.get("hints", "")
            overlay_sections["hints"] = (
                prev + ("\n" if prev else "") + "\n".join(blines)
            ).strip()
    except Exception:
        pass

    # 5) 버전 해시(베이스 전용 요소로 구성)
    version_obj = {
        "guard": guard_items,
        "core": prefs,
        "dynamic_keys": [it.get("key_path") for it in (dynamic_items or [])],
        "directives": directives,
        "persona": {"bigfive": bf} if bf else {},
    }
    base_version = version_of(version_obj)

    # 6) 토큰 예산 적용 + (세션6) 계층별 압축 + 2계층 압축 캐시/락
    # 6-1) 1차 간이 예산 적용으로 과도한 섹션 드랍(힌트/토픽 축소)
    base_parts, overlay_sections = _apply_token_budget(
        base_parts=base_parts, overlay_sections=overlay_sections, budget_tokens=300
    )

    # 플래그로 압축 기능 온/오프 (기본 on)
    CTX_COMPRESS_ENABLED = bool(get_settings().CTX_COMPRESS_ENABLED)

    if CTX_COMPRESS_ENABLED:
        try:
            # 캐시 조회 및 단일비행 락
            from backend.context.compressor import compress_by_tier_async
            from backend.directives.store import (
                acquire_compress_lock,
                get_compressed_base,
                get_compressed_overlay,
                release_compress_lock,
                set_compressed_base,
                set_compressed_overlay,
            )
            from backend.utils.logger import log_event as _log

            settings_local = get_settings()
            model_name = settings_local.LLM_MODEL
            token_budget = 300

            # Guard/Core/Dynamic 텍스트 (구조 보존: key_path: value)
            guard_text = "\n".join(
                [
                    f"⛔ {it.get('key_path')}: {it.get('value')}"
                    for it in (guard_items or [])
                ]
            )
            core_text = "\n".join(
                [f"{it.get('key_path')}: {it.get('value')}" for it in (prefs or [])]
            )
            dynamic_text = overlay_sections.get("hints", "")

            # 2계층 캐시 확인
            cached_base = get_compressed_base(
                user_id, base_version, model_name, token_budget
            )
            cached_overlay = get_compressed_overlay(
                session_id, user_query, model_name, token_budget
            )

            if cached_base and cached_overlay:
                try:
                    _log("compress_cache_hits", {"base": True, "overlay": True})
                except Exception:
                    pass
                # base_parts 재구성
                base_parts = [
                    base_parts[0],
                    base_parts[1],
                    cached_base,
                ]
                overlay_sections["hints"] = cached_overlay
            else:
                # 단일비행 락
                lock_key, ok = acquire_compress_lock(
                    session_id, user_query, model_name, token_budget, ex_sec=5
                )
                if not ok and (cached_base or cached_overlay):
                    # 부분 캐시라도 있으면 사용
                    try:
                        _log(
                            "compress_lock_denied_use_partial_cache",
                            {
                                "base_cached": bool(cached_base),
                                "overlay_cached": bool(cached_overlay),
                            },
                        )
                    except Exception:
                        pass
                    if cached_base:
                        base_parts = [base_parts[0], base_parts[1], cached_base]
                    if cached_overlay:
                        overlay_sections["hints"] = cached_overlay
                else:
                    try:
                        compressed = await compress_by_tier_async(
                            guard_text=guard_text,
                            core_text=core_text,
                            dynamic_text=dynamic_text,
                            total_budget=token_budget,
                        )

                        base_comp = []
                        if compressed.get("guard"):
                            base_comp.append(
                                "[Tier 1: Guard - Immutable Rules]\n"
                                + compressed["guard"]
                            )
                        if compressed.get("core"):
                            base_comp.append(
                                "[Tier 2: Core - Long-term Preferences]\n"
                                + compressed["core"]
                            )

                        base_joined = "\n\n".join(base_comp)
                        dyn_joined = (
                            "[Tier 3: Dynamic - Current Context]\n"
                            + compressed.get("dynamic", "")
                            if compressed.get("dynamic")
                            else ""
                        )

                        # 캐시 저장
                        set_compressed_base(
                            user_id, base_version, model_name, token_budget, base_joined
                        )
                        set_compressed_overlay(
                            session_id, user_query, model_name, token_budget, dyn_joined
                        )

                        # 반영
                        base_parts = [base_parts[0], base_parts[1]] + (
                            [base_joined] if base_joined else []
                        )
                        overlay_sections["hints"] = dyn_joined

                        try:
                            _log(
                                "compress_compiled",
                                {
                                    "base_tokens": max(1, len(base_joined) // 4),
                                    "dyn_tokens": max(0, len(dyn_joined) // 4),
                                    "budget": token_budget,
                                },
                            )
                        except Exception:
                            pass
                    finally:
                        if ok:
                            release_compress_lock(lock_key)
        except Exception:
            # 실패 시 기존 경로 유지(간이 토큰 예산만 적용)
            pass

    base_prompt = "\n\n".join([p for p in base_parts if p])
    overlay_prompt = "\n\n".join(
        [v for v in (overlay_sections.get("style"), overlay_sections.get("hints")) if v]
    )

    # 로깅
    try:
        log_event(
            "unified_prompt_compiled_split",
            {
                "session_id": session_id,
                "user_id": user_id,
                "version": base_version,
                "base_len": len(base_prompt or ""),
                "overlay_len": len(overlay_prompt or ""),
            },
        )
        # profile.loaded 텔레메트리 (계층별 로드 수, 증거/플래그)
        log_event(
            "profile.loaded",
            {
                "guard_count": len(guard_items or []),
                "core_count": len((merged.get("preferences") or [])),
                "dynamic_count": len(dynamic_items or []),
                "has_evidence": has_evidence,
                "on_demand_enabled": on_demand_enabled,
            },
        )
    except Exception:
        pass

    return base_prompt, overlay_prompt, base_version


async def compile_unified_prompt(
    user_id: str,
    session_id: str,
    user_query: str,
    top_k: int = 5,
) -> tuple[str, str]:
    """
    통합 시스템 프롬프트 컴파일러
    - RAG(Profile) + Directives + Signals + Persona → System Prompt
    - explicit > directives > inferred 병합 정책 적용
    - 버전 해시를 함께 반환
    """
    base_prompt, overlay_prompt, base_version = await compile_unified_prompt_split(
        user_id=user_id, session_id=session_id, user_query=user_query, top_k=top_k
    )
    final_prompt = "\n\n".join([p for p in (base_prompt, overlay_prompt) if p])
    return final_prompt, base_version
