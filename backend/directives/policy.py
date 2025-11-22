# # 완만 업데이트(하이시스/쿨다운/EMA)
import time
from copy import deepcopy
from typing import Dict, Tuple

from backend.config import get_settings

from .schema import Directives, DirectiveSnapshot

# 업데이트 정책 파라미터 (완만/자연스럽게)
_s = get_settings()
CONF_THRESH = float(getattr(_s, "DIR_CONF_THRESH", 0.55))  # 바꿀 확신도 최소
COOLDOWN_S = int(getattr(_s, "DIR_COOLDOWN_S", 3600))  # 필드별 쿨다운(초)
EMA_ALPHA = float(getattr(_s, "DIR_EMA_ALPHA", 0.35))  # verbosity 등 숫자 EMA
MAX_CHANGES_PER_UPDATE = int(
    getattr(_s, "DIR_MAX_CHANGES", 3)
)  # 한 번에 바뀔 필드 수 제한


# 메타: 마지막 변경 시각 기록
def _now() -> int:
    return int(time.time())


def apply_update_policy(
    prev: Directives, cand: DirectiveSnapshot, meta: Dict
) -> Tuple[Directives, Dict]:
    """
    prev: 기존 지시문
    cand: 에이전트가 뽑은 후보(확신도 포함)
    meta: {"last_changed": {field: epoch_sec}}
    """
    if not cand or cand.get("confidence", 0.0) < CONF_THRESH:
        return prev, meta  # 확신 낮으면 무시

    new = deepcopy(prev)
    last = meta.get("last_changed", {})
    changed = 0
    now = _now()

    def try_set(field, val):
        nonlocal changed
        if val is None:
            return
        # 쿨다운 & 변경예산
        if changed >= MAX_CHANGES_PER_UPDATE:
            return
        if now - int(last.get(field, 0)) < COOLDOWN_S:
            return
        if prev.get(field) == val:
            return
        new[field] = val
        last[field] = now
        changed += 1

    d = cand.get("directives") or {}
    # 범주형: 톤/말투/정서 등은 히스테리시스(쿨다운+예산)
    try_set("tone", d.get("tone"))
    try_set("formality", d.get("formality"))
    try_set("emotion", d.get("emotion"))
    # 리스트는 상위 2~3개만 유지하여 과잉 증가 방지
    style = d.get("style")
    if style:
        try_set("style", list(dict.fromkeys(style))[:3])
    taboo = d.get("taboo_phrases")
    if taboo:
        try_set("taboo_phrases", list(dict.fromkeys(taboo))[:5])
    do = d.get("do")
    dont = d.get("dont")
    if do:
        try_set("do", list(dict.fromkeys(do))[:3])
    if dont:
        try_set("dont", list(dict.fromkeys(dont))[:3])
    # 부울
    if "emojis" in d:
        try_set("emojis", bool(d["emojis"]))
    if "markdown" in d:
        try_set("markdown", bool(d["markdown"]))
    # 언어는 고정 ko로만 허용
    if d.get("language") == "ko" and prev.get("language") != "ko":
        try_set("language", "ko")

    # 숫자(verbosity)는 EMA로 부드럽게
    if "verbosity" in d and isinstance(d["verbosity"], int):
        old_v = int(prev.get("verbosity", 2))
        tgt_v = int(d["verbosity"])
        ema_v = round((1 - EMA_ALPHA) * old_v + EMA_ALPHA * tgt_v)
        if ema_v != old_v and now - int(last.get("verbosity", 0)) >= COOLDOWN_S:
            new["verbosity"] = ema_v
            last["verbosity"] = now
            changed += 1

    meta["last_changed"] = last
    return new, meta


# ---------------- Signals/Persona 보수적 병합 ----------------


def ema_merge_signals(prev: Dict, cand: Dict, alpha: float = 0.3) -> Dict:
    """
    수치 신호는 EMA, 리스트/카테고리는 상위 소수만 유지하여 급격한 변화를 방지.
    """
    if not cand:
        return prev or {}
    out = dict(prev or {})

    # language/meta/affect/style 내부의 수치 항목은 EMA
    def _ema_block(block_name: str, keys=None):
        cand_b = cand.get(block_name) or {}
        prev_b = out.get(block_name) or {}
        merged = dict(prev_b)
        for k, v in cand_b.items():
            if isinstance(v, (int, float)):
                old = float(prev_b.get(k, v))
                merged[k] = round((1 - alpha) * old + alpha * float(v), 3)
            else:
                merged[k] = v
        out[block_name] = merged

    _ema_block("language")
    _ema_block("meta")
    _ema_block("affect")
    _ema_block("style")

    # topics: 상위 4개만 유지, weight는 EMA
    cand_topics = cand.get("topics") or []
    prev_topics = {
        t.get("label"): t.get("weight", 0.0) for t in (out.get("topics") or [])
    }
    merged_topics: Dict[str, float] = dict(prev_topics)
    # 단발성/저가중치는 보수 반영(감쇠)
    try:
        decay_factor = float(getattr(_s, "SIG_DECAY_FACTOR", 0.5))
    except Exception:
        decay_factor = 0.5
    for t in cand_topics:
        lab = t.get("label")
        w = float(t.get("weight", 0.0))
        old = float(merged_topics.get(lab, w))
        eff_alpha = alpha * (decay_factor if w < 0.1 else 1.0)
        merged_topics[lab] = round((1 - eff_alpha) * old + eff_alpha * w, 3)
    top = sorted(merged_topics.items(), key=lambda x: x[1], reverse=True)[:4]
    out["topics"] = [{"label": k, "weight": v} for k, v in top if v > 0]

    # mobile: 덮어쓰기(ingest가 이미 집계된 값)
    if cand.get("mobile"):
        out["mobile"] = cand["mobile"]

    return out


def conservative_persona_merge(
    prev: Dict, cand: Dict, conf: float, min_conf: float = 0.6, alpha: float = 0.2
) -> Dict:
    """
    BigFive는 EMA, MBTI는 확신 높을 때만 세팅/변경.
    """
    if not cand:
        return prev or {}
    out = dict(prev or {})

    bf_c = cand.get("bigfive") or {}
    bf_p = out.get("bigfive") or {}
    merged_bf = dict(bf_p)
    for k, v in bf_c.items():
        try:
            old = float(bf_p.get(k, v))
            merged_bf[k] = round((1 - alpha) * old + alpha * float(v), 3)
        except Exception:
            merged_bf[k] = v
    if merged_bf:
        out["bigfive"] = merged_bf

    mbti_c = cand.get("mbti")
    if conf >= min_conf and mbti_c:
        out["mbti"] = mbti_c
    # conf 낮으면 기존 유지
    return out
