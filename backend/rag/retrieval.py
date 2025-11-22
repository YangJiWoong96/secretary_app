# backend/rag/retrieval.py
import hashlib
import logging
import math
import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from backend.config import get_settings
from backend.utils.tracing import traceable
from backend.memory.summarizer import get_tokenizer
from backend.utils.datetime_utils import now_kst

from .config import METRIC
from .embeddings import embed_query_cached
from .milvus import ensure_collections
from .utils import hit_similarity
from backend.routing.router_context import user_for_session


def _parse_domain_from_text(text: str) -> str:
    """
    enriched_text 내 '출처: <url>' 라인을 파싱하여 도메인을 추출한다.
    - 기존 내부 중첩 함수와 동일한 로직을 모듈 수준으로 승격하여 재사용성을 높인다.
    - 동작/결과/에러 처리(실패 시 빈 문자열)는 기존과 동일하다.
    """
    try:
        for ln in (text or "").split("\n"):
            ln1 = ln.strip()
            if ln1.startswith("출처:"):
                url = ln1.split("출처:", 1)[1].strip()
                from urllib.parse import urlparse as _urlparse

                dom = (_urlparse(url).netloc or "").lower()
                if dom.startswith("www."):
                    dom = dom[4:]
                return dom
    except Exception:
        return ""
    return ""


def _hit_similarity(hit) -> float:
    # 하위 호환: 외부에서 import하는 기존 심볼 유지
    return hit_similarity(hit)


settings = get_settings()
RAG_THRESHOLD = float(getattr(settings, "RAG_THRESHOLD", 0.35))


# ===== 필수 보강 유틸 =====
def _evidence_budget(
    ctx_window: int,
    used_tokens: int,
    ratio: float = 0.35,
    headroom: int = 256,
) -> int:
    """
    증거 예산 = min(EVIDENCE_TOKEN_CAP, (ctx_window - used_tokens - headroom) * ratio)
    - ctx_window: 모델 컨텍스트 창 크기(토큰)
    - used_tokens: 이미 조립된 프롬프트/히스토리 토큰 수
    - ratio: 증거 채널 상대 비율(기본 35%)
    - headroom: 안전 여유 토큰
    """
    try:
        cap_env = int(settings.EVIDENCE_TOKEN_CAP)
    except Exception:
        cap_env = 1600
    remain = max(0, ctx_window - used_tokens - headroom)
    return max(0, min(cap_env, int(remain * max(0.0, min(1.0, ratio)))))


def _apply_time_decay(score: float, age_days: int, lam: float) -> float:
    """시간 감쇠 적용: score * exp(-lam * age_days)"""
    try:
        return float(score) * math.exp(-float(lam) * max(0, int(age_days)))
    except Exception:
        return float(score)


def _age_days_from_ymd(ymd: int) -> int:
    """YYYYMMDD 정수를 KST now 대비 일수로 변환(미제공 시 0)."""
    try:
        y = int(ymd // 10000)
        m = int((ymd % 10000) // 100)
        d = int(ymd % 100)
        end_dt = __import__("datetime").datetime(y, m, d)
        delta = now_kst().date() - end_dt.date()
        return max(0, int(delta.days))
    except Exception:
        return 0


def _dedup_by_doc(
    hits,
    key_fields: tuple[str, ...] = ("doc_id", "section_id", "domain", "title_norm"),
):
    """
    안정적 dedup 키: doc_id + section_id + domain + title_norm
    키가 없으면 텍스트 50자 해시로 폴백.
    """
    seen = set()
    out = []
    for h in hits:
        e = getattr(h, "entity", {}) or {}
        try:
            key = tuple(str(e.get(k, "")) for k in key_fields)
            # 폴백: 키가 모두 빈 값이면 텍스트 앞 50자 해시
            if not any(key):
                txt = str(e.get("text", ""))[:50]
                key = (hashlib.md5(txt.encode()).hexdigest(),)
        except Exception:
            txt = str(getattr(h, "entity", {}).get("text", ""))[:50]
            key = (hashlib.md5(txt.encode()).hexdigest(),)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _format_line(hit) -> str:
    """
    RAG 라인 표준화: [출처: 타입|도메인|YYYY-MM] 텍스트
    - 타입은 entity.type 또는 프로필/로그 추정
    - 도메인은 entity.domain 없으면 '-'
    - YYYY-MM은 date_ym 정수
    """
    e = getattr(hit, "entity", {}) or {}
    typ = str(
        e.get("type") or ("profile" if "name" in (e.get("text") or "") else "log")
    )
    dom = str(e.get("domain") or "-")
    yym = str(e.get("date_ym") or "")
    text = str(e.get("text") or "")
    header = f"[출처: {typ} | {dom} | {yym}] "
    return header + text


def _pack_evidence(
    profile_hits: List,
    log_hits: List,
    web_hits: List,
    budget: int,
    enc,
) -> tuple[str, int]:
    """
    채널 인터리브 Packing
    - 최소 쿼터: 프로필≥1 + 로그≥1 + 웹≥1 (존재할 경우)
    - 남는 예산은 interleave로 채움
    """
    buckets: List[str] = []
    picks: List = []

    if profile_hits:
        picks.append(profile_hits[0])
    if log_hits:
        picks.append(log_hits[0])
    if web_hits:
        picks.append(web_hits[0])

    rest = []
    if len(profile_hits) > 1:
        rest.extend(profile_hits[1:])
    if len(log_hits) > 1:
        rest.extend(log_hits[1:])
    if len(web_hits) > 1:
        rest.extend(web_hits[1:])

    stream = picks + rest

    out_lines: List[str] = []
    used = 0
    for h in stream:
        line = _format_line(h)
        t = len(enc.encode(line))
        if used + t > budget:
            break
        out_lines.append(line)
        used += t

    return "\n\n".join(out_lines), used


@traceable(name="RAG: retrieve_from_rag", run_type="chain", tags=["rag", "retrieval"])
def retrieve_from_rag(
    session_id: str,
    query: str,
    top_k: int | None = None,
    date_filter: Optional[Tuple[int, int]] = None,
    exclude_ids: Optional[List[str]] = None,
    expand_entities: bool = True,
) -> str:
    """
    동적 K RAG 검색

    알고리즘:
    1. 점수 임계값(RAG_THR) 이상만 선택
    2. MMR 다양성 필터링
    3. 토큰 버짓(EVIDENCE_TOKEN_CAP) 내에서 자동 K 결정
    4. norm_key 중복 제거
    5. 출처/타임스탬프 자동 주입
    """
    logger = logging.getLogger("rag")

    # 설정값 (RAG_THR/RAG_THRESHOLD 하위 호환 포함)
    rag_thr = float(getattr(settings, "RAG_THRESHOLD", settings.RAG_THR))
    mmr_lambda = float(settings.RAG_MMR_LAMBDA)
    # 시간 감쇠 람다(채널별): 로그에 더 큰 감쇠, 프로필에 작은 감쇠 권장
    decay_lam_log = float(settings.RAG_DECAY_LAM_LOG)
    decay_lam_prof = float(settings.RAG_DECAY_LAM_PROFILE)

    # 상대예산 기반 증거 토큰 캡 계산 (보수적으로 used_tokens=0 가정)
    ctx_window = int(settings.CTX_WINDOW)
    ratio = float(settings.EVIDENCE_RATIO)
    headroom = int(settings.EVIDENCE_HEADROOM)
    evidence_cap = _evidence_budget(
        ctx_window, used_tokens=0, ratio=ratio, headroom=headroom
    )

    enc = get_tokenizer()
    t0 = time.time()

    try:
        # 세션→사용자 매핑: 영구 저장소 검색은 user_id 기준으로만 수행
        uid = user_for_session(session_id) or session_id

        # 신규: 엔티티 기반 쿼리 확장
        expanded_query = query
        if expand_entities:
            try:
                from backend.rag.entity_expander import get_entity_expander

                expander = get_entity_expander()
                expanded_query = expander.expand_query(query, uid)

                if expanded_query != query:
                    logger.info(f"[rag] Query expanded: '{query}' → '{expanded_query}'")
            except Exception as e:
                logger.warning(f"[rag] Entity expansion failed: {e}")
                expanded_query = query

        # logs/legacy profile JSON
        prof_coll, log_coll = ensure_collections()
        # structured profile chunks (active preferences)
        try:
            from backend.rag.milvus import (
                ensure_profile_collection as _ensure_prof_chunks,
            )

            prof_chunks = _ensure_prof_chunks()
        except Exception:
            prof_chunks = None

        # 쿼리 임베딩 (확장된 쿼리 사용)
        query_emb = embed_query_cached(expanded_query)
        search_params = {"metric_type": METRIC, "params": {"ef": 64}}
        expr = f"user_id == '{uid}'"
        if date_filter:
            s, e = date_filter
            expr += f" and date_end >= {s} and date_start <= {e}"

        # 충분한 후보 검색
        cand_k = 20

        # 프로필 채널: profile_chunks 우선 검색 (active/pending만). 폴백: legacy profile 컬렉션
        prof_hits = []
        beh_hits = []
        if prof_chunks is not None:
            try:
                expr_chunks = f"user_id == '{uid}' and (status == 'active' or status == 'pending')"
                res_chunks = prof_chunks.search(
                    data=[query_emb],
                    anns_field="embedding",
                    param=search_params,
                    limit=cand_k,
                    expr=expr_chunks,
                    output_fields=[
                        "norm_key",
                        "tier",
                        "status",
                        "confidence",
                        "category",
                        "value",
                        "updated_at",
                        "date_ym",
                        "embedding",
                    ],
                )
                prof_hits = res_chunks[0] if res_chunks else []
            except Exception:
                prof_hits = []
        # Behavior 채널(별도 컬렉션)
        try:
            from backend.rag.milvus import ensure_behavior_collection as _ens_beh

            beh_coll = _ens_beh()
            res_beh = beh_coll.search(
                data=[query_emb],
                anns_field="embedding",
                param=search_params,
                limit=cand_k,
                expr=f"user_id == '{uid}' and (status == 'active' or status == 'pending')",
                output_fields=[
                    "slot_key",
                    "norm_key",
                    "value",
                    "status",
                    "confidence",
                    "updated_at",
                ],
            )
            beh_hits = res_beh[0] if res_beh else []
        except Exception:
            beh_hits = []
        if not prof_hits:
            # 폴백: legacy profile_json 컬렉션 (검색 오염 방지를 위해 type 필터 유지)
            prof_res = prof_coll.search(
                data=[query_emb],
                anns_field="embedding",
                param=search_params,
                limit=cand_k,
                expr=expr,
                output_fields=[
                    "text",
                    "date_start",
                    "date_end",
                    "date_ym",
                    "embedding",
                ],
            )
            prof_hits = prof_res[0] if prof_res else []
        log_res = log_coll.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=cand_k,
            expr=expr,
            output_fields=[
                "text",
                "date_start",
                "date_end",
                "date_ym",
                "embedding",
            ],
        )

        # 점수 임계값 컷
        log_hits = log_res[0] if log_res else []

        # 타입 필터링: 스냅샷 프로필 JSON은 검색 오염 방지를 위해 제외
        def _type_ok(hit) -> bool:
            try:
                t = str(getattr(hit, "entity", {}).get("type", "") or "")
                return t != "profile_json"
            except Exception:
                return True

        # legacy 컬렉션 결과에만 타입 필터 적용
        if prof_chunks is None:
            prof_hits = [h for h in prof_hits if _type_ok(h)]

        # 제외 ID 필터 (Milvus expr 대신 결과 후처리로 안전 적용)
        if exclude_ids:
            try:
                ex = {str(x) for x in exclude_ids if x}
                prof_hits = [
                    h for h in prof_hits if str(getattr(h, "id", "")) not in ex
                ]
                log_hits = [h for h in log_hits if str(getattr(h, "id", "")) not in ex]
            except Exception:
                pass

        # 스코어 보정(시간 감쇠) 후 임계값 컷 + 정렬
        def _adj_score_profile(h) -> float:
            # 기본 유사도 + 시간 감쇠(업데이트 기준) + 선호 가중
            base = hit_similarity(h)
            e = getattr(h, "entity", {}) or {}
            # updated_at(나노초) → days
            try:
                upd_ns = int(e.get("updated_at") or 0)
                age = int(max(0, (time.time_ns() - upd_ns) / (1e9 * 86400)))
            except Exception:
                # 폴백: date_ym 사용
                yym = int(e.get("date_ym") or 0)
                age = 0
                if yym:
                    # YYYYMM → 일수 상한 30으로 근사
                    age = 30
            s = _apply_time_decay(base, age, decay_lam_prof)

            # 선호 가중치: Scoreboard의 score를 가중으로 반영
            try:
                nk = str(e.get("norm_key") or "")
                from backend.personalization.preference_scoreboard import (
                    PreferenceScoreboard as _SB,
                )

                sb = _SB(settings.REDIS_URL)
                entry = sb.get(uid, nk) if nk else {}
                pref_score = float(entry.get("score", 0.0))
                lam = float(getattr(settings, "PREFERENCE_WEIGHT_LAMBDA", 0.15))
                s = s * (1.0 + lam * pref_score)
            except Exception:
                pass
            return s

        def _adj_score_log(h, lam: float) -> float:
            base = hit_similarity(h)
            e = getattr(h, "entity", {}) or {}
            ymd_end = int(e.get("date_end") or 0)
            age = _age_days_from_ymd(ymd_end) if ymd_end else 0
            return _apply_time_decay(base, age, lam)

        prof_scored = [(h, _adj_score_profile(h)) for h in prof_hits]
        log_scored = [(h, _adj_score_log(h, decay_lam_log)) for h in log_hits]

        prof_filtered = [h for (h, s) in prof_scored if s >= rag_thr]
        log_filtered = [h for (h, s) in log_scored if s >= rag_thr]

        # 정렬(동점 시 최근 우선)
        prof_filtered.sort(key=lambda h: _adj_score_profile(h), reverse=True)
        log_filtered.sort(key=lambda h: _adj_score_log(h, decay_lam_log), reverse=True)

        # Behavior 스코어(시간 감쇠 + confidence 보정)
        def _adj_score_behavior(h) -> float:
            try:
                e = getattr(h, "entity", {}) or {}
                upd_ns = int(e.get("updated_at") or 0)
                age = int(max(0, (time.time_ns() - upd_ns) / (1e9 * 86400)))
            except Exception:
                age = 0
            base = hit_similarity(h)
            conf = 0.5
            try:
                conf = float(getattr(h, "entity", {}).get("confidence", 0.5))
            except Exception:
                conf = 0.5
            return _apply_time_decay(base, age, decay_lam_prof) * (0.9 + 0.2 * conf)

        beh_filtered = [h for h in beh_hits if _adj_score_behavior(h) >= rag_thr]
        beh_filtered.sort(key=lambda h: _adj_score_behavior(h), reverse=True)

        # MMR 다양성 필터링
        prof_mmr = _apply_mmr(prof_filtered, mmr_lambda, max_k=3)
        log_mmr = _apply_mmr(log_filtered, mmr_lambda, max_k=10)

        # norm_key 중복 제거
        prof_deduped = _dedup_by_doc(prof_mmr)
        log_deduped = _dedup_by_doc(log_mmr)

        # 토큰 버짓 내에서 자동 K 결정
        # 포맷 함수: profile_chunks와 legacy 결과를 모두 처리
        def _format_profile_line(hit) -> str:
            e = getattr(hit, "entity", {}) or {}
            if "norm_key" in e:
                # chunks
                cat = str(e.get("category") or "-")
                tier = str(e.get("tier") or "-")
                nk = str(e.get("norm_key") or "-")
                val = str(e.get("value") or "")
                header = f"[출처: profile | {cat}:{tier} | {nk}] "
                return header + val
            # legacy
            return _format_line(hit)

        def _format_behavior_line(hit) -> str:
            e = getattr(hit, "entity", {}) or {}
            sk = str(e.get("slot_key") or "-")
            nk = str(e.get("norm_key") or "-")
            status = str(e.get("status") or "-")
            val = str(e.get("value") or "")
            header = f"[출처: behavior | {sk} | {status} | {nk}] "
            return header + val

        def _pack_evidence_profiles(profile_hits, log_hits, budget, enc):
            lines: List[str] = []
            used = 0
            # 우선 최소 1개씩
            picks = []
            if profile_hits:
                picks.append(profile_hits[0])
            if log_hits:
                picks.append(log_hits[0])
            rest = []
            if len(profile_hits) > 1:
                rest.extend(profile_hits[1:])
            if len(log_hits) > 1:
                rest.extend(log_hits[1:])
            stream = picks + rest
            for h in stream:
                line = _format_profile_line(h)
                t = len(enc.encode(line))
                if used + t > budget:
                    break
                lines.append(line)
                used += t
            return "\n\n".join(lines), used

        # Behavior는 최소 1개 쿼터(있을 때)만 보장, 나머지 예산은 profile/log에 우선
        beh_line = ""
        if beh_filtered:
            try:
                beh_line = _format_behavior_line(beh_filtered[0])
            except Exception:
                beh_line = ""

        ctx, current_tokens = _pack_evidence_profiles(
            prof_deduped, log_deduped, evidence_cap, enc
        )
        if beh_line:
            # 여유 있을 때만 추가
            t = len(enc.encode(beh_line))
            if current_tokens + t <= evidence_cap:
                ctx = (ctx + ("\n\n" if ctx else "") + beh_line).strip()
                current_tokens += t

        try:
            dynamic_k = 0
            if ctx:
                # 블록 간 구분은 빈 줄(\n\n)
                dynamic_k = ctx.count("\n\n") + 1
            logger.info(
                "[rag] expr=%s thr=%.2f mmr=%.2f ctx_len=%d tokens=%d dynamic_k=%d",
                expr,
                rag_thr,
                mmr_lambda,
                len(ctx),
                current_tokens,
                dynamic_k,
            )
        except Exception:
            pass

        return ctx
    except Exception as e:
        try:
            logging.getLogger("rag").error(f"[rag] Error: {e}")
        except Exception:
            pass
        return ""


# ─────────────────────────────────────────────────────────────
# Subtractive Search + Reranking (Enhanced)
# ─────────────────────────────────────────────────────────────
import logging as _logging


@traceable(name="Retrieval: retrieve_enhanced", run_type="chain", tags=["retrieval"])
async def retrieve_enhanced(
    query: str,
    route: Literal["rag", "web", "weather"],
    session_id: str,
    top_k: int = 5,
    date_filter: Optional[Tuple[int, int]] = None,
) -> str:
    """
    차감 검색 + (옵션) 재랭킹 통합 함수

    - route='web': 0-hop 캐시(web_archived)에서 제외 도메인 생성 → build_web_context에 -site: 도메인 주입
    - route='rag': 향후 evidence_feedback negative 기반 exclude_ids 적용(스키마 제약으로 현재는 미적용)

    Returns:
        블록 문자열(web) 또는 RAG 컨텍스트 문자열(rag)
    """
    try:
        from backend.config import get_settings as _gs
        from backend.rag.embeddings import embed_query_openai as _embed
        from backend.rag.milvus import ensure_collections as _ens
        from backend.rag.reranker import rerank_with_confidence as _rerank
        from backend.search_engine.service import build_web_context as _build_web
        from backend.utils.logger import log_event as _log_event

        # settings 로컬 바인딩을 선행하여 UnboundLocalError를 방지한다.
        _settings = _gs()

        # (신규) 캐시 가드: 동일 쿼리/도메인 반복 시 즉시 반환
        if _settings.FEATURE_CACHE_GUARD and route in {"web", "rag"}:
            try:
                from backend.search_engine.cache_guard import get as _cg_get
                from backend.search_engine.cache_guard import put as _cg_put

                hit = _cg_get(query, domain=route)
                if hit and isinstance(hit.get("body"), str):
                    return str(hit.get("body") or "")
            except Exception:
                pass

        _log_event("retrieve_enhanced_start", {"route": route})

        # Step 1: 0-hop 캐시 프로브 (web_archived 중심)
        _uid = user_for_session(session_id) or session_id
        exclude_sites: List[str] = []
        exclude_ids: List[str] = []
        try:
            _, log_coll = _ens()
            qvec = _embed(query)
            # web_archived 우선: 과거 아카이브에서 동일/유사 쿼리의 출처 도메인 제외
            expr = f"user_id == '{_uid}' and type == 'web_archived'"
            res = log_coll.search(
                data=[qvec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=expr,
                output_fields=["text", "date_end"],
            )
            hits = res[0] if res else []

            seen = set()
            for h in hits:
                txt = str(getattr(h, "entity", {}).get("text", ""))
                dom = _parse_domain_from_text(txt)
                if dom and dom not in seen:
                    seen.add(dom)
                    exclude_sites.append(dom)
        except Exception as _e:
            _logging.getLogger("retrieve_enhanced").warning(
                f"cache probe skipped: {_e}"
            )

        if route == "weather":
            try:
                from backend.search_engine.service import (
                    retrieve_weather_context as _weather_ctx,
                )

                blk = await _weather_ctx(query)
                return blk or ""
            except Exception:
                return ""

        if route == "youtube":
            try:
                from backend.search_engine.service import (
                    retrieve_youtube_context as _yt_ctx,
                )

                blk = await _yt_ctx(query)
                return blk or ""
            except Exception:
                return ""

        if route == "web":
            # Step 2: 차감 검색(Web)
            _kind, ctx = await _build_web(
                _settings.MCP_SERVER_URL,
                query,
                display=10,
                timeout_s=_settings.TIMEOUT_WEB,
                endpoints=None,
                exclude_sites=exclude_sites,
            )

            # Step 3: (선택) LLM/링킹 기반 신뢰도 스코어 반영 재랭킹
            try:
                # build_web_context가 반환한 블록 문자열을 후보 리스트로 구조화
                # 후보 포맷: {id, embedding, date_end}
                cands = []
                if ctx:
                    blocks = (ctx or "").split("\n\n")
                    for i, b in enumerate(blocks):
                        lines = [ln for ln in b.split("\n") if ln.strip()]
                        if not lines:
                            continue
                        text = " ".join(lines[:-1]) if len(lines) >= 2 else lines[0]
                        vec = _embed(text)
                        # 날짜 추출: 두 번째 라인에서 YYYY-MM-DD 접두를 우선 시도
                        date_end = 0
                        try:
                            import re as _re2

                            if len(lines) >= 2:
                                m = _re2.search(
                                    r"(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",
                                    lines[1],
                                )
                                if m:
                                    y, mo, d = (
                                        int(m.group(1)),
                                        int(m.group(2)),
                                        int(m.group(3)),
                                    )
                                    date_end = y * 10000 + mo * 100 + d
                        except Exception:
                            date_end = 0
                        cands.append(
                            {
                                "id": f"web:{i}",
                                "embedding": (
                                    vec.tolist()
                                    if hasattr(vec, "tolist")
                                    else list(vec)
                                ),
                                "date_end": date_end,
                            }
                        )
                import numpy as _np

                qv = _embed(query)
                ranked = _rerank(cands, _np.array(qv, dtype=_np.float32), session_id)
                # 상위 K만 다시 블록 순서로 재구성
                if ranked and ctx:
                    idxs = [int(d.get("id", "web:0").split(":")[1]) for d in ranked]
                    blocks = (ctx or "").split("\n\n")
                    reord = [blocks[j] for j in idxs if 0 <= j < len(blocks)]
                    ctx = "\n\n".join(reord)
                    _log_event("web_reranked", {"count": len(blocks)})
            except Exception:
                pass

            # 캐시 저장
            try:
                if _settings.FEATURE_CACHE_GUARD:
                    from backend.search_engine.cache_guard import put as _cg_put

                    _cg_put(query, {"body": ctx or ""}, domain="web")
            except Exception:
                pass
            return ctx or ""

        # route == 'rag'
        # Step 2: 차감 검색(RAG) — negative 피드백 기반 exclude_ids 적용
        try:
            # 스키마 호환: log 컬렉션에는 feedback_type/original_evidence_id 컬럼이 없으므로
            # 텍스트 메타([meta] ...)를 파싱해서 negative 및 original_evidence_id를 추출한다.
            _, log_coll = _ens()
            rows = log_coll.query(
                expr=f"user_id == '{_uid}' and type == 'evidence_feedback'",
                output_fields=["text"],
                limit=200,
            )
            import re as _re

            for row in rows or []:
                t = str(row.get("text", ""))
                m_type = _re.search(r"feedback_type=([^\s]+)", t)
                m_oid = _re.search(r"original_evidence_id=([^\s]+)", t)
                if m_type and m_type.group(1) == "negative" and m_oid:
                    oid = m_oid.group(1)
                    if oid:
                        exclude_ids.append(oid)
        except Exception:
            pass

        rag_ctx = retrieve_from_rag(
            _uid,
            query,
            top_k=None,
            date_filter=date_filter,
            exclude_ids=exclude_ids or None,
        )
        try:
            if _settings.FEATURE_CACHE_GUARD:
                from backend.search_engine.cache_guard import put as _cg_put

                _cg_put(query, {"body": rag_ctx or ""}, domain="rag")
        except Exception:
            pass
        return rag_ctx or ""
    except Exception as e:
        try:
            _logging.getLogger("retrieve_enhanced").error(f"[retrieve_enhanced] {e}")
        except Exception:
            pass
        return ""


def _apply_mmr(candidates, lambda_param: float, max_k: int):
    """MMR 다양성 필터링"""
    if not candidates:
        return []

    # 입력은 유사도 순으로 정렬되어 있다고 가정 (Milvus 결과)
    selected = [candidates[0]]
    remaining = list(candidates[1:])

    while len(selected) < max_k and remaining:
        best_idx = -1
        best_mmr = -float("inf")

        for i, cand in enumerate(remaining):
            # 관련도: 쿼리-문서 유사도
            rel = hit_similarity(cand)

            # 다양성: 선택된 항목들과의 최대 유사도
            try:
                cand_emb = np.array(
                    cand.entity.get("embedding", []) or [], dtype=np.float32
                )
            except Exception:
                cand_emb = np.array([], dtype=np.float32)

            max_sim = 0.0
            if cand_emb.size > 0:
                for sel in selected:
                    try:
                        sel_emb = np.array(
                            sel.entity.get("embedding", []) or [], dtype=np.float32
                        )
                    except Exception:
                        sel_emb = np.array([], dtype=np.float32)
                    if sel_emb.size > 0:
                        sim = float(
                            np.dot(cand_emb, sel_emb)
                            / (
                                (np.linalg.norm(cand_emb) or 1.0)
                                * (np.linalg.norm(sel_emb) or 1.0)
                            )
                        )
                        if sim > max_sim:
                            max_sim = sim

            mmr_score = lambda_param * rel - (1.0 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


# (삭제됨) _dedup_by_norm_key: 사용 중단, _dedup_by_doc 사용
