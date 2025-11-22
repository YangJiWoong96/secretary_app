import logging
import os
import time
from typing import List, Tuple
from urllib.parse import urlparse

import httpx
from backend.utils.http_client import get_async_client

from backend.config import get_settings
from backend.utils.tracing import traceable
from backend.memory.summarizer import get_tokenizer
from backend.utils.datetime_utils import now_kst, ym, ymd

from .client import MCPClient, NaverDirectClient
from .formatter import format_items_to_blocks
from .router import pick_endpoint, pick_endpoints
from backend.search_engine.ranking import score_item
from backend.search_engine.filters import match_interest

import json
from pathlib import Path
from datetime import datetime, timedelta
import re

settings = get_settings()

# (신규) YouTube 블록 생성기
try:
    from backend.services.youtube import get_youtube_block as _yt_block
except Exception:
    _yt_block = None  # type: ignore

# 간단 TTL 캐시(프로세스 메모리): 동일 질의 단시간 반복 요청 방지
_SEARCH_TTL_SEC = float(getattr(settings, "SEARCH_TTL_SEC", 120.0))  # 기본 120s
_CACHE: dict[str, tuple[float, tuple[str, str]]] = {}

# ── MCP 서킷 브레이커(간단): 성공 TTL/실패 백오프
_MCP_OK_UNTIL_TS: float = 0.0
_MCP_BACKOFF_UNTIL_TS: float = 0.0
_MCP_FAILS: int = 0


def _norm_q(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


# 공통 URL/도메인 정규화 헬퍼 (내부 전용)
def _url_of_common(it: dict) -> str:
    return it.get("originallink") or it.get("link") or it.get("url") or ""


def _norm_domain_common(url: str) -> str:
    try:
        u = urlparse(url or "")
        host = (u.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


# 로컬 검색시 허용 도메인/URL 가드 (지도/플레이스 제공자 위주)
_ALLOWED_LOCAL_DOMAINS: set[str] = {
    # Naver
    "map.naver.com",
    "m.map.naver.com",
    "place.naver.com",
    "store.naver.com",
    "naver.me",  # 네이버 단축 URL (지도/플레이스)
    # Kakao
    "map.kakao.com",
    "m.map.kakao.com",
    "place.map.kakao.com",
    "kko.to",  # 카카오 단축 URL (지도)
    # Google Maps
    "maps.google.com",
    "www.google.com",
    "maps.app.goo.gl",
    "goo.gl",
    # MangoPlate
    "www.mangoplate.com",
    "mangoplate.com",
}


def _has_disallowed_extension(url: str) -> bool:
    try:
        path = (urlparse(url or "").path or "").lower()
        return any(
            path.endswith(ext)
            for ext in (".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx")
        )
    except Exception:
        return False


def _is_allowed_local_url(url: str) -> bool:
    """
    로컬(vertical)에서 사용할 수 있는 지도/플레이스 제공자 URL만 허용한다.
    Google은 도메인이 넓어 경로에 '/maps' 포함 여부를 확인한다.
    단축 URL(naver.me, kko.to, goo.gl, maps.app.goo.gl)은 허용한다.
    """
    try:
        u = urlparse(url or "")
        host = (u.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in _ALLOWED_LOCAL_DOMAINS:
            return False
        # Google 도메인의 경우 경로 확인 (너무 광범위 차단 방지)
        if host in {"www.google.com"}:
            p = (u.path or "").lower()
            return p.startswith("/maps") or "/maps/" in p
        return True
    except Exception:
        return False


def _is_youtube_url(url: str) -> bool:
    try:
        host = (urlparse(url or "").netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return ("youtube.com" in host) or ("youtu.be" in host)
    except Exception:
        return False


def _project_root() -> Path:
    # backend/search_engine/service.py → backend
    return Path(__file__).resolve().parents[2]


def _snapshot_root() -> Path:
    # 스냅샷 저장 루트: backend/evidence/websnap
    return _project_root() / "evidence" / "websnap"


def _normalize_title_text(title: str) -> str:
    # Node MCP가 반환하는 HTML <b>태그 제거 등 간단 정규화
    t = str(title or "")
    t = re.sub(r"</?b>", "", t, flags=re.I)
    t = " ".join(t.split())
    return t


def _write_snapshot(kind: str, query_original: str, items: List[dict]) -> None:
    """
    시간대별 raw 스냅샷 저장(JSONL)
    - 경로: backend/evidence/websnap/YYYYMMDD/{kind}-{HHMM}.jsonl
    - 필드: title, url, rank, kind, query, timestamp
    """
    try:
        root = _snapshot_root()
        now = datetime.now()
        day_dir = root / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{kind}-{now.strftime('%H%M')}.jsonl"
        fpath = day_dir / fname

        with fpath.open("w", encoding="utf-8") as f:
            for idx, it in enumerate(items, 1):
                title = _normalize_title_text(it.get("title") or it.get("name") or "")
                url = (
                    it.get("originallink") or it.get("link") or it.get("url") or ""
                ).strip()
                rec = {
                    "title": title,
                    "url": url,
                    "rank": idx,
                    "kind": kind,
                    "query": str(query_original or ""),
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # 스냅샷 실패는 검색 경로를 막지 않음
        pass


def _load_recent_title_stats(kind: str, lookback_hours: int = 24) -> dict:
    """
    최근 스냅샷에서 제목별 통계를 수집한다.
    반환:
        {
          normalized_title: {
            "count": 등장 총 횟수,
            "occurrences": 등장한 스냅샷 수(중복 포함),
            "top_hits": 상위 TOP-5 안에 든 횟수
          },
          ...
        }
    """
    out: dict = {}
    try:
        root = _snapshot_root()
        if not root.exists():
            return out
        now = datetime.now()
        # 하루 단위 폴더만 스캔(최근 24h 기준으로 오늘/어제 폴더 후보)
        candidate_days = [now.strftime("%Y%m%d")]
        prev = now - timedelta(days=1)
        if prev.strftime("%Y%m%d") != candidate_days[0]:
            candidate_days.append(prev.strftime("%Y%m%d"))

        cutoff = now - timedelta(hours=max(1, int(lookback_hours)))

        for day in candidate_days:
            d = root / day
            if not d.exists():
                continue
            for fp in sorted(d.glob(f"{kind}-*.jsonl")):
                try:
                    # 파일명 시간 파싱으로 대략적인 컷 적용(엄격하지 않아도 충분)
                    m = re.match(rf"{re.escape(kind)}-(\d{{4}})\.jsonl", fp.name)
                    if m:
                        hhmm = m.group(1)
                        t = datetime.strptime(f"{day}{hhmm}", "%Y%m%d%H%M")
                        if t < cutoff:
                            continue
                    with fp.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                title = _normalize_title_text(rec.get("title", ""))
                                rank = int(rec.get("rank", 999))
                                if not title:
                                    continue
                                s = out.setdefault(
                                    title, {"count": 0, "occurrences": 0, "top_hits": 0}
                                )
                                s["count"] += 1
                                s["occurrences"] += 1
                                if rank <= 5:
                                    s["top_hits"] += 1
                            except Exception:
                                continue
                except Exception:
                    continue
    except Exception:
        return out
    return out


@traceable(name="WebSearch: build_web_context", run_type="tool", tags=["web", "search"])
async def build_web_context(
    base_url: str,
    query: str,
    display: int = 5,
    timeout_s: float = 15.0,
    endpoints: List[str] | None = None,
    exclude_sites: List[str] | None = None,
    interest: dict | None = None,
    use_scoring: bool = True,
    lookback_hours: int = 24,
) -> Tuple[str, str]:
    """
    MCP 기반 네이버 검색을 호출해 3줄 블록 컨텍스트를 만든다.
    반환: (kind, ctx)
    kind: 'local' | 'news' | 'webkr' | 'error'
    ctx: 블록 문자열 (없으면 빈 문자열)
    """
    logger = logging.getLogger("router")

    # 0) 프리플라이트: base_url/alt_url 중 응답하는 쪽을 선별(환경 불일치/부트 지연 대응)

    # 0-0) 차감 검색: 제외 도메인 쿼리 주입(-site:domain)
    query_original = query
    query_search = query
    try:
        if exclude_sites:
            # 중복 제거 및 간단 정규화
            uniq = []
            seen = set()
            for d in exclude_sites:
                dom = (d or "").strip().lower()
                if dom.startswith("www."):
                    dom = dom[4:]
                if dom and dom not in seen:
                    seen.add(dom)
                    uniq.append(dom)
            if uniq:
                exclude_str = " " + " ".join(f"-site:{dom}" for dom in uniq)
                query_search = (query or "").strip() + exclude_str
    except Exception:
        # 안전 폴백: 제외 목록 적용 실패 시 원 쿼리 사용
        query_search = query

    async def _probe(url: str) -> bool:
        try:
            to = httpx.Timeout(min(timeout_s, 1.0))
            client = get_async_client()
            r = await client.get(f"{url.rstrip('/')}/health", timeout=to)
            return r.status_code == 200
        except Exception:
            return False

    alt_url = (
        "http://localhost:5000"
        if (base_url or "").startswith("http://mcp:")
        else "http://mcp:5000"
    )
    effective_url = base_url
    try:
        import time as _t

        now_ts = _t.time()
        global _MCP_OK_UNTIL_TS, _MCP_BACKOFF_UNTIL_TS, _MCP_FAILS
        # 백오프 기간이면 프로브 생략
        if now_ts >= _MCP_BACKOFF_UNTIL_TS:
            # OK TTL 기간 내면 생략(최근 정상)
            if now_ts < _MCP_OK_UNTIL_TS:
                ok_base = True
                ok_alt = False
            else:
                ok_base = await _probe(base_url)
                ok_alt = False
                if not ok_base and alt_url != base_url:
                    ok_alt = await _probe(alt_url)
            if ok_base:
                _MCP_OK_UNTIL_TS = now_ts + 120.0  # 2분 TTL
                _MCP_FAILS = 0
            elif ok_alt:
                effective_url = alt_url
                _MCP_OK_UNTIL_TS = now_ts + 120.0
                _MCP_FAILS = 0
                logging.getLogger("router").info(
                    f"[web:mcp:probe] switch base_url -> {effective_url}"
                )
            else:
                # 실패: 지수 백오프(최대 60s)
                _MCP_FAILS += 1
                backoff = min(60.0, 2.0 * _MCP_FAILS)
                _MCP_BACKOFF_UNTIL_TS = now_ts + backoff
    except Exception:
        pass

    # 0.5) TTL 캐시 조회(effective_url 기준)
    key = f"{effective_url.rstrip('/')}:d={int(display)}:q={_norm_q(query_search)}"
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) <= _SEARCH_TTL_SEC:
        kind_cached, ctx_cached = hit[1]
        return kind_cached, ctx_cached
    # 1차: MCP 프록시 경유 (기본 URL) — 적응 fanout과 엔드포인트 병렬 지원
    eps = endpoints or [pick_endpoint(query)]
    client = MCPClient(effective_url, timeout_s)
    items = []
    # 라우터 결정 엔드포인트를 신뢰(응답의 kind 필드에 종속되지 않음)
    selected_kind = eps[0] if eps else "webkr"
    kind = selected_kind
    for ep in eps:
        # 후보 수집은 최대 10개까지 요청
        cand_display = 10
        data = await client.naver_search(
            query_search, display=cand_display, endpoint=ep
        )
        # kind는 라우팅된 엔드포인트(ep)를 유지
        batch = (
            (data.get("data", {}) or {}).get("items", [])
            if isinstance(data, dict)
            else []
        )
        items.extend(batch)
    # 0.9) 스냅샷 저장(원시 배치)
    try:
        if items:
            _write_snapshot(
                kind=selected_kind, query_original=query_original, items=items
            )
    except Exception:
        pass
    # 동적 K 적용 준비
    status = data.get("status") if isinstance(data, dict) else None
    if status and status != 200:
        logger.info(
            f"[web:mcp] status={status} kind={kind} q='{query_original[:60]}' url={effective_url}"
        )

    # 안전 범위: URL 추출기는 이후 분기 어디서든 사용되므로 선제 정의
    def _url_of(it: dict) -> str:
        return _url_of_common(it)

    if items:
        # 1) 점수 임계값 컷 (가능한 경우)
        try:
            web_thr = float(settings.WEB_THR)
        except Exception:
            web_thr = 0.25

        def _get_score(it):
            try:
                return float(it.get("score"))
            except Exception:
                return None

        # 로컬(vertical)은 스코어 미정/저점 빈도가 높아 컷을 적용하지 않고 전량 보류
        if selected_kind == "local":
            items_thr = list(items)
            # 하드필터 1: 카테고리/제목에 질의 핵심어 포함 여부(간단 포함검사)
            try:
                q_tokens = [t for t in (query_original or "").split() if len(t) >= 2]
            except Exception:
                q_tokens = []

            def _has_keyword(it: dict) -> bool:
                try:
                    cand = " ".join(
                        [str(it.get("title") or ""), str(it.get("category") or "")]
                    ).lower()
                    return (
                        any(t.lower() in cand for t in q_tokens) if q_tokens else True
                    )
                except Exception:
                    return True

            items_thr = [it for it in items_thr if _has_keyword(it)]
        else:
            items_thr = [
                it
                for it in items
                if (_get_score(it) is None) or (_get_score(it) >= web_thr)
            ]

        # 공통: 파일 확장자 필터링(pdf/hwp 등) 제거
        items_thr = [
            it for it in items_thr if not _has_disallowed_extension(_url_of(it))
        ]

        # 2) 도메인 단위 중복 제거 (동일 사이트 최대 2개)
        def _norm_domain(url: str) -> str:
            return _norm_domain_common(url)

        per_domain_count: dict[str, int] = {}
        deduped: list[dict] = []
        for it in items_thr:
            url = _url_of(it)
            dom = _norm_domain(url)
            cnt = per_domain_count.get(dom, 0)
            if dom and cnt >= 2:
                continue
            per_domain_count[dom] = cnt + 1
            deduped.append(it)

        # 로컬(vertical)일 때: 허용되지 않은 도메인의 링크는 제거하여 포매터가 지도 검색 URL을 합성하도록 유도
        if selected_kind == "local":
            new_deduped: list[dict] = []
            for it in deduped:
                url = _url_of(it)
                if url and (not _is_allowed_local_url(url)):
                    it2 = dict(it)
                    it2.pop("originallink", None)
                    it2.pop("link", None)
                    it2.pop("url", None)
                    new_deduped.append(it2)
                else:
                    new_deduped.append(it)
            deduped = new_deduped
            # 하드필터 2: 좌표 기반 반경 필터(distance_km가 제공되면 1.0km 초과 제거)
            try:
                radius_km = float(settings.LOCAL_RADIUS_KM)
            except Exception:
                radius_km = 1.0
            deduped = [
                it
                for it in deduped
                if (not isinstance(it.get("distance_km"), (int, float)))
                or float(it.get("distance_km", 0)) <= radius_km
            ]

        # 3) 토큰 버짓 내에서 자동 K 결정
        enc = get_tokenizer()
        token_cap = int(settings.EVIDENCE_TOKEN_CAP)
        blocks_selected: list[str] = []
        used_tokens = 0

        # (신규) 개인화 필터 적용
        if interest:
            req = list(interest.get("required") or [])
            nor = list(interest.get("normal") or [])
            den = list(interest.get("denied") or [])
            deduped = [
                it
                for it in deduped
                if match_interest(
                    _normalize_title_text(it.get("title") or it.get("name") or ""),
                    req,
                    nor,
                    den,
                )
            ]

        # (신규) 최근 스냅샷 기반 freq/hot 계산 및 가중합 스코어 정렬
        if use_scoring and deduped:
            stats = _load_recent_title_stats(
                kind=selected_kind, lookback_hours=lookback_hours
            )
            scored: list[tuple[float, dict]] = []
            for idx, it in enumerate(deduped, 1):
                title = _normalize_title_text(it.get("title") or it.get("name") or "")
                # 현재 순서를 랭킹으로 사용(1부터 시작)
                rank_val = idx
                st = stats.get(title, {"count": 0, "occurrences": 0, "top_hits": 0})
                # freq 정규화(최대 5회를 1.0으로 캡)
                freq_norm = min(float(st.get("count", 0)) / 5.0, 1.0)
                occ = max(1, int(st.get("occurrences", 0)))
                hot_norm = float(st.get("top_hits", 0)) / float(occ) if occ > 0 else 0.0
                sc = score_item(rank=rank_val, freq=freq_norm, hot=hot_norm)
                scored.append((sc, it))
            # 점수 내림차순
            scored.sort(key=lambda x: x[0], reverse=True)
            deduped = [it for _, it in scored]
        for it in deduped:
            # 실제 스니펫(설명)을 포함시켜 날짜/사실 검증이 가능하도록 llm_desc=False
            block = None
            try:
                url = _url_of(it)
                use_yt = bool(getattr(settings, "FEATURE_YOUTUBE_TRANSCRIPT", True))
                if (
                    selected_kind != "local"
                    and _yt_block is not None
                    and use_yt
                    and _is_youtube_url(url)
                ):
                    # YouTube URL: 트랜스크립트/폴백 STT를 활용해 블록 생성
                    block = await _yt_block(url)  # type: ignore[misc]
            except Exception:
                block = None
            if not block:
                block = format_items_to_blocks([it], selected_kind, llm_desc=False)
            if not block:
                continue
            tk = len(enc.encode(block))
            if used_tokens + tk > token_cap:
                break
            blocks_selected.append(block)
            used_tokens += tk

        ctx = "\n\n".join(blocks_selected)
        # 최종 폴백: 로컬인데 블록이 비었으면 지도 검색 링크 1개 생성
        if (not ctx) and selected_kind == "local":
            try:
                from urllib.parse import quote as _quote
            except Exception:
                _quote = lambda x: x  # type: ignore
            title_fb = f"{query_original} (지도 검색)"
            desc_fb = "-"
            url_fb = f"https://map.naver.com/v5/search/{_quote(query_original)}"
            ctx = "\n".join([title_fb, desc_fb, url_fb])
        _CACHE[key] = (now, (selected_kind, ctx))
        return selected_kind, ctx

    # 1.5차: MCP 대체 URL 재시도 (이중 안전장치)
    if alt_url != effective_url:
        client_alt = MCPClient(alt_url, timeout_s)
        data_alt = await client_alt.naver_search(
            query_search, display=10, endpoint=eps[0]
        )
        kind_alt = eps[0]
        items_alt = (
            (data_alt.get("data", {}) or {}).get("items", [])
            if isinstance(data_alt, dict)
            else []
        )
        blocks_alt = None
        if isinstance(data_alt, dict):
            blocks_alt = data_alt.get("blocks") or (
                (data_alt.get("data") or {}).get("blocks")
                if isinstance(data_alt.get("data"), dict)
                else None
            )
        status_alt = data_alt.get("status") if isinstance(data_alt, dict) else None
        if status_alt and status_alt != 200:
            logger.info(
                f"[web:mcp:alt] status={status_alt} kind={kind_alt} q='{query_original[:60]}' url={alt_url}"
            )
        if items_alt:
            # 동일한 동적 K 처리
            try:
                web_thr = float(settings.WEB_THR)
            except Exception:
                web_thr = 0.25

            def _get_score2(it):
                try:
                    return float(it.get("score"))
                except Exception:
                    return None

            if kind_alt == "local":
                items_thr2 = list(items_alt)
            else:
                items_thr2 = [
                    it
                    for it in items_alt
                    if (_get_score2(it) is None) or (_get_score2(it) >= web_thr)
                ]

            # 공통 확장자 필터
            items_thr2 = [
                it for it in items_thr2 if not _has_disallowed_extension(_url_of(it))
            ]

            def _norm_domain2(url: str) -> str:
                return _norm_domain_common(url)

            per_domain_count2: dict[str, int] = {}
            deduped2: list[dict] = []
            for it in items_thr2:
                url = _url_of(it)
                dom = _norm_domain2(url)
                cnt = per_domain_count2.get(dom, 0)
                if dom and cnt >= 2:
                    continue
                per_domain_count2[dom] = cnt + 1
                deduped2.append(it)

            # 로컬(vertical): 허용되지 않은 도메인의 링크 제거 → 포매터가 지도 검색 URL 합성
            if kind_alt == "local":
                nd2: list[dict] = []
                for it in deduped2:
                    url = _url_of(it)
                    if url and (not _is_allowed_local_url(url)):
                        it2 = dict(it)
                        it2.pop("originallink", None)
                        it2.pop("link", None)
                        it2.pop("url", None)
                        nd2.append(it2)
                    else:
                        nd2.append(it)
                deduped2 = nd2

            enc = get_tokenizer()
            token_cap = int(settings.EVIDENCE_TOKEN_CAP)
            blocks_selected2: list[str] = []
            used_tokens2 = 0
            for it in deduped2:
                block = None
                try:

                    def _url_of2(x: dict) -> str:
                        return _url_of_common(x)

                    url2 = _url_of2(it)
                    use_yt = bool(getattr(settings, "FEATURE_YOUTUBE_TRANSCRIPT", True))
                    if (
                        kind_alt != "local"
                        and _yt_block is not None
                        and use_yt
                        and _is_youtube_url(url2)
                    ):
                        block = await _yt_block(url2)  # type: ignore[misc]
                except Exception:
                    block = None
                if not block:
                    block = format_items_to_blocks([it], kind_alt, llm_desc=False)
                if not block:
                    continue
                tk = len(enc.encode(block))
                if used_tokens2 + tk > token_cap:
                    break
                blocks_selected2.append(block)
                used_tokens2 += tk

            ctx_alt = "\n\n".join(blocks_selected2)
            if (not ctx_alt) and kind_alt == "local":
                try:
                    from urllib.parse import quote as _quote2
                except Exception:
                    _quote2 = lambda x: x  # type: ignore
                ctx_alt = "\n".join(
                    [
                        f"{query_original} (지도 검색)",
                        "-",
                        f"https://map.naver.com/v5/search/{_quote2(query_original)}",
                    ]
                )
            _CACHE[key] = (now, (kind_alt, ctx_alt))
            return kind_alt, ctx_alt

    # 2차 폴백: 환경변수/설정 자격으로 직접 호출 (MCP 서버 이슈/폴백 미동작 대비)

    _settings_local = get_settings()
    cid = (
        (_settings_local.CLIENT_ID or "")
        or os.getenv("CLIENT_ID")
        or os.getenv("NAVER_CLIENT_ID")
    )
    csec = (
        (_settings_local.CLIENT_SECRET or "")
        or os.getenv("CLIENT_SECRET")
        or os.getenv("NAVER_CLIENT_SECRET")
    )
    if cid and csec:
        ndc = NaverDirectClient(cid, csec, timeout_s)
        # 엔드포인트 선택(임베딩 라우팅)
        eps2 = endpoints or pick_endpoints(query)
        ep = eps2[0]
        res = await ndc.search(ep, query_search, 10)
        logger.info(
            f"[web:direct] status={res.get('status')} ep={ep} q='{query_original[:60]}'"
        )
        items2 = (res.get("data", {}) or {}).get("items", [])
        k2 = ep
        if (not items2) and len(eps2) > 1:
            alt_ep = eps2[1]
            res2 = await ndc.search(alt_ep, query_search, 10)
            logger.info(
                f"[web:direct:fallback] status={res2.get('status')} ep={alt_ep} q='{query_original[:60]}'"
            )
            items2 = (res2.get("data", {}) or {}).get("items", [])
            k2 = alt_ep
        if items2:
            # 동적 K 처리(직접 호출 폴백)
            try:
                web_thr = float(settings.WEB_THR)
            except Exception:
                web_thr = 0.25

            def _get_score3(it):
                try:
                    return float(it.get("score"))
                except Exception:
                    return None

            if k2 == "local":
                items2_thr = list(items2)
            else:
                items2_thr = [
                    it
                    for it in items2
                    if (_get_score3(it) is None) or (_get_score3(it) >= web_thr)
                ]

            # 공통 확장자 필터
            items2_thr = [
                it for it in items2_thr if not _has_disallowed_extension(_url_of(it))
            ]

            per_domain_count3: dict[str, int] = {}
            deduped3: list[dict] = []
            for it in items2_thr:
                url = _url_of(it)
                dom = (urlparse(url or "").netloc or "").lower()
                if dom.startswith("www."):
                    dom = dom[4:]
                cnt = per_domain_count3.get(dom, 0)
                if dom and cnt >= 2:
                    continue
                per_domain_count3[dom] = cnt + 1
                deduped3.append(it)

            # 로컬(vertical): 허용되지 않은 도메인의 링크 제거 → 포매터가 지도 검색 URL 합성
            if k2 == "local":
                nd3: list[dict] = []
                for it in deduped3:
                    url = _url_of(it)
                    if url and (not _is_allowed_local_url(url)):
                        it2 = dict(it)
                        it2.pop("originallink", None)
                        it2.pop("link", None)
                        it2.pop("url", None)
                        nd3.append(it2)
                    else:
                        nd3.append(it)
                deduped3 = nd3

            enc = get_tokenizer()
            token_cap = int(settings.EVIDENCE_TOKEN_CAP)
            blocks_selected3: list[str] = []
            used_tokens3 = 0
            for it in deduped3:
                block = None
                try:
                    url3 = _url_of(it)
                    use_yt = bool(getattr(settings, "FEATURE_YOUTUBE_TRANSCRIPT", True))
                    if (
                        k2 != "local"
                        and _yt_block is not None
                        and use_yt
                        and _is_youtube_url(url3)
                    ):
                        block = await _yt_block(url3)  # type: ignore[misc]
                except Exception:
                    block = None
                if not block:
                    block = format_items_to_blocks([it], k2, llm_desc=False)
                if not block:
                    continue
                tk = len(enc.encode(block))
                if used_tokens3 + tk > token_cap:
                    break
                blocks_selected3.append(block)
                used_tokens3 += tk

            ctx2 = "\n\n".join(blocks_selected3)
            if (not ctx2) and k2 == "local":
                try:
                    from urllib.parse import quote as _quote3
                except Exception:
                    _quote3 = lambda x: x  # type: ignore
                ctx2 = "\n".join(
                    [
                        f"{query_original} (지도 검색)",
                        "-",
                        f"https://map.naver.com/v5/search/{_quote3(query_original)}",
                    ]
                )
            _CACHE[key] = (now, (k2, ctx2))
            return k2, ctx2
    else:
        logger.info(
            "[web:direct] credentials missing (CLIENT_ID/CLIENT_SECRET or NAVER_CLIENT_ID/NAVER_CLIENT_SECRET)"
        )

    # 실패 시 빈 컨텍스트 (로컬은 마지막 폴백)
    try:
        from urllib.parse import quote as _quote4
    except Exception:
        _quote4 = lambda x: x  # type: ignore
    if selected_kind == "local":
        ctx_fb = "\n".join(
            [
                f"{query_original} (지도 검색)",
                "-",
                f"https://map.naver.com/v5/search/{_quote4(query_original)}",
            ]
        )
        _CACHE[key] = (now, (selected_kind, ctx_fb))
        return selected_kind, ctx_fb
    _CACHE[key] = (now, ("error", ""))
    return "error", ""


async def retrieve_weather_context(query: str) -> str:
    """
    날씨 전용 컨텍스트 생성(구조화 데이터 소스 → 3줄 블록 변환)
    - 지명→좌표(내부 geocode) → 날씨 API(Open-Meteo 등) → 3줄 블록(title/desc/url)
    - 실패 시 빈 문자열
    """
    try:
        from backend.services.weather import (  # (별도 모듈) title, desc, url 반환
            get_weather_block,
        )

        block = await get_weather_block(query)
        if not block:
            return ""
        # 이미 title\n\n형식으로 오면 그대로, 아니면 3줄로 강제
        try:
            title = str(block.get("title") or "").strip()
            desc = str(block.get("desc") or "").strip() or "-"
            url = str(block.get("url") or "").strip()
            if not url:
                return ""
            return "\n".join([title, desc, url])
        except Exception:
            return ""
    except Exception:
        return ""


async def retrieve_youtube_context(query_or_url: str) -> str:
    """
    YouTube 전용 컨텍스트 생성
    - 입력이 YouTube URL일 때만 동작(최소 구현)
    - 성공 시 3줄 블록(title/desc/url), 실패 시 빈 문자열
    """
    try:
        url = (query_or_url or "").strip()
        if not url or (not _is_youtube_url(url)):
            return ""
        if _yt_block is None:
            return ""
        block = await _yt_block(url)  # type: ignore[misc]
        return block or ""
    except Exception:
        return ""
