from typing import Tuple, List
import logging
import os
import time
from .client import MCPClient, NaverDirectClient
from .router import pick_endpoint, pick_endpoints
from .formatter import format_items_to_blocks

# 간단 TTL 캐시(프로세스 메모리): 동일 질의 단시간 반복 요청 방지
_SEARCH_TTL_SEC = float(os.getenv("SEARCH_TTL_SEC", "120"))  # 기본 120s
_CACHE: dict[str, tuple[float, tuple[str, str]]] = {}


def _norm_q(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


async def build_web_context(
    base_url: str,
    query: str,
    display: int = 5,
    timeout_s: float = 2.5,
    endpoints: List[str] | None = None,
) -> Tuple[str, str]:
    """
    MCP 기반 네이버 검색을 호출해 3줄 블록 컨텍스트를 만든다.
    반환: (kind, ctx)
    kind: 'local' | 'news' | 'webkr' | 'error'
    ctx: 블록 문자열 (없으면 빈 문자열)
    """
    logger = logging.getLogger("router")

    # 0) TTL 캐시 조회
    key = f"{base_url.rstrip('/')}:d={int(display)}:q={_norm_q(query)}"
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) <= _SEARCH_TTL_SEC:
        kind_cached, ctx_cached = hit[1]
        return kind_cached, ctx_cached
    # 1차: MCP 프록시 경유 (기본 URL) — 적응 fanout과 엔드포인트 병렬 지원
    eps = endpoints or [pick_endpoint(query)]
    client = MCPClient(base_url, timeout_s)
    items = []
    kind = "error"
    for ep in eps:
        data = await client.naver_search(query, display=display, endpoint=ep)
        kind = data.get("kind", ep) if isinstance(data, dict) else ep
        batch = (
            (data.get("data", {}) or {}).get("items", [])
            if isinstance(data, dict)
            else []
        )
        items.extend(batch)
    # MCP blocks 우선 사용(플래그)
    prefer_blocks = os.getenv(
        "SEARCH_USE_MCP_BLOCKS", os.getenv("USE_MCP_FORMATS", "0")
    ) in ("1", "true", "True")
    blocks_val = None
    if isinstance(data, dict):
        blocks_val = data.get("blocks") or (
            (data.get("data") or {}).get("blocks")
            if isinstance(data.get("data"), dict)
            else None
        )
    status = data.get("status") if isinstance(data, dict) else None
    if status and status != 200:
        logger.info(
            f"[web:mcp] status={status} kind={kind} q='{query[:60]}' url={base_url}"
        )
    if prefer_blocks and blocks_val:
        ctx = str(blocks_val)
        _CACHE[key] = (now, (kind, ctx))
        return kind, ctx
    if items:
        ctx = format_items_to_blocks(items[:display], kind)
        _CACHE[key] = (now, (kind, ctx))
        return kind, ctx

    # 1.5차: MCP 대체 URL 재시도 (host/dev ↔ docker 간 환경 불일치 대비)
    alt_url = (
        "http://localhost:5000"
        if (base_url or "").startswith("http://mcp:")
        else "http://mcp:5000"
    )
    if alt_url != base_url:
        client_alt = MCPClient(alt_url, timeout_s)
        data_alt = await client_alt.naver_search(
            query, display=display, endpoint=eps[0]
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
                f"[web:mcp:alt] status={status_alt} kind={kind_alt} q='{query[:60]}' url={alt_url}"
            )
        if prefer_blocks and blocks_alt:
            ctx_alt = str(blocks_alt)
            _CACHE[key] = (now, (kind_alt, ctx_alt))
            return kind_alt, ctx_alt
        if items_alt:
            ctx_alt = format_items_to_blocks(items_alt, kind_alt)
            _CACHE[key] = (now, (kind_alt, ctx_alt))
            return kind_alt, ctx_alt

    # 2차 폴백: 환경변수 자격으로 직접 호출 (MCP 서버 이슈/폴백 미동작 대비)

    cid = os.getenv("CLIENT_ID")
    csec = os.getenv("CLIENT_SECRET")
    if cid and csec:
        ndc = NaverDirectClient(cid, csec, timeout_s)
        # 엔드포인트 선택(임베딩 라우팅)
        eps2 = endpoints or pick_endpoints(query)
        ep = eps2[0]
        res = await ndc.search(ep, query, display)
        logger.info(f"[web:direct] status={res.get('status')} ep={ep} q='{query[:60]}'")
        items2 = (res.get("data", {}) or {}).get("items", [])
        k2 = ep
        if (not items2) and len(eps2) > 1:
            alt_ep = eps2[1]
            res2 = await ndc.search(alt_ep, query, display)
            logger.info(
                f"[web:direct:fallback] status={res2.get('status')} ep={alt_ep} q='{query[:60]}'"
            )
            items2 = (res2.get("data", {}) or {}).get("items", [])
            k2 = alt_ep
        if items2:
            ctx2 = format_items_to_blocks(items2[:display], k2)
            _CACHE[key] = (now, (k2, ctx2))
            return k2, ctx2
    else:
        logger.info(
            "[web:direct] credentials missing in Python env (NAVER_CLIENT_ID/CLIENT_ID)"
        )

    # 실패 시 빈 컨텍스트
    _CACHE[key] = (now, ("error", ""))
    return "error", ""
