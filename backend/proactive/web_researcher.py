from __future__ import annotations

"""
WebResearchAgent

기능:
- Google News RSS, Reddit, (선택) Google CSE(Programmable Search)까지 병렬 검색
- 도메인 카탈로그/선택 소스에서 전달된 seed_domains 기반 site: 제한 검색 지원
- 계층적 콘텐츠 추출(trafilatura → readability → (있으면) 사이트 어댑터)
- TrustScorer로 신뢰도 산출 및 필터(trust_score ≥ 0.7)
- ConsistencyChecker로 일관성 점수 계산
- (옵션) BM25/임베딩/cross-encoder 기반 재랭킹

주의:
- 외부 API 키/크리덴셜이 없으면 해당 소스는 자동 폴백(빈 결과)
"""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import httpx

from backend.config import get_settings
from backend.proactive.consistency_checker import ConsistencyChecker
from backend.proactive.schemas import ConsistencyReport, WebSearchResult
from backend.proactive.trust_scorer import calculate_trust_score
from backend.utils.logger import log_event

try:
    from backend.proactive.observability import emit_metric  # type: ignore
except Exception:  # pragma: no cover
    emit_metric = None  # type: ignore

# ------------------------------
# User-Agent 로테이션 (가중치)
# ------------------------------
USER_AGENTS: List[Tuple[str, float]] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        0.40,
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        0.25,
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        0.20,
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        0.10,
    ),
    (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
        0.05,
    ),
]


def _random_user_agent() -> str:
    agents, weights = zip(*USER_AGENTS)
    return random.choices(list(agents), weights=list(weights))[0]


class RateLimiter:
    def __init__(self, base_delay: float = 0.5) -> None:
        self.base_delay = base_delay
        self._last: Dict[str, float] = {}

    async def wait(self, domain: str) -> None:
        now = time.time()
        last = self._last.get(domain, 0.0)
        jitter = random.uniform(0.5, 1.5)
        total = self.base_delay + jitter
        if now - last < total:
            await asyncio.sleep(total - (now - last))
        self._last[domain] = time.time()


class RobotsChecker:
    def __init__(self) -> None:
        self.cache: Dict[str, Any] = {}

    async def can_fetch(self, url: str, user_agent: str) -> bool:
        try:
            import urllib.robotparser as rp  # 표준 라이브러리
        except Exception:
            return True
        domain = urlparse(url).netloc
        if domain not in self.cache:
            r = rp.RobotFileParser()
            r.set_url(f"https://{domain}/robots.txt")
            try:
                # blocking → 스레드로 위임
                await asyncio.to_thread(r.read)
                self.cache[domain] = r
            except Exception:
                self.cache[domain] = None
        parser = self.cache.get(domain)
        if parser is None:
            return True
        try:
            return bool(parser.can_fetch(user_agent, url))
        except Exception:
            return True


class WebResearchAgent:
    def __init__(self) -> None:
        self.limiter = RateLimiter(0.5)
        self.robots = RobotsChecker()
        self.consistency = ConsistencyChecker()

    async def research(
        self,
        query: str,
        context_hints: Optional[List[str]] = None,
        max_sources: int = 10,
        timeout_sec: float = 2.0,
        seed_domains: Optional[List[str]] = None,
        seed_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """단일 쿼리에 대한 리서치 수행.

        Returns: {"results": List[WebSearchResult], "consistency": ConsistencyReport, "search_latencies": Dict[str,int]}
        """
        q = (query or "").strip()
        hints = list(context_hints or [])

        log_event(
            "web.search_start",
            {
                "session_id": None,
                "query": q,
                "sources": ["google_news_rss", "reddit", "cse?"],
            },
        )

        t0 = time.time()

        # 0) 캐시 조회(있으면 즉시 반환)
        try:
            import hashlib, json  # 표준 라이브러리

            try:
                import redis  # type: ignore
            except Exception:
                redis = None  # type: ignore
            if redis is not None:
                r = redis.Redis.from_url(get_settings().REDIS_URL)
                key = f"web_search:{hashlib.md5((q + str(max_sources)).encode()).hexdigest()}"
                raw = r.get(key)
                if raw:
                    cached = json.loads(raw)
                    return cached
        except Exception:
            pass
        # 1) 소스 병렬 검색
        tasks = [
            asyncio.create_task(
                self._search_google_news_rss(q, limit=max_sources // 2 or 5)
            ),
            asyncio.create_task(self._search_reddit(q, limit=max_sources // 2 or 5)),
        ]
        # CSE 활성 시: 일반 검색 + seed_domains별 site: 제한 검색
        try:
            tasks.append(asyncio.create_task(self._search_cse(q, limit=max_sources)))
        except Exception:
            pass
        try:
            for dom in list(seed_domains or [])[:5]:
                tasks.append(
                    asyncio.create_task(self._search_cse(q, limit=5, site=dom))
                )
        except Exception:
            pass
        # 도메인별 네이티브 소스(가능 시 추가, 실패는 무시)
        try:
            tasks.append(asyncio.create_task(self._search_arxiv_native(q, limit=5)))
        except Exception:
            pass
        try:
            tasks.append(asyncio.create_task(self._search_hf_hub(q, limit=5)))
        except Exception:
            pass
        try:
            tasks.append(asyncio.create_task(self._search_openai_blog(q, limit=5)))
        except Exception:
            pass
        done, pending = await asyncio.wait(tasks, timeout=timeout_sec)

        raw_items: List[Dict[str, Any]] = []
        latencies: Dict[str, int] = {}
        for t in done:
            try:
                src_name, took_ms, items = await t
                raw_items.extend(items)
                latencies[src_name] = took_ms
            except Exception:
                continue
        for t in pending:
            t.cancel()

        # 2) 콘텐츠 추출(병렬)
        # 중복 URL 제거
        uniq = {}
        for it in raw_items:
            u = it.get("url")
            if u and u not in uniq:
                uniq[u] = it
        candidates = list(uniq.values())[:max_sources]
        extracted = await self._extract_many(candidates)

        # 3) TrustScore 계산 및 필터링
        scored: List[WebSearchResult] = []
        all_for_cv = [e for e in extracted if e.get("content")]  # 교차검증 대상
        for item in extracted:
            if not item.get("content"):
                continue
            score = calculate_trust_score(
                item,
                all_for_cv,
                key_phrases=[q],  # 핵심 키구문 최소화(추후 개선 가능)
                community_mentions={},
            )
            if score >= 0.7:
                wi: WebSearchResult = WebSearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    excerpt=item.get("excerpt", ""),
                    content=item.get("content", ""),
                    trust_score=float(score),
                    domain=item.get("domain", ""),
                    published_at=item.get("published_at"),
                    source_type=item.get("source_type", "web"),
                    cross_validated=True,
                    evidence_ids=[],
                )
                scored.append(wi)

        # 4) 일관성 점수
        consistency_score = self.consistency.calculate_consistency(scored)
        report: ConsistencyReport = ConsistencyReport(
            overall_score=float(consistency_score),
            cluster_count=0,  # 간단 리포트(상세 클러스터 메타는 추후 확장)
            noise_count=0,
            largest_cluster_size=0,
        )

        # (옵션) 재랭킹: BM25/임베딩/크로스 인코더 가용 시 활용
        try:
            from backend.proactive.reranker import rerank_results  # type: ignore

            scored = rerank_results(q, scored, top_k=min(len(scored), max_sources))
        except Exception:
            pass

        took_ms_all = int((time.time() - t0) * 1000)
        log_event(
            "web.search_complete",
            {
                "session_id": None,
                "results_count": len(scored),
                "trust_scores": {r["url"]: r["trust_score"] for r in scored[:10]},
                "consistency_score": consistency_score,
                "took_ms": took_ms_all,
            },
        )
        try:
            if emit_metric is not None:
                emit_metric("web.results.count", float(len(scored)), {"agent": "web"})
                emit_metric(
                    "web.consistency", float(consistency_score), {"agent": "web"}
                )
        except Exception:
            pass

        # 신뢰도 기준 내림차순(재랭킹 결과가 이미 반영되었어도 신뢰도 우선 정렬 유지)
        scored.sort(key=lambda x: float(x.get("trust_score", 0.0)), reverse=True)
        result_blob = {
            "results": scored,
            "consistency": report,
            "search_latencies": latencies,
        }

        # 5) 캐시 저장(검색 결과 TTL)
        try:
            import hashlib, json  # 표준 라이브러리

            try:
                import redis  # type: ignore
            except Exception:
                redis = None  # type: ignore
            if redis is not None:
                r = redis.Redis.from_url(get_settings().REDIS_URL)
                key = f"web_search:{hashlib.md5((q + str(max_sources)).encode()).hexdigest()}"
                ttl = int(get_settings().SEARCH_TTL_SEC)
                r.setex(key, ttl, json.dumps(result_blob, ensure_ascii=False))
        except Exception:
            pass

        return result_blob

    # ------------------------------
    # 내부 구현
    # ------------------------------
    async def _search_google_news_rss(
        self, query: str, limit: int = 5
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Google News RSS로 뉴스 검색(키 기반 API 없이 동작)."""
        base = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=ko&gl=KR&ceid=KR:ko"
        t0 = time.time()
        try:
            import feedparser  # type: ignore

            feed = await asyncio.to_thread(feedparser.parse, base)
            items: List[Dict[str, Any]] = []
            for e in feed.entries[:limit]:
                url = e.get("link") or ""
                items.append(
                    {
                        "url": url,
                        "title": e.get("title") or "",
                        "excerpt": e.get("summary") or "",
                        "domain": urlparse(url).netloc,
                        "published_at": self._entry_published_iso(e),
                        "source_type": "news",
                    }
                )
            took_ms = int((time.time() - t0) * 1000)
            return ("google", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("google", took_ms, [])

    async def _search_reddit(
        self, query: str, limit: int = 5
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        t0 = time.time()
        try:
            import praw  # type: ignore

            _s = get_settings()
            cid = _s.REDDIT_CLIENT_ID
            sec = _s.REDDIT_CLIENT_SECRET
            ua = _s.REDDIT_USER_AGENT or "ProactiveAgent/1.0 (research)"
            if not cid or not sec:
                took_ms = int((time.time() - t0) * 1000)
                return ("reddit", took_ms, [])

            reddit = praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)
            items: List[Dict[str, Any]] = []
            for s in reddit.subreddit("all").search(query, limit=limit):
                url = f"https://www.reddit.com{s.permalink}"
                items.append(
                    {
                        "url": url,
                        "title": str(getattr(s, "title", "") or ""),
                        "excerpt": str(getattr(s, "selftext", "") or "")[:280],
                        "domain": urlparse(url).netloc,
                        "published_at": self._utc_iso(getattr(s, "created_utc", None)),
                        "source_type": "community",
                    }
                )
            took_ms = int((time.time() - t0) * 1000)
            return ("reddit", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("reddit", took_ms, [])

    async def _search_arxiv_native(
        self, query: str, limit: int = 5
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """arXiv API(feedparser) 사용. source_type='paper' 부여"""
        t0 = time.time()
        try:
            import re
            import feedparser  # type: ignore

            en = " ".join(re.findall(r"[A-Za-z0-9:+#\\-/]+", query or ""))
            url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(en)}&max_results={limit}"
            feed = await asyncio.to_thread(feedparser.parse, url)
            items: List[Dict[str, Any]] = []
            for e in feed.entries[:limit]:
                items.append(
                    {
                        "url": e.get("link") or "",
                        "title": e.get("title") or "",
                        "excerpt": e.get("summary") or "",
                        "domain": "arxiv.org",
                        "published_at": None,
                        "source_type": "paper",
                    }
                )
            took_ms = int((time.time() - t0) * 1000)
            return ("arxiv", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("arxiv", took_ms, [])

    async def _search_hf_hub(
        self, query: str, limit: int = 5
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """HuggingFace Hub 모델 검색(API 키 불요)."""
        t0 = time.time()
        try:
            from backend.utils.http_client import get_async_client
            client = get_async_client()
            r = await client.get(
                f"https://huggingface.co/api/models?search={quote_plus(query)}",
                timeout=httpx.Timeout(10.0),
            )
                arr = r.json() if r.status_code == 200 else []
                items: List[Dict[str, Any]] = []
                for m in arr[:limit]:
                    mid = m.get("modelId")
                    if not mid:
                        continue
                    items.append(
                        {
                            "url": f"https://huggingface.co/{mid}",
                            "title": mid,
                            "excerpt": (m.get("pipeline_tag") or "")[:240],
                            "domain": "huggingface.co",
                            "published_at": None,
                            "source_type": "official",
                        }
                    )
                took_ms = int((time.time() - t0) * 1000)
                return ("hf", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("hf", took_ms, [])

    async def _search_openai_blog(
        self, query: str, limit: int = 5
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """OpenAI 블로그 RSS 파싱(키 불요)."""
        t0 = time.time()
        try:
            import feedparser  # type: ignore

            url = "https://openai.com/blog/rss"
            feed = await asyncio.to_thread(feedparser.parse, url)
            items: List[Dict[str, Any]] = []
            for e in feed.entries[:limit]:
                items.append(
                    {
                        "url": e.get("link") or "",
                        "title": e.get("title") or "",
                        "excerpt": e.get("summary") or "",
                        "domain": "openai.com",
                        "published_at": None,
                        "source_type": "official",
                    }
                )
            took_ms = int((time.time() - t0) * 1000)
            return ("openai_blog", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("openai_blog", took_ms, [])

    async def _search_cse(
        self, query: str, limit: int = 10, site: Optional[str] = None
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """
        Google CSE(JSON API) 검색.
        - 키/엔진 미설정 시 빈 결과 반환(안전 폴백).
        - 최신성 우선(dateRestrict="d7").
        """
        t0 = time.time()
        try:
            from backend.search_engine.cse_client import CSEClient  # type: ignore
            from backend.config import get_settings as _gs  # 지연 임포트

            s = _gs()
            client = CSEClient(
                ttl_sec=int(getattr(s, "SEARCH_TTL_SEC", 120)),
                qps=2.0,
                redis_url=getattr(s, "REDIS_URL", None),
            )
            items = []
            if client.enabled:
                items = await client.search(
                    query=query,
                    limit=min(10, max(1, int(limit))),
                    site=site,
                    date_restrict="d7",
                )
            took_ms = int((time.time() - t0) * 1000)
            return ("cse", took_ms, items)
        except Exception:
            took_ms = int((time.time() - t0) * 1000)
            return ("cse", took_ms, [])

    async def _extract_many(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(6)

        async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
            url = item.get("url") or ""
            if not url:
                return {**item, "content": ""}
            domain = urlparse(url).netloc
            await self.limiter.wait(domain)
            ua = _random_user_agent()
            allowed = await self.robots.can_fetch(url, ua)
            if not allowed:
                log_event(
                    "web.robots_violation_avoided",
                    {"url": url, "domain": domain, "user_agent": ua},
                )
                return {**item, "content": ""}
            async with sem:
                content = await self._extract_content(url, ua)
                return {**item, "content": content}

        return await asyncio.gather(*[_one(x) for x in items])

    async def _extract_content(self, url: str, user_agent: str) -> str:
        # 1) trafilatura
        try:
            import trafilatura  # type: ignore

            html = await asyncio.to_thread(trafilatura.fetch_url, url)
            if html:
                text = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    include_comments=False,
                    include_tables=False,
                )
                if text:
                    return text.strip()
        except Exception:
            pass

        # 2) readability 폴백
        try:
            from readability import Document  # type: ignore
            from backend.utils.http_client import get_async_client

            client = get_async_client()
            r = await client.get(
                url, headers={"User-Agent": user_agent}, timeout=httpx.Timeout(10.0)
            )
                if r.status_code == 200:
                    doc = Document(r.text)
                    summary_html = doc.summary(html_partial=True)
                    base_text = self._strip_html(summary_html).strip()
                    # 2.1) 사이트 어댑터 적용(가능 시)
                    try:
                        from backend.proactive.site_adapters import extract_from_html  # type: ignore

                        adapted = extract_from_html(url, r.text) or ""
                        # 어댑터 결과가 더 풍부하면 교체
                        if len(adapted) > max(200, len(base_text)):
                            return adapted.strip()
                    except Exception:
                        pass
                    return base_text
        except Exception:
            pass

        return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        try:
            # 매우 간단한 제거(정확도보다는 안정 위주)
            import re

            txt = re.sub(r"<script[\s\S]*?</script>", " ", html or "", flags=re.I)
            txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt)
            return txt
        except Exception:
            return html or ""

    @staticmethod
    def _entry_published_iso(entry: Dict[str, Any]) -> Optional[str]:
        try:
            import datetime as _dt

            pp = entry.get("published_parsed")
            if not pp:
                return None
            dt = _dt.datetime(*pp[:6], tzinfo=_dt.timezone.utc)
            return dt.isoformat()
        except Exception:
            return None

    @staticmethod
    def _utc_iso(ts: Any) -> Optional[str]:
        try:
            import datetime as _dt

            if ts is None:
                return None
            dt = _dt.datetime.utcfromtimestamp(float(ts)).replace(
                tzinfo=_dt.timezone.utc
            )
            return dt.isoformat()
        except Exception:
            return None


__all__ = ["WebResearchAgent"]
