const express = require('express');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const app = express();
app.use(express.json());  // JSON 바디 파싱 활성화

// ----------------------------
// 간단 TTL 캐시 및 메트릭
// ----------------------------
const CACHE_TTL_MS = Number(process.env.SEARCH_TTL_MS || 120000); // 120s
const cache = new Map(); // key -> {ts, data}
const metrics = {
  total: 0,
  hits: 0,
  misses: 0,
  upstream_ms: 0,
};

function makeKey(path, body) {
  return `${path}:${JSON.stringify(body || {})}`;
}

app.get('/metrics', (_req, res) => {
  res.json({ ...metrics, cache_size: cache.size });
});

// MCP 기본 테스트 엔드포인트
app.post('/mcp/context', (req, res) => {
    console.log('✅ MCP 수신 데이터:', req.body);
    res.json({ status: 'success', message: 'MCP 컨텍스트 처리 완료' });
});

// 헬스체크
app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

// 네이버 검색 통합: 의도 기반 엔드포인트 선택 (local/news/webkr)
function pickNaverEndpoint(query) {
  const t = String(query || '').trim();
  const has = (xs) => xs.some(k => t.includes(k));
  if (has(["맛집","카페","식당","근처","주변","주소","영업시간","위치","전화","리뷰","후기"])) {
    return { kind: 'local', url: 'https://openapi.naver.com/v1/search/local.json' };
  }
  if (has(["주가","종가","시세","환율","실적","공시","뉴스","속보","브리핑","증시","가격","코스피","코스닥","나스닥","다우","NASDAQ","S&P"])) {
    return { kind: 'news', url: 'https://openapi.naver.com/v1/search/news.json' };
  }
  return { kind: 'webkr', url: 'https://openapi.naver.com/v1/search/webkr.json' };
}

// 요청 -> 네이버 API 프록시
app.post('/mcp/search/naver', async (req, res) => {
  try {
    const { query, display = 5 } = req.body || {};
    const clientId = process.env.CLIENT_ID;
    const clientSecret = process.env.CLIENT_SECRET;
    if (!clientId || !clientSecret) {
      return res.status(400).json({ error: 'Naver credentials missing: set CLIENT_ID and CLIENT_SECRET' });
    }
    if (!query || String(query).trim().length < 2) {
      return res.status(400).json({ error: 'invalid query' });
    }
    // 캐시 조회
    metrics.total += 1;
    const key = makeKey('/mcp/search/naver', { query, display });
    const now = Date.now();
    const c = cache.get(key);
    if (c && (now - c.ts) <= CACHE_TTL_MS) {
      metrics.hits += 1;
      return res.status(200).json(c.data);
    }
    metrics.misses += 1;

    const picked = pickNaverEndpoint(query);
    const url = new URL(picked.url);
    url.searchParams.set('query', query);
    url.searchParams.set('display', String(display));
    const r0 = Date.now();
    const t0 = Date.now();
    let resp = await fetch(url.toString(), {
      headers: {
        'X-Naver-Client-Id': clientId,
        'X-Naver-Client-Secret': clientSecret,
      },
    });
    let took = Date.now() - r0;
    metrics.upstream_ms += (Date.now() - t0);
    let data = await resp.json().catch(() => ({}));
    let items = (data && data.items) || [];
    console.log(`[mcp:naver:${picked.kind}] status=${resp.status} took_ms=${took} q='${String(query).slice(0,60)}' items=${items.length}`);

    // 빈 결과 폴백: news/local -> webkr 재시도
    if (resp.status === 200 && (!items || items.length === 0) && picked.kind !== 'webkr') {
      const fallback = { kind: 'webkr', url: 'https://openapi.naver.com/v1/search/webkr.json' };
      const furl = new URL(fallback.url);
      furl.searchParams.set('query', query);
      furl.searchParams.set('display', String(display));
      const r1 = Date.now();
      const t1 = Date.now();
      const resp2 = await fetch(furl.toString(), {
        headers: {
          'X-Naver-Client-Id': clientId,
          'X-Naver-Client-Secret': clientSecret,
        },
      });
      took = Date.now() - r1;
      metrics.upstream_ms += (Date.now() - t1);
      data = await resp2.json().catch(() => ({}));
      console.log(`[mcp:naver:fallback->webkr] status=${resp2.status} took_ms=${took} q='${String(query).slice(0,60)}' items=${(data && data.items && data.items.length) || 0}`);
      const out = { kind: fallback.kind, data };
      cache.set(key, { ts: now, data: out });
      return res.status(200).json(out);
    }

    const out = { kind: picked.kind, data };
    cache.set(key, { ts: now, data: out });
    return res.status(200).json(out);
  } catch (e) {
    console.error('[mcp:naver] error', e);
    return res.status(500).json({ error: String(e) });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => console.log(`🚀 MCP 서버 실행: http://0.0.0.0:${PORT}`)); // 컨테이너 바인딩