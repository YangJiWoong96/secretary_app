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

// 헬스체크
app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

// ----------------------------
// 유틸리티 함수들
// ----------------------------

// HTML 태그 제거
function stripBold(s) {
  if (typeof s !== 'string') return '';
  return s.replace(/<\/?b>/g, '').trim();
}

// 날짜 파싱
function parseDate(s) {
  if (!s) return 0;
  const d = new Date(s);
  return isNaN(d.getTime()) ? 0 : d.getTime();
}

// 날짜 필터 적용
function applyDateFilter(items, freshness_days) {
  const days = Number(freshness_days || process.env.DEFAULT_FRESHNESS_DAYS || 365);
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  
  if (!Array.isArray(items) || items.length === 0) return items;
  
  return items.filter(it => {
    const dt = parseDate(it.pubDate || it.datetime || it.date || it.postdate);
    return dt === 0 || dt >= cutoff;
  });
}

// ----------------------------
// 검색 도구 함수들
// ----------------------------

/**
 * 검색 도구: 웹 문서 (webkr)
 */
async function searchWebkr(query, display = 5, freshness_days) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/webkr.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  let items = data.items || [];
  items = applyDateFilter(items, freshness_days);
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.originallink || it.link || '';
    return link ? [title, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'webkr', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 뉴스 (news)
 */
async function searchNews(query, display = 5, freshness_days) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/news.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  let items = data.items || [];
  items = applyDateFilter(items, freshness_days);
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.originallink || it.link || '';
    return link ? [title, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'news', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 블로그 (blog)
 */
async function searchBlog(query, display = 5, freshness_days) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/blog.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  let items = data.items || [];
  items = applyDateFilter(items, freshness_days);
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    const bloggername = it.bloggername ? ` (by ${it.bloggername})` : '';
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.link || '';
    return link ? [title + bloggername, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'blog', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 카페글 (cafearticle)
 */
async function searchCafearticle(query, display = 5, freshness_days) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/cafearticle.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  let items = data.items || [];
  items = applyDateFilter(items, freshness_days);
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    const cafename = it.cafename ? ` [${it.cafename}]` : '';
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.link || '';
    return link ? [title + cafename, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'cafearticle', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 쇼핑 (shop)
 */
async function searchShop(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/shop.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    const lprice = it.lprice ? `최저가: ${Number(it.lprice).toLocaleString()}원` : '';
    const link = it.link || '';
    return link ? [title, lprice || '(가격 정보 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'shop', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 이미지 (image)
 */
async function searchImage(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/image.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    const link = it.link || '';
    const thumbnail = it.thumbnail || '';
    return thumbnail ? [title, thumbnail, link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'image', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 지식iN (kin)
 */
async function searchKin(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/kin.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.link || '';
    return link ? [title, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'kin', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 책 (book)
 */
async function searchBook(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/book.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    const author = it.author ? `저자: ${it.author}` : '';
    const link = it.link || '';
    return link ? [title, author || '(저자 정보 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'book', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 백과사전 (encyc)
 */
async function searchEncyc(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/encyc.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.link || '';
    return link ? [title, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'encyc', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 학술 논문 (academic)
 */
async function searchAcademic(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/doc.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || '(이름 없음)');
    let desc = stripBold(it.description || '');
    if (desc.length > 140) desc = desc.slice(0, 137).trimEnd() + '...';
    const link = it.link || '';
    return link ? [title, desc || '(설명 없음)', link].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'academic', status: resp.status, data, blocks };
}

/**
 * 검색 도구: 지역 장소 (local)
 */
async function searchLocal(query, display = 5) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const url = new URL('https://openapi.naver.com/v1/search/local.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', String(display));
  
  const resp = await fetch(url.toString(), {
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
    },
  });
  
  const data = await resp.json().catch(() => ({}));
  const items = data.items || [];
  
  const blocks = items.slice(0, display).map(it => {
    const title = stripBold(it.title || it.name || '(이름 없음)');
    const desc = stripBold(it.category || it.description || '(설명 없음)');
    const address = it.roadAddress || it.address || '';
    return address ? [title, desc, address].join('\n') : null;
  }).filter(Boolean).join('\n\n');
  
  return { kind: 'local', status: resp.status, data, blocks };
}

// ----------------------------
// DataLab 도구 함수들
// ----------------------------

/**
 * DataLab: 검색어 트렌드 분석
 */
async function datalabSearch(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/search', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  // 트렌드 데이터를 간단한 텍스트 블록으로 변환
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(키워드 없음)';
      const keywords = result.keywords ? result.keywords.join(', ') : '';
      const dataPoints = result.data ? result.data.length : 0;
      return `키워드: ${title}\n검색어: ${keywords}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_search', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 카테고리 트렌드 분석
 */
async function datalabShoppingCategory(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/keyword/age', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(카테고리 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `카테고리: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_category', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 기기별 트렌드
 */
async function datalabShoppingDevice(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/device', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(기기 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `기기: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_device', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 성별 트렌드
 */
async function datalabShoppingGender(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/gender', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(성별 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `성별: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_gender', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 연령별 트렌드
 */
async function datalabShoppingAge(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/age', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(연령 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `연령대: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_age', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 키워드 트렌드
 */
async function datalabShoppingKeywords(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/categories', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(키워드 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `키워드: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_keywords', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 키워드 기기별
 */
async function datalabShoppingKeywordDevice(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/keyword/device', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(키워드 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `키워드: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_keyword_device', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 키워드 성별
 */
async function datalabShoppingKeywordGender(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/keyword/gender', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(키워드 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `키워드: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_keyword_gender', status: resp.status, data, blocks };
}

/**
 * DataLab: 쇼핑 키워드 연령별
 */
async function datalabShoppingKeywordAge(body) {
  const clientId = process.env.CLIENT_ID;
  const clientSecret = process.env.CLIENT_SECRET;
  
  const resp = await fetch('https://openapi.naver.com/v1/datalab/shopping/category/keyword/age', {
    method: 'POST',
    headers: {
      'X-Naver-Client-Id': clientId,
      'X-Naver-Client-Secret': clientSecret,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  const data = await resp.json().catch(() => ({}));
  
  let blocks = '';
  if (data.results && Array.isArray(data.results)) {
    blocks = data.results.map(result => {
      const title = result.title || '(키워드 없음)';
      const dataPoints = result.data ? result.data.length : 0;
      return `키워드: ${title}\n데이터 포인트: ${dataPoints}개`;
    }).join('\n\n');
  }
  
  return { kind: 'datalab_shopping_keyword_age', status: resp.status, data, blocks };
}

/**
 * 카테고리 검색 (간단 구현 - 실제로는 카테고리 DB 필요)
 */
async function findCategory(query) {
  // 카테고리 매핑 (README의 빠른 참조 기반)
  const categoryMap = {
    '패션': '50000000',
    '의류': '50000000',
    '옷': '50000000',
    '화장품': '50000002',
    '뷰티': '50000002',
    '미용': '50000002',
    '디지털': '50000003',
    '전자제품': '50000003',
    '가전': '50000003',
    '스포츠': '50000004',
    '레저': '50000004',
    '운동': '50000004',
    '식품': '50000008',
    '음료': '50000008',
    '건강': '50000009',
    '의료': '50000009',
  };
  
  const queryLower = query.toLowerCase();
  const matches = [];
  
  for (const [keyword, code] of Object.entries(categoryMap)) {
    if (queryLower.includes(keyword)) {
      matches.push({ keyword, code });
    }
  }
  
  const blocks = matches.length > 0
    ? matches.map(m => `카테고리: ${m.keyword}\n코드: ${m.code}`).join('\n\n')
    : '일치하는 카테고리를 찾을 수 없습니다.';
  
  return { 
    kind: 'find_category', 
    status: 200, 
    data: { matches },
    blocks 
  };
}

/**
 * 현재 한국 시간 조회
 */
async function getCurrentKoreanTime() {
  const now = new Date();
  const kstOffset = 9 * 60; // KST는 UTC+9
  const kstTime = new Date(now.getTime() + (kstOffset - now.getTimezoneOffset()) * 60000);
  
  const year = kstTime.getFullYear();
  const month = String(kstTime.getMonth() + 1).padStart(2, '0');
  const day = String(kstTime.getDate()).padStart(2, '0');
  const hours = String(kstTime.getHours()).padStart(2, '0');
  const minutes = String(kstTime.getMinutes()).padStart(2, '0');
  const seconds = String(kstTime.getSeconds()).padStart(2, '0');
  
  const formatted = `${year}-${month}-${day} ${hours}:${minutes}:${seconds} KST`;
  
  const blocks = `현재 한국 시간(KST)\n${formatted}\n년: ${year}, 월: ${month}, 일: ${day}\n시: ${hours}, 분: ${minutes}, 초: ${seconds}`;
  
  return {
    kind: 'get_current_korean_time',
    status: 200,
    data: { kst: formatted, year, month, day, hours, minutes, seconds },
    blocks
  };
}

// ----------------------------
// 도구 맵 (디스패처)
// ----------------------------
const toolMap = {
  // 검색 도구
  'webkr': searchWebkr,
  'news': searchNews,
  'blog': searchBlog,
  'cafearticle': searchCafearticle,
  'shop': searchShop,
  'image': searchImage,
  'kin': searchKin,
  'book': searchBook,
  'encyc': searchEncyc,
  'academic': searchAcademic,
  'local': searchLocal,
  
  // DataLab 도구
  'datalab_search': datalabSearch,
  'datalab_shopping_category': datalabShoppingCategory,
  'datalab_shopping_device': datalabShoppingDevice,
  'datalab_shopping_gender': datalabShoppingGender,
  'datalab_shopping_age': datalabShoppingAge,
  'datalab_shopping_keywords': datalabShoppingKeywords,
  'datalab_shopping_keyword_device': datalabShoppingKeywordDevice,
  'datalab_shopping_keyword_gender': datalabShoppingKeywordGender,
  'datalab_shopping_keyword_age': datalabShoppingKeywordAge,
  
  // 유틸리티 도구
  'find_category': findCategory,
  'get_current_korean_time': getCurrentKoreanTime,
};

// ----------------------------
// 통합 MCP 엔드포인트
// ----------------------------
app.post('/mcp/search/naver', async (req, res) => {
  try {
    const { query, display = 5, endpoint, freshness_days, body: datalabBody } = req.body || {};
    const clientId = process.env.CLIENT_ID;
    const clientSecret = process.env.CLIENT_SECRET;
    
    if (!clientId || !clientSecret) {
      return res.status(400).json({ error: 'Naver credentials missing: set CLIENT_ID and CLIENT_SECRET' });
    }
    
    // 캐시 조회
    metrics.total += 1;
    const key = makeKey('/mcp/search/naver', req.body);
    const now = Date.now();
    const c = cache.get(key);
    if (c && (now - c.ts) <= CACHE_TTL_MS) {
      metrics.hits += 1;
      return res.status(200).json(c.data);
    }
    metrics.misses += 1;
    
    // 엔드포인트 결정
    let targetEndpoint = endpoint || 'webkr';
    
    // 도구 함수 선택
    const toolFunc = toolMap[targetEndpoint];
    if (!toolFunc) {
      return res.status(400).json({ error: `Unknown endpoint: ${targetEndpoint}` });
    }
    
    const t0 = Date.now();
    let result;
    
    // DataLab 도구는 body를 전달, 검색 도구는 query/display/freshness_days 전달
    if (targetEndpoint.startsWith('datalab_')) {
      if (!datalabBody) {
        return res.status(400).json({ error: 'DataLab tools require a body parameter' });
      }
      result = await toolFunc(datalabBody);
    } else if (targetEndpoint === 'find_category') {
      if (!query || String(query).trim().length < 1) {
        return res.status(400).json({ error: 'find_category requires a query' });
      }
      result = await toolFunc(query);
    } else if (targetEndpoint === 'get_current_korean_time') {
      result = await toolFunc();
    } else {
      // 일반 검색 도구
      if (!query || String(query).trim().length < 2) {
        return res.status(400).json({ error: 'invalid query' });
      }
      result = await toolFunc(query, display, freshness_days);
    }
    
    const took = Date.now() - t0;
    metrics.upstream_ms += took;
    
    const out = {
      schema_version: 'naver.search.v2',
      provider: 'naver',
      kind: result.kind,
      endpoint: targetEndpoint,
      status: result.status,
      took_ms: took,
      data: result.data,
      blocks: result.blocks,
    };
    
    console.log(`[mcp:naver:${targetEndpoint}] status=${result.status} took_ms=${took} q='${String(query || '').slice(0,60)}'`);
    
    cache.set(key, { ts: now, data: out });
    return res.status(200).json(out);
    
  } catch (e) {
    console.error('[mcp:naver] error', e);
    return res.status(500).json({ error: String(e) });
  }
});

// 레거시 호환성을 위한 기본 테스트 엔드포인트
app.post('/mcp/context', (req, res) => {
    console.log('✅ MCP 수신 데이터:', req.body);
    res.json({ status: 'success', message: 'MCP 컨텍스트 처리 완료' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => console.log(`🚀 MCP 서버 실행 (다중 도구): http://0.0.0.0:${PORT}`));
