# 범용 모바일 챗봇(Flutter + FastAPI) 아키텍처 

## 전체 구조도 및 역할

### 프로젝트 루트
- **`requirements.txt`**: 백엔드 Python 의존성 목록. 설치: `pip install -r requirements.txt`
- **`docker/`**: Redis, Milvus(ETCD/MinIO 포함), MCP 서버(Node) 구성을 위한 `docker-compose.yml` 제공
- **`models/`**: 로컬 임베딩/NER 모델(예: Sentence-Transformers 기반 Gemma, 한국어 NER) 파일 저장소
- **`data/`**: 샘플 데이터(rag/web/conv) 텍스트 파일
- **`mcp_server/`**: 네이버 검색을 프록시하는 MCP(모델 컨텍스트 프로토콜) Node 서버
- **`naver-search-mcp/`**: TypeScript 기반의 대안 MCP 구현(선택). 기본 사용은 `mcp_server/`
- **`flutter_app/timely_agent/`**: Flutter 모바일 앱. 웹소켓/푸시 연동 클라이언트

---

### backend/ (FastAPI 서버, 핵심 비즈니스 로직)
- **`backend/app.py`**: 메인 애플리케이션 파일
  - 웹소켓: `GET /ws/{session_id}`
    - 실시간 채팅 스트리밍. 사용자 입력 수신 → 라우팅/증거수집(Evidence) → 답변 스트림 전송
  - 헬스체크: `GET /`
  - 내부 유틸 엔드포인트(서버 간 호출/테스트용)
    - `POST /internal/rag/retrieve` — RAG 컨텍스트 문자열 반환
    - `POST /internal/mobile/context` — Firestore에서 오늘 일정/최근 위치 요약 문자열 생성
    - `POST /internal/evidence/bundle` — web/rag 증거를 묶은 Evidence 번들 반환(캐시 포함)
    - `GET  /internal/directives/{session_id}/compiled` — 지시문 시스템 프롬프트(컴파일 결과) 조회
  - 프로액티브 트리거(테스트용): `POST /proactive/trigger/{user_id}` — 특정 유저에 대해 즉시 푸시 시도
  - 내부 핵심 함수 예시
    - `build_mobile_ctx(user_id)` — Firestore에서 오늘 일정/최근 위치를 요약(스레드 오프로딩)
    - `filter_semantic_mismatch(user_input, rag_ctx)` — LLM으로 RAG-질의 의미 불일치 필터(짧은 타임박스)
    - `filter_web_ctx(user_input, web_ctx)` — LLM으로 웹 컨텍스트 불일치 필터(짧은 타임박스)
  - 주요 환경 변수(발췌)
    - `OPENAI_API_KEY`, `LLM_MODEL`, `THINKING_MODEL`
    - `EMBEDDING_MODEL`, `MILVUS_HOST`, `MILVUS_PORT`, `REDIS_URL`
    - `FIRESTORE_ENABLE`, `FIRESTORE_USERS_COLL`, `FIRESTORE_EVENTS_SUB`
    - `MCP_SERVER_URL`, `CLIENT_ID`, `CLIENT_SECRET`
    - 타임아웃/임계치: `TIMEOUT_RAG`, `TIMEOUT_WEB`, `REWRITE_TIMEOUT_S`, `AMBIGUITY_BAND` 등
  - 주의사항
    - Firestore 사용 시 Windows에서 `GOOGLE_APPLICATION_CREDENTIALS` 파일 경로가 유효해야 함
    - Milvus 임베딩 차원/메트릭과 RAG 설정이 일치해야 검색 품질/오류 방지
    - 스트리밍 응답은 plain 텍스트. 필요 시 앱에서 특정 프리픽스(`[탐색 중]`, `[검토]`)를 분기 처리

- `backend/rag/` (RAG: 벡터 검색/임베딩)
  - `config.py`: 임베딩/밀버스 환경 변수→상수화. 임베딩 백엔드 전환(openai|gemma) 시 컬렉션 버전 분리(`v3|v4`)
    - 핵심 ENV: `EMBEDDING_BACKEND`, `GEMMA_MODEL_PATH`, `EMBEDDING_DIM`, `MILVUS_HOST/PORT`, `MILVUS_METRIC`
  - `embeddings.py`: 임베딩 백엔드(OpenAI 또는 로컬 Gemma) 지연 초기화 + 캐시(`lru_cache`)
    - Gemma 사용 시 `sentence-transformers` 필수 설치(requirements 반영 필요)
  - `milvus.py`: 컬렉션 보장/인덱스 생성/로드(`ensure_collections`) — dim 불일치 시 명시적 오류 발생
  - `retrieval.py`: RAG 검색 → 프로필/로그 문맥을 단순 문자열로 구축(스코어/임계치 반영)
  - `utils.py`: Milvus hit 유사도 스키마(COSINE은 `1 - distance`)
  - `refs.py`: web/rag 컨텍스트로부터 참조 포인터를 Milvus/Redis에 경량 업서트(추후 회고/추적용)
  - 주의사항
    - 컬렉션 dim이 ENV상의 `EMBEDDING_DIM`과 다르면 런타임 예외 → ENV와 모델 차원 동기화 필수
    - Gemma 전환 시 `RAG_COLL_VER`가 자동으로 v4로 분기되어 기존(v3) 데이터와 충돌 방지

- `backend/search_engine/` (웹검색 MCP 통합)
  - `client.py`: MCP 프록시/네이버 직접호출 클라이언트(httpx)
  - `router.py`: 의도 기반 엔드포인트 선택(키워드 폴백 + 임베딩 기반 분기)
  - `formatter.py`: 네이버 items → 3줄 블록(이름/설명/링크) 문자열 변환
  - `service.py`: `build_web_context()` — TTL 캐시(기본 120s), MCP 기본→대체 URL 폴백→직접 호출 2단 폴백
    - ENV: `SEARCH_TTL_SEC`, `SEARCH_USE_MCP_BLOCKS|USE_MCP_FORMATS`, `CLIENT_ID`, `CLIENT_SECRET`
  - 주의사항
    - Docker 네트워크 환경에서 `MCP_SERVER_URL`이 `http://mcp:5000`인지 확인(호스트/컨테이너 경로 스위칭 지원)
    - 네이버 직접 호출은 `CLIENT_ID/SECRET` 없으면 비활성. 429/인증 실페 대비 타임박스 짧게 유지

- `backend/evidence/builder.py` (증거 번들)
  - web/rag 컨텍스트를 묶어 EvidenceBundle을 구성, 3분 TTL 인메모리 캐시
  - STWM 앵커(최근 위치/행위/대상/주제)와의 단순 매칭 점수로 웹 블록 재정렬(실패 시 무시)
  - 반환: `(EvidenceBundle, web_ctx_blocks, rag_ctx_blocks)` — 상위 레이어에서 바로 사용 가능

- `backend/memory/` (단기/중기 메모리)
  - `stwm.py`: STWM(10분 TTL, 최대 30개), 규칙/NER 기반 앵커 추출(LLM 금지). ko-ner 로컬 모델 자동 로드(선택)
  - `turns.py`: 턴 버퍼/요약(요약 트리거: 토큰 수/턴 수/주제 이동 코사인). LLM JSON 스키마 응답 사용
  - `selector.py`: 요약 선택(BM25+임베딩 혼합 스코어) — 쿼리와 가장 관련 높은 N개를 선택
  - 주의사항
    - ko-ner 모델이 없으면 NER은 자동 비활성(빈 결과). 기능 저하일 뿐 오류 아님
    - 요약은 비용 절감을 위해 토큰 상한/EMA 등 가드가 있음

- `backend/directives/` (사용자 취향 지시문/페르소나/신호)
  - `agent.py`: 대화 로그에서 고정 취향 지시문(JSON), 휴리스틱 신호, 페르소나(BigFive/선택적 MBTI) 추출
    - `OPENAI_API_KEY` 없으면 즉시 런타임 오류를 던져 조기 실패(환경 설정 필수)
  - `schema.py`: JSON 스키마 정의(Directives/Signals/Persona/Report)
  - `store.py`: Redis 저장/TTL, active 사용자 세트 관리, 컴파일된 시스템 프롬프트 캐시
  - `compiler.py`: 시스템 프롬프트로 주입할 JSON을 최소화해서 컴파일(토큰 절약 규칙 포함)
  - `validator.py`: LLM 검증 그래프(보안/정책/스타일 최소 보정). 승인 실패 시 fixed안 반영
  - `pipeline.py`: 비동기 워커 큐(디바운스/락/쿨다운), 지시문 업데이트 스케줄링
  - `scheduler.py`: 매일 KST 03:00 배치 업데이트 예약(활성 사용자 기준)
  - `signals_ingest.py`: Firestore 기반 모바일 요약(피크 시간대, 주소 상위, 일정 밀도 등)
  - 주요 ENV: `DIR_*` 접두(큐/워커/쿨다운/락), `REDIS_URL`
  - 주의사항
    - 멀티 인스턴스 시 Redis 락(`DIR_USE_REDIS_LOCK=1`) 사용 권장
    - 과격한 스타일 변경 방지: 변경 예산/쿨다운/EMA로 스로틀링

- `backend/proactive/` (선제 푸시)
  - `planner.py`: LangGraph 파이프라인 — 최근 이벤트/RAG를 요약해 질문 후보 Top-N 산출
  - `agent.py`: seed별 컨텍스트(RAG/Web/모바일) 구성 → LLM 초안 → 가드/보정 → 후보 생성/선별/전송
  - `notifier.py`: FCM 전송(GCP 서비스 계정으로 OAuth 토큰 획득)
  - `scheduler.py`: 주기 실행(기본 30분, 동시성 제한)
  - 주요 ENV: `PROACTIVE_*`, `GCP_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`
  - 주의사항
    - 야간/이른 아침 조용 시간 게이트, 일일 레이트리밋(기본 3회)
    - 토큰 목록 조회 경로(users/{uid}/fcm_tokens | devices | 루트 필드) 모두 시도

- `backend/policy/state.py` (보안/운영 정책)
  - PII 마스킹 정규식, 보관 기간(요약 개수/캐시 분/TTL 분기), 운영 임계치(bm25/emb 가중 등)

- `backend/generation/` (출력 래핑/컴포지션)
  - `composer.py`: Evidence가 있을 때만 사실형 답변을 구성(웹 블록 직접 나열, RAG는 상위 레이어 요약)
  - `wrapper.py`: 간단 말투/출력 포매팅(과도한 변환 금지, JSON/코드 블록 중립화)

- `backend/ingest/main.py` (모바일 컨텍스트 수집 API)
  - 통합 엔드포인트(Flutter → Firestore 저장)
    - `POST /api/v1/location` — 위경도 수신→카카오 역지오코딩→주소 포함 저장(ENV `KAKAO_REST_API_KEY` 필요)
    - `POST /api/v1/calendar` — 다가오는 일정 리스트 저장
    - `POST /api/v1/preferences` — 자유 키-값 선호 설정 저장
  - 주의사항
    - Firestore 클라이언트 초기화 실패 시 저장 불가(환경 구성 필요). 에러 메시지로 가이드 출력

- 기타
  - `backend/rewrite/log.py` — 질의 재작성 로그 메모리 저장
  - `backend/planner/logger.py` — 라우팅/의사결정 로깅 DTO
  - `backend/test/*.ipynb` — 기능별 노트북(참고/실험)

---

### mcp_server/ (네이버 검색 MCP, Node.js)
- `index.js`
  - 엔드포인트
    - `GET  /health` — 헬스체크
    - `GET  /metrics` — 캐시/메트릭 노출
    - `POST /mcp/context` — 데모용
    - `POST /mcp/search/naver` — 네이버 검색 프록시(local/news/webkr 자동 선택, 빈 결과 시 webkr 폴백)
  - 캐시: 메모리 TTL(기본 120s), 간단 메트릭 수집
  - ENV: `CLIENT_ID`, `CLIENT_SECRET`, `PORT`, `SEARCH_TTL_MS`
  - 주의사항: 컨테이너 바인딩 `0.0.0.0`, Docker Compose에 포함되어 백엔드에서 `http://mcp:5000`로 접근

### docker/
- `docker-compose.yml`
  - `redis:7` — 세션/지시문/캐시 저장
  - `etcd`, `minio`, `milvusdb/milvus:v2.4.0` — Milvus 스탠드얼론 구성(포트 19530/9091)
  - `mcp-server` — 네이버 검색 MCP(Node) 빌드/실행
  - 볼륨: `./volumes_milvus/*`에 영속 데이터 저장(etcd/minio/milvus)
  - 주의사항: Windows 경로 매핑 시 권한/경로 구분자 확인, 포트 충돌 여부 체크

### models/
- `EmbeddingGemma/` — Sentence-Transformers 형식 로컬 임베딩 모델(기본 dim=768)
- `ko-ner/` — 한국어 NER 토크나이저/모델(선택). 없으면 NER 기능 자동 비활성
- `router_kor_electra_small/` — (참고) 라우팅용 소형 분류기 아티팩트

### data/
- `conv_data.txt`, `rag_data.txt`, `web_data.txt` — 샘플 데이터. 개발/테스트 참고용

### flutter_app/timely_agent/
- 실시간 채팅(WebSocket)과 FCM 푸시 수신. 연결 정보
  - 웹소켓 URL: `ws://<backend-host>:8000/ws/{session_id}`
  - 수신 텍스트: 스트리밍 line-by-line(일부 프리픽스 존재)
  - 푸시 클릭: data에 포함된 `session_id`, `question`, `kind`를 이용해 새 세션 화면으로 이동

---

## 엔드포인트 카탈로그

### FastAPI (backend/app.py)
- `GET /` — 헬스체크
- `GET /ws/{session_id}` — 채팅 웹소켓 스트림
- `POST /internal/rag/retrieve` — RAG 컨텍스트 반환
- `POST /internal/mobile/context` — 오늘 일정/최근 위치 요약
- `POST /internal/evidence/bundle` — Evidence 번들(web/rag)
- `GET  /internal/directives/{session_id}/compiled` — 컴파일 지시문 프롬프트
- `POST /proactive/trigger/{user_id}` — 프로액티브 테스트 트리거

### Ingest API (backend/ingest/main.py)
- `POST /api/v1/location`
- `POST /api/v1/calendar`
- `POST /api/v1/preferences`

### MCP(Node) (mcp_server/index.js)
- `GET /health`, `GET /metrics`, `POST /mcp/context`, `POST /mcp/search/naver`

---

## 환경 변수(필수/권장)

### 공통/플랫폼
- **OpenAI/LLM**: `OPENAI_API_KEY`, `LLM_MODEL`, `THINKING_MODEL`
- **임베딩/RAG**: `EMBEDDING_MODEL`, `EMBEDDING_BACKEND`(openai|gemma), `GEMMA_MODEL_PATH`, `EMBEDDING_DIM`, `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_METRIC`
- **Redis**: `REDIS_URL`
- **MCP/네이버**: `MCP_SERVER_URL`, `CLIENT_ID`, `CLIENT_SECRET`, `SEARCH_TTL_SEC`, `SEARCH_TTL_MS`, `SEARCH_USE_MCP_BLOCKS|USE_MCP_FORMATS`
- **Firestore/GCP**: `FIRESTORE_ENABLE`, `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT_ID`, `FIRESTORE_USERS_COLL`, `FIRESTORE_EVENTS_SUB`
- **프로액티브**: `PROACTIVE_INTERVAL_SEC`, `PROACTIVE_CONCURRENCY`, `PROACTIVE_DAILY_LIMIT`, `PROACTIVE_USE_MCP`
- **지시문 파이프라인**: `DIR_QUEUE_MAX`, `DIR_WORKERS`, `DIR_DEBOUNCE_TURNS`, `DIR_ENQ_GUARD_S`, `DIR_USE_REDIS_LOCK`, `DIR_LOCK_EX_S`, `DIR_CONF_THRESH`, `DIR_COOLDOWN_S`, `DIR_EMA_ALPHA`, `DIR_MAX_CHANGES`
- **운영 정책/튜닝**: `TIMEBOX_MS`, `CACHE_MINUTES`, `TOPK_SUMMARIES`, `BM25_WEIGHT`, `EMB_WEIGHT`, `TAU`, `DELTA`, `LOW_CONF_MARGIN`, `FASTPATH_MARGIN`, `STWM_TTL_MIN`, `STWM_MAX_SLOTS`, `STWM_FLUSH_BATCH`, `SUM_BUF_TOKENS`, `SUM_MIN_TURNS`, `SUM_TOPIC_COS`
- **기타**: `KAKAO_REST_API_KEY`(역지오코딩), `DEBUG_META`, `WS_DEBUG_META`

---

## 운영 가이드/주의사항(필독)

- **자격 증명**
  - OpenAI 키 미설정 시 지시문/요약 등 LLM 호출 경로가 즉시 실패. `OPENAI_API_KEY` 필수
  - Firestore 사용 시 `GOOGLE_APPLICATION_CREDENTIALS` JSON 경로와 `GCP_PROJECT_ID`가 유효해야 함
  - 네이버 검색 직접 호출은 `CLIENT_ID/CLIENT_SECRET` 필요. 없으면 MCP 경유만 사용됨

- **Milvus 차원/메트릭 정합성**
  - 컬렉션 생성 시 `EMBEDDING_DIM`과 모델 차원이 다르면 에러. 백엔드 전환(openai→gemma) 시 `RAG_COLL_VER`로 네임스페이스 분리됨

- **캐시/타임박스 전략**
  - 웹검색 TTL(기본 120s), Evidence(180s), Redis TTL(지시문 7d). 외부 호출은 짧은 타임아웃과 적은 재시도 유지

- **콘텍스트 안전성**
  - 답변은 web_ctx/rag_ctx가 있을 때 그 “안에서만” 인용. 추측 금지. 불일치 필터로 관련 없는 블록 제거

- **푸시 정책**
  - 조용 시간(22:30~07:30) 자동 차단, 일일 레이트리밋(기본 3회). 클릭 시 새 세션으로 분리 권장

- **Windows 환경 주의**
  - 경로 구분자/권한 이슈로 Docker 볼륨/자격 파일 경로를 절대경로로 지정 권장

---

## 트러블슈팅(자주 묻는 오류)

- "OPENAI_API_KEY not loaded" — `.env` 또는 시스템 ENV에 키 등록 후 재시작
- Milvus dim mismatch — `EMBEDDING_DIM`과 컬렉션 dim 불일치. 모델/ENV 동기화 후 컬렉션 재생성
- MCP 연결 실패 — `MCP_SERVER_URL` 확인. Docker 내부는 `http://mcp:5000`, 호스트는 `http://localhost:5000`
- Firestore 저장 실패 — 서비스 계정/프로젝트 권한 확인, `FIRESTORE_ENABLE=1`, 네트워크/프록시 점검
- 카카오 역지오코딩 401 — `KAKAO_REST_API_KEY` 미설정/오타. 콘솔에서 애플리케이션 키 확인

---

## TEST 시 

1) Docker로 인프라 기동(로컬 - 추후 EC2)
```bash
cd docker && docker compose up -d
```
2) MCP(Node)만 별도 실행 시
```bash
cd mcp_server && npm ci && npm run start
```
3) 백엔드 실행(FastAPI/Uvicorn)
```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
4) Flutter 앱에서 웹소켓 연결
```text
ws://<backend-host>:8000/ws/{session_id}
```


## 프로액티브 플로우(요약)

1) 스케줄러가 `get_active_users()`로 타겟 사용자 목록 취득  
2) 플래너가 최근 Firestore 이벤트/푸시로그 + Milvus(프로필/로그)를 요약 → 개인화된 “질문 후보 Top-N” 생성  
3) 에이전트가 seed마다 컨텍스트를 구성(RAG/Search/Mobile), LLM 초안 생성 → 안전/스타일 검증/보정  
4) FCM 푸시 전송(data에 `{"session_id": "<uid>", "question":"...","kind":"proactive"}`)  
5) Flutter에서 클릭 시 “새로운 대화 세션” 화면으로 진입(기존 세션과 분리)

- 주의: 프로액티브는 채팅과 **약간 별개 프로세스**로 동작. RAG/검색 모듈을 “툴”처럼 호출하여 컨텍스트를 만든 후, 푸시만 보냄.

## Flutter 연동 관련하여 

- WebSocket
  - URL: `ws://<backend-host>:8000/ws/{session_id}`
  - 송신: 사용자 입력(Plain text)
  - 수신: 스트리밍 텍스트(plain). 일부 라인 프리픽스:
    - `[탐색 중] ...` (검색 타임박스 워밍업)
    - `[검토] ...` (사후 경량 검토)
    - `[debug_meta] {...}` (옵션)
  - 권장: 앱 내부에서 이 라인을 구분 처리하거나, 차후 JSON 프로토콜로 승격

- FCM Push
  - data 예시: `{"session_id":"u123","question":"퇴근길 카페?","kind":"proactive"}`
  - 클릭 액션: “새 세션” 화면으로 이동(기존 채팅 세션과 분리 권장)


## 성능 최적화 체크리스트

- **타임박스/재시도**: 외부 호출은 `timeout + 적은 재시도`(기존 유지), LLM `response_format`은 지원 모델에서만 강제
- **임베딩 프리워밍**: 재작성된 질의어에 대해 `embed_query_cached`를 백그라운드 프리워밍(완)
- **검색 LRU/TTL 캐시**: 동일 질의 단시간 반복 시 `search_engine.build_web_context`에 120s TTL 캐시 권장
- **증거 캐시 범위**: Evidence 캐시는 사용자 단위로 분리
- **병렬화 레벨**: 프로액티브 컨텍스트/채팅 내 증거 수집은 가능한 범위에서 `gather`로 병렬화(완)
- **메모리 관리**: Redis 3000 토큰 임계 초과 시 요약 교체(비동기), Milvus 스냅샷은 근사중복/신규성 게이트로 비용 절감
- **모델 인스턴스 공유**: Embeddings/LLM 인스턴스는 지연 초기화 + 모듈 단일 인스턴스 재사용

---

## Flutter 프로토콜

- 스트리밍 메시지를 차후 JSON으로 정규화(앱 오프라인 상에서 파싱 단순화)
```json
{
  "type": "chunk|final|notice|debug",
  "text": "...",
  "meta": { "route": "rag|web|conv", "prob": {"rag":0.73,"web":0.22} }
}
```

- 푸시 클릭 → new session 생성 규칙
  - 새 session_id = `user_id:ts` 등으로 생성하여 기존 세션과 독립
  - 최초 발화로 `question` seed를 자동 입력(선상호작용 컨티뉴로 )

---