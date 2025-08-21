### 초기 구상안
<img width="1564" height="575" alt="image" src="https://github.com/user-attachments/assets/36e8bc6d-4715-4634-addb-2b096951beb0" />

###  Flutter 기반 애플리케이션 

* public 제한 예정 
* RAG(Milvus -HNSW) Redis를 장/단기 메모리로 구현 
* 에이전트를 활용한 Operating in the Giant's Blind Spot
* 최종적으로는 Kubernetes (Helm chart) + AWS EC2 환경에 배포 예정

### 구조도
```
.
├─ .env
├─ backend/
│ ├─ app.py # FastAPI - wscat(스트리밍) 로컬용, main.py와 병합 후, AWS EC2 배포
│ ├─ requirements.txt
│ ├─ .gitignore # backend 전용 ignore
│ │
│ ├─ directives/ # 동적 시스템 프롬프트
│ │ ├─ init.py
│ │ ├─ agent.py
│ │ ├─ compiler.py
│ │ ├─ pipeline.py
│ │ ├─ policy.py
│ │ ├─ schema.py
│ │ └─ store.py
│ │
│ ├─ ingest/ # GCP Firestore - 데이터수집 - 카카오 맵 역지오코딩 처리 후, 저장
│ │ ├─ init.py
│ │ └─ main.py # AWS EC2
│ │
│ ├─ proactive/ # 선 상호작용 골격
│ │ ├─ notifier.py
│ │ ├─ planner.py
│ │ └─ scheduler.py
│ │
│ └─ router_kor_electra_small/ # 대화 소분류기 가중치 -
│ ├─ pytorch_model.bin
│ ├─ tokenizer.json
│ ├─ tokenizer_config.json
│ ├─ vocab.txt
│ ├─ config.json
│ ├─ training_args.bin
│ └─ checkpoint-128/
│
├─ test/ # 참조 X
│ ├─ baseline_test.ipynb
│ ├─ realtime_context_events.ipynb
│ ├─ search_engine.ipynb
│ └─ small_classifier_train_data.txt # electra_small # 모델 학습 데이터
│
├─ flutter_app/
│ └─ timely_agent/ # Flutter(frontend) 캘린더 & 백그라운드/포그라운드 위치 서비스
│
├─ mcp_server/ # MCP(Node) 서버
│
└─ docker/
├─ docker-compose.yml # 로컬 redis, milvus -> 최종은 쿠버네틱스 Helm
├─ etcd/ milvus/ minio/
```
