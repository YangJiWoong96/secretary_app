### 초기 구상안
<img width="1564" height="575" alt="image" src="https://github.com/user-attachments/assets/36e8bc6d-4715-4634-addb-2b096951beb0" />
```
.
├─ .env  
├─ backend/                     
│  ├─ app.py                    # FastAPI 엔트리포인트 (WS 포함)
│  ├─ requirements.txt          
│  ├─ .gitignore
│  │
│  ├─ directives/               # 동적 시스템 프롬프트 모듈
│  │  ├─ agent.py
│  │  ├─ compiler.py
│  │  ├─ pipeline.py
│  │  ├─ policy.py
│  │  ├─ schema.py
│  │  └─ store.py
│  │
│  ├─ ingest/                   # GCP Firestore 데이터 수집
│  │  └─ main.py                # (카카오맵 역지오코딩 처리 포함)
│  │
│  ├─ proactive/                # 선제적 상호작용 모듈
│  │  ├─ notifier.py
│  │  ├─ planner.py
│  │  └─ scheduler.py
│  │
│  └─ router_kor_electra_small/ # 소분류기 가중치
│     ├─ pytorch_model.bin
│     ├─ tokenizer.json
│     ├─ tokenizer_config.json
│     ├─ vocab.txt
│     ├─ config.json
│     ├─ training_args.bin
│     └─ checkpoint-128/
│
├─ test/                        # 실험/검증 (배포 미사용)
│  ├─ baseline_test.ipynb
│  ├─ realtime_context_events.ipynb
│  ├─ search_engine.ipynb
│  └─ small_classifier_train_data.txt
│
├─ flutter_app/
│  └─ timely_agent/             # Flutter (Frontend)
│
├─ mcp_server/                  # MCP(Node) 서버
│
└─ docker/
   ├─ docker-compose.yml         # 로컬용 Redis + Milvus
   ├─ etcd/  milvus/  minio/
```
