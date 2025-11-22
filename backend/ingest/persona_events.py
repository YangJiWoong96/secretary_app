from __future__ import annotations

import json
import time
from typing import Dict

from backend.directives.events import PersonaEvent, serialize_event
from backend.rag import ensure_collections
from backend.rag.embeddings import embed_query_openai
from backend.utils.tracing import traceable
from backend.utils.logger import safe_log_event


@traceable(name="Ingest: upsert_bot_event", run_type="tool", tags=["ingest", "rag"])
def upsert_bot_event(session_id: str, ev: PersonaEvent) -> Dict[str, str]:
    """bot_event를 RAG Log 컬렉션에 업서트.
    - text 필드로 직렬화 저장 → 유사 이벤트 검색/요약 가능
    """
    prof_coll, log_coll = ensure_collections()
    doc = serialize_event(ev)
    emb = embed_query_openai(doc["text"])  # 로그 전용 OpenAI 임베딩
    rid = f"{session_id}:bot_event:{doc['ts']}:{doc['k']}"
    log_coll.upsert(
        [
            {
                "id": rid,
                "embedding": emb,
                "text": json.dumps(doc, ensure_ascii=False),
                "user_id": session_id,
                "type": "bot_event",
                "created_at": int(time.time_ns()),
                "date_start": 0,
                "date_end": 99999999,
                "date_ym": 0,
            }
        ]
    )
    try:
        safe_log_event(
            "rag.log_upsert",
            {
                "user_id": session_id,
                "collection": "logs",
                "chunk_count": 1,
                "vector_dim": len(emb or []),
                "reason": "bot_event",
            },
        )
    except Exception:
        pass
    return {"id": rid}
