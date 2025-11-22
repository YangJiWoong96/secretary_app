# -*- coding: utf-8 -*-
"""
backend.rag.behavior_writer - behavior_slots upsert 경로
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.rag.embeddings import embed_query_openai
from backend.utils.tracing import traceable
from backend.utils.logger import safe_log_event


def _ensure_behavior_collection():
    from pymilvus import Collection, utility
    from backend.rag.behavior_schema import BEHAVIOR_SCHEMA, BEHAVIOR_INDEXES
    from backend.rag.config import EMBEDDING_DIM  # noqa: F401
    from backend.rag.milvus import ensure_milvus  # reuse connection mgmt
    from pymilvus import DataType

    ensure_milvus()
    if utility.has_collection("behavior_slots"):
        coll = Collection("behavior_slots")
    else:
        coll = Collection(name="behavior_slots", schema=BEHAVIOR_SCHEMA)
        for idx in BEHAVIOR_INDEXES:
            try:
                coll.create_index(field_name=idx["field"], index_params=idx)
            except Exception:
                pass
    try:
        # 스칼라 인덱스: 숫자형만 STL_SORT 적용(문자열은 스킵)
        def _safe_scalar_index(field_name: str) -> None:
            try:
                dtype = None
                for f in coll.schema.fields:
                    if getattr(f, "name", None) == field_name:
                        dtype = getattr(f, "dtype", None)
                        break
                if dtype in (
                    DataType.INT32,
                    DataType.INT64,
                    DataType.FLOAT,
                    DataType.DOUBLE,
                ):
                    coll.create_index(
                        field_name=field_name, index_params={"index_type": "STL_SORT"}
                    )
                # VARCHAR/JSON 등은 인덱스 생략
            except Exception:
                pass

        for fld in ["user_id", "slot_key", "status"]:
            _safe_scalar_index(fld)
    except Exception:
        pass
    coll.load()
    return coll


@traceable(name="Behavior: upsert_slots", run_type="tool", tags=["behavior", "rag"])
async def upsert_slots(user_id: str, items: List[Dict[str, Any]]) -> None:
    """
    items: [{"norm_key","slot_key","value","confidence","scoreboard":{...},"evidence":str}, ...]
    """
    if not user_id or not items:
        return
    coll = _ensure_behavior_collection()
    now_ns = int(time.time_ns())

    rows: List[Dict[str, Any]] = []
    for it in items:
        try:
            norm_key = str(it.get("norm_key") or "")
            slot_key = str(it.get("slot_key") or "")
            value = it.get("value")
            conf = float(it.get("confidence") or 0.5)
            sb = it.get("scoreboard") or {}
            status = str(sb.get("status") or "pending")

            # 임베딩: slot_key + value 합성
            composite = f"{slot_key}:{value}"
            emb = embed_query_openai(composite)

            row_id = f"{user_id}:{norm_key}"
            rows.append(
                {
                    "id": row_id,
                    "user_id": user_id,
                    "slot_key": slot_key,
                    "norm_key": norm_key,
                    "value": (
                        json.dumps(value, ensure_ascii=False)
                        if not isinstance(value, str)
                        else value
                    ),
                    "status": status,
                    "confidence": float(conf),
                    "tags": {"scoreboard": sb},
                    "embedding": emb,
                    "created_at": now_ns,
                    "updated_at": now_ns,
                }
            )
        except Exception:
            continue
    if rows:
        try:
            coll.upsert(rows)
            # 구조화 로깅: behavior_slots 업서트 요약
            try:
                safe_log_event(
                    "rag.behavior_upsert",
                    {
                        "user_id": user_id,
                        "collection": "behavior_slots",
                        "row_count": len(rows),
                        "vector_dim": len(rows[0].get("embedding", []) or []),
                        "reason": "behavior_update",
                    },
                )
            except Exception:
                pass
        except Exception:
            pass


__all__ = ["upsert_slots"]
