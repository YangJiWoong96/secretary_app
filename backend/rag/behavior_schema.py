# -*- coding: utf-8 -*-
"""
backend.rag.behavior_schema - Milvus behavior_slots 컬렉션 스키마/인덱스
"""
from pymilvus import CollectionSchema, DataType, FieldSchema

from .config import EMBEDDING_DIM

# behavior_slots: 100 슬롯 결과를 장기 보존/검색 가능하게 유지
BEHAVIOR_SCHEMA = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="slot_key", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="norm_key", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="confidence", dtype=DataType.FLOAT),
        FieldSchema(name="tags", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
    ],
    description="Behavior Slots (100) Collection",
)

BEHAVIOR_INDEXES = [
    {
        "field": "embedding",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 200},
    },
]

__all__ = ["BEHAVIOR_SCHEMA", "BEHAVIOR_INDEXES"]
