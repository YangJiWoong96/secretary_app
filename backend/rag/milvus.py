import time
from threading import Lock
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from .config import (
    MILVUS_HOST,
    MILVUS_PORT,
    EMBEDDING_DIM,
    PROFILE_COLLECTION_NAME,
    LOG_COLLECTION_NAME,
)


_conn_alias = "default"
_coll_lock = Lock()
_prof_coll = None
_log_coll = None
_ready = False


def ensure_milvus():
    if not connections.has_connection(_conn_alias):
        connections.connect(_conn_alias, host=MILVUS_HOST, port=MILVUS_PORT)


def _create_collection(name: str, desc: str) -> Collection:
    if utility.has_collection(name):
        coll = Collection(name)
        # dim 검증
        for f in coll.schema.fields:
            if f.name == "embedding":
                dim = f.params.get("dim")
                if dim != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Milvus collection dim mismatch: {dim} != {EMBEDDING_DIM}"
                    )
        return coll

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("user_id", DataType.VARCHAR, max_length=256),
            FieldSchema("type", DataType.VARCHAR, max_length=50),
            FieldSchema("created_at", DataType.INT64),
            FieldSchema("date_start", DataType.INT64),
            FieldSchema("date_end", DataType.INT64),
            FieldSchema("date_ym", DataType.INT64),
        ],
        desc,
    )
    coll = Collection(name, schema)
    coll.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return coll


def ensure_collections():
    global _prof_coll, _log_coll, _ready
    if _ready and _prof_coll and _log_coll:
        return _prof_coll, _log_coll
    with _coll_lock:
        if _ready and _prof_coll and _log_coll:
            return _prof_coll, _log_coll
        ensure_milvus()
        _prof_coll = _create_collection(PROFILE_COLLECTION_NAME, "User Profiles")
        _log_coll = _create_collection(LOG_COLLECTION_NAME, "Conversation Logs")
        _prof_coll.load()
        _log_coll.load()
        _ready = True
        return _prof_coll, _log_coll
