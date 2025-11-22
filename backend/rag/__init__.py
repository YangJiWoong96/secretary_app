# C:\My_Business\backend\rag\__init__.py
from .config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOG_COLLECTION_NAME,
    METRIC,
    MILVUS_HOST,
    MILVUS_PORT,
    OPENAI_API_KEY,
    PROFILE_COLLECTION_NAME,
)
from .embeddings import embed_query_cached, get_embeddings
from .milvus import (
    create_milvus_collection,
    ensure_collections,
    ensure_milvus,
    ensure_partition,
    ensure_profile_collection,
)
from .profile_rag import ProfileRAG
from .profile_writer import ProfileWriter
from .retrieval import retrieve_from_rag
from .retrieval_utils import (
    filter_by_date,
    format_rag_blocks,
    hit_similarity_wrapper,
    milvus_hits_to_ctx,
)
from .snapshot_pipeline import (
    SnapshotPipeline,
    get_snapshot_pipeline,
    update_long_term_memory,
)
from .utils import hit_similarity
