import os
from pathlib import Path

# OpenAI/Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Embedding backend: "openai" | "gemma" (sentence-transformers)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai").lower()

# Local Gemma/Sentence-Transformers model path (for EMBEDDING_BACKEND="gemma")
GEMMA_MODEL_PATH = os.getenv(
    "GEMMA_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "models" / "EmbeddingGemma"),
)

# Embedding dimensions
_EMBEDDING_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Gemma 임베딩 차원: 기본 768 (models/EmbeddingGemma/1_Pooling/config.json 기준)
_DEFAULT_GEMMA_DIM = int(os.getenv("GEMMA_DIM", "768"))

if EMBEDDING_BACKEND == "gemma":
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(_DEFAULT_GEMMA_DIM)))
else:
    EMBEDDING_DIM = int(
        os.getenv("EMBEDDING_DIM", _EMBEDDING_DIM_MAP.get(EMBEDDING_MODEL, 1536))
    )

# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
METRIC = os.getenv("MILVUS_METRIC", "COSINE").upper()

# Collections (dimension-coupled)
# - 무중단 운영: 기본(v3)은 OpenAI, Gemma 전환시 자동 v4 네임스페이스 사용
_DEFAULT_COLL_VER = "v4" if EMBEDDING_BACKEND == "gemma" else "v3"
RAG_COLL_VER = os.getenv("RAG_COLL_VER", _DEFAULT_COLL_VER)
PROFILE_COLLECTION_NAME = f"user_profiles_{RAG_COLL_VER}_{EMBEDDING_DIM}d"
LOG_COLLECTION_NAME = f"conversation_logs_{RAG_COLL_VER}_{EMBEDDING_DIM}d"
