import time
import numpy as np
from typing import Optional, Tuple
from .config import METRIC
from .milvus import ensure_collections
from .embeddings import embed_query_cached
from .utils import hit_similarity


def _hit_similarity(hit) -> float:
    # 하위 호환: 외부에서 import하는 기존 심볼 유지
    return hit_similarity(hit)


def _hits_to_ctx(hits, score_min: float = 0.45, top_k: int = 3) -> str:
    if not hits:
        return ""
    picked = []
    for hit in hits[:top_k]:
        sim = _hit_similarity(hit)
        if sim >= score_min:
            picked.append(hit.entity.get("text"))
    return "\n".join(p for p in picked if p)


def retrieve_from_rag(
    session_id: str,
    query: str,
    top_k: int = 2,
    date_filter: Optional[Tuple[int, int]] = None,
) -> str:
    t0 = time.time()
    try:
        prof_coll, log_coll = ensure_collections()
        query_emb = embed_query_cached(query)
        search_params = {"metric_type": METRIC, "params": {"ef": 64}}
        expr = f"user_id == '{session_id}'"
        if date_filter:
            s, e = date_filter
            expr += f" and date_end >= {s} and date_start <= {e}"

        prof_res = prof_coll.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=1,
            expr=expr,
            output_fields=["text", "date_start", "date_end", "date_ym"],
        )
        log_res = log_coll.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "date_start", "date_end", "date_ym"],
        )

        ctx_parts = []
        if prof_res and prof_res[0]:
            prof_ctx = _hits_to_ctx(prof_res[0], score_min=0.45, top_k=1)
            if prof_ctx:
                ctx_parts.append(f"[RAG 프로필]\n{prof_ctx}")
        if log_res and log_res[0]:
            log_ctx = _hits_to_ctx(log_res[0], score_min=0.45, top_k=top_k)
            if log_ctx:
                ctx_parts.append(f"[RAG 로그]\n{log_ctx}")

        ctx = "\n".join(ctx_parts)
        return ctx
    except Exception:
        return ""
