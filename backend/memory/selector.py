from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
import re
from backend.rag.embeddings import embed_query_cached
from .turns import TurnSummary


@dataclass
class SelectedContext:
    persona10: str
    stwm_snapshot: Dict[str, Any]
    selected_summaries: List[str]
    query_rewritten: str
    reason: str | None = None


def _tokenize_ko(text: str) -> List[str]:
    # 간단 토크나이저: 한글/영문/숫자 토큰화
    return re.findall(r"[가-힣A-Za-z0-9]{2,}", text or "")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    v = float(np.dot(a, b) / denom)
    return max(0.0, v)


def select_summaries(
    query: str,
    candidates: List[TurnSummary],
    topk_bm25: int = 10,
    topk_final: int = 3,
    bm25_weight: float = 0.6,
    emb_weight: float = 0.4,
) -> List[TurnSummary]:
    docs = [c.answer_summary for c in candidates]
    tokenized = [_tokenize_ko(t) for t in docs]
    if not tokenized:
        return []
    bm25 = BM25Okapi(tokenized)
    q_tokens = _tokenize_ko(query)
    bm_scores = bm25.get_scores(q_tokens)
    # 상위 N 후보
    idxs = list(np.argsort(-np.array(bm_scores))[:topk_bm25])
    qv = embed_query_cached(query)
    # 코사인 계산
    final: List[Tuple[float, int]] = []
    for i in idxs:
        dv = embed_query_cached(candidates[i].answer_summary)
        sim = _cosine(qv, dv)
        score = bm25_weight * float(bm_scores[i]) + emb_weight * sim
        final.append((score, i))
    final.sort(key=lambda x: -x[0])
    picked = [candidates[i] for _, i in final[:topk_final]]
    return picked
