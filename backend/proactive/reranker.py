from __future__ import annotations

"""
웹 검색 재랭커

역할:
- 질의와 결과(title+excerpt+content) 기반으로 BM25/임베딩 유사도를 계산하고,
  TrustScore와 결합하여 최종 점수로 재정렬한다.

의존:
- rank-bm25 (필수)
- sentence-transformers (선택)
- CrossEncoder(bge-reranker 등)는 선택(현재 미사용, 확장 지점)
"""

from typing import Dict, List

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore

_ST_MODEL = None


def _lazy_st_model():
    """
    sentence-transformers 지연 로드.
    - 환경에 설치되어 있지 않다면 None을 반환(옵션).
    """
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _ST_MODEL = SentenceTransformer("BAAI/bge-m3")
        return _ST_MODEL
    except Exception:
        _ST_MODEL = None
        return None


def _tokenize_ko_en(text: str) -> List[str]:
    import re

    t = (text or "").lower()
    t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
    return [x for x in t.split() if x]


def rerank_results(query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    BM25 + (옵션) 임베딩 유사도 기반 재랭킹.
    - 반환: 점수 재계산 후 상위 top_k를 유지한 리스트(원소는 입력과 동일 dict)
    """
    if not results or BM25Okapi is None:
        return results

    docs = []
    for r in results:
        txt = " ".join(
            [
                str(r.get("title") or ""),
                str(r.get("excerpt") or ""),
                str(r.get("content") or ""),
            ]
        )
        docs.append(_tokenize_ko_en(txt))
    bm25 = BM25Okapi(docs)
    qtok = _tokenize_ko_en(query or "")
    bm_scores = list(bm25.get_scores(qtok))
    bm_max = max(1e-6, max(bm_scores) if bm_scores else 1.0)

    # (옵션) 임베딩 코사인 유사도
    emb_sim = [0.0] * len(results)
    st = _lazy_st_model()
    if st is not None:
        try:
            import numpy as np  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

            qv = st.encode([query or ""], normalize_embeddings=True)
            dv = st.encode(
                [
                    (str(r.get("title") or "") + " " + str(r.get("content") or ""))
                    for r in results
                ],
                normalize_embeddings=True,
            )
            sim = cosine_similarity(qv, dv)[0]  # shape: (N,)
            emb_sim = [float(max(0.0, min(1.0, s))) for s in sim]
        except Exception:
            emb_sim = [0.0] * len(results)

    # 최종 점수: 신뢰도(0.5) + BM25 정규화(0.35) + 임베딩(0.15)
    # 임베딩 불가 시: 신뢰도(0.5) + BM25(0.5)
    out = []
    for i, r in enumerate(results):
        trust = float(r.get("trust_score", 0.0))
        bm_norm = float(bm_scores[i] / bm_max) if bm_max > 0 else 0.0
        if st is None:
            final = 0.5 * trust + 0.5 * bm_norm
        else:
            final = 0.5 * trust + 0.35 * bm_norm + 0.15 * emb_sim[i]
        rr = dict(r)
        rr["_final_score"] = float(max(0.0, min(1.0, final)))
        out.append(rr)

    out.sort(key=lambda x: float(x.get("_final_score", 0.0)), reverse=True)
    return out[: top_k or len(out)]


__all__ = ["rerank_results"]
