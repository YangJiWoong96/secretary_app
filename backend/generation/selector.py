import re
from typing import Any
from typing import Any as _Any
from typing import Dict, List, Tuple

import numpy as np

from backend.rag.embeddings import embed_documents, embed_query_openai


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    v = float(np.dot(a, b) / d)
    return 0.0 if v < 0 else v


def build_evidence_meta(
    question: str,
    bundle: Any,
    web_blocks: str,
    rag_blocks: str,
    top_m: int = 8,
) -> Dict[str, Any]:
    """
    본문은 태깅 체인에 전달하지 않고, 여기서 유사도/recency 등 메타만 계산해 전달한다.
    """
    qv = embed_query_openai(question)
    webs = [(i, b) for i, b in enumerate((web_blocks or "").split("\n\n")) if b.strip()]
    rags = [(i, b) for i, b in enumerate((rag_blocks or "").split("\n\n")) if b.strip()]

    def _meta(items: List[Tuple[int, str]], kind: str):
        metas: List[Dict[str, Any]] = []
        texts = [it[1][:256] for it in items]
        if not texts:
            return metas
        vecs = embed_documents(texts)
        for (i, _), dv in zip(items, vecs):
            sim = _cos(qv, dv)
            metas.append(
                {"id": f"{kind}:{i}", "domain": kind, "sim": float(sim), "age_s": 0}
            )
        metas.sort(key=lambda x: x["sim"], reverse=True)
        return metas[:top_m]

    return {"web": _meta(webs, "web"), "rag": _meta(rags, "rag")}


def select_blocks_by_ids(
    web_blocks: str,
    rag_blocks: str,
    selected_web_ids: List[_Any],
    selected_rag_ids: List[_Any],
    max_blocks_per_channel: int = 3,
) -> Tuple[str, str]:
    """
    태깅 체인이 고른 ID만 본문으로 전달.
    """

    def _parse_idx(sid: _Any) -> int | None:
        try:
            if isinstance(sid, int):
                return sid
            s = str(sid).strip()
            m = re.search(r"(\d+)$", s)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def _pick(blocks: str, ids: List[_Any], kind: str) -> str:
        arr = (blocks or "").split("\n\n")
        out: List[str] = []
        for sid in ids[:max_blocks_per_channel]:
            idx = _parse_idx(sid)
            if idx is None:
                continue
            if 0 <= idx < len(arr):
                out.append(arr[idx])
        return "\n\n".join(out)

    return _pick(web_blocks, selected_web_ids, "web"), _pick(
        rag_blocks, selected_rag_ids, "rag"
    )
