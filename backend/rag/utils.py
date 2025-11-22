# C:\My_Business\backend\rag\utils.py
import json
from pathlib import Path
from typing import Any, Dict, List

from .embeddings import embed_query_gemma


def build_gazetteer_embeddings(
    input_path: str, output_path: str, overwrite: bool = False
) -> Dict[str, Any]:
    """
    입력 JSON이 두 형태 모두 가능하도록 처리:
    1) ["이름1", "이름2", ...]
    2) [{"name": "이름", "emb": [..]}, ...]  (이미 일부 임베딩 포함 가능)

    - emb가 비어있는 항목만 gemma/openai 백엔드로 임베딩 계산 후 채움
    - 결과를 동일 포맷(리스트[dict])으로 output_path에 저장
    - 반환: {count_total, count_built, output}
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    if not in_p.exists():
        raise FileNotFoundError(f"gazetteer input not found: {input_path}")
    if out_p.exists() and not overwrite:
        # 이미 존재하면 그대로 로드 후 반환
        with open(out_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "count_total": len(data),
            "count_built": 0,
            "output": str(out_p),
        }

    with open(in_p, "r", encoding="utf-8") as f:
        base = json.load(f)

    # 통합 리스트[dict]로 변환
    items: List[Dict[str, Any]] = []
    if isinstance(base, list):
        for x in base:
            if isinstance(x, str):
                items.append({"name": x, "emb": None})
            elif isinstance(x, dict) and "name" in x:
                items.append({"name": x.get("name"), "emb": x.get("emb")})
    else:
        raise ValueError("gazetteer input must be list")

    # 임베딩 채우기(emb==None)
    built = 0
    for it in items:
        if not it.get("emb"):
            # Gazetteer는 Gemma로 구축
            vec = embed_query_gemma(it["name"]).tolist()
            it["emb"] = vec
            built += 1

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    return {"count_total": len(items), "count_built": built, "output": str(out_p)}


from .config import METRIC


def hit_similarity(hit) -> float:
    """
    Milvus 검색 hit에서 통일된 유사도 점수를 계산한다.
    - METRIC=="IP": distance/score를 그대로 사용
    - METRIC=="COSINE": 1 - distance
    - 그 외: -distance (낮을수록 유사)
    """
    d = getattr(hit, "distance", None)
    s = getattr(hit, "score", None)
    if METRIC == "IP":
        return float(d if d is not None else s)
    elif METRIC == "COSINE":
        dist = float(d if d is not None else 1.0)
        return 1.0 - dist
    else:
        return -float(d if d is not None else 1e9)
