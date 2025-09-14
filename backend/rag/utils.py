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
