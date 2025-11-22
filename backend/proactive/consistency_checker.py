from __future__ import annotations

"""
ConsistencyChecker - SBERT 임베딩 + DBSCAN 기반 내용 일관성 점수 계산기

주의:
- sentence-transformers, scikit-learn 의존. requirements.txt에 이미 존재.
- 모델: paraphrase-multilingual-MiniLM-L12-v2 (가벼운 다국어)
"""

from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class ConsistencyChecker:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # 지연 임포트로 초기 로딩 지연을 줄임
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def _extract_sentences(
        self, sources: List[Dict], max_per_source: int = 3
    ) -> List[str]:
        sents: List[str] = []
        for src in sources:
            content = (src.get("content") or "").strip()
            if not content:
                continue
            # 간단 분할: 마침표 기준. 한국어도 처리 위해 추가 구두점 포함
            parts = [
                p.strip() for p in content.replace("\n", " ").split(". ") if p.strip()
            ]
            sents.extend(parts[:max_per_source])
        return sents

    def calculate_consistency(self, sources: List[Dict]) -> float:
        sentences = self._extract_sentences(sources)
        if len(sentences) < 2:
            return 0.0

        embeddings = self.model.encode(sentences)
        sim_matrix = cosine_similarity(embeddings)

        # DBSCAN 클러스터링 (거리를 metric으로 사용)
        distances = 1 - sim_matrix
        clustering = DBSCAN(eps=0.3, min_samples=2, metric="precomputed")
        labels = clustering.fit_predict(distances)

        unique_labels = list(set(labels) - {-1})
        if not unique_labels:
            return 0.0

        cluster_scores = []
        for lab in unique_labels:
            mask = labels == lab
            sub = sim_matrix[mask][:, mask]
            size = int(mask.sum())
            std = float(np.std(sub)) if sub.size > 0 else 1.0
            cluster_scores.append(size * (1.0 - std))

        consistency = float(sum(cluster_scores) / max(1, len(sentences)))
        return min(1.0, max(0.0, consistency))


__all__ = ["ConsistencyChecker"]
