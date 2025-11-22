# backend/routing/intent_router.py
"""
backend.routing.intent_router - 임베딩 기반 의도 라우터

사용자 입력을 임베딩 유사도로 분석하여 conv/rag/web 중 적절한 경로를 선택합니다.
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("intent_router")


class IntentRouter:
    """
    임베딩 기반 의도 라우터

    사용자 입력과 시드 예시들의 임베딩 유사도를 계산하여
    가장 적합한 라우팅 경로(conv/rag/web)를 결정합니다.

    특징:
    - 센트로이드 임베딩 + Top-K 평균 혼합 스코어링
    - 파일 기반 시드 데이터 확장 지원
    - 지연 초기화로 부팅 시간 최적화
    - OpenAI 임베딩 전용 (전역 백엔드 설정과 독립)
    """

    # ===== 기본 시드 예시 (라벨별) =====
    INTENT_EXAMPLES: Dict[str, List[str]] = {
        "rag": [
            "내가 저번에 설정한 목표 다시 알려줘.",
            "내가 너에게 알려준 내 이름이 뭐라고 했지?",
            "너가 어제 추천해줬던 브렌드가 뭐였지?",
            "우리 지난주에 무슨 얘기까지 했지?",
            "진행 중인 프로젝트의 진행 상황 요약해봐.",
            "내가 엊그제 너한테 말했던 고민 기억해?",
            "오늘 대화를 바탕으로 내 최종 목표를 업데이트해서 정리해줘.",
        ],
        "web": [
            "오늘 서울 날씨 어때?",
            "가장 가까운 스타벅스 어디야?",
            "엔비디아의 가장 최근 GPU 성능에 대한 평가가 어때?",
            "한국의 현 정치 상황에 대해 알려줘.",
            "트럼프가 유엔총회에서 뭐라고 했어?",
            "저녁 먹으려는데 강남역 맛집 추천해줄래?",
            "지난 주 미국 증시에 대한 기사들 요약 보고해줘.",
        ],
        "conv": [
            "양자역학은 왜 이렇게 어려울까?",
            "심심해. 재밌는 농담 하나 해줘.",
            "만나서 반가워.",
        ],
    }

    def __init__(self):
        """
        IntentRouter 초기화

        임베딩은 지연 로딩되므로, 생성자는 즉시 반환됩니다.
        """
        self._ready = False
        self._lock = Lock()
        self._embeddings: Dict[str, np.ndarray] = {}  # 센트로이드
        self._embeddings_list: Dict[str, List[np.ndarray]] = {}  # 개별 예시
        self._seed_files: Dict[str, str] = {}

        # 파일 기반 시드 확장 경로 설정
        try:
            # My_Business/data 디렉토리
            data_dir = str(Path(__file__).resolve().parents[2] / "data")
            self._seed_files = {
                "conv": os.path.join(data_dir, "conv_data.txt"),
                "web": os.path.join(data_dir, "web_data.txt"),
                "rag": os.path.join(data_dir, "rag_data.txt"),
            }
        except Exception:
            self._seed_files = {}

    def _load_seed_texts(self) -> Dict[str, List[str]]:
        """
        시드 예시 로드 (기본 + 파일 확장)

        Returns:
            Dict[str, List[str]]: 라벨별 예시 문장 리스트
        """
        labels = list(self.INTENT_EXAMPLES.keys())
        result = {}

        for label in labels:
            texts = list(self.INTENT_EXAMPLES.get(label, []))

            # 파일 기반 확장 데이터 병합
            try:
                file_path = self._seed_files.get(label)
                if file_path and Path(file_path).exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw = f.read()
                        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
                        extra = [
                            seg.strip()
                            for seg in re.split(r"\s*\n\s*\n\s*", raw)
                            if seg.strip()
                        ]
                        # 과도한 노이즈 방지: 최대 1000 라인까지 사용
                        texts.extend(extra[:1000])
                        logger.info(
                            f"[intent_router] Loaded {len(extra[:1000])} extra examples for '{label}' from file"
                        )
            except Exception as e:
                logger.warning(
                    f"[intent_router] Failed to load seed file for '{label}': {e}"
                )

            result[label] = texts

        return result

    def _embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        라우터 전용 임베딩 (배치) — 전역 백엔드 추상화 사용

        Args:
            texts: 임베딩할 문자열 리스트

        Returns:
            List[np.ndarray]: 임베딩 벡터 리스트
        """
        from backend.rag.embeddings import embed_documents

        vecs = embed_documents([(x or "").strip() for x in texts])
        return [np.array(v, dtype=np.float32) for v in vecs]

    def _embed_query(self, text: str) -> np.ndarray:
        """
        라우터 전용 임베딩 (단일 쿼리) — 전역 백엔드 추상화 사용

        Args:
            text: 임베딩할 문자열

        Returns:
            np.ndarray: 임베딩 벡터
        """
        from backend.rag.embeddings import embed_query_cached

        return embed_query_cached((text or "").strip())

    def ensure_embeddings(self) -> None:
        """
        임베딩 지연 초기화

        시드 예시들의 임베딩을 계산하여 센트로이드와 개별 벡터를 캐시합니다.
        스레드 안전하게 구현되어 있으며, 이미 초기화된 경우 즉시 반환됩니다.
        """

        def _cache_paths(seed_signature: str, labels: List[str]):
            """
            캐시 파일 경로 계산 헬퍼.
            - 외부 시그니처/경로 규약은 변경하지 않는다.
            """
            cache_dir = Path(__file__).resolve().parents[2] / ".cache"
            cache_dir.mkdir(exist_ok=True)
            vec_path = cache_dir / "intent_router_seeds.npz"
            sig_path = cache_dir / "intent_router_seeds.sig"
            return cache_dir, vec_path, sig_path

        def _try_load_cache(
            vec_path: Path, sig_path: Path, sig: str, labels: List[str]
        ) -> bool:
            """
            디스크 캐시를 시도해 로드한다. 성공 시 True.
            로깅 메시지 키/텍스트는 상위에서 유지하므로 여기서는 예외만 전달한다.
            """
            if vec_path.exists() and sig_path.exists():
                saved_sig = sig_path.read_text(encoding="utf-8").strip()
                if saved_sig == sig:
                    data = np.load(vec_path, allow_pickle=True)
                    for label in labels:
                        vecs = data[f"{label}_list"].tolist()
                        centroid = data[f"{label}_centroid"].astype(np.float32)
                        self._embeddings[label] = centroid
                        self._embeddings_list[label] = [
                            np.array(v, dtype=np.float32) for v in vecs
                        ]
                    return True
            return False

        def _try_save_cache(vec_path: Path, sig_path: Path, labels: List[str]) -> None:
            """
            현재 메모리 임베딩을 디스크 캐시에 저장한다.
            실패 시 예외를 상위로 전달하여 기존 경고 로깅을 유지한다.
            """
            save_map = {}
            for label in labels:
                save_map[f"{label}_list"] = np.stack(
                    [
                        np.array(v, dtype=np.float32)
                        for v in self._embeddings_list[label]
                    ],
                    axis=0,
                )
                save_map[f"{label}_centroid"] = np.array(
                    self._embeddings[label], dtype=np.float32
                )
            np.savez_compressed(vec_path, **save_map)
            # 시그니처 파일 기록은 상위 컨텍스트에서 수행한다.

        if self._ready:
            return

        with self._lock:
            if self._ready:
                return

            logger.info(
                "[intent_router] Preparing intent embeddings (lazy initialization)..."
            )
            start_time = time.time()

            # 시드 텍스트 로드 (기본 + 파일)
            seed_data = self._load_seed_texts()
            labels = list(seed_data.keys())
            groups = [seed_data[label] for label in labels]

            # 캐시 파일 경로 및 시그니처 계산
            def _sig(obj: dict) -> str:
                return hashlib.sha256(
                    json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()

            sig = _sig(seed_data)
            _cache_dir, vec_path, sig_path = _cache_paths(sig, labels)

            loaded_from_cache = False
            try:
                if _try_load_cache(vec_path, sig_path, sig, labels):
                    loaded_from_cache = True
                    logger.info("[intent_router] 시드 임베딩 캐시 로드")
            except Exception as e:
                logger.warning(f"[intent_router] 캐시 로드 실패: {e}")

            if not loaded_from_cache:
                # 전체 텍스트 평탄화 → 배치 임베딩
                flat_texts = [t for texts in groups for t in texts]
                flat_vecs = self._embed_documents(flat_texts)

                # 라벨별로 분할 및 센트로이드 계산
                idx = 0
                for label, texts in zip(labels, groups):
                    n = len(texts)
                    vecs = flat_vecs[idx : idx + n]
                    idx += n

                    # 센트로이드 (평균)
                    avg = np.mean(np.stack(vecs, axis=0), axis=0)
                    self._embeddings[label] = avg
                    self._embeddings_list[label] = vecs

                    logger.info(
                        f"[intent_router] Label '{label}': {len(texts)} examples, "
                        f"centroid dim={len(avg)}"
                    )

                try:
                    # 디스크 캐시 저장 (시그니처 파일 포함)
                    _try_save_cache(vec_path, sig_path, labels)
                    sig_path.write_text(sig, encoding="utf-8")
                except Exception as se:
                    logger.warning(f"[intent_router] 캐시 저장 실패: {se}")

            self._ready = True
            took = (time.time() - start_time) * 1000
            logger.info(
                f"[intent_router] Intent embeddings ready: {list(self._embeddings.keys())} "
                f"(took {took:.1f}ms)"
            )

    def route(self, query: str, threshold: float = 0.7) -> Optional[str]:
        """
        쿼리를 라우팅하여 최적 라벨만 반환

        Args:
            query: 사용자 입력 문자열
            threshold: 최소 유사도 임계값 (이 값 이하면 None 반환)

        Returns:
            Optional[str]: 'conv', 'rag', 'web' 중 하나, 또는 None

        Example:
            >>> router = IntentRouter()
            >>> label = router.route("오늘 날씨 어때?")
            >>> print(label)  # 'web'

            >>> label = router.route("우리 지난주 회의 내용")
            >>> print(label)  # 'rag'
        """
        best, _ = self.route_with_scores(query, threshold)
        return best

    def route_with_scores(
        self,
        query: str,
        threshold: float = 0.7,
        prev_turn_ctx: str = "",
        user_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        쿼리를 라우팅하여 라벨 + 유사도 스코어 반환

        센트로이드 유사도와 Top-K 평균 유사도를 혼합하여 최종 스코어를 계산합니다.
        이전 턴 요약 컨텍스트가 제공되면 쿼리와 결합하여 임베딩합니다.

        Args:
            query: 사용자 입력 문자열
            threshold: 최소 유사도 임계값
            prev_turn_ctx: 이전 턴 요약 컨텍스트 (라우팅용 초경량 요약 전용).\n"
                          "⚠️ 이 값은 프롬프트에 주입되지 않으며 임베딩 입력으로만 사용됨

        Returns:
            Tuple[Optional[str], Dict[str, float]]: (최적 라벨 or None, 라벨별 유사도)

        Example:
            >>> router = IntentRouter()
            >>> label, scores = router.route_with_scores("오늘 날씨")
            >>> print(label)  # 'web'
            >>> print(scores)  # {'conv': 0.42, 'rag': 0.35, 'web': 0.88}
        """
        # 임베딩 초기화 (지연)
        self.ensure_embeddings()

        start = time.time()

        # 쿼리 임베딩 (query-only; prev_turn_ctx 결합 금지)
        q_emb = self._embed_query(query)
        q_norm = float(np.linalg.norm(q_emb) or 1.0)

        # 설정에서 Top-K/온도/정규화 로드
        from backend.config import get_settings as _gs

        _s = _gs()
        topk = int(getattr(_s, "ROUTER_SEED_TOPK", 8))
        topk_weight = float(getattr(_s, "ROUTER_TOPK_WEIGHT", 0.5))
        topk_weight = max(0.0, min(1.0, topk_weight))
        # 예시 Top-K 가중 평균 사용 여부 및 최소 가중치(선형 감소)
        example_weighted = bool(getattr(_s, "ROUTER_TOPK_EXAMPLE_WEIGHTED", True))
        example_weight_min = float(getattr(_s, "ROUTER_TOPK_WEIGHT_MIN", 0.2))
        # 스코어 정규화 및 소프트맥스 설정
        use_zscore = bool(getattr(_s, "ROUTER_USE_ZSCORE", True))
        base_tau = float(getattr(_s, "ROUTER_TEMP", 0.5))  # 0.3~0.7 권장
        adaptive_temp = bool(getattr(_s, "ROUTER_ADAPTIVE_TEMP", True))
        # 범위 클램프
        base_tau = max(0.1, min(1.5, base_tau))

        sims: Dict[str, float] = {}

        for label, centroid_emb in self._embeddings.items():
            # 1) 센트로이드 유사도
            centroid_norm = float(np.linalg.norm(centroid_emb) or 1.0)
            sim_centroid = float(np.dot(q_emb, centroid_emb) / (q_norm * centroid_norm))

            # 2) 예시별 Top-K 평균 유사도
            example_vecs = self._embeddings_list.get(label, [])
            if example_vecs:
                per_example_sims = []
                for vec in example_vecs:
                    vec_norm = float(np.linalg.norm(vec) or 1.0)
                    sim = float(np.dot(q_emb, vec) / (q_norm * vec_norm))
                    per_example_sims.append(sim)

                # Top-K 선택
                per_example_sims.sort(reverse=True)
                k = max(1, min(topk, len(per_example_sims)))
                # 선형 가중 평균(상위에 더 큰 가중치)
                topk_vals = np.array(per_example_sims[:k], dtype=np.float32)
                if example_weighted and k > 1:
                    w_min = max(0.01, min(1.0, example_weight_min))
                    weights = np.linspace(1.0, w_min, k, dtype=np.float32)
                    sim_topk = float(np.average(topk_vals, weights=weights))
                else:
                    sim_topk = float(np.mean(topk_vals))

                # 혼합 스코어
                sims[label] = (
                    topk_weight * sim_topk + (1.0 - topk_weight) * sim_centroid
                )
            else:
                sims[label] = sim_centroid

        # ===== 점수 보정: z-score → Temperature Softmax(Adaptive) =====
        labels_list = list(sims.keys())
        vals = np.array([float(sims[l]) for l in labels_list], dtype=np.float32)

        # z-score 정규화로 분포 표준화(옵션)
        if use_zscore and vals.size > 0:
            mu = float(vals.mean())
            sigma = float(vals.std())
            scores = (vals - mu) / (sigma + 1e-6)
        else:
            mu = float(vals.mean()) if vals.size > 0 else 0.0
            sigma = float(vals.std()) if vals.size > 0 else 1.0
            scores = vals

        # 엔트로피 기반 적응형 온도 산출
        def _softmax(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            m = float(np.max(arr))
            ex = np.exp(np.clip(arr - m, -50.0, 50.0))
            s = float(np.sum(ex))
            if s <= 0.0:
                return np.ones_like(ex) / float(len(ex))
            return ex / s

        probs_pre = _softmax(scores)
        entropy = (
            float(-(probs_pre * np.log(probs_pre + 1e-9)).sum())
            if probs_pre.size > 0
            else 0.0
        )
        tau = base_tau
        if adaptive_temp:
            tau_entropy = max(0.3, 1.0 - entropy)
            tau = min(tau, tau_entropy)

        # 최종 확률 계산(Temperature Softmax)
        scores_scaled = scores / max(0.1, tau)
        probs_arr = _softmax(scores_scaled)
        probs: Dict[str, float] = {
            label: float(p) for label, p in zip(labels_list, probs_arr.tolist())
        }

        # 최적 라벨 선택(코사인 유사도 기준)
        best = max(sims, key=sims.get) if sims else None

        took = (time.time() - start) * 1000
        logger.info(
            f"[intent_router] query='{query[:60]}...' sims_raw={sims} probs={probs} "
            f"tau={tau:.3f} z_mu={mu:.4f} z_std={sigma:.4f} entropy={entropy:.4f} "
            f"best={best} threshold={threshold} took_ms={took:.1f}"
        )

        # 실패 케이스 로깅 (최적 없음 또는 임계 미달)
        try:
            should_log_failure = (best is None) or (probs.get(best, 0.0) < threshold)
            if should_log_failure and user_id:
                try:
                    import redis as _redis

                    from backend.config import get_settings

                    r = _redis.Redis.from_url(
                        get_settings().REDIS_URL, decode_responses=True
                    )
                    failure_data = {
                        "query": query,
                        "prev_turn_ctx": prev_turn_ctx,
                        "sims": probs,
                        "threshold": threshold,
                        "timestamp": time.time(),
                    }
                    r.lpush(f"routing:failures:{user_id}", json.dumps(failure_data))
                    r.ltrim(f"routing:failures:{user_id}", 0, 99)
                except Exception:
                    pass
        except Exception:
            pass

        if best is None:
            return None, sims

        # 임계값 체크 (코사인 유사도 기반)
        if float(sims.get(best, 0.0)) >= threshold:
            return best, sims
        else:
            return None, sims

    def route_with_scores_by_vec(
        self,
        q_emb: np.ndarray,
        threshold: float = 0.7,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        사전 계산된 쿼리 임베딩(q_emb)에 대해 라벨 + 스코어를 반환.
        query-only 결정 이후 보조 신호(EMA 등) 적용 시 사용.
        """
        self.ensure_embeddings()

        q_norm = float(np.linalg.norm(q_emb) or 1.0)

        from backend.config import get_settings as _gs

        _s = _gs()
        topk = int(getattr(_s, "ROUTER_SEED_TOPK", 8))
        topk_weight = float(getattr(_s, "ROUTER_TOPK_WEIGHT", 0.5))
        topk_weight = max(0.0, min(1.0, topk_weight))
        example_weighted = bool(getattr(_s, "ROUTER_TOPK_EXAMPLE_WEIGHTED", True))
        example_weight_min = float(getattr(_s, "ROUTER_TOPK_WEIGHT_MIN", 0.2))
        use_zscore = bool(getattr(_s, "ROUTER_USE_ZSCORE", True))
        base_tau = float(getattr(_s, "ROUTER_TEMP", 0.5))
        adaptive_temp = bool(getattr(_s, "ROUTER_ADAPTIVE_TEMP", True))
        base_tau = max(0.1, min(1.5, base_tau))

        sims: Dict[str, float] = {}
        for label, centroid_emb in self._embeddings.items():
            centroid_norm = float(np.linalg.norm(centroid_emb) or 1.0)
            sim_centroid = float(np.dot(q_emb, centroid_emb) / (q_norm * centroid_norm))

            example_vecs = self._embeddings_list.get(label, [])
            if example_vecs:
                per_example_sims = []
                for vec in example_vecs:
                    vec_norm = float(np.linalg.norm(vec) or 1.0)
                    sim = float(np.dot(q_emb, vec) / (q_norm * vec_norm))
                    per_example_sims.append(sim)
                per_example_sims.sort(reverse=True)
                k = max(1, min(topk, len(per_example_sims)))
                topk_vals = np.array(per_example_sims[:k], dtype=np.float32)
                if example_weighted and k > 1:
                    w_min = max(0.01, min(1.0, example_weight_min))
                    weights = np.linspace(1.0, w_min, k, dtype=np.float32)
                    sim_topk = float(np.average(topk_vals, weights=weights))
                else:
                    sim_topk = float(np.mean(topk_vals))
                sims[label] = (
                    topk_weight * sim_topk + (1.0 - topk_weight) * sim_centroid
                )
            else:
                sims[label] = sim_centroid

        labels_list = list(sims.keys())
        vals = np.array([float(sims[l]) for l in labels_list], dtype=np.float32)
        if use_zscore and vals.size > 0:
            mu = float(vals.mean())
            sigma = float(vals.std())
            scores = (vals - mu) / (sigma + 1e-6)
        else:
            scores = vals

        def _softmax(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            m = float(np.max(arr))
            ex = np.exp(np.clip(arr - m, -50.0, 50.0))
            s = float(np.sum(ex))
            if s <= 0.0:
                return np.ones_like(ex) / float(len(ex))
            return ex / s

        tau = base_tau
        probs_arr = _softmax(scores / max(0.1, tau))
        _ = probs_arr  # 확률은 현재 로깅 생략

        best = max(sims, key=sims.get) if sims else None
        if best is None:
            return None, sims
        if float(sims.get(best, 0.0)) >= threshold:
            return best, sims
        else:
            return None, sims

    # ===== 추가 유틸리티 =====
    @staticmethod
    def prefer_rag_when_recall(user_input: str) -> bool:
        """
        회상(이전 대화/사용자 프로필 상기) 의도가 강할 때 RAG 선호 여부 판단.

        구현 원칙:
        - 휴리스틱 키워드가 아닌 임베딩 기반 라우팅 점수를 사용
        - rag 확률이 충분히 높고 다른 라벨 대비 여유 마진이 있을 때만 True

        환경 변수:
        - RAG_RECALL_THR (기본 0.44): rag 확률 하한
        - RAG_RECALL_MARGIN (기본 0.03): 타 라벨 대비 여유 마진
        """
        try:
            import os as _os

            router = get_intent_router()
            best, scores = router.route_with_scores(
                (user_input or "").strip(), threshold=0.0, prev_turn_ctx=""
            )
            p_rag = float(scores.get("rag", 0.0))
            p_web = float(scores.get("web", 0.0))
            p_conv = float(scores.get("conv", 0.0))
            from backend.config import get_settings as _gs2

            _s2 = _gs2()
            thr = float(getattr(_s2, "RAG_RECALL_THR", 0.44))
            margin = float(getattr(_s2, "RAG_RECALL_MARGIN", 0.03))
            return (
                (best == "rag")
                and (p_rag >= thr)
                and (p_rag - max(p_web, p_conv) >= margin)
            )
        except Exception:
            return False


# ===== 싱글톤 인스턴스 =====
_intent_router_instance: Optional[IntentRouter] = None
_router_lock = Lock()


def get_intent_router() -> IntentRouter:
    """
    전역 IntentRouter 싱글톤 인스턴스 반환

    애플리케이션 전역에서 동일한 라우터를 공유하며,
    임베딩은 지연 초기화됩니다.

    Returns:
        IntentRouter: 전역 의도 라우터 인스턴스

    Example:
        >>> from backend.routing import get_intent_router
        >>> router = get_intent_router()
        >>> label = router.route("오늘 날씨")
        >>> print(label)  # 'web'
    """
    global _intent_router_instance

    if _intent_router_instance is None:
        with _router_lock:
            if _intent_router_instance is None:
                _intent_router_instance = IntentRouter()
                logger.info("[intent_router] IntentRouter instance created")

    return _intent_router_instance


# ===== 호환성을 위한 함수형 인터페이스 =====


def ensure_intent_embeddings() -> None:
    """
    임베딩 초기화 (호환성 래퍼)

    기존 코드 호환성을 위한 함수형 인터페이스입니다.
    """
    router = get_intent_router()
    router.ensure_embeddings()


def embedding_router(query: str, threshold: float = 0.7) -> Optional[str]:
    """
    임베딩 라우터 (라벨만 반환, 호환성 래퍼)

    Args:
        query: 사용자 입력
        threshold: 최소 유사도 임계값

    Returns:
        Optional[str]: 최적 라벨 또는 None
    """
    router = get_intent_router()
    return router.route(query, threshold)


def embedding_router_scores(
    query: str,
    threshold: float = 0.7,
    prev_turn_ctx: str = "",
    user_id: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    임베딩 라우터 (라벨 + 스코어 반환, 호환성 래퍼)

    Args:
        query: 사용자 입력
        threshold: 최소 유사도 임계값
        prev_turn_ctx: 이전 턴 요약 컨텍스트 (선택 사항)

    Returns:
        Tuple[Optional[str], Dict[str, float]]: (최적 라벨, 라벨별 유사도)
    """
    router = get_intent_router()
    return router.route_with_scores(query, threshold, prev_turn_ctx, user_id)


def router_embed_query(text: str) -> np.ndarray:
    """
    라우터 전용 임베딩 (호환성 래퍼)

    Args:
        text: 임베딩할 문자열

    Returns:
        np.ndarray: 임베딩 벡터
    """
    router = get_intent_router()
    return router._embed_query(text)


def prefer_rag_when_recall(user_input: str) -> bool:
    """
    회상 키워드 감지 (호환성 래퍼)

    Args:
        user_input: 사용자 입력

    Returns:
        bool: 회상 키워드 포함 여부
    """
    return IntentRouter.prefer_rag_when_recall(user_input)
