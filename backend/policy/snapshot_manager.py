"""
backend.policy.snapshot_manager - 스냅샷 및 프로필 관리

장기 메모리 스냅샷 생성, 멱등성 관리, 중복 감지, 워커 큐 관리를 담당합니다.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("snapshot")


class SnapshotManager:
    """
    스냅샷 워커 및 멱등성 관리

    대화 히스토리를 장기 메모리로 스냅샷할 때 사용하는 워커 큐,
    멱등성 캐시, 중복 감지 등을 관리합니다.

    특징:
    - 비동기 워커 큐 (백그라운드 처리)
    - 멱등성 캐시 (동일 스냅샷 중복 방지)
    - 에지 트리거 + 디바운스 (최적화된 스냅샷 타이밍)
    - 세션별 상태 관리
    """

    def __init__(self):
        """
        SnapshotManager 초기화

        설정은 지연 로딩하여 순환 의존성을 방지합니다.
        """
        self._settings = None
        self._session_state: Dict[str, Dict[str, Any]] = {}
        self._idempotency_cache: Dict[str, str] = {}
        self._snapshot_queue: Optional[asyncio.Queue] = None
        self._embed_sem: Optional[asyncio.Semaphore] = None
        self._workers_started = False

    @property
    def settings(self):
        """설정 인스턴스 (지연 로딩)"""
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def session_state(self) -> Dict[str, Dict[str, Any]]:
        """세션별 상태 딕셔너리"""
        return self._session_state

    @property
    def idempotency_cache(self) -> Dict[str, str]:
        """멱등성 캐시 (session_id → last_hash)"""
        return self._idempotency_cache

    @property
    def snapshot_queue(self) -> asyncio.Queue:
        """스냅샷 워커 큐 (지연 초기화)"""
        if self._snapshot_queue is None:
            maxsize = self.settings.SNAPSHOT_QUEUE_MAXSIZE
            self._snapshot_queue = asyncio.Queue(maxsize=maxsize)
        return self._snapshot_queue

    @property
    def embed_sem(self) -> asyncio.Semaphore:
        """임베딩 동시성 제어 세마포어 (지연 초기화)"""
        if self._embed_sem is None:
            concurrency = self.settings.EMBED_CONCURRENCY
            self._embed_sem = asyncio.Semaphore(concurrency)
        return self._embed_sem

    @staticmethod
    def sha256(s: str) -> str:
        """
        문자열의 SHA-256 해시 계산

        Args:
            s: 해시할 문자열

        Returns:
            str: 16진수 해시 문자열
        """
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @staticmethod
    def flatten_profile_items(profile: dict) -> Set[str]:
        """
        프로필 JSON에서 핵심 필드만 평탄화하여 비교용 집합 생성

        소문자 변환 및 공백 정규화를 적용하여 신규성 비교에 사용합니다.

        Args:
            profile: 사용자 프로필 딕셔너리

        Returns:
            Set[str]: 정규화된 항목들의 집합

        Example:
            >>> profile = {
            ...     "facts": ["서울 거주", "개발자"],
            ...     "goals": ["운동 시작"],
            ... }
            >>> items = SnapshotManager.flatten_profile_items(profile)
            >>> print(items)  # {'서울 거주', '개발자', '운동 시작'}
        """
        keys = ("facts", "goals", "tasks", "decisions", "constraints")
        out = set()

        for k in keys:
            v = profile.get(k)
            if isinstance(v, list):
                for x in v:
                    s = str(x).strip().lower()
                    if s:
                        out.add(s)
            elif isinstance(v, str):
                s = v.strip().lower()
                if s:
                    out.add(s)

        return out

    def near_duplicate_log(
        self, session_id: str, log_emb: List[float], ym_min: int
    ) -> Tuple[bool, float]:
        """
        최근 N개월 범위에서 근사중복 로그 검색 (Top-3 평균 유사도)

        벡터 유사도 검색으로 가장 유사한 로그 3개를 찾아
        평균 유사도를 계산하여 중복 여부를 판단합니다.

        Args:
            session_id: 세션 ID
            log_emb: 로그 임베딩 벡터
            ym_min: 최소 년월 (YYYYMM 형식)

        Returns:
            Tuple[bool, float]: (중복 여부, 평균 유사도 스코어)

        Example:
            >>> manager = SnapshotManager()
            >>> is_dup, avg_sim = manager.near_duplicate_log("user123", embedding, 202408)
            >>> if is_dup:
            ...     print(f"중복 감지 (평균 유사도: {avg_sim:.3f})")
        """
        from backend.rag import METRIC, ensure_collections
        from backend.rag.utils import hit_similarity

        try:
            prof_coll, log_coll = ensure_collections()
            search_params = {"metric_type": METRIC, "params": {"ef": 32}}
            expr = f"user_id == '{session_id}' and date_ym >= {ym_min}"

            # Top-3 조회로 변경
            res = log_coll.search(
                data=[log_emb],
                anns_field="embedding",
                param=search_params,
                limit=3,  # Top-3로 확장
                expr=expr,
                output_fields=["text", "date_ym"],
            )

            if res and res[0]:
                # Top-3 평균 유사도 계산
                sims = [hit_similarity(hit) for hit in res[0]]
                avg_sim = sum(sims) / len(sims)

                # 임계값 완화 (0.90 → 0.85)
                threshold = 0.85

                logger.info(
                    f"[snapshot] Near-dup Top-3 sims={[f'{s:.3f}' for s in sims]} "
                    f"avg={avg_sim:.3f} threshold={threshold}"
                )

                return (avg_sim >= threshold, avg_sim)
        except Exception as e:
            logger.warning(f"[snapshot] Near-duplicate search error: {e}")

        return (False, 0.0)

    async def ensure_workers(self) -> None:
        """
        스냅샷 워커들을 초기화 (백그라운드 태스크)

        이미 실행 중이면 무시합니다.
        """
        if self._workers_started:
            return

        self._workers_started = True
        concurrency = self.settings.WORKER_CONCURRENCY

        for i in range(concurrency):
            asyncio.create_task(self._snapshot_worker(i))

        logger.info(f"[snapshot] Started {concurrency} snapshot workers")

    async def _snapshot_worker(self, worker_id: int) -> None:
        """
        스냅샷 워커 (백그라운드 무한 루프)

        큐에서 (user_id, session_id)를 가져와 장기 메모리 업데이트를 수행합니다.

        Args:
            worker_id: 워커 식별자
        """
        while True:
            item = await self.snapshot_queue.get()
            t0 = time.time()

            try:
                # update_long_term_memory는 동기 함수이므로 스레드로 오프로딩
                from backend.rag.snapshot_pipeline import update_long_term_memory

                # 강제 형식: (user_id, session_id)
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    user_id, session_id = str(item[0]), str(item[1])
                else:
                    raise ValueError(
                        "Snapshot queue item must be (user_id, session_id)"
                    )

                await asyncio.to_thread(update_long_term_memory, user_id, session_id)

                took = (time.time() - t0) * 1000
                logger.info(
                    f"[snapshot:worker-{worker_id}] Done user={user_id} session={session_id} took_ms={took:.1f}"
                )
            except Exception as e:
                logger.warning(
                    f"[snapshot:worker-{worker_id}] Error item={item} err={e}"
                )
            finally:
                self.snapshot_queue.task_done()

    def enqueue_snapshot(self, user_id: str, session_id: str) -> None:
        """
        스냅샷 큐에 작업 추가

        큐가 가득 차면 무시하고 경고 로깅합니다.

        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
        """
        try:
            if not user_id or not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("enqueue_snapshot requires non-empty user_id")
            self.snapshot_queue.put_nowait((user_id, session_id))
            logger.info(
                f"[snapshot:q] Enqueued user={user_id} session={session_id} qsize={self.snapshot_queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"[snapshot:q] Queue full → drop user={user_id} session={session_id}"
            )

    def edge_and_debounce(
        self, user_id: str, session_id: str, tokens_prev: int, tokens_now: int
    ) -> None:
        """
        에지 트리거 + 디바운스로 스냅샷 타이밍 결정

        토큰 수가 임계값을 넘고, 최소 턴 수 및 시간이 경과하면 스냅샷을 예약합니다.

        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            tokens_prev: 이전 토큰 수
            tokens_now: 현재 토큰 수

        Example:
            >>> manager = SnapshotManager()
            >>> manager.edge_and_debounce("user123", "sessA", tokens_prev=4000, tokens_now=4600)
            # → 4500 임계값 초과 시 스냅샷 예약
        """
        # 세션 상태 안전 초기화
        st = self._session_state.setdefault(session_id, {})
        if "last_flush_at" not in st:
            st["last_flush_at"] = 0.0
        if "turns_since_last" not in st:
            st["turns_since_last"] = 0
        if "prev_tokens" not in st:
            st["prev_tokens"] = 0

        now = time.time()

        # 에지 트리거: 토큰 수가 임계값을 넘는 순간
        edge_threshold = self.settings.SNAPSHOT_EDGE_TOKENS
        edge = tokens_prev < edge_threshold and tokens_now >= edge_threshold

        # 디바운스 조건
        elapsed = now - st["last_flush_at"]
        turns_ok = st["turns_since_last"] >= self.settings.DEBOUNCE_TURNS
        time_ok = elapsed >= self.settings.DEBOUNCE_SECONDS

        if edge and time_ok and turns_ok:
            # 스냅샷 예약
            self.enqueue_snapshot(user_id, session_id)

            # Directive 업데이트도 예약
            try:
                from backend.directives.pipeline import schedule_directive_update

                schedule_directive_update(session_id)
            except Exception:
                pass

            # 상태 초기화
            st["last_flush_at"] = now
            st["turns_since_last"] = 0
        else:
            st["turns_since_last"] += 1

        st["prev_tokens"] = tokens_now

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        세션 상태 가져오기 (없으면 생성)

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 세션 상태 딕셔너리
        """
        return self._session_state.setdefault(session_id, {})

    def set_session_value(self, session_id: str, key: str, value: Any) -> None:
        """
        세션 상태에 값 설정

        Args:
            session_id: 세션 ID
            key: 키
            value: 값
        """
        st = self.get_session_state(session_id)
        st[key] = value

    def get_session_value(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        세션 상태에서 값 가져오기

        Args:
            session_id: 세션 ID
            key: 키
            default: 기본값

        Returns:
            Any: 저장된 값 또는 기본값
        """
        st = self.get_session_state(session_id)
        return st.get(key, default)

    def check_idempotency(self, session_id: str, content_hash: str) -> bool:
        """
        멱등성 체크 (중복 스냅샷 방지)

        Args:
            session_id: 세션 ID
            content_hash: 콘텐츠 해시

        Returns:
            bool: True면 중복 (스킵해야 함)
        """
        cached_hash = self._idempotency_cache.get(session_id)
        return cached_hash == content_hash

    def update_idempotency(self, session_id: str, content_hash: str) -> None:
        """
        멱등성 캐시 업데이트

        Args:
            session_id: 세션 ID
            content_hash: 새 콘텐츠 해시
        """
        self._idempotency_cache[session_id] = content_hash


# ===== 싱글톤 인스턴스 =====
_snapshot_manager_instance: Optional[SnapshotManager] = None


def get_snapshot_manager() -> SnapshotManager:
    """
    전역 SnapshotManager 싱글톤 인스턴스 반환

    Returns:
        SnapshotManager: 전역 스냅샷 관리자 인스턴스

    Example:
        >>> from backend.policy.snapshot_manager import get_snapshot_manager
        >>> manager = get_snapshot_manager()
        >>> manager.enqueue_snapshot("user123")
    """
    global _snapshot_manager_instance

    if _snapshot_manager_instance is None:
        _snapshot_manager_instance = SnapshotManager()
        logger.info("[snapshot] SnapshotManager instance created")

    return _snapshot_manager_instance


# ===== 호환성을 위한 함수형 인터페이스 및 전역 상태 =====

# 전역 상태 참조 (호환성)
SESSION_STATE: Dict[str, Dict[str, Any]] = {}
IDEMPOTENCY_CACHE: Dict[str, str] = {}


def get_session_state_dict() -> Dict[str, Dict[str, Any]]:
    """
    전역 세션 상태 딕셔너리 반환 (호환성)

    Returns:
        Dict: 세션 상태 딕셔너리
    """
    manager = get_snapshot_manager()
    return manager.session_state


def get_idempotency_cache_dict() -> Dict[str, str]:
    """
    전역 멱등성 캐시 딕셔너리 반환 (호환성)

    Returns:
        Dict: 멱등성 캐시
    """
    manager = get_snapshot_manager()
    return manager.idempotency_cache


# 전역 변수를 싱글톤으로 리다이렉트
def _init_global_refs():
    """전역 변수를 싱글톤 속성으로 리다이렉트"""
    global SESSION_STATE, IDEMPOTENCY_CACHE
    manager = get_snapshot_manager()
    SESSION_STATE = manager.session_state
    IDEMPOTENCY_CACHE = manager.idempotency_cache


def sha256(s: str) -> str:
    """SHA-256 해시 (호환성 래퍼)"""
    return SnapshotManager.sha256(s)


def flatten_profile_items(profile: dict) -> Set[str]:
    """프로필 평탄화 (호환성 래퍼)"""
    return SnapshotManager.flatten_profile_items(profile)


def near_duplicate_log(
    session_id: str, log_emb: List[float], ym_min: int
) -> Tuple[bool, float]:
    """근사중복 로그 검색 (호환성 래퍼)"""
    manager = get_snapshot_manager()
    return manager.near_duplicate_log(session_id, log_emb, ym_min)


async def ensure_workers() -> None:
    """스냅샷 워커 초기화 (호환성 래퍼)"""
    manager = get_snapshot_manager()
    await manager.ensure_workers()


def enqueue_snapshot(user_id: str, session_id: str) -> None:
    """스냅샷 큐에 추가 (호환성 래퍼)"""
    manager = get_snapshot_manager()
    manager.enqueue_snapshot(user_id, session_id)


def edge_and_debounce(
    user_id: str, session_id: str, tokens_prev: int, tokens_now: int
) -> None:
    """에지 트리거 + 디바운스 (호환성 래퍼)"""
    manager = get_snapshot_manager()
    manager.edge_and_debounce(user_id, session_id, tokens_prev, tokens_now)
