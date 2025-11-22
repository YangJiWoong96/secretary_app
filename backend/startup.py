"""
backend.startup - 애플리케이션 시작 시 초기화

모든 서비스, 워커, 스케줄러를 초기화합니다.
"""

import asyncio
import logging

from backend.search_engine.router import initialize_seed_vectors

logger = logging.getLogger("startup")


async def initialize_all_services() -> None:
    """
    모든 백그라운드 서비스 및 리소스 초기화

    초기화 항목:
    1. 스냅샷 워커 (장기 메모리 업데이트 큐 처리)
    2. Directive 워커 및 배치 스케줄러
    3. 임베딩 라우터 시드 데이터 로드
    4. NER/Gemma 모델 프리워밍
    5. Milvus 컬렉션 확인
    6. 프로액티브 스케줄러
    7. 프로액티브 랭커 재학습 스케줄러
    """
    logger.info("[startup] Initializing all services...")

    # 1. 스냅샷 워커 (장기 메모리)
    try:
        from backend.policy.snapshot_manager import get_snapshot_manager

        manager = get_snapshot_manager()
        await manager.ensure_workers()
        logger.info("[startup] ✅ Snapshot workers initialized")
    except Exception as e:
        logger.warning(f"[startup] ❌ Snapshot workers init failed: {e}")

    # 2. Directive 워커
    try:
        from backend.directives.pipeline import ensure_directive_workers

        await ensure_directive_workers()
        logger.info("[startup] ✅ Directive workers initialized")
    except Exception as e:
        logger.warning(f"[startup] ❌ Directive workers init failed: {e}")

    # 3. Directive 배치 스케줄러
    try:
        from backend.directives.scheduler import ensure_daily_scheduler

        await ensure_daily_scheduler()
        logger.info("[startup] ✅ Directive scheduler ready")
    except Exception as e:
        logger.warning(f"[startup] ❌ Directive scheduler init failed: {e}")

    # 4. 임베딩 라우터 시드 로드 (비동기 프리로드)
    try:
        from backend.routing.intent_router import ensure_intent_embeddings

        asyncio.create_task(asyncio.to_thread(ensure_intent_embeddings))
        logger.info("[startup] ✅ Intent embeddings preload scheduled")
    except Exception as e:
        logger.warning(f"[startup] ❌ Intent embeddings init failed: {e}")

    # 4.5 JSON Schema 레지스트리 부트 검증
    try:
        from backend.utils.schema_registry import validate_schema_registry

        validate_schema_registry()
        logger.info("[startup] ✅ Schema registry validated")
    except Exception as e:
        logger.error(f"[startup] ❌ Schema registry invalid: {e}")
        raise

    # 5. 검색 엔진 라우터 시드 로드 (Search Engine Router)
    try:
        await asyncio.to_thread(initialize_seed_vectors)
        logger.info("[startup] ✅ Search Engine router seed vectors ready")
    except Exception as e:
        logger.warning(f"[startup] ❌ Search Engine router init failed: {e}")

    # 6. NER/Gemma 모델 프리워밍
    try:

        async def _warmup_models():
            try:
                from backend.memory.stwm import update_stwm

                # NER 모델 로드 (첫 호출 트리거)
                update_stwm("_warmup_", "안녕하세요")
            except Exception:
                pass

            try:
                from backend.rag.embeddings import embed_query_gemma

                # Gemma 모델 로드
                embed_query_gemma("서울")
            except Exception:
                pass

        asyncio.create_task(_warmup_models())
        logger.info("[startup] ✅ Model warmup scheduled")
    except Exception as e:
        logger.warning(f"[startup] ❌ Model warmup failed: {e}")

    # 6.1 Kiwi 프리워밍(이벤트 루프 비차단 방식)
    try:
        from backend.memory.stwm import _ensure_kiwi  # type: ignore

        await asyncio.to_thread(_ensure_kiwi)  # heavy init은 스레드로 오프로딩
        logger.info("[startup] ✅ Kiwi prewarm done")
    except Exception as e:
        logger.warning(f"[startup] ❌ Kiwi prewarm failed: {e}")

    # 7. Milvus 컬렉션 확인
    try:
        from backend.rag import ensure_collections
        from backend.rag.milvus import ensure_behavior_collection  # type: ignore

        ensure_collections()
        try:
            ensure_behavior_collection()
        except Exception:
            pass
        logger.info("[startup] ✅ Milvus collections ready")
    except Exception as e:
        logger.warning(f"[startup] ❌ Milvus init failed: {e}")

    # 8. 프로액티브 스케줄러
    try:
        from backend.proactive.scheduler import ensure_proactive_scheduler

        await ensure_proactive_scheduler()
        logger.info("[startup] ✅ Proactive scheduler ready")
    except Exception as e:
        logger.warning(f"[startup] ❌ Proactive scheduler init failed: {e}")

    # 9. 프로액티브 랭커 재학습 스케줄러
    try:
        from backend.proactive.retrain import ensure_daily_retrainer

        await ensure_daily_retrainer()
        logger.info("[startup] ✅ Proactive retrainer ready")
    except Exception as e:
        logger.warning(f"[startup] ❌ Proactive retrainer init failed: {e}")

    # 10. Preference 이벤트 버스 소비자
    try:
        from backend.config import get_settings as _gs
        from backend.personalization.events_bus import ensure_scoreboard_consumer

        asyncio.create_task(ensure_scoreboard_consumer(_gs().REDIS_URL))
        logger.info("[startup] ✅ Preference scoreboard consumer scheduled")
    except Exception as e:
        logger.warning(f"[startup] ❌ Preference consumer init failed: {e}")

    logger.info("[startup] ✨ All services initialized successfully!")
