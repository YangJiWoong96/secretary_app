# C:\My_Business\backend\rag\milvus.py
import importlib
import inspect as _inspect
import logging
import os
import time
from threading import Lock

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .config import (
    EMBEDDING_DIM,
    LOG_COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_PORT,
    PROFILE_COLLECTION_NAME,
)

logger = logging.getLogger("milvus")


_conn_alias = "default"
_coll_lock = Lock()
_prof_coll = None
_log_coll = None
_profile_chunks_coll = None
_behavior_coll = None
_ready = False


def ensure_milvus():
    if not connections.has_connection(_conn_alias):
        connections.connect(_conn_alias, host=MILVUS_HOST, port=MILVUS_PORT)


def _safe_create_scalar_index(coll: Collection, field_name: str) -> None:
    """
    STL_SORT는 숫자형 필드에서만 지원된다.
    - 숫자형(INT32/INT64/FLOAT/DOUBLE)인 경우에만 STL_SORT 적용
    - 그 외(VARCHAR/JSON/ARRAY 등)에는 인덱스 생성을 건너뛰고 경고 로그
    """
    try:
        ftype = None
        for f in coll.schema.fields:
            if getattr(f, "name", None) == field_name:
                ftype = getattr(f, "dtype", None)
                break
        if ftype in (DataType.INT64, DataType.INT32, DataType.FLOAT, DataType.DOUBLE):
            try:
                coll.create_index(
                    field_name=field_name, index_params={"index_type": "STL_SORT"}
                )
                logger.info(
                    f"[milvus] Created STL_SORT index on numeric field '{field_name}' in {coll.name}"
                )
            except Exception as ie:
                logger.warning(
                    f"[milvus] scalar index create failed for field='{field_name}': {ie}"
                )
        else:
            logger.warning(
                f"[milvus] Skip STL_SORT for non-numeric field '{field_name}' "
                f"(dtype={ftype}) in collection={coll.name}"
            )
    except Exception as e:
        logger.warning(
            f"[milvus] safe scalar index error for field='{field_name}': {e}"
        )


def _log_profile_schema_origin() -> None:
    """profile_schema 모듈의 실제 로딩 경로를 로깅한다."""
    try:
        mod = importlib.import_module("backend.rag.profile_schema")
        logger.info(
            "[milvus] profile_schema module=%s path=%s",
            getattr(mod, "__name__", "<unknown>"),
            _inspect.getfile(mod),
        )
    except Exception as e:
        logger.warning(f"[milvus] profile_schema origin inspect failed: {e}")


def _log_schema_params(schema: CollectionSchema) -> None:
    """VARCHAR/ARRAY 필드 파라미터를 상세 로깅한다."""
    try:
        for f in schema.fields:
            try:
                params = getattr(f, "params", {})
                # ARRAY(VARCHAR)의 element_type/params 추정 로깅 (버전별 속성 상이할 수 있음)
                element_type = getattr(f, "element_type", None)
                element_type_params = getattr(f, "element_type_params", None)
                logger.info(
                    "[milvus] field name=%s params=%s element_type=%s element_type_params=%s",
                    getattr(f, "name", "<unknown>"),
                    params,
                    element_type,
                    element_type_params,
                )
            except Exception as ie:
                logger.warning(f"[milvus] field inspect failed: {ie}")
    except Exception as e:
        logger.warning(f"[milvus] schema params inspect failed: {e}")

    # 간결 요약 로깅으로 대체(중앙 로거 사용)
    try:
        from pymilvus import DataType

        from backend.rag.profile_schema import PROFILE_SCHEMA
        from backend.utils.logger import log_event

        fields = []
        for f in PROFILE_SCHEMA.fields:
            item = {
                "name": f.name,
                "dtype": str(f.dtype),
                "params": getattr(f, "params", {}),
            }
            if f.dtype == DataType.ARRAY:
                item["element_type"] = str(getattr(f, "element_type", None))
                item["element_type_params"] = getattr(f, "element_type_params", None)
            fields.append(item)
        log_event("milvus_profile_schema_summary", {"fields": fields})
    except Exception:
        pass


def _create_collection(name: str, desc: str) -> Collection:
    if utility.has_collection(name):
        coll = Collection(name)
        # dim 검증
        for f in coll.schema.fields:
            if f.name == "embedding":
                dim = f.params.get("dim")
                if dim != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Milvus collection dim mismatch: {dim} != {EMBEDDING_DIM}"
                    )
        return coll

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("user_id", DataType.VARCHAR, max_length=256),
            FieldSchema("type", DataType.VARCHAR, max_length=50),
            FieldSchema("created_at", DataType.INT64),
            FieldSchema("date_start", DataType.INT64),
            FieldSchema("date_end", DataType.INT64),
            FieldSchema("date_ym", DataType.INT64),
        ],
        desc,
    )
    coll = Collection(name, schema)
    coll.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return coll


def ensure_partition(coll: Collection, ym: int) -> str:
    """
    Milvus 컬렉션에 파티션 생성 (월 단위)

    Args:
        coll: Milvus Collection 객체
        ym: YYYYMM 형식 정수 (예: 202508)

    Returns:
        str: 파티션 이름 (예: "ym_202508")
    """
    part_name = f"ym_{ym}"
    try:
        if part_name not in [p.name for p in coll.partitions]:
            coll.create_partition(partition_name=part_name, description=f"YYYYMM={ym}")
            logger.info(f"[milvus] Created partition {part_name} in {coll.name}")
    except Exception as e:
        logger.warning(f"[milvus] Ensure partition error: {e}")
    return part_name


def create_milvus_collection(name: str, desc: str) -> Collection:
    """
    Milvus 컬렉션 생성 또는 재사용 (호환성 래퍼)

    기존 컬렉션이 있으면 dim 검증 후 재사용, 없으면 새로 생성합니다.

    Args:
        name: 컬렉션 이름
        desc: 컬렉션 설명

    Returns:
        Collection: Milvus Collection 객체

    Note:
        이 함수는 호환성을 위해 유지되며, 내부적으로는 _create_collection을 사용합니다.
    """
    if utility.has_collection(name):
        coll = Collection(name)

        # Embedding dim 검증
        for f in coll.schema.fields:
            if f.name == "embedding":
                existing_dim = f.params.get("dim")
                if existing_dim != EMBEDDING_DIM:
                    logger.error(
                        f"[milvus] Dim mismatch for {name}: "
                        f"existing={existing_dim} expected={EMBEDDING_DIM}"
                    )
                    raise RuntimeError(
                        f"Milvus collection dim mismatch. "
                        f"Create a new collection with correct dim."
                    )

        # 날짜 필드 검증
        have_dates = set(x.name for x in coll.schema.fields)
        expected = {"date_start", "date_end", "date_ym"}
        missing = expected - have_dates
        if missing:
            logger.warning(
                f"[milvus] Collection {name} missing date fields {missing}. "
                f"Consider migrating to v3."
            )

        logger.info(f"[milvus] Reuse collection={name}")
        return coll

    # 새 컬렉션 생성
    return _create_collection(name, desc)


def ensure_collections():
    global _prof_coll, _log_coll, _profile_chunks_coll, _ready
    if _ready and _prof_coll and _log_coll:
        return _prof_coll, _log_coll
    with _coll_lock:
        if _ready and _prof_coll and _log_coll:
            return _prof_coll, _log_coll
        ensure_milvus()
        _prof_coll = _create_collection(PROFILE_COLLECTION_NAME, "User Profiles")
        _log_coll = _create_collection(LOG_COLLECTION_NAME, "Conversation Logs")
        _prof_coll.load()
        _log_coll.load()
        try:
            # profile_chunks 컬렉션 확보 (존재 시 재사용, 없으면 생성)
            _log_profile_schema_origin()
            from .profile_schema import PROFILE_INDEXES, PROFILE_SCHEMA

            # 진단용: 우리가 사용할 스키마 파라미터를 로그로 남긴다
            _log_schema_params(PROFILE_SCHEMA)

            # 옵션: 초기화 시 강제 드롭 (메타 꼬임/이전 스키마 충돌 방지)
            # 운영 안정성을 위해 기본값을 0으로 둔다.
            from backend.config import get_settings as _gs

            if bool(getattr(_gs(), "MILVUS_RESET_PROFILE_CHUNKS", False)):
                try:
                    if utility.has_collection("profile_chunks"):
                        utility.drop_collection("profile_chunks")
                        logger.warning(
                            "[milvus] Dropped existing profile_chunks (RESET flag)"
                        )
                except Exception as de:
                    logger.warning(
                        f"[milvus] drop_collection(profile_chunks) failed: {de}"
                    )

            if utility.has_collection("profile_chunks"):
                _profile_chunks_coll = Collection("profile_chunks")
                try:
                    # 현재 서버에 존재하는 실제 스키마도 로깅해서 충돌 여부 진단
                    _log_schema_params(_profile_chunks_coll.schema)
                except Exception:
                    pass
            else:
                _profile_chunks_coll = Collection(
                    name="profile_chunks", schema=PROFILE_SCHEMA
                )
                # 인덱스 생성 (스칼라/HNSW)
                for idx in PROFILE_INDEXES:
                    field = idx.get("field")
                    params = {k: v for k, v in idx.items() if k != "field"}
                    _profile_chunks_coll.create_index(
                        field_name=field, index_params=params
                    )
            # 세션3: Tier 파티션 보장 (guard/core/dynamic)
            try:
                existing_parts = {p.name for p in _profile_chunks_coll.partitions}
                for pname in ("tier_guard", "tier_core", "tier_dynamic"):
                    if pname not in existing_parts:
                        _profile_chunks_coll.create_partition(
                            partition_name=pname, description=f"tier={pname}"
                        )
            except Exception as pe:
                logger.warning(f"[milvus] ensure tier partitions skipped: {pe}")
            _profile_chunks_coll.load()
            logger.info("[milvus] profile_chunks collection ready")
        except Exception as e:
            logger.warning(f"[milvus] profile_chunks init skipped: {e}")
            _profile_chunks_coll = None

        _ready = True
        return _prof_coll, _log_coll


def ensure_profile_collection() -> Collection:
    """
    profile_chunks 컬렉션을 보장하고 반환한다.

    - 존재하면 차원 검증 후 재사용
    - 없으면 스키마/인덱스 생성 후 로드
    """
    global _profile_chunks_coll
    if _profile_chunks_coll is not None:
        return _profile_chunks_coll
    with _coll_lock:
        if _profile_chunks_coll is not None:
            return _profile_chunks_coll
        ensure_milvus()
        try:
            _log_profile_schema_origin()
            from .profile_schema import PROFILE_INDEXES, PROFILE_SCHEMA

            # 진단용 스키마 파라미터 로깅
            _log_schema_params(PROFILE_SCHEMA)

            # 옵션: 강제 드롭 플래그 처리 (기본 0)
            from backend.config import get_settings as _gs2

            if bool(getattr(_gs2(), "MILVUS_RESET_PROFILE_CHUNKS", False)):
                try:
                    if utility.has_collection("profile_chunks"):
                        utility.drop_collection("profile_chunks")
                        logger.warning(
                            "[milvus] Dropped existing profile_chunks (RESET flag)"
                        )
                except Exception as de:
                    logger.warning(
                        f"[milvus] drop_collection(profile_chunks) failed: {de}"
                    )

            if utility.has_collection("profile_chunks"):
                coll = Collection("profile_chunks")
                # dim 검증
                for f in coll.schema.fields:
                    if f.name == "embedding":
                        dim = f.params.get("dim")
                        if dim != EMBEDDING_DIM:
                            raise RuntimeError(
                                f"Milvus profile_chunks dim mismatch: {dim} != {EMBEDDING_DIM}"
                            )
                try:
                    _log_schema_params(coll.schema)
                except Exception:
                    pass
                # 스칼라 인덱스 보강(숫자형만 STL_SORT 적용)
                for field in ["user_id", "category", "norm_key", "status", "source"]:
                    _safe_create_scalar_index(coll, field)
                _profile_chunks_coll = coll
            else:
                coll = Collection(name="profile_chunks", schema=PROFILE_SCHEMA)
                # 인덱스 생성 (존재 시 에러 무시)
                for idx in PROFILE_INDEXES:
                    try:
                        field = idx.get("field")
                        params = {k: v for k, v in idx.items() if k != "field"}
                        coll.create_index(field_name=field, index_params=params)
                    except Exception as ie:
                        logger.warning(f"[milvus] index create skipped for {idx}: {ie}")
                # 스칼라 인덱스 생성(숫자형만 STL_SORT 적용)
                for field in ["user_id", "category", "norm_key", "status", "source"]:
                    _safe_create_scalar_index(coll, field)
                _profile_chunks_coll = coll

            # 세션3: Tier 파티션 보장 (guard/core/dynamic)
            try:
                existing_parts = {p.name for p in _profile_chunks_coll.partitions}
                for pname in ("tier_guard", "tier_core", "tier_dynamic"):
                    if pname not in existing_parts:
                        _profile_chunks_coll.create_partition(
                            partition_name=pname, description=f"tier={pname}"
                        )
            except Exception as pe:
                logger.warning(f"[milvus] ensure tier partitions skipped: {pe}")

            _profile_chunks_coll.load()
            logger.info("[milvus] ensure_profile_collection ready")
            return _profile_chunks_coll
        except Exception as e:
            logger.error(f"[milvus] ensure_profile_collection error: {e}")
            raise


def ensure_behavior_collection() -> Collection:
    """
    behavior_slots 컬렉션을 보장하고 반환한다.
    - 존재 시 차원 검증 후 재사용
    - 없으면 스키마/인덱스 생성 후 로드
    """
    global _behavior_coll
    if _behavior_coll is not None:
        return _behavior_coll
    with _coll_lock:
        if _behavior_coll is not None:
            return _behavior_coll
        ensure_milvus()
        from pymilvus import Collection, utility

        try:
            from backend.rag.behavior_schema import BEHAVIOR_SCHEMA, BEHAVIOR_INDEXES

            if utility.has_collection("behavior_slots"):
                coll = Collection("behavior_slots")
                # dim 검증
                for f in coll.schema.fields:
                    if f.name == "embedding":
                        dim = f.params.get("dim")
                        if dim != EMBEDDING_DIM:
                            raise RuntimeError(
                                f"Milvus behavior_slots dim mismatch: {dim} != {EMBEDDING_DIM}"
                            )
            else:
                coll = Collection(name="behavior_slots", schema=BEHAVIOR_SCHEMA)
                for idx in BEHAVIOR_INDEXES:
                    try:
                        coll.create_index(field_name=idx["field"], index_params=idx)
                    except Exception:
                        pass
            # 스칼라 인덱스(숫자형만 STL_SORT 적용)
            for field in ["user_id", "slot_key", "status"]:
                _safe_create_scalar_index(coll, field)
            coll.load()
            _behavior_coll = coll
            logger.info("[milvus] ensure_behavior_collection ready")
            return _behavior_coll
        except Exception as e:
            logger.error(f"[milvus] ensure_behavior_collection error: {e}")
            raise
