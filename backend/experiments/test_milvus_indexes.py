"""
간이 테스트: Milvus 컬렉션 인덱스 생성/보장 시 STL_SORT 오류가 발생하지 않는지 확인

실행 예:
    python -m backend.experiments.test_milvus_indexes
"""

from __future__ import annotations

from pymilvus import Collection

from backend.rag.milvus import (
    ensure_behavior_collection,
    ensure_collections,
    ensure_profile_collection,
)


def _describe(coll: Collection) -> str:
    fields = [(f.name, str(f.dtype)) for f in coll.schema.fields]
    return f"{coll.name}: {fields}"


def main() -> None:
    prof, log = ensure_collections()
    prof_chunks = ensure_profile_collection()
    beh = ensure_behavior_collection()
    print(_describe(prof))
    print(_describe(log))
    print(_describe(prof_chunks))
    print(_describe(beh))
    print("OK")


if __name__ == "__main__":
    main()
