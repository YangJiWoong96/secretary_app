"""
backend/scripts/debug_behavior_flow.py - Behavior 파이프라인 디버깅

특정 사용자의 Behavior Slot → Scoreboard → RAG 파이프라인 전체 흐름 출력.

실행 방법:
    python -m backend.scripts.debug_behavior_flow <user_id>
"""

import json
import sys

import redis

from backend.config import get_settings
from backend.personalization.preference_scoreboard import PreferenceScoreboard


def audit_behavior_pipeline(user_id: str) -> None:
    """
    Behavior 파이프라인 디버깅

    출력 내용:
    - Scoreboard 등록 키 (pref:idx:{user_id})
    - 각 norm_key의 점수/상태/증거 수
    - Milvus behavior_slots 컬렉션 데이터
    """
    settings = get_settings()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    print(f"\n===== Behavior Pipeline Audit: {user_id} =====\n")

    # 1) Scoreboard 인덱스
    idx_key = f"pref:idx:{user_id}"
    norm_keys = r.smembers(idx_key) or []
    print(f"[Scoreboard Index] {len(norm_keys)} norm_keys registered:")

    if not norm_keys:
        print("  - (no keys found)")
        return

    # 2) 각 norm_key 상세 정보
    for norm_key in sorted(norm_keys):
        pref_key = f"pref:{user_id}:{norm_key}"
        data = r.get(pref_key)

        if not data:
            print(f"  - {norm_key}: (data not found)")
            continue

        try:
            entry = json.loads(data)
            score = entry.get("score", 0.0)
            status = entry.get("status", "pending")
            evidence_count = entry.get("evidence_count", 0)
            stability = entry.get("stability", 0.5)

            print(f"  - {norm_key}:")
            print(
                f"      score={score:.3f}, status={status}, evidence={evidence_count}, stability={stability:.3f}"
            )
        except Exception as e:
            print(f"  - {norm_key}: (parse error: {e})")

    # 3) Milvus behavior_slots 데이터
    print(f"\n[Milvus behavior_slots]:")
    try:
        from backend.rag.milvus import ensure_behavior_collection

        coll = ensure_behavior_collection()
        results = coll.query(
            expr=f"user_id == '{user_id}'",
            output_fields=[
                "slot_key",
                "norm_key",
                "value",
                "status",
                "confidence",
                "updated_at",
            ],
            limit=100,
        )

        if not results:
            print("  - (no slots found)")
        else:
            print(f"  - Found {len(results)} slots:")
            for item in results[:20]:  # 최대 20개만 출력
                slot_key = item.get("slot_key", "?")
                norm_key = item.get("norm_key", "?")
                value = item.get("value", "?")
                status = item.get("status", "pending")
                confidence = item.get("confidence", 0.5)

                print(
                    f"    - {slot_key} ({norm_key}): value={value}, status={status}, conf={confidence:.2f}"
                )
    except Exception as e:
        print(f"  - (Milvus query failed: {e})")

    # 4) Top-5 선호도
    print(f"\n[Top-5 Preferences]:")
    try:
        sb = PreferenceScoreboard(settings.REDIS_URL)
        top_prefs = sb.get_top_n(user_id, n=5)

        if not top_prefs:
            print("  - (no preferences found)")
        else:
            for i, item in enumerate(top_prefs, 1):
                norm_key = item["norm_key"]
                score = item["score"]
                status = item["status"]
                print(f"  {i}. {norm_key}: score={score:.3f}, status={status}")
    except Exception as e:
        print(f"  - (Top-5 query failed: {e})")

    print(f"\n===== End Audit =====\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.debug_behavior_flow <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    audit_behavior_pipeline(user_id)


if __name__ == "__main__":
    main()
