"""
backend/scripts/audit_memory_scope.py - 메모리 스코핑 디버깅 도구

특정 사용자의 모든 Redis 메모리 키를 출력하여 스코핑 규칙 검증.

실행 방법:
    python -m backend.scripts.audit_memory_scope <user_id>
"""

import sys

import redis

from backend.config import get_settings


def audit_user_memory(user_id: str) -> None:
    """
    사용자의 모든 메모리 키 출력 (디버깅용)

    출력 내용:
    - Redis Index 등록 키
    - STM 키 (message_store:{user_id}:{session_id})
    - MTM 키 (mtm:{user_id})
    - MTM Item 키 (mtm:item:{item_id})
    - 세션→사용자 매핑 (router:session_user:*)
    """
    settings = get_settings()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    print(f"\n===== Memory Audit: {user_id} =====\n")

    # 1) Redis Index
    idx_key = f"idx:{user_id}"
    registered_keys = r.smembers(idx_key)
    print(f"[Redis Index] {len(registered_keys)} keys registered:")
    for k in sorted(registered_keys):
        print(f"  - {k}")

    # 2) STM Keys (복합 키 패턴)
    stm_pattern = f"message_store:{user_id}:*"
    stm_keys = list(r.scan_iter(match=stm_pattern, count=1000))
    print(f"\n[STM Keys] {len(stm_keys)} found:")
    for k in stm_keys:
        msg_count = r.llen(k)
        print(f"  - {k} ({msg_count} messages)")

    # 3) MTM Keys (Cross-Session 단일 키)
    mtm_key = f"mtm:{user_id}"
    mtm_exists = r.exists(mtm_key)
    print(f"\n[MTM Keys] Cross-Session key: {mtm_key}")
    if mtm_exists:
        summary_count = r.zcard(mtm_key)
        print(f"  - {mtm_key} ({summary_count} summaries)")

        # MTM Item 샘플 출력
        item_ids = r.zrevrange(mtm_key, 0, 4)  # 최신 5개
        for iid in item_ids:
            hkey = f"mtm:item:{iid}"
            item = r.hgetall(hkey)
            sess = item.get("session_id", "?")
            ts = item.get("ts", "?")
            access = item.get("access_count", "0")
            print(f"    - {iid} | session={sess} | ts={ts} | access={access}")
    else:
        print(f"  - {mtm_key} (not found)")

    # 4) 레거시 MTM Keys (마이그레이션 후에는 없어야 함)
    legacy_mtm_pattern = f"mtm:{user_id}:*"
    legacy_mtm_keys = list(r.scan_iter(match=legacy_mtm_pattern, count=1000))
    legacy_mtm_keys = [k for k in legacy_mtm_keys if ":item:" not in k]
    if legacy_mtm_keys:
        print(
            f"\n[LEGACY MTM Keys] {len(legacy_mtm_keys)} found (should be 0 after migration):"
        )
        for k in legacy_mtm_keys:
            summary_count = r.zcard(k) if r.type(k) == "zset" else 0
            print(f"  - {k} ({summary_count} summaries)")

    # 5) Session Mapping (역방향 조회)
    print(f"\n[Session Mapping] (sessions mapped to user={user_id}):")
    mapping_pattern = "router:session_user:*"
    found_sessions = []
    for k in r.scan_iter(match=mapping_pattern, count=1000):
        v = r.get(k)
        if v == user_id:
            session_id = k.split(":", 2)[2]
            found_sessions.append(session_id)
            print(f"  - session={session_id} → user={v}")

    if not found_sessions:
        print("  - (no sessions found)")

    print(f"\n===== End Audit =====\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.audit_memory_scope <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    audit_user_memory(user_id)


if __name__ == "__main__":
    main()
