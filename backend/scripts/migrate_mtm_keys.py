"""
backend/scripts/migrate_mtm_keys.py - MTM Cross-Session 마이그레이션

기존 세션별 MTM 키를 사용자별 단일 키로 병합한다.

실행 방법:
    python -m backend.scripts.migrate_mtm_keys [--dry-run]

옵션:
    --dry-run: 실제 변경 없이 마이그레이션 계획만 출력
"""

import argparse
import logging
import sys
from collections import defaultdict
from typing import Dict, List

import redis

from backend.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_mtm")


def migrate_to_cross_session_mtm(dry_run: bool = False) -> None:
    """
    MTM 키 마이그레이션: 세션별 → 사용자별 Cross-Session

    절차:
    1. 모든 mtm:{user_id}:{session_id} 키 발견
    2. user_id별로 그룹화
    3. 각 user_id에 대해:
       a. 신규 mtm:{user_id} ZSET 생성
       b. 모든 세션의 item_id를 타임스탬프 순으로 병합
       c. 각 mtm:item:{item_id} HASH에 user_id 필드 추가
    4. 레거시 mtm:{user_id}:{session_id} 키 삭제

    Args:
        dry_run: True이면 실제 변경 없이 계획만 출력
    """
    settings = get_settings()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Step 1: 레거시 MTM 키 발견
    logger.info("[migrate] Scanning legacy MTM keys: mtm:*:*")
    legacy_keys: List[str] = []
    try:
        for k in r.scan_iter(match="mtm:*:*", count=1000):
            # mtm:item:* 제외
            if ":item:" in k:
                continue
            # mtm:{user_id}:{session_id} 형태만 선택
            parts = k.split(":")
            if len(parts) == 3 and parts[0] == "mtm":
                legacy_keys.append(k)
    except Exception as e:
        logger.error(f"[migrate] SCAN failed: {e}")
        sys.exit(1)

    if not legacy_keys:
        logger.info("[migrate] No legacy MTM keys found. Migration complete.")
        return

    logger.info(f"[migrate] Found {len(legacy_keys)} legacy MTM keys")

    # Step 2: user_id별 그룹화
    user_groups: Dict[str, List[str]] = defaultdict(list)
    for k in legacy_keys:
        # mtm:{user_id}:{session_id}
        parts = k.split(":", 2)
        if len(parts) != 3:
            continue
        user_id = parts[1]
        user_groups[user_id].append(k)

    logger.info(f"[migrate] Grouped into {len(user_groups)} users")

    # Step 3: 사용자별 마이그레이션
    for user_id, keys in user_groups.items():
        logger.info(f"[migrate] Processing user: {user_id} ({len(keys)} sessions)")

        new_zkey = f"mtm:{user_id}"

        # 이미 신규 키가 존재하는지 확인
        if r.exists(new_zkey) and not dry_run:
            logger.warning(
                f"[migrate] {new_zkey} already exists, skipping user {user_id}"
            )
            continue

        # 모든 세션의 item_id와 score 수집
        all_items: List[tuple] = []
        for legacy_key in keys:
            try:
                # ZSET 멤버 및 score 조회
                members = r.zrange(legacy_key, 0, -1, withscores=True)
                for item_id, score in members:
                    all_items.append((item_id, score))
            except Exception as e:
                logger.warning(f"[migrate] ZRANGE failed for {legacy_key}: {e}")

        if not all_items:
            logger.info(f"[migrate] No items found for user {user_id}, skipping")
            continue

        logger.info(f"[migrate] Collected {len(all_items)} items for user {user_id}")

        if dry_run:
            logger.info(
                f"[DRY-RUN] Would create {new_zkey} with {len(all_items)} items"
            )
            logger.info(f"[DRY-RUN] Would delete {len(keys)} legacy keys")
            continue

        # Step 3a: 신규 ZSET 생성
        try:
            pipe = r.pipeline()
            for item_id, score in all_items:
                pipe.zadd(new_zkey, {item_id: score})
            pipe.execute()
            logger.info(f"[migrate] Created {new_zkey}")
        except Exception as e:
            logger.error(f"[migrate] Failed to create {new_zkey}: {e}")
            continue

        # Step 3b: 각 item HASH에 user_id 필드 추가
        try:
            pipe = r.pipeline()
            for item_id, _ in all_items:
                hkey = f"mtm:item:{item_id}"
                pipe.hset(hkey, "user_id", user_id)
            pipe.execute()
            logger.info(f"[migrate] Updated {len(all_items)} item HASHes with user_id")
        except Exception as e:
            logger.warning(f"[migrate] Failed to update item HASHes: {e}")

        # Step 3c: 레거시 키 삭제
        try:
            pipe = r.pipeline()
            for legacy_key in keys:
                pipe.delete(legacy_key)
            pipe.execute()
            logger.info(f"[migrate] Deleted {len(keys)} legacy keys")
        except Exception as e:
            logger.error(f"[migrate] Failed to delete legacy keys: {e}")

        # Step 3d: 인덱스 갱신
        try:
            from backend.memory.redis_index import register_key

            register_key(user_id, new_zkey)
            logger.info(f"[migrate] Registered {new_zkey} in index")
        except Exception as e:
            logger.warning(f"[migrate] Failed to register index: {e}")

    logger.info("[migrate] Migration complete")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate MTM to Cross-Session structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without making changes",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("[migrate] Running in DRY-RUN mode")

    migrate_to_cross_session_mtm(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
