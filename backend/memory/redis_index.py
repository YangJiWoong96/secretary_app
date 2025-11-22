"""
backend.memory.redis_index - Redis 키 인덱스 관리 유틸리티

목적:
- KEYS 명령 금지 정책 하에서 사용자별로 생성되는 Redis 키를 인덱스 세트에 등록/관리한다.
- 모든 신규 키 생성 시 `idx:{user_id}` 세트에 등록하여 O(1) 조회 및 일괄 삭제(GDPR 대응)를 가능하게 한다.
- 인덱스 누락 시 SCAN 기반 폴백으로 키를 재발견하고 인덱스를 재구축한다.
"""

from __future__ import annotations

import logging
from typing import List, Set

import redis

from backend.config import get_settings

logger = logging.getLogger("redis_index")


def _get_client() -> redis.Redis:
    """설정에서 Redis 클라이언트를 생성/반환"""
    settings = get_settings()
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _index_key(user_id: str) -> str:
    """사용자 인덱스 세트 키 생성"""
    return f"idx:{user_id}"


def register_key(user_id: str, key: str) -> None:
    """
    인덱스 세트에 키를 등록한다.

    주의: 키가 아직 실제로 생성되지 않았더라도 등록은 허용한다.
    이는 사전 등록을 통해 삭제/조회 경로를 단순화하기 위함이다.
    """
    try:
        r = _get_client()
        r.sadd(_index_key(user_id), key)
    except Exception as e:
        logger.warning(f"[idx] register_key error: user={user_id} key={key} err={e}")


def get_user_keys(user_id: str) -> List[str]:
    """
    인덱스 세트에서 사용자 키 목록을 조회한다. 인덱스가 비어있으면 SCAN 폴백을 수행한다.

    현재 키 스키마:
    - 트랜잭션 로그:  txlog:{user_id}:{session_id}
    - MTM 인덱스:    mtm:{user_id} (Cross-Session 단일 키)
    - 대화 히스토리: message_store:{user_id}:{session_id} (복합 키)
    """
    try:
        r = _get_client()
        idx_key = _index_key(user_id)
        members = list(r.smembers(idx_key) or [])
        if members:
            return members

        # 폴백: SCAN으로 재발견 후 인덱스 재구축
        patterns = [
            f"txlog:{user_id}:*",
            f"mtm:{user_id}",  # Cross-Session 단일 키
            f"message_store:{user_id}:*",  # 복합 키 패턴
        ]
        found: Set[str] = set()
        for pat in patterns:
            try:
                for k in r.scan_iter(match=pat, count=1000):
                    found.add(k)
            except Exception as se:
                logger.warning(f"[idx] scan_iter pattern={pat} error: {se}")

        if found:
            try:
                # 재구축
                r.sadd(idx_key, *list(found))
            except Exception as se:
                logger.warning(f"[idx] rebuild index error: {se}")
        return list(found)
    except Exception as e:
        logger.warning(f"[idx] get_user_keys error: user={user_id} err={e}")
        return []


def delete_user_data(user_id: str) -> int:
    """
    사용자 데이터 키를 일괄 삭제한다. (GDPR 대응)

    절차:
    1) 인덱스/폴백으로 키 수집
    2) MTM ZSET이 발견되면 해당 멤버로부터 `mtm:item:{item_id}`도 함께 삭제
    3) 실제 키 삭제 후 인덱스 세트 자체도 삭제

    Returns:
        삭제 시도한 키 개수(인덱스 세트 포함)
    """
    try:
        r = _get_client()
        keys = get_user_keys(user_id)

        # MTM 항목 해시도 함께 제거
        mtm_item_keys: List[str] = []
        for k in keys:
            # Cross-Session MTM 키: mtm:{user_id}
            if k == f"mtm:{user_id}":
                try:
                    members = r.zrange(k, 0, -1) or []
                    for mid in members:
                        mtm_item_keys.append(f"mtm:item:{mid}")
                except Exception as se:
                    logger.warning(f"[idx] zrange error on {k}: {se}")
            # 레거시 MTM 키도 처리 (마이그레이션 완료 전 호환성)
            elif k.startswith(f"mtm:{user_id}:"):
                try:
                    members = r.zrange(k, 0, -1) or []
                    for mid in members:
                        mtm_item_keys.append(f"mtm:item:{mid}")
                except Exception as se:
                    logger.warning(f"[idx] zrange error on {k}: {se}")

        all_keys = list(dict.fromkeys(keys + mtm_item_keys))  # 중복 제거, 순서 유지

        pipe = r.pipeline()
        for k in all_keys:
            pipe.delete(k)
        # 인덱스 세트 삭제
        pipe.delete(_index_key(user_id))
        try:
            results = pipe.execute()
            # DELETE 결과 합산(성공/실패 무관, 시도한 개수 반환)
            return len(all_keys) + 1
        except Exception as pe:
            logger.warning(f"[idx] pipeline delete error: {pe}")
            return 0
    except Exception as e:
        logger.warning(f"[idx] delete_user_data error: user={user_id} err={e}")
        return 0
