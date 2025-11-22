# C:\My_Business\backend\rag\snapshot_pipeline.py
"""
backend.rag.snapshot_pipeline - 장기 메모리 업데이트 파이프라인

Redis 단기 메모리를 Milvus 장기 메모리로 스냅샷하는 전체 파이프라인을 제공합니다.
"""

import json
import logging
import os
import time
from typing import Dict

from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from backend.routing.router_context import user_for_session
from backend.utils.tracing import traceable
from backend.utils.logger import safe_log_event

logger = logging.getLogger("snapshot_pipeline")


class SnapshotPipeline:
    """
    장기 메모리 스냅샷 파이프라인

    Redis 단기 메모리의 대화 히스토리를 분석하여:
    1. 구조화 + 요약 스냅샷 생성
    2. 사용자 프로필 갱신
    3. Milvus에 업서트 (중복 방지)

    특징:
    - 멱등성 보장 (sha256 해시)
    - 신규성 평가 (프로필 delta, 벡터 유사도)
    - 근사중복 감지 (SimHash + 벡터 검색)
    - TAGS 헤더 주입 (검색 최적화)
    """

    def __init__(self):
        """SnapshotPipeline 초기화"""
        self._settings = None
        self._llm_cold = None

    @property
    def settings(self):
        """설정 인스턴스 (지연 로딩)"""
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    @property
    def llm_cold(self):
        """메모리/요약 전용 LLM (지연 로딩)"""
        if self._llm_cold is None:
            from backend.config import get_llm_cold

            self._llm_cold = get_llm_cold()
        return self._llm_cold

    @traceable(
        name="Snapshot: update_long_term_memory",
        run_type="chain",
        tags=["memory", "snapshot", "rag"],
    )
    def update_long_term_memory(self, session_id: str) -> None:
        """
        장기 메모리 업데이트 (Redis → Milvus)

        Redis 대화 히스토리를 분석하여 스냅샷과 프로필을 Milvus에 저장합니다.

        처리 순서:
        1. Redis에서 메시지 로드
        2. old/recent 분리
        3. 타임스탬프 범위 추출
        4. 스냅샷 생성 (구조화 + 요약)
        5. 멱등성 체크
        6. 프로필 요약 및 갱신
        7. 신규성 평가
        8. Milvus 업서트 (profile, log)
        9. 상태 업데이트

        Args:
            session_id: 세션 ID

        Example:
            >>> pipeline = SnapshotPipeline()
            >>> pipeline.update_long_term_memory("user123")
        """
        logger.info(f"[snapshot_pipeline] Start for session_id={session_id}")
        # 세션→사용자 매핑(영구 스토어는 user_id 기준)
        mapped_user_id = user_for_session(session_id) or session_id

        # 1. Redis 메시지 로드
        history = RedisChatMessageHistory(
            session_id=session_id, url=self.settings.REDIS_URL
        )
        messages = history.messages

        if not messages:
            logger.info("[snapshot_pipeline] No messages → skip")
            return

        # 2. 메시지 분리 (old/recent)
        from backend.memory import split_for_summary

        old_msgs, recent_msgs = split_for_summary(
            messages, recent_budget=self.settings.RECENT_RAW_TOKENS_BUDGET
        )

        if not old_msgs:
            logger.info("[snapshot_pipeline] Nothing to summarize (old empty) → skip")
            return

        # 3. 타임스탬프 범위 추출
        from backend.utils.datetime_utils import extract_ts_bounds, now_kst, ym, ymd

        snap_start_dt, snap_end_dt = extract_ts_bounds(old_msgs, now_kst())
        ymd_start = ymd(snap_start_dt)
        ymd_end = ymd(snap_end_dt)
        ym_end = ym(snap_end_dt)

        logger.info(
            f"[snapshot_pipeline] Timestamp bounds: "
            f"start={snap_start_dt.isoformat()} end={snap_end_dt.isoformat()} "
            f"(ymd_start={ymd_start} ymd_end={ymd_end} ym_end={ym_end})"
        )

        # 4. 스냅샷 생성 (구조화 + 요약)
        from backend.memory import build_structured_and_summary, messages_to_text
        from backend.policy import sha256

        old_text = messages_to_text(old_msgs)
        snap_text, meta = build_structured_and_summary(session_id, old_msgs)
        snap_hash = sha256(old_text)

        # 5. 멱등성 체크
        from backend.policy import get_snapshot_manager

        manager = get_snapshot_manager()

        if manager.check_idempotency(session_id, snap_hash):
            logger.info("[snapshot_pipeline] Idempotent skip (same hash)")
            return

        manager.update_idempotency(session_id, snap_hash)

        # 6. 프로필 요약 및 갱신
        conv_all = messages_to_text(messages)

        # 6-1. 대화 전체 요약
        summary_chain = (
            ChatPromptTemplate.from_template(
                "다음 대화에서 사용자의 특징과 관계없는 인사말 등 불필요한 잡담과 내용은 모두 제거하고, "
                "사용자 프로필에 유의미한 핵심 정보만 요약해라.\n{conversation}"
            )
            | self.llm_cold
            | StrOutputParser()
        )
        summary_text = summary_chain.invoke({"conversation": conv_all})
        logger.info(
            f"[snapshot_pipeline] Profile summary length: {len(summary_text or 0)}"
        )

        # 6-2. 프로필 생성/갱신
        from backend.memory import model_supports_response_format
        from backend.policy import get_global_state
        from backend.policy.profile_schema import validate_user_profile

        state = get_global_state()
        old_prof = state.get_profile(session_id)
        old_prof_str = json.dumps(old_prof, ensure_ascii=False)

        # JSON 강제 모드
        if model_supports_response_format(self.settings.LLM_MODEL):
            llm_profile = self.llm_cold.bind(response_format={"type": "json_object"})
        else:
            llm_profile = self.llm_cold

        profile_chain = (
            ChatPromptTemplate.from_template(
                "[기존 프로필]\n{old}\n[요약된 최신 대화]\n{sum}\n"
                "다음 JSON 스키마에 맞춘 사용자 프로필을 생성/갱신하라."
                " 스키마 필수 키: name, location, occupation, interests(array), preferences(object), constraints(object)."
                " 기존 값과 충돌 시 최신 대화를 우선 반영하되, 빈 값은 내지 말고 적절히 보수적으로 채워라."
            )
            | llm_profile
            | StrOutputParser()
        )
        new_prof_str = profile_chain.invoke({"old": old_prof_str, "sum": summary_text})

        # 프로필 파싱 및 검증
        try:
            new_prof = json.loads(new_prof_str)

            # 스키마 검증
            try:
                validate_user_profile(new_prof)
            except Exception as e:
                logger.warning(
                    f"[snapshot_pipeline] Profile schema invalid → keep previous: {e}"
                )
                new_prof = old_prof

            state.set_profile(session_id, new_prof)
            logger.info(
                f"[snapshot_pipeline] Profile updated, keys={list(new_prof.keys())}"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"[snapshot_pipeline] Profile JSON decode failed: {e}")
            new_prof = old_prof

        # 7. Milvus 컬렉션 확보
        from backend.rag import (
            LOG_COLLECTION_NAME,
            PROFILE_COLLECTION_NAME,
            ensure_collections,
            ensure_partition,
        )
        from backend.rag.milvus import create_milvus_collection, ensure_milvus

        try:
            prof_coll, log_coll = ensure_collections()
        except Exception as e:
            logger.warning(f"[snapshot_pipeline] ensure_collections error: {e}")
            # 폴백
            ensure_milvus()
            prof_coll = create_milvus_collection(
                PROFILE_COLLECTION_NAME, "User Profiles"
            )
            log_coll = create_milvus_collection(
                LOG_COLLECTION_NAME, "Conversation Logs"
            )

        # 파티션 생성
        try:
            part_prof = ensure_partition(prof_coll, ym_end)
        except Exception:
            part_prof = None

        try:
            part_log = ensure_partition(log_coll, ym_end)
        except Exception:
            part_log = None

        # 8. 프로필 임베딩 및 업서트
        from backend.rag.embeddings import embed_query_openai

        prof_emb = embed_query_openai(json.dumps(new_prof, ensure_ascii=False))

        try:
            prof_coll.upsert(
                [
                    {
                        "id": f"{mapped_user_id}:profile_json",
                        "embedding": prof_emb,
                        "text": json.dumps(new_prof, ensure_ascii=False),
                        "user_id": mapped_user_id,
                        # 전체 JSON은 검색 오염을 막기 위해 별도 유형으로 표시
                        "type": "profile_json",
                        "created_at": int(time.time_ns()),
                        "date_start": ymd_start,
                        "date_end": ymd_end,
                        "date_ym": ym_end,
                    }
                ],
                partition_name=part_prof if part_prof else None,
            )
            logger.info(
                f"[snapshot_pipeline] Upserted profile_json for user_id={mapped_user_id}"
            )
            # 구조화 로깅(rag.profile_upsert)
            try:
                safe_log_event(
                    "rag.profile_upsert",
                    {
                        "user_id": mapped_user_id,
                        "collection": PROFILE_COLLECTION_NAME,
                        "text_len": len(json.dumps(new_prof, ensure_ascii=False)),
                        "vector_dim": len(prof_emb or []),
                        "chunk_count": 1,
                        "reason": "snapshot_pipeline",
                    },
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[snapshot_pipeline] Profile upsert error: {e}")

        # 9. 신규성 평가
        from backend.policy import flatten_profile_items

        try:
            old_prof_obj = json.loads(old_prof_str) if old_prof_str else {}
        except Exception:
            old_prof_obj = {}

        new_items = flatten_profile_items(
            new_prof if isinstance(new_prof, dict) else {}
        )
        old_items = flatten_profile_items(
            old_prof_obj if isinstance(old_prof_obj, dict) else {}
        )
        profile_delta_cnt = len(new_items - old_items)

        logger.info(
            f"[snapshot_pipeline] Profile delta: {profile_delta_cnt} "
            f"(min={self.settings.NOVELTY_MIN_PROFILE_DELTA})"
        )

        # 10. 정규화 해시 (완전중복 체크)
        try:
            norm = " ".join((snap_text or "").strip().lower().split())
            norm_hash = sha256(norm)
        except Exception:
            norm_hash = sha256(snap_text or "")

        # 11. SimHash 서명 (근사중복 빠른 배제)
        sig64 = 0
        try:
            from backend.rag.refs import _simhash64

            sig64 = _simhash64(snap_text or "")

            # Redis에 서명 저장 (중복 힌트)
            try:
                import redis

                r = redis.Redis.from_url(self.settings.REDIS_URL, decode_responses=True)
                sigkey = f"snap:sig:{session_id}"

                if r.sismember(sigkey, str(sig64)):
                    logger.info(
                        "[snapshot_pipeline] SimHash immediate-hit → likely duplicate"
                    )
            except Exception:
                pass
        except Exception:
            pass

        # 12. TAGS 헤더 생성 (검색 가중치용)
        from backend.memory import get_structure_schema
        from backend.policy.profile_utils import get_pinned_facts

        tags_line = ""
        try:
            struct_for_tags = (
                ChatPromptTemplate.from_template(
                    "대화에서 사실을 추출하라. JSON만 출력. 스키마:\n{schema_json}\n"
                    "[핀 고정(변경 금지)]:\n{pinned_json}\n\n[과거 대화]:\n{text_block}"
                )
                | self.llm_cold
                | StrOutputParser()
            ).invoke(
                {
                    "schema_json": json.dumps(
                        get_structure_schema(), ensure_ascii=False
                    ),
                    "pinned_json": json.dumps(
                        get_pinned_facts(session_id), ensure_ascii=False
                    ),
                    "text_block": old_text,
                }
            )

            struct_dict = json.loads(struct_for_tags)
            ents = (
                struct_dict.get("entities", []) if isinstance(struct_dict, dict) else []
            )
            kps = struct_dict.get("facts", []) if isinstance(struct_dict, dict) else []
            tags = list(dict.fromkeys([str(x) for x in (ents + kps) if x]))[:10]

            if tags:
                tags_line = "[TAGS] " + ",".join(tags) + "\n"
        except Exception:
            pass

        # 13. 의미 단위 청킹 (RecursiveCharacterTextSplitter)
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 청킹 설정: 문장 경계 보존, 오버랩 50 토큰
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            separators=["\n\n", ".", "?", "!", "\n", " "],
            length_function=len,
        )

        # 스냅샷 텍스트를 의미 단위로 분할
        chunks = splitter.split_text(snap_text)

        logger.info(
            f"[snapshot_pipeline] Semantic chunking: {len(chunks)} chunks "
            f"(avg ~{len(snap_text) // max(len(chunks), 1)} chars/chunk)"
        )

        # 14. 청크별 로그 텍스트 블록 및 임베딩 생성
        chunk_logs = []
        for idx, chunk in enumerate(chunks):
            text_blob = (
                f"[SNAPSHOT meta:turn_range=?, token_count=?, chunk={idx+1}/{len(chunks)}, "
                f"ver={meta['summary_version']}, model={meta['model']}]\n"
                + tags_line
                + chunk
            )

            log_emb = embed_query_openai(text_blob)
            chunk_logs.append(
                {
                    "text": text_blob,
                    "embedding": log_emb,
                    "chunk_idx": idx,
                }
            )

        # 15. 근사중복 검색 및 로그 업서트 (청크별)
        from backend.policy import near_duplicate_log
        from backend.utils.datetime_utils import ym_minus_months

        ym_min = ym_minus_months(now_kst(), self.settings.SNAPSHOT_LOOKBACK_MONTHS)

        # 16. 완전중복 체크 (정규화 해시)
        prev_hash = manager.get_session_value(session_id, "last_norm_hash")

        # 청크별로 중복 검사 및 업서트
        upserted_count = 0
        skipped_count = 0

        for chunk_log in chunk_logs:
            chunk_text = chunk_log["text"]
            chunk_emb = chunk_log["embedding"]
            chunk_idx = chunk_log["chunk_idx"]

            # 정규화 해시 (청크별)
            try:
                norm = " ".join((chunk_text or "").strip().lower().split())
                norm_hash = sha256(norm)
            except Exception:
                norm_hash = sha256(chunk_text or "")

            # 완전중복 체크
            if prev_hash == norm_hash:
                skipped_count += 1
                logger.info(f"[snapshot_pipeline] Chunk {chunk_idx}: 완전중복 스킵")
                continue

            # 근사중복 검색
            is_dup, dup_sim = near_duplicate_log(mapped_user_id, chunk_emb, ym_min)

            # 17. 신규성 게이트: 중복이거나 프로필 delta 부족하면 스킵
            if is_dup or profile_delta_cnt < self.settings.NOVELTY_MIN_PROFILE_DELTA:
                reason = "near-duplicate" if is_dup else "low-novelty(profile)"
                skipped_count += 1
                logger.info(
                    f"[snapshot_pipeline] Chunk {chunk_idx}: {reason} (sim={dup_sim:.3f})"
                )
                continue

            # 18. 로그 업서트
            try:
                chunk_id = f"{session_id}:{snap_hash}:chunk{chunk_idx}"
                log_coll.upsert(
                    [
                        {
                            "id": chunk_id,
                            "embedding": chunk_emb,
                            "text": chunk_text,
                            "user_id": mapped_user_id,
                            "type": "log",
                            "created_at": int(time.time_ns()),
                            "date_start": ymd_start,
                            "date_end": ymd_end,
                            "date_ym": ym_end,
                        }
                    ],
                    partition_name=part_log if part_log else None,
                )
                upserted_count += 1
                logger.info(
                    f"[snapshot_pipeline] Upserted chunk {chunk_idx} id={chunk_id} user_id={mapped_user_id}"
                )
                # per-chunk는 과도하므로 집계 로그에서 요약
            except Exception as e:
                logger.warning(
                    f"[snapshot_pipeline] Chunk {chunk_idx} upsert error: {e}"
                )

        logger.info(
            f"[snapshot_pipeline] Chunking results: {upserted_count} upserted, "
            f"{skipped_count} skipped (total {len(chunk_logs)} chunks)"
        )
        # 구조화 로깅(rag.log_upsert) - 집계
        try:
            vec_dim = 0
            try:
                if chunk_logs:
                    vec_dim = len(chunk_logs[0].get("embedding", []) or [])
            except Exception:
                vec_dim = 0
            safe_log_event(
                "rag.log_upsert",
                {
                    "user_id": mapped_user_id,
                    "collection": LOG_COLLECTION_NAME,
                    "chunk_count": upserted_count,
                    "skipped_count": skipped_count,
                    "vector_dim": vec_dim,
                    "reason": "snapshot_pipeline",
                },
            )
        except Exception:
            pass

        # 19. 상태 업데이트 (정규화 해시, SimHash)
        try:
            manager.set_session_value(mapped_user_id, "last_norm_hash", norm_hash)

            # Redis에 SimHash 집합 업데이트
            try:
                import redis

                r = redis.Redis.from_url(self.settings.REDIS_URL, decode_responses=True)
                sigkey = f"snap:sig:{mapped_user_id}"
                r.sadd(sigkey, str(sig64))

                ttl = int(os.getenv("SNAP_SIG_TTL_SEC", "15552000"))  # 180일
                r.expire(sigkey, ttl)
            except Exception:
                pass
        except Exception:
            pass

        logger.info(f"[snapshot_pipeline] Completed for session_id={session_id}")


# ===== 싱글톤 인스턴스 =====
_pipeline_instance = None


def get_snapshot_pipeline() -> SnapshotPipeline:
    """
    전역 SnapshotPipeline 싱글톤 인스턴스 반환

    Returns:
        SnapshotPipeline: 전역 파이프라인 인스턴스
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = SnapshotPipeline()
        logger.info("[snapshot_pipeline] SnapshotPipeline instance created")

    return _pipeline_instance


# ===== 호환성을 위한 함수형 인터페이스 =====


def update_long_term_memory(session_id: str) -> None:
    """
    장기 메모리 업데이트 (호환성 래퍼)

    Args:
        session_id: 세션 ID
    """
    pipeline = get_snapshot_pipeline()
    pipeline.update_long_term_memory(session_id)
