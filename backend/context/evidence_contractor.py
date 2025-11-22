"""
backend.context.evidence_contractor - 증거 계약 생성 및 외부화

원문을 Redis에 저장하고, 프롬프트에는 요약 계약(Claim)만 주입한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger("evidence_contractor")


class EvidenceContract(BaseModel):
    """증거 계약 스키마"""

    eid: str  # 증거 ID (E_RAG_xxx, E_WEB_xxx)
    claim: str  # 핵심 요지 (50자 이내)
    source: str  # 출처 (도메인 또는 "RAG")
    timestamp: str  # 수집 시각 (ISO 8601)
    confidence: float  # 신뢰도 (0~1)
    scope: str  # 적용 범위 ("session" | "user")


class EvidenceContractor:
    """증거 계약 생성 및 외부화 관리자"""

    def __init__(self):
        import redis

        from backend.config import get_settings

        self.settings = get_settings()
        self.redis = redis.Redis.from_url(
            self.settings.REDIS_URL, decode_responses=True
        )

        # 설정 기반
        self.enabled = bool(self.settings.EVIDENCE_REF_ENABLED)
        self.max_claims = int(self.settings.EVIDENCE_MAX_CLAIMS)
        # 4턴 × 150초/턴 = 600초 (기본)
        self.ttl_seconds = int(self.settings.EVIDENCE_TTL_TURNS) * 150

    def store_and_contract(
        self,
        session_id: str,
        rag_ctx: str,
        web_ctx: str,
        user_query: str,
        active_eids: Optional[List[str]] = None,
        save_active_eids: bool = True,
    ) -> Tuple[List[str], str]:
        """
        증거 원문을 Redis에 저장하고, 계약 텍스트를 생성한다.

        Returns:
            (eids, contract_text): 증거 ID 리스트, 계약 텍스트
        """
        if not self.enabled:
            # 플래그 OFF: 기존 방식 (원문 그대로 반환)
            return [], (rag_ctx or "") + ("\n\n" + web_ctx if web_ctx else "")

        eids: List[str] = []
        contracts: List[EvidenceContract] = []

        # Output-Aware 재사용 경로: 이전 턴에서 인용된 EID만 복원하여 계약 구성
        try:
            pruning_on = bool(self.settings.OUTPUT_PRUNING_ENABLED)
        except Exception:
            pruning_on = True

        if pruning_on and active_eids:
            restored: List[EvidenceContract] = []
            for eid in active_eids:
                try:
                    body = self.redis.get(f"evidence:{eid}")
                except Exception:
                    body = None
                if not body:
                    continue
                # 타입 판정 및 소스 설정
                src = (
                    "RAG"
                    if eid.startswith("E_RAG_")
                    else (self._extract_domain(body) or "web")
                )
                conf = 0.9 if eid.startswith("E_RAG_") else 0.85
                restored.append(
                    EvidenceContract(
                        eid=eid,
                        claim=self._extract_claim(body, user_query),
                        source=src,
                        timestamp=self._now_iso(),
                        confidence=conf,
                        scope="session",
                    )
                )
            if restored:
                contracts = restored
                eids = [c.eid for c in contracts]
                contract_text = self._format_contracts(contracts)
                # 최신 계약 저장 (사후 사용: 미사용 보류 등)
                try:
                    self._set_latest_contracts(session_id, contracts)
                except Exception:
                    pass
                logger.info(
                    f"[evidence_contractor] Reused {len(contracts)} contracts via active_eids"
                )
                return eids, contract_text

        # RAG 증거 처리
        if rag_ctx and rag_ctx.strip():
            rag_blocks = [b.strip() for b in rag_ctx.split("\n\n") if b.strip()]
            for block in rag_blocks[: max(1, self.max_claims // 2)]:
                eid = self._generate_eid(session_id, "RAG", block)
                claim = self._extract_claim(block, user_query)

                contract = EvidenceContract(
                    eid=eid,
                    claim=claim,
                    source="RAG",
                    timestamp=self._now_iso(),
                    confidence=0.9,
                    scope="session",
                )

                # Redis 저장
                try:
                    self.redis.setex(f"evidence:{eid}", self.ttl_seconds, block)
                except Exception as e:
                    logger.warning(f"[evidence_contractor] Redis setex failed: {e}")
                eids.append(eid)
                contracts.append(contract)

        # Web 증거 처리
        if web_ctx and web_ctx.strip():
            web_blocks = [b.strip() for b in web_ctx.split("\n\n") if b.strip()]
            for block in web_blocks[: max(1, self.max_claims // 2)]:
                eid = self._generate_eid(session_id, "WEB", block)
                claim = self._extract_claim(block, user_query)
                source_domain = self._extract_domain(block) or "web"

                contract = EvidenceContract(
                    eid=eid,
                    claim=claim,
                    source=source_domain,
                    timestamp=self._now_iso(),
                    confidence=0.85,
                    scope="session",
                )

                # Redis 저장
                try:
                    self.redis.setex(f"evidence:{eid}", self.ttl_seconds, block)
                except Exception as e:
                    logger.warning(f"[evidence_contractor] Redis setex failed: {e}")
                eids.append(eid)
                contracts.append(contract)

        # 계약 텍스트 생성
        contract_text = self._format_contracts(contracts)

        # 최신 계약 저장 (사후 사용: 미사용 보류 등)
        try:
            self._set_latest_contracts(session_id, contracts)
        except Exception:
            pass

        # 신규: TurnState에 active_eids 저장
        if save_active_eids and eids:
            try:
                from backend.routing.turn_state import get_turn_state, set_turn_state

                state = get_turn_state(session_id) or {}
                state["active_eids"] = eids
                set_turn_state(session_id, state)

                logger.info(
                    f"[evidence_contractor] Saved {len(eids)} active_eids to TurnState"
                )
            except Exception as e:
                logger.warning(f"[evidence_contractor] Failed to save active_eids: {e}")

        # 텔레메트리
        try:
            from backend.utils.logger import log_event

            log_event(
                "evidence.stored",
                {
                    "session_id": session_id,
                    "eid_count": len(eids),
                    "ttl": self.ttl_seconds,
                },
            )
        except Exception:
            pass

        logger.info(
            f"[evidence_contractor] Stored {len(eids)} evidences with TTL={self.ttl_seconds}s"
        )
        return eids, contract_text

    def retrieve_evidence(self, eid: str) -> Optional[str]:
        """증거 원문 조회 (사용자가 상세 요청 시)"""
        try:
            evidence = self.redis.get(f"evidence:{eid}")
            if evidence:
                try:
                    from backend.utils.logger import log_event

                    log_event("evidence.retrieved", {"eid": eid})
                except Exception:
                    pass
                return evidence
            else:
                logger.warning(f"[evidence_contractor] Evidence expired: {eid}")
                try:
                    from backend.utils.logger import log_event

                    log_event("evidence.expired", {"eid": eid})
                except Exception:
                    pass
                return None
        except Exception as e:
            logger.error(f"[evidence_contractor] Retrieve error: {e}")
            return None

    def _generate_eid(self, session_id: str, typ: str, content: str) -> str:
        """증거 ID 생성: E_{TYPE}_{SESSION}_{HASH}"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        session_short = session_id[-8:] if len(session_id) > 8 else session_id
        return f"E_{typ}_{session_short}_{content_hash}"

    def _extract_claim(self, block: str, query: str) -> str:
        """핵심 요지 추출 (규칙 기반, 50자 이내)"""
        lines = [ln.strip() for ln in (block or "").split("\n") if ln.strip()]
        if not lines:
            return "(증거 없음)"

        # 첫 줄이 제목/출처 형식이면 제거
        if lines[0].startswith("[출처:") or lines[0].startswith("http"):
            lines = lines[1:]

        # 첫 문장 추출 (50자 제한)
        first_sentence = lines[0] if lines else ""
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:47] + "..."

        return first_sentence or "(증거 없음)"

    def _extract_domain(self, block: str) -> Optional[str]:
        """도메인 추출 (URL에서)"""
        import re
        from urllib.parse import urlparse

        urls = re.findall(r"https?://[^\s]+", block or "")
        if urls:
            try:
                domain = urlparse(urls[0]).netloc
                if domain.startswith("www."):
                    domain = domain[4:]
                return domain
            except Exception:
                pass
        return None

    def _format_contracts(self, contracts: List[EvidenceContract]) -> str:
        """계약 리스트를 프롬프트용 텍스트로 포맷"""
        if not contracts:
            return ""

        lines: List[str] = ["[증거 요약]"]
        for c in contracts:
            line = f"[{c.eid}] {c.claim} - 출처: {c.source}, 신뢰도: {c.confidence:.2f}"
            lines.append(line)

        lines.append("")
        lines.append(
            '💡 증거 상세 확인: AI 답변의 [E_XXX] 클릭 또는 "E_XXX 자세히" 요청'
        )
        return "\n".join(lines)

    def _now_iso(self) -> str:
        """현재 시각 ISO 8601 형식"""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    # ─────────────────────────────────────────────────────────────
    # Output-Aware Pruning 확장 메서드
    # ─────────────────────────────────────────────────────────────
    def extract_cited_eids(self, ai_output: str) -> List[str]:
        """
        AI 출력 텍스트에서 인용된 EID 식별자 추출.

        구현 원칙:
        - 정규식: 대괄호에 둘러싸인 [E_RAG_xxx], [E_WEB_xxx] 패턴 탐지
        - 캡처 그룹 주의: 전체 토큰을 1그룹으로 캡처(r"\[(E_(?:RAG|WEB)_[\w]+)\]")
        - 토큰 기반 이중 검증: 접두사가 E_RAG_/E_WEB_인지 확인
        - 중복 제거: 첫 등장을 우선으로 보존(안정성 및 재현 가능성)
        """
        try:
            pattern = r"\[(E_(?:RAG|WEB)_[\w]+)\]"
            eids: List[str] = [
                m.group(1) for m in re.finditer(pattern, ai_output or "")
            ]

            # 토큰 기반 이중 검증
            verified = [
                eid
                for eid in eids
                if eid.startswith("E_RAG_") or eid.startswith("E_WEB_")
            ]

            # 순서 보존 중복 제거
            unique_eids = list(dict.fromkeys(verified))

            logger.info(
                f"[evidence_contractor] Extracted {len(unique_eids)} cited EIDs"
            )
            return unique_eids
        except Exception as e:
            logger.warning(f"[evidence_contractor] extract_cited_eids failed: {e}")
            return []

    def filter_contracts_by_eids(
        self,
        contracts: List["EvidenceContract"],
        cited_eids: List[str],
    ) -> Tuple[List["EvidenceContract"], List["EvidenceContract"]]:
        """
        계약 리스트를 인용된 EID 기준으로 분할하여 반환.

        Returns:
            (cited_contracts, unused_contracts)
        """
        cited_set = set(cited_eids or [])
        cited = [c for c in contracts if c.eid in cited_set]
        unused = [c for c in contracts if c.eid not in cited_set]
        logger.info(
            f"[evidence_contractor] Filtered by EIDs → cited={len(cited)}, unused={len(unused)}"
        )
        return cited, unused

    def _contract_to_dict(self, c: "EvidenceContract") -> Dict[str, Any]:
        """EvidenceContract를 직렬화 가능한 dict로 변환 (pydantic v1/v2 호환)."""
        try:
            # pydantic v2
            return c.model_dump()  # type: ignore[attr-defined]
        except Exception:
            try:
                # pydantic v1
                return c.dict()  # type: ignore[call-arg]
            except Exception:
                # 최후 수단: 필드 수동 매핑
                return {
                    "eid": getattr(c, "eid", ""),
                    "claim": getattr(c, "claim", ""),
                    "source": getattr(c, "source", ""),
                    "timestamp": getattr(c, "timestamp", ""),
                    "confidence": float(getattr(c, "confidence", 0.0)),
                    "scope": getattr(c, "scope", "session"),
                }

    def store_unused_evidence(
        self,
        session_id: str,
        turn_id: str,
        unused_contracts: List["EvidenceContract"],
    ) -> None:
        """
        미사용 증거를 별도 Redis 키에 보류 (사용자 후속 요청 시 재로드 대비)

        키: unused_evidence:{session}:{turn}
        TTL: 600초 (10분)
        """
        if not unused_contracts:
            return
        key = f"unused_evidence:{session_id}:{turn_id}"
        try:
            payload = json.dumps(
                [self._contract_to_dict(c) for c in unused_contracts],
                ensure_ascii=False,
            )
            self.redis.setex(key, 600, payload)
            logger.info(
                f"[evidence_contractor] Stored {len(unused_contracts)} unused evidences for session={session_id}, turn={turn_id}"
            )
        except Exception as e:
            logger.error(f"[evidence_contractor] Store unused error: {e}")

    def retrieve_unused_evidence(
        self,
        session_id: str,
        turn_id: str,
    ) -> List["EvidenceContract"]:
        """미사용 증거 조회 (사용자가 '아까/다른/추가/더' 등 요청 시)"""
        key = f"unused_evidence:{session_id}:{turn_id}"
        try:
            raw = self.redis.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            contracts: List[EvidenceContract] = []
            for item in data:
                try:
                    contracts.append(EvidenceContract(**item))
                except Exception:
                    continue
            logger.info(
                f"[evidence_contractor] Retrieved {len(contracts)} unused evidences for session={session_id}, turn={turn_id}"
            )
            return contracts
        except Exception as e:
            logger.error(f"[evidence_contractor] Retrieve unused error: {e}")
            return []

    # 내부: 최신 계약 저장/조회 (턴 종료 후 미사용 보류 산출용)
    def _latest_contracts_key(self, session_id: str) -> str:
        return f"contracts_latest:{session_id}"

    def _set_latest_contracts(
        self, session_id: str, contracts: List["EvidenceContract"]
    ) -> None:
        try:
            key = self._latest_contracts_key(session_id)
            payload = json.dumps(
                [self._contract_to_dict(c) for c in contracts], ensure_ascii=False
            )
            # 10분 보존
            self.redis.setex(key, 600, payload)
        except Exception as e:
            logger.warning(f"[evidence_contractor] set_latest_contracts failed: {e}")

    def get_latest_contracts(self, session_id: str) -> List["EvidenceContract"]:
        try:
            key = self._latest_contracts_key(session_id)
            raw = self.redis.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            out: List[EvidenceContract] = []
            for item in data:
                try:
                    out.append(EvidenceContract(**item))
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.warning(f"[evidence_contractor] get_latest_contracts failed: {e}")
            return []


# 싱글톤 접근자
_CONTRACTOR_INSTANCE: Optional[EvidenceContractor] = None


def get_evidence_contractor() -> EvidenceContractor:
    """프로세스 전역 EvidenceContractor 인스턴스"""
    global _CONTRACTOR_INSTANCE
    if _CONTRACTOR_INSTANCE is None:
        _CONTRACTOR_INSTANCE = EvidenceContractor()
    return _CONTRACTOR_INSTANCE
