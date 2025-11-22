from __future__ import annotations

"""
backend.context.adaptive_budget - 적응형 토큰 버짓 관리자

증거(Evidence) > 메모리(Memory) > 프로필(Profile) 우선순위로 토큰을 배분한다.
메모리 채널의 최근 4턴은 삭제하지 않고 문장 압축만 허용한다.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ContextBundle:
    evidence: str
    memory: str
    profile: str


class AdaptiveBudgetManager:
    """적응형 토큰 버짓 관리자"""

    def __init__(self, tokenizer: Any | None = None) -> None:
        """LLM과 동일한 토크나이저를 주입 받는다. 미제공 시 합리적 기본값 사용.

        tiktoken.encoding_for_model("gpt-4o-mini")가 실패하는 환경을 고려해
        'cl100k_base'로 폴백한다.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
            return
        try:
            import tiktoken

            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # pragma: no cover - 런타임 의존성 이슈 회피
            raise RuntimeError(
                "tiktoken 토크나이저를 초기화하지 못했습니다. requirements를 확인하세요."
            ) from e

    def allocate(self, bundle: ContextBundle, total_cap: int = 3400) -> ContextBundle:
        """
        채널별 토큰 동적 할당

        우선순위:
        1) Evidence (최우선, 1000~1600)
        2) Memory (안정성, 400~800, 최근 4턴 보호)
        3) Profile (잔여, 200~400)

        환경변수로 캡 조정 가능(ADAPTIVE_BUDGET_TUNING_ENABLED=true 일 때 적용):
        - BUDGET_EVIDENCE_CAP, BUDGET_EVIDENCE_MIN
        - BUDGET_MEMORY_CAP, BUDGET_MEMORY_MIN
        - BUDGET_PROFILE_CAP, BUDGET_PROFILE_MIN
        """
        # 실제 사용량 측정
        evidence_used = self._count_tokens(bundle.evidence)
        memory_used = self._count_tokens(bundle.memory)
        profile_used = self._count_tokens(bundle.profile)

        total_used = evidence_used + memory_used + profile_used
        if total_used <= total_cap:
            # 총합이 한도를 넘지 않으면 원문 유지
            return bundle

        # 플래그: OFF면 기존 로직 유지, ON이면 설정 기반 캡 적용(6:3:1 기본값)
        try:
            from backend.config import get_settings as _gs

            _s = _gs()
            tuning_on = bool(getattr(_s, "ADAPTIVE_BUDGET_TUNING_ENABLED", True))
        except Exception:
            tuning_on = True

        if tuning_on:

            try:
                E_MIN = int(getattr(_s, "BUDGET_EVIDENCE_MIN", 1000))
                E_MAX = int(getattr(_s, "BUDGET_EVIDENCE_CAP", 1600))
                M_MIN = int(getattr(_s, "BUDGET_MEMORY_MIN", 400))
                M_MAX = int(getattr(_s, "BUDGET_MEMORY_CAP", 800))
                P_MIN = int(getattr(_s, "BUDGET_PROFILE_MIN", 200))
                P_MAX = int(getattr(_s, "BUDGET_PROFILE_CAP", 400))
            except Exception:
                E_MIN, E_MAX = 1000, 1600
                M_MIN, M_MAX = 400, 800
                P_MIN, P_MAX = 200, 400
        else:
            # 기존 상수(이전 로직으로 복귀)
            E_MIN, E_MAX = 1000, 1600
            M_MIN, M_MAX = 400, 800
            P_MIN, P_MAX = 200, 600

        # 상한은 실제 사용량과의 min으로 계산 (과대평가 방지)
        e_upper = min(evidence_used, E_MAX)
        m_upper = min(memory_used, M_MAX)
        p_upper = min(profile_used, P_MAX)

        # 하한은 채널이 존재할 때만 적용
        e_floor = E_MIN if evidence_used > 0 else 0
        m_floor = M_MIN if memory_used > 0 else 0
        p_floor = P_MIN if profile_used > 0 else 0

        # 하한 총합이 total_cap을 초과하면 우선순위(E > M > P) 기반 비례 축소
        floor_sum = e_floor + m_floor + p_floor
        if floor_sum > total_cap:
            # Evidence 최소 50% 보호 + 비례 축소
            e_cap = max(
                int(total_cap * 0.5), int(e_floor * total_cap / max(1, floor_sum))
            )
            remain = max(0, total_cap - e_cap)
            if (m_floor + p_floor) > 0:
                m_cap = min(int(m_floor * remain / (m_floor + p_floor)), remain)
                p_cap = max(0, remain - m_cap)
            else:
                m_cap = 0
                p_cap = 0
        else:
            # 하한 배치 후 남은 예산을 상한까지 우선순위(E→M→P)로 증액
            e_cap, m_cap, p_cap = e_floor, m_floor, p_floor
            remain = total_cap - floor_sum

            # Evidence 증액
            e_add = min(remain, max(0, e_upper - e_cap))
            e_cap += e_add
            remain -= e_add

            # Memory 증액
            if remain > 0:
                m_add = min(remain, max(0, m_upper - m_cap))
                m_cap += m_add
                remain -= m_add

            # Profile 증액
            if remain > 0:
                p_add = min(remain, max(0, p_upper - p_cap))
                p_cap += p_add
                remain -= p_add

        # 로깅(베타): 분배 결과 스냅샷
        try:
            from backend.utils.logger import log_event

            log_event(
                "adaptive_budget_allocated",
                {
                    "evidence_used": evidence_used,
                    "memory_used": memory_used,
                    "profile_used": profile_used,
                    "evidence_cap": e_cap,
                    "memory_cap": m_cap,
                    "profile_cap": p_cap,
                    "total_cap": total_cap,
                    "caps": {
                        "E_MIN": E_MIN,
                        "E_MAX": E_MAX,
                        "M_MIN": M_MIN,
                        "M_MAX": M_MAX,
                        "P_MIN": P_MIN,
                        "P_MAX": P_MAX,
                    },
                },
            )
        except Exception:
            pass

        return ContextBundle(
            evidence=self._trim_channel(bundle.evidence, e_cap, protect_recent=False),
            memory=self._trim_channel(bundle.memory, m_cap, protect_recent=True),
            profile=self._trim_channel(bundle.profile, p_cap, protect_recent=False),
        )

    # ─────────────────────────────────────────────────────────────
    # 세션6: 프로필 상대 예산 계산 보조 함수
    # Evidence 우선 할당 후 남은 창에서 비율로 프로필/히스토리 분배
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def allocate_profile_budget(
        total_cap: int,
        used_now: int,
        evidence_reserved: int,
        headroom: int = 256,
        ratio: float = 0.35,
    ) -> int:
        """
        남은 창으로부터 프로필 예산을 상대 비율로 산출한다.

        Args:
            total_cap: 전체 컨텍스트 창 용량(토큰)
            used_now: 현재까지 조립된 토큰 수(시스템/프롬프트 등)
            evidence_reserved: Evidence에 이미 예약된 토큰 수
            headroom: 안전 여유 토큰(초과 방지)
            ratio: 프로필 채널 상대 비율(0~1)

        Returns:
            int: 프로필 채널에 할당할 토큰 수(음수 방지)
        """
        remain = max(
            0, int(total_cap) - int(used_now) - int(evidence_reserved) - int(headroom)
        )
        return max(0, int(remain * float(ratio)))

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _trim_channel(self, text: str, cap: int, protect_recent: bool = False) -> str:
        if not text:
            return text

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= cap:
            return text

        # 메모리 최근 4턴 보호: 서문([최근 대화])은 압축, 과거 대화는 트림
        if protect_recent and "[최근 대화]" in text:
            parts = text.split("[관련 과거 대화]")
            recent_part = parts[0]
            past_part = parts[1] if len(parts) > 1 else ""

            # 한국어 문장 경계 인식 기반 경량 압축으로 위임
            try:
                from backend.context.korean_sentence_utils import (
                    compress_middle_sentences_rule_based,
                    extract_head_tail_sentences,
                )

                head, middle, tail = extract_head_tail_sentences(recent_part)
                middle_c = compress_middle_sentences_rule_based(middle)
                recent_compressed = f"{head} ... {middle_c} ... {tail}".strip()
            except Exception:
                # 폴백: 기존 간이 압축
                recent_compressed = self._compress_sentences(recent_part)
            combined = recent_compressed + (
                "\n[관련 과거 대화]\n" + past_part if past_part else ""
            )
            if self._count_tokens(combined) <= cap:
                return combined

            # 과거 대화 우선 트림
            recent_tokens = self._count_tokens(recent_compressed)
            past_cap = max(0, cap - recent_tokens)
            if past_cap > 50:
                past_trimmed = self.tokenizer.decode(
                    self.tokenizer.encode(past_part)[:past_cap]
                )
                return recent_compressed + "\n[관련 과거 대화]\n" + past_trimmed
            return recent_compressed

        # 일반 트림: 문장 경계 보존을 위해 마지막 문장 제거
        trimmed = self.tokenizer.decode(tokens[:cap])
        sentences = trimmed.split(". ")
        if len(sentences) > 1:
            return (". ".join(sentences[:-1])).rstrip()
        return trimmed

    def _compress_sentences(self, text: str) -> str:
        """
        문장 단위 압축(한국어 문장 경계 인식 기반):
        - 앞/뒤 소수 문장을 보존하고, 중간 문장을 규칙 기반으로 경량화한다.
        - 외부 의존 없이 동작하며, 실패 시 간이 방식으로 폴백한다.
        """
        try:
            from backend.context.korean_sentence_utils import (
                compress_middle_sentences_rule_based,
                extract_head_tail_sentences,
            )

            head, middle, tail = extract_head_tail_sentences(text or "")
            mid_c = compress_middle_sentences_rule_based(middle)
            return f"{head} ... {mid_c} ... {tail}".strip()
        except Exception:
            # 폴백: 중복 단어 제거 기반의 간이 압축
            sentences = (text or "").split(". ")
            compressed: list[str] = []
            for sent in sentences:
                words = sent.split()
                unique: list[str] = []
                seen: set[str] = set()
                for w in words:
                    lw = w.lower()
                    if lw not in seen:
                        unique.append(w)
                        seen.add(lw)
                line = " ".join(unique).strip()
                if line:
                    compressed.append(line)
            return ". ".join(compressed)

    def _compute_query_complexity(self, user_input: str) -> int:
        """
        쿼리 복잡도를 엔티티 개수로 측정

        Args:
            user_input: 사용자 쿼리

        Returns:
            int: 추출된 엔티티 개수 (LOC, ORG, PERSON)
        """
        try:
            from backend.memory.stwm_plugins import _ensure_ner

            # NER 파이프라인 직접 호출
            pipe = _ensure_ner()
            results = pipe(user_input or "")
            if not results:
                return 0

            # LOC, ORG, PER 엔티티만 카운트
            entity_count = 0
            for entity in results:
                label = (entity.get("entity_group") or "").upper()
                if any(label.endswith(tag) for tag in ("LOC", "ORG", "PER")):
                    entity_count += 1

            return entity_count
        except Exception:
            return 0

    def allocate_dynamic(
        self,
        bundle: ContextBundle,
        user_input: str,
        history_tokens: int,
        total_cap: int = 3400,
    ) -> ContextBundle:
        """
        동적 예산 할당 (쿼리 복잡도 및 히스토리 길이 기반)

        Args:
            bundle: 원본 컨텍스트 번들
            user_input: 사용자 쿼리 (복잡도 계산용)
            history_tokens: 현재 히스토리 토큰 수
            total_cap: 전체 컨텍스트 예산 (기본 3400)

        Returns:
            ContextBundle: 예산 조정된 컨텍스트

        동작:
            1. 쿼리 엔티티 개수 계산
            2. 엔티티 개수에 따른 증거 비율 결정:
               - 3개 이상: 45% (복잡한 쿼리)
               - 2개: 40%
               - 1개 이하: 30% (단순 쿼리)
            3. 히스토리가 길면 (> 2000 토큰) 증거 비율 80%로 축소
            4. 남는 예산을 Memory/Profile에 분배
        """
        # 실제 사용량 측정
        evidence_used = self._count_tokens(bundle.evidence)
        memory_used = self._count_tokens(bundle.memory)
        profile_used = self._count_tokens(bundle.profile)

        total_used = evidence_used + memory_used + profile_used
        if total_used <= total_cap:
            # 총합이 한도를 넘지 않으면 원문 유지
            return bundle

        # Step 1: 쿼리 복잡도 계산
        entity_count = self._compute_query_complexity(user_input)

        # Step 2: 엔티티 개수 기반 증거 비율 결정
        if entity_count >= 3:
            evidence_ratio = 0.45  # 복잡한 쿼리
        elif entity_count >= 2:
            evidence_ratio = 0.40
        else:
            evidence_ratio = 0.30  # 단순 쿼리

        # Step 3: 히스토리 길이 보정
        if history_tokens > 2000:
            evidence_ratio *= 0.8  # 히스토리가 길면 증거 축소

        # Step 4: 예산 계산
        available = total_cap - history_tokens
        evidence_budget = int(available * evidence_ratio)
        memory_budget = int(available * 0.25)  # Memory 고정 25%
        profile_budget = available - evidence_budget - memory_budget

        # 최소/최대 제약 적용
        evidence_budget = max(1000, min(1600, evidence_budget))
        memory_budget = max(400, min(800, memory_budget))
        profile_budget = max(200, min(400, profile_budget))

        # 로깅 (선택적)
        try:
            from backend.utils.logger import log_event

            log_event(
                "adaptive_budget_dynamic",
                {
                    "entity_count": entity_count,
                    "history_tokens": history_tokens,
                    "evidence_ratio": evidence_ratio,
                    "evidence_budget": evidence_budget,
                    "memory_budget": memory_budget,
                    "profile_budget": profile_budget,
                    "total_cap": total_cap,
                },
            )
        except Exception:
            pass

        # Step 5: 채널별 트림
        return ContextBundle(
            evidence=self._trim_channel(
                bundle.evidence, evidence_budget, protect_recent=False
            ),
            memory=self._trim_channel(
                bundle.memory, memory_budget, protect_recent=True
            ),
            profile=self._trim_channel(
                bundle.profile, profile_budget, protect_recent=False
            ),
        )
