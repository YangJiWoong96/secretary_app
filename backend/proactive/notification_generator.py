from __future__ import annotations

"""
NotificationGenerator

역할(세션4):
- 퍼소나/정서 상태를 반영해 알림 후보 생성
- 긴급도(0~10) 계산, 카테고리 분류
- 금지 패턴 검증 및 보정
- 최종 알림(FinalNotification) 구성

설계 원칙:
- 한국어 존댓말/톤 일관성
- PII 제거(로그 시 해시만)
- 빠른 응답(≤ 400ms 목표) → 후보 3개 병렬 생성
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.proactive.observability import hash_text, hash_user_id
from backend.proactive.schemas import (
    FinalNotification,
    NotificationCandidate,
    SMARTAction,
)
from backend.utils.logger import log_event

try:
    from langchain_openai import ChatOpenAI

    from backend.config.settings import get_settings
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    get_settings = None  # type: ignore


FORBIDDEN_PATTERNS = [
    r"반드시\s+\S+하세요",
    r"약\s*복용",
    r"주식\s*사",
    r"\d{3}-\d{4}-\d{4}",
]


class NotificationGenerator:
    """퍼소나 기반 알림 생성기."""

    def __init__(self) -> None:
        try:
            from backend.config import get_settings as _gs

            self.model_name = _gs().NOTIF_MODEL
        except Exception:
            self.model_name = "gpt-4o-mini"

    async def generate(
        self,
        actionable_items: List[SMARTAction],
        user_id: str,
        session_id: str,
        internal_contexts: Optional[Dict[str, str]] = None,
        verified_facts: Optional[List[Dict[str, Any]]] = None,
        base_confidence: float = 0.7,
    ) -> Tuple[List[NotificationCandidate], Optional[FinalNotification]]:
        """
        알림 후보를 생성하고, 최고 긴급도 1개를 최종으로 선택하여 반환한다.
        Returns: (candidates, final_notification)
        """

        persona = self._load_persona(session_id)
        emotion = self._load_emotion(session_id)

        actions = list(actionable_items or [])[:3]
        if not actions:
            return [], None

        # 후보 병렬 생성
        tasks = [
            asyncio.create_task(
                self._draft_and_verify_candidate(
                    action, persona, emotion, internal_contexts
                )
            )
            for action in actions
        ]
        candidates = await asyncio.gather(*tasks)
        candidates = [c for c in candidates if c]
        if not candidates:
            return [], None

        best = max(candidates, key=lambda x: int(x.get("urgency", 0)))

        # 최종 알림 구성
        now_iso = datetime.now(timezone.utc).isoformat()
        meta = {
            "rag_len": len((internal_contexts or {}).get("rag_ctx", "")),
            "web_len": len((internal_contexts or {}).get("web_ctx", "")),
            "mobile_len": len((internal_contexts or {}).get("mobile_ctx", "")),
        }
        # 밴딧 변이 라벨(기본값 → 추후 밴딧 적용 시 업데이트)
        try:
            from backend.config import get_settings as _gs1

            variant_label = _gs1().PROACTIVE_VARIANT
        except Exception:
            variant_label = "baseline_v1"

        # 밴딧이 적용된 후보의 변이명 반영(있다면 bandit:<arm>)
        try:
            from backend.config import get_settings as _gs2

            if bool(_gs2().FEATURE_BANDIT):
                arm_label = str(best.get("bandit_arm") or "").strip()
                if arm_label:
                    variant_label = f"bandit:{arm_label}"
        except Exception:
            pass

        fn = FinalNotification(
            push_id=str(uuid.uuid4()),
            user_id=user_id,
            title=best["title"],
            body=best["body"],
            urgency=int(best["urgency"]),
            category=best["category"],
            tone=best["tone"],
            emoji_level=best["emoji_level"],
            contexts_meta=meta,
            confidence_score=float(best.get("confidence", base_confidence)),
            timestamp=now_iso,
            variant=variant_label,
        )

        # Why-Tag(설명가능 푸시) 주입
        try:
            from backend.config import get_settings as _gs3

            if bool(_gs3().FEATURE_WHY_TAG):
                from backend.proactive.why_explainer import build_why_tag

                fn.why_tag = build_why_tag(internal_contexts or {})
        except Exception:
            pass

        # 최소 로그(PII 제거)
        try:
            log_event(
                "notification.generated",
                {
                    "user_id": hash_user_id(user_id),
                    "title_hash": hash_text(fn.title),
                    "body_hash": hash_text(fn.body),
                    "urgency": fn.urgency,
                    "category": fn.category,
                    "tone": fn.tone,
                    "emoji_level": fn.emoji_level,
                },
            )
        except Exception:
            pass

        return candidates, fn

    # -----------------------------
    # 내부 구현
    # -----------------------------
    def _load_persona(self, session_id: str) -> Dict[str, Any]:
        try:
            from backend.directives.store import load_persona

            return load_persona(session_id) or {}
        except Exception:
            return {}

    def _load_emotion(self, session_id: str) -> str:
        try:
            from backend.memory.stwm import get_stwm_snapshot

            snap = get_stwm_snapshot(session_id) or {}
            return (snap or {}).get("last_emotion") or ""
        except Exception:
            return ""

    async def _draft_and_verify_candidate(
        self,
        action: SMARTAction,
        persona: Dict[str, Any],
        emotion: str,
        internal_contexts: Optional[Dict[str, str]],
    ) -> Optional[NotificationCandidate]:
        draft = await self._draft_notification(
            action, persona, emotion, internal_contexts
        )
        ok, reason = self._verify_safety(draft.get("body", ""))
        if not ok:
            # 보수적 보정: 금지 표현 제거
            body = draft.get("body", "")
            for pat in FORBIDDEN_PATTERNS:
                body = re.sub(pat, "", body)
            draft["body"] = body.strip()
        return draft  # 검증 결과는 후보에 반영

    async def _draft_notification(
        self,
        action: SMARTAction,
        persona: Dict[str, Any],
        emotion: str,
        internal_contexts: Optional[Dict[str, str]],
    ) -> NotificationCandidate:
        # LLM 준비(없으면 보수적 생성)
        llm = None
        if ChatOpenAI and get_settings:
            try:
                settings = get_settings()
                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model=self.model_name,
                    temperature=0.7,
                    max_tokens=200,
                    max_retries=int(getattr(settings, "MAX_RETRIES_OPENAI", 2)),
                )
            except Exception:
                llm = None

        persona_style = persona.get("communication_style", {}) if persona else {}
        honorifics = bool(persona_style.get("honorifics", True))
        emoji_level = str(persona.get("emoji_level", "medium"))
        preferred_tone = str(persona.get("preferred_tone", "formal"))

        # (신규) 반응학습 밴딧 변이 적용: 톤/이모지 레벨 조정
        bandit_arm: Optional[str] = None
        try:
            if os.getenv("FEATURE_BANDIT", "0") == "1":
                from backend.experiments.bandit import Bandit

                arms = {
                    "tone_formal": {"tone": "formal"},
                    "tone_friendly": {"tone": "friendly"},
                    "tone_empathetic": {"tone": "empathetic"},
                    "tone_concise": {
                        "tone": "formal"
                    },  # 간결 톤은 하위 규칙에서 문장 길이로 제어 가능
                    "emoji_none": {"emoji_level": "none"},
                    "emoji_low": {"emoji_level": "low"},
                    "emoji_medium": {"emoji_level": "medium"},
                    "emoji_high": {"emoji_level": "high"},
                }
                bandit_arm = Bandit().select(arms)
                # 선택된 변이를 현재 초안 생성 파라미터에 반영
                preferred_tone = arms.get(bandit_arm, {}).get("tone", preferred_tone)  # type: ignore[arg-type]
                emoji_level = arms.get(bandit_arm, {}).get("emoji_level", emoji_level)  # type: ignore[arg-type]
        except Exception:
            bandit_arm = None

        title = self._short_title_from(action.get("action", "알림"))

        if llm is None:
            # 보수적 초안
            body = self._render_body_baseline(
                action, honorifics, emoji_level, preferred_tone, emotion
            )
        else:
            sys = (
                "너는 한국어 알림 카피라이터다. 주어진 SMART 액션을 바탕으로 존댓말 알림 본문을 2~4문장으로 작성하라.\n"
                "규칙: 1) 과장/추측 금지, 2) 지금 바로 실행 가능한 제안 포함, 3) 존댓말 유지, 4) 부정 정서 시 공감 톤으로 시작."
            )
            user = json.dumps(
                {
                    "action": action,
                    "persona": persona,
                    "emotion": emotion,
                    "contexts": {
                        k: (v[:600]) for k, v in (internal_contexts or {}).items()
                    },
                },
                ensure_ascii=False,
            )
            try:
                resp = await llm.ainvoke(
                    [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ]
                )
                body = (getattr(resp, "content", "") or "").strip()
            except Exception:
                body = self._render_body_baseline(
                    action, honorifics, emoji_level, preferred_tone, emotion
                )

        urgency = self._calculate_urgency(action, internal_contexts)
        category = self._classify_category(action)

        draft: NotificationCandidate = NotificationCandidate(
            title=title,
            body=self._apply_persona_tone(
                body, honorifics, emoji_level, preferred_tone, emotion
            ),
            urgency=int(urgency),
            category=category,
            tone=preferred_tone,
            emoji_level=emoji_level,
            confidence=0.8,
        )
        # 밴딧 변이 라벨(후속 최종 알림 variant 연결용 힌트)
        if bandit_arm:
            try:
                draft["bandit_arm"] = str(bandit_arm)
            except Exception:
                pass
        return draft

    # -------- 렌더/규칙 --------
    def _render_body_baseline(
        self,
        action: SMARTAction,
        honorifics: bool,
        emoji_level: str,
        tone: str,
        emotion: str,
    ) -> str:
        lead = "최근 힘드셨죠. " if emotion in ["슬픔", "분노", "불안"] else ""
        base = f"{lead}{action.get('action','').strip()} — {action.get('rationale','').strip()} (권장 시한: {action.get('time_bound','').strip()})"
        if honorifics:
            base = self._to_honorifics(base)
        return self._add_emojis(base, emoji_level)

    def _apply_persona_tone(
        self, text: str, honorifics: bool, emoji_level: str, tone: str, emotion: str
    ) -> str:
        out = text
        if honorifics:
            out = self._to_honorifics(out)
        out = self._add_emojis(out, emoji_level)
        if emotion in ["슬픔", "분노", "불안"] and not out.startswith("최근 힘드셨죠"):
            out = f"최근 힘드셨죠. {out}"
        return out

    def _to_honorifics(self, text: str) -> str:
        # 매우 보수적 변환: '~해요' → '~하세요' 패턴 일부만 치환
        try:
            return re.sub(r"해요\b", "하세요", text)
        except Exception:
            return text

    def _add_emojis(self, text: str, level: str) -> str:
        count = {"none": 0, "low": 1, "medium": 2, "high": 3}.get(level, 1)
        if count <= 0:
            return text
        # 간단 맥락 이모지(보수적): 문장 끝에만 추가
        emoji = "✨"
        return (text + " " + ("✨" * min(3, count))).strip()

    def _calculate_urgency(
        self, action: SMARTAction, internal_contexts: Optional[Dict[str, str]]
    ) -> int:
        base = int(action.get("urgency", 5))
        time_bound = action.get("time_bound", "") or ""
        if "오늘" in time_bound or "지금" in time_bound:
            base += 3
        elif "내일" in time_bound:
            base += 2
        return max(0, min(10, base))

    def _classify_category(self, action: SMARTAction) -> str:
        """LLM 기반 대분류(건강/관계/재무/여가/업무) 분류.

        - LLM 불가 시 보수적 폴백을 사용(최소 규칙), 단 정상 경로는 LLM 필수.
        """
        # 1) LLM 시도
        if ChatOpenAI and get_settings:
            try:
                settings = get_settings()
                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=10,
                    max_retries=int(getattr(settings, "MAX_RETRIES_OPENAI", 2)),
                )
                sys = (
                    "아래 액션을 다음 다섯 범주 중 하나로만 분류하라: 건강, 관계, 재무, 여가, 업무.\n"
                    "출력은 해당 범주 단어만 반환."
                )
                user = json.dumps(action, ensure_ascii=False)
                resp = llm.invoke(
                    [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ]
                )
                out = (getattr(resp, "content", "") or "").strip()
                if out in {"건강", "관계", "재무", "여가", "업무"}:
                    return out
            except Exception:
                pass

        # 2) 폴백(키워드 최소 규칙)
        txt = f"{action.get('action','')} {action.get('rationale','')}"
        if any(k in txt for k in ["병원", "운동", "약", "건강"]):
            return "건강"
        if any(k in txt for k in ["미팅", "선물", "가족", "친구"]):
            return "관계"
        if any(k in txt for k in ["할인", "가격", "구매", "환율", "투자"]):
            return "재무"
        if any(k in txt for k in ["전시", "공연", "여행", "취미", "맛집"]):
            return "여가"
        return "업무"

    def _verify_safety(self, body: str) -> Tuple[bool, str]:
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, body):
                return False, f"금지 패턴 감지: {pat}"
        return True, "ok"


__all__ = ["NotificationGenerator"]
