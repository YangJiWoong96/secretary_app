import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from backend.memory.stwm import STWMSnapshot
from backend.rag.embeddings import (
    embed_query_cached,
    embed_query_gemma,
)

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Turn:
    role: str  # "user"|"assistant"|"system"
    text: str
    ts: float
    token_est: int


@dataclass
class TurnSummary:
    id: str
    schema_ver: int
    ts_range: Dict[str, float]
    answer_summary: str
    decisions: Dict[str, Any]
    commits: List[Dict[str, Any]]
    updates: Dict[str, Any]
    entities: List[str]
    refs: List[Dict[str, Any]]
    safety_flags: List[str]
    quote_snippet: str
    keyphrases: List[str]
    emb_dim: int
    emb_id: str
    token_cost_est: int
    extras: Dict[str, Any]


class TurnBuffer:
    """
    세션별 턴 버퍼와 요약을 관리한다.
    - 버퍼: 최근 턴 텍스트(유저/봇)
    - 요약: TurnSummary 레코드(최대 60개 유지)
    트리거: buf_tokens>=400 or turns>=4 or topic_shift_cos<0.7
    """

    def __init__(self, max_summaries: int = 60):
        self.max_summaries = max_summaries
        self._buf: Dict[str, List[Turn]] = {}
        self._sums: Dict[str, List[TurnSummary]] = {}

    def add_turn(
        self, session_id: str, role: str, text: str, ts: Optional[float] = None
    ):
        t = Turn(
            role=role,
            text=text or "",
            ts=ts or time.time(),
            token_est=len(enc.encode(text or "")),
        )
        self._buf.setdefault(session_id, []).append(t)

    def _buf_tokens(self, session_id: str) -> int:
        return sum(t.token_est for t in self._buf.get(session_id, []))

    def _topic_shift_cos(self, session_id: str) -> float:
        turns = self._buf.get(session_id, [])
        if len(turns) < 2:
            return 1.0
        # 최근 두 유저/봇 텍스트 평균 임베딩 코사인
        # 주제 변화 감지: 의미적 일관성 → Gemma 전용
        a = embed_query_gemma(turns[-1].text)
        b = embed_query_gemma(turns[-2].text)
        import numpy as np

        denom = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
        cos = float(np.dot(a, b) / denom)
        if cos < 0:
            cos = 0.0
        return cos

    def _should_summarize(self, session_id: str) -> bool:
        turns = self._buf.get(session_id, [])
        return (
            self._buf_tokens(session_id) >= 400
            or len(turns) >= 4
            or self._topic_shift_cos(session_id) < 0.7
        )

    async def maybe_summarize(
        self, session_id: str, stwm: Optional[STWMSnapshot] = None
    ) -> Optional[TurnSummary]:
        if not self._should_summarize(session_id):
            return None
        turns = self._buf.get(session_id, [])
        if not turns:
            return None
        start_ts = turns[0].ts
        end_ts = turns[-1].ts
        text_block = "\n".join(f"{t.role}: {t.text}" for t in turns)
        # 최신 RAG 스냅샷 프리뷰(0~2개)를 요약 입력에 포함(비동기 MILVUS 조회)
        try:
            from backend.rag.config import METRIC as _METRIC
            from backend.rag.milvus import ensure_collections as _ens

            prof_coll, log_coll = _ens()
            # 최신 2개 로그 샘플
            hits = log_coll.query(
                expr=f"user_id == '{session_id}'",
                output_fields=["text", "created_at"],
                limit=2,
                order_by="-created_at",
            )
            previews = []
            for h in hits or []:
                t = (h.get("text") or "").splitlines()
                snippet = " ".join(t[:3])
                if len(snippet) > 160:
                    snippet = snippet[:157].rstrip() + "..."
                previews.append(snippet)
            # 동적 제한: 요약 목표 토큰 여유가 적으면 프리뷰 0~1개만 포함
            # 간단히 길이 기반으로 컷(≤ 220자만 허용)
            joined = ("\n- ".join(previews)).strip()
            if len(joined) > 220:
                previews = previews[:1]
            if previews:
                text_block = (
                    "[최근 스냅샷 프리뷰]\n- "
                    + "\n- ".join(previews)
                    + "\n\n"
                    + text_block
                )
        except Exception:
            pass

        schema = {
            "name": "TurnSummary",
            "schema": {
                "type": "object",
                "properties": {
                    "answer_summary": {"type": "string"},
                    "decisions": {"type": "object"},
                    "commits": {"type": "array", "items": {"type": "object"}},
                    "updates": {"type": "object"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "refs": {"type": "array", "items": {"type": "object"}},
                    "safety_flags": {"type": "array", "items": {"type": "string"}},
                    "quote_snippet": {"type": "string"},
                    "keyphrases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer_summary"],
                "additionalProperties": True,
            },
        }
        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 대화 요약기다. 입력 턴들을 보고 200~350 토큰 내로 핵심 요약과 상태 델타를 JSON으로 반환하라."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[턴]\n{text_block}\n\n규칙: 새로운 사실 추가 금지. 필요한 경우 최소 키만 채워라. JSON만."
                ),
            },
        ]
        try:
            # 순환 의존성 방지를 위해 지연 임포트
            from backend.config import get_settings
            from backend.utils.retry import openai_chat_with_retry
            from backend.utils.schema_builder import build_json_schema as _bjs

            settings_local = get_settings()

            resp = await openai_chat_with_retry(
                model=settings_local.LLM_MODEL,
                messages=msgs,
                response_format=_bjs("TurnSummary", schema, strict=True),
                max_tokens=380,
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content) if content.startswith("{") else {}
        except Exception:
            data = {
                "answer_summary": text_block[:300],
                "decisions": {},
                "commits": [],
                "updates": {},
                "entities": [],
                "refs": [],
                "safety_flags": [],
                "quote_snippet": text_block[:150],
                "keyphrases": [],
            }

        # 임베딩 생성: OpenAI 1536차원 그대로 보존 (정보 손실 방지)
        import numpy as np

        # 회차 요약 벡터: 기본 백엔드(OpenAI) 사용, 전체 차원 유지
        v = embed_query_cached(data.get("answer_summary", "") or text_block)
        v_full = np.array(v, dtype=np.float32)
        norm = float(np.linalg.norm(v_full) or 1.0)
        v_normalized = (v_full / norm).tolist()
        emb_dim = len(v_normalized)

        sum_id = f"{int(end_ts)}:{len(self._sums.get(session_id, []))}"
        summary = TurnSummary(
            id=sum_id,
            schema_ver=1,
            ts_range={"start": start_ts, "end": end_ts},
            answer_summary=data.get("answer_summary", ""),
            decisions=data.get("decisions", {}),
            commits=data.get("commits", [])[:8],
            updates=data.get("updates", {}),
            entities=data.get("entities", [])[:12],
            refs=data.get("refs", [])[:10],
            safety_flags=data.get("safety_flags", [])[:6],
            quote_snippet=data.get("quote_snippet", "")[:150],
            keyphrases=data.get("keyphrases", [])[:20],
            emb_dim=emb_dim,
            emb_id=sum_id,
            token_cost_est=sum(t.token_est for t in turns),
            extras={"stwm": asdict(stwm) if stwm else None},
        )
        # 저장
        arr = self._sums.setdefault(session_id, [])
        arr.append(summary)
        # 유지: 최근 60개, 초과 시 20개 롤업 삭제
        if len(arr) > self.max_summaries:
            del arr[:20]
        # 버퍼 비우기
        self._buf[session_id] = []
        return summary

    def get_summaries(self, session_id: str) -> List[TurnSummary]:
        return list(self._sums.get(session_id, []))


# 전역 인스턴스
_TURN_BUFFER = TurnBuffer()


def add_turn(session_id: str, role: str, text: str, ts: Optional[float] = None):
    _TURN_BUFFER.add_turn(session_id, role, text, ts)


async def maybe_summarize(
    session_id: str, stwm: Optional[STWMSnapshot] = None
) -> Optional[TurnSummary]:
    return await _TURN_BUFFER.maybe_summarize(session_id, stwm)


def get_summaries(session_id: str) -> List[TurnSummary]:
    return _TURN_BUFFER.get_summaries(session_id)
