import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import json
from backend.rag.embeddings import embed_query_cached
from backend.memory.stwm import STWMSnapshot


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
        a = embed_query_cached(turns[-1].text)
        b = embed_query_cached(turns[-2].text)
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
        # 최신 RAG 스냅샷 프리뷰 1~2개를 요약 입력에 포함(비동기 MILVUS 조회)
        try:
            from backend.rag.milvus import ensure_collections as _ens
            from backend.rag.config import METRIC as _METRIC

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
                previews.append(" ".join(t[:3]))
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
                    "commits": {"type": "array"},
                    "updates": {"type": "object"},
                    "entities": {"type": "array"},
                    "refs": {"type": "array"},
                    "safety_flags": {"type": "array"},
                    "quote_snippet": {"type": "string"},
                    "keyphrases": {"type": "array"},
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
            from backend.app import openai_chat_with_retry, LLM_MODEL

            resp = await openai_chat_with_retry(
                model=LLM_MODEL,
                messages=msgs,
                response_format={"type": "json_schema", "json_schema": schema},
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

        # 임베딩(512d) 생성: 기존 임베딩을 이용해 512로 축소 후 재정규화
        import numpy as np

        v = embed_query_cached(data.get("answer_summary", "") or text_block)
        v512 = v[:512].astype(float)
        norm = float(np.linalg.norm(v512) or 1.0)
        v512 = (v512 / norm).tolist()

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
            emb_dim=512,
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
