from __future__ import annotations

"""
SynthesizerAgent

역할:
- 내부 컨텍스트(ContextBundle)와 웹 결과(WebSearchResult)를 교차 검증
- 교차 검증된 사실(VerifiedFact) 도출
- SMART 액션(SMARTAction) 생성(LLM 기반)
- 전체 신뢰도(confidence_score) 집계

주의:
- 원문/PII는 외부 저장하지 않으며, 로깅은 메타 정보만 기록
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from backend.proactive.consistency_checker import ConsistencyChecker
from backend.proactive.schemas import (
    ProactiveState,
    SMARTAction,
    SynthesizerOutput,
    VerifiedFact,
    FinanceFact,
    WeatherFact,
    TrafficFact,
)
from backend.utils.logger import log_event

try:
    # LLM (LangChain OpenAI)
    from langchain_openai import ChatOpenAI

    from backend.config.settings import get_settings
except Exception:  # pragma: no cover - 실행 환경에 따라 미존재 가능
    ChatOpenAI = None  # type: ignore
    get_settings = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Inputs:
    internal_contexts: Dict[str, str]
    web_results: List[Dict[str, Any]]


class SynthesizerAgent:
    def __init__(self) -> None:
        self.consistency = ConsistencyChecker()

    # -----------------------------
    # Public API
    # -----------------------------
    def synthesize(self, state: ProactiveState) -> ProactiveState:
        t0 = time.perf_counter_ns()
        inputs = self._collect_inputs(state)

        # 1) 검증된 사실 도출(LLM 사용, 교차 검증 지시)
        verified_facts = self._derive_verified_facts(inputs)

        # 2) SMART 액션 생성(최대 3개)
        actions = self._generate_smart_actions(verified_facts, inputs.internal_contexts)

        # 3) 도메인 구조화 팩트 도출
        domain_facts = self._derive_domain_facts(state)

        # 4) 신뢰도 집계
        confidence, contradictions = self._aggregate_confidence(verified_facts, inputs)

        took_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        try:
            log_event(
                "proactive.agent_complete",
                {
                    "agent": "synthesizer",
                    "took_ms": took_ms,
                    "confidence": confidence,
                    "output_size": len(verified_facts) + len(actions),
                },
            )
        except Exception:
            pass

        out: ProactiveState = {
            **state,
            "synthesized_insights": [
                {
                    "type": "verified_summary",
                    "count": len(verified_facts),
                    "timestamp": _now_iso(),
                }
            ],
            "verified_facts": verified_facts,
            "actionable_items": actions,
            "confidence_score": float(confidence),
            "domain_facts": domain_facts,
        }
        # 모순 수 기록(필요 시 참조)
        out.setdefault("meta", {})
        try:
            out["meta"]["contradictions_found"] = int(contradictions)
        except Exception:
            pass
        return out

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _collect_inputs(self, state: ProactiveState) -> _Inputs:
        internal = dict(state.get("internal_contexts", {}) or {})
        web_results = list(state.get("web_results", []) or [])
        return _Inputs(internal_contexts=internal, web_results=web_results)

    def _derive_verified_facts(self, inp: _Inputs) -> List[VerifiedFact]:
        # 신뢰도 상위 문서 5개로 축약 (LLM 프롬프트 크기 제한)
        docs = sorted(
            [d for d in inp.web_results if float(d.get("trust_score", 0.0)) >= 0.7],
            key=lambda x: float(x.get("trust_score", 0.0)),
            reverse=True,
        )[:5]
        if not docs:
            return []

        # LLM 준비
        llm = None
        if ChatOpenAI and get_settings:
            try:
                settings = get_settings()
                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
                    temperature=0.1,
                    max_retries=int(getattr(settings, "MAX_RETRIES_OPENAI", 2)),
                )
            except Exception:
                llm = None

        # LLM 부재 시 보수적 요약(상위 1~2개 문서 제목 기반 추출)
        if llm is None:
            top = docs[:2]
            facts: List[VerifiedFact] = []
            if top:
                merged_title = ", ".join(
                    [(t.get("title") or "").strip() for t in top if t.get("title")]
                )
                sources = [t.get("url", "") for t in top]
                ts = sum(float(t.get("trust_score", 0.0)) for t in top) / max(
                    1, len(top)
                )
                facts.append(
                    VerifiedFact(
                        fact=(merged_title[:200] or "신뢰도 높은 출처 요약"),
                        sources=sources,
                        trust_score=float(ts),
                        evidence_count=len(sources),
                    )
                )
            return facts

        # LLM 프롬프트 구성: 중복/추측 금지, 다중 출처 일치하는 주장만
        context_text = inp.internal_contexts
        doc_snippets = []
        for d in docs:
            content = (d.get("content") or "").replace("\n", " ")
            doc_snippets.append(
                {
                    "url": d.get("url", ""),
                    "title": d.get("title", ""),
                    "domain": d.get("domain", ""),
                    "trust": float(d.get("trust_score", 0.0)),
                    "snippet": content[:700],  # 토큰 제한 고려
                }
            )

        sys = (
            "너는 검증된 사실만을 요약하는 분석가다. \n"
            "요구사항: 1) 최소 두 개 이상의 출처에서 동일하게 언급된 주장만 포함, 2) 추측/예상 금지, 3) 과장/광고성 표현 금지, 4) 한국어 1~2문장으로 간결하게.\n"
            '출력 형식(JSON 배열): [{"fact":str, "sources":[url...], "trust_score":float, "evidence_count":int}]'
        )
        user = json.dumps(
            {
                "internal_contexts": {k: (v[:600]) for k, v in context_text.items()},
                "web_documents": doc_snippets,
            },
            ensure_ascii=False,
        )

        try:
            resp = llm.invoke(
                [{"role": "system", "content": sys}, {"role": "user", "content": user}]
            )
            content = (getattr(resp, "content", "") or "").strip()
            data = json.loads(content) if content.startswith("[") else []
        except Exception:
            data = []

        facts: List[VerifiedFact] = []
        for it in data[:5]:
            try:
                fact = (it.get("fact") or "").strip()
                sources = [
                    s for s in (it.get("sources") or []) if isinstance(s, str) and s
                ]
                evc = int(it.get("evidence_count", len(sources)))
                ts = float(it.get("trust_score", 0.7))
                if not fact or len(sources) < 1:
                    continue
                facts.append(
                    VerifiedFact(
                        fact=fact[:300],
                        sources=sources[:5],
                        trust_score=max(0.0, min(1.0, ts)),
                        evidence_count=max(1, evc),
                    )
                )
            except Exception:
                continue
        return facts

    def _derive_domain_facts(self, state: ProactiveState) -> Dict[str, List[Dict]]:
        """
        웹 결과 중 도메인 소스(weather/finance/traffic)에서 구조화 data를 사용해
        신뢰도 높은 도메인 팩트를 생성한다(LLM 불필요).
        """
        results = list(state.get("web_results", []) or [])
        out: Dict[str, List[Dict]] = {"finance": [], "weather": [], "traffic": []}

        # Weather
        for it in results:
            try:
                if (it.get("source_type") == "weather") and isinstance(
                    it.get("data"), dict
                ):
                    d = it.get("data") or {}
                    cur = d.get("current") or {}
                    lat = float(d.get("lat") or 0.0)
                    lon = float(d.get("lon") or 0.0)
                    wf: WeatherFact = {
                        "location": it.get("title", "") or "현재 위치",
                        "lat": lat,
                        "lon": lon,
                        "temperature_c": float(cur.get("temperature_2m") or 0.0),
                        "apparent_c": float(cur.get("apparent_temperature") or 0.0),
                        "precipitation_mm": float(cur.get("precipitation") or 0.0),
                        "humidity_pct": float(cur.get("relative_humidity_2m") or 0.0),
                        "wind_ms": float(cur.get("wind_speed_10m") or 0.0),
                        "time": str(cur.get("time") or ""),
                        "sources": ["open-meteo"],
                        "trust_score": float(it.get("trust_score", 0.8)),
                    }
                    out["weather"].append(wf)
            except Exception:
                continue

        # Finance
        for it in results:
            try:
                if (it.get("source_type") == "finance") and isinstance(
                    it.get("data"), dict
                ):
                    d = it.get("data") or {}
                    ff: FinanceFact = {
                        "symbol": str(d.get("01. symbol") or ""),
                        "price": float(d.get("05. price") or 0.0),
                        "change": float(d.get("09. change") or 0.0),
                        "change_percent": str(d.get("10. change percent") or ""),
                        "timestamp": str(d.get("07. latest trading day") or ""),
                        "sources": ["alphavantage"],
                        "trust_score": float(it.get("trust_score", 0.8)),
                    }
                    if ff["symbol"]:
                        out["finance"].append(ff)
            except Exception:
                continue

        # Traffic
        for it in results:
            try:
                if (it.get("source_type") == "traffic") and isinstance(
                    it.get("data"), dict
                ):
                    d = it.get("data") or {}
                    org = d.get("origin") or {}
                    dst = d.get("destination") or {}
                    tf: TrafficFact = {
                        "origin": f"({org.get('lat')},{org.get('lon')})",
                        "destination": f"({dst.get('lat')},{dst.get('lon')})",
                        "distance_km": float(d.get("distance_km") or 0.0),
                        "duration_min": float(d.get("duration_min") or 0.0),
                        "sources": ["osrm"],
                        "trust_score": float(it.get("trust_score", 0.75)),
                    }
                    out["traffic"].append(tf)
            except Exception:
                continue

        return out

    def _generate_smart_actions(
        self, verified_facts: List[VerifiedFact], internal_contexts: Dict[str, str]
    ) -> List[SMARTAction]:
        if not verified_facts:
            return []
        llm = None
        if ChatOpenAI and get_settings:
            try:
                settings = get_settings()
                llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model=getattr(settings, "LLM_MODEL", "gpt-4o-mini"),
                    temperature=0.2,
                    max_retries=int(getattr(settings, "MAX_RETRIES_OPENAI", 2)),
                )
            except Exception:
                llm = None
        if llm is None:
            return []

        actions: List[SMARTAction] = []
        for fact in verified_facts[:3]:
            prompt = (
                "아래의 사용자 맥락과 검증된 사실을 바탕으로 SMART 원칙을 충족하는 즉시 실행 가능한 행동 1개를 생성하라.\n"
                "- Specific: 장소/제품 등 구체적\n- Measurable: 시간/거리 등 측정 가능\n- Achievable: 현재 상황 고려\n- Relevant: 맥락 관련성 높음\n- Time-bound: 명확한 시한\n"
                '출력(JSON 단일 객체): {"action":str, "rationale":str, "urgency":0-10, "time_bound":str, "measurable":str}'
            )
            user = json.dumps(
                {
                    "internal_contexts": {
                        k: (v[:600]) for k, v in internal_contexts.items()
                    },
                    "verified_fact": fact,
                },
                ensure_ascii=False,
            )
            try:
                resp = llm.invoke(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user},
                    ]
                )
                content = (getattr(resp, "content", "") or "").strip()
                data = json.loads(content) if content.startswith("{") else {}
                if not data:
                    continue
                action = SMARTAction(
                    action=(data.get("action") or "").strip()[:200],
                    rationale=(data.get("rationale") or "").strip()[:300],
                    urgency=max(0, min(10, int(data.get("urgency", 5)))),
                    time_bound=(data.get("time_bound") or "").strip()[:80],
                    measurable=(data.get("measurable") or "").strip()[:80],
                )
                actions.append(action)
            except Exception:
                continue
        return actions

    def _aggregate_confidence(
        self, verified_facts: List[VerifiedFact], inp: _Inputs
    ) -> Tuple[float, int]:
        if not verified_facts:
            return 0.0, 0

        # mean trust
        mean_trust = 0.0
        try:
            if verified_facts:
                mean_trust = sum(
                    float(f.get("trust_score", 0.0)) for f in verified_facts
                ) / max(1, len(verified_facts))
        except Exception:
            mean_trust = 0.0

        # consistency (웹 결과 기반 재계산)
        try:
            consistency_score = float(
                self.consistency.calculate_consistency(inp.web_results)
            )
        except Exception:
            consistency_score = 0.0

        # 보수적 집계: 0.7*신뢰 + 0.3*일관성, [0,1]
        confidence = max(0.0, min(1.0, 0.7 * mean_trust + 0.3 * consistency_score))

        # 모순 추정(보수적): 일관성 낮으면 1, 아니면 0
        contradictions_found = 1 if consistency_score < 0.4 else 0

        return confidence, contradictions_found


__all__ = ["SynthesizerAgent"]
