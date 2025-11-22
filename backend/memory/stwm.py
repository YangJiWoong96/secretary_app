import logging
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as _np

from backend.config import get_settings
from backend.memory.stwm_plugins import (
    extract_goal_with_qa,
    extract_koner_spans,
    extract_mecab_intent,
    mecab_tokens,
)
from backend.rag.embeddings import (
    embed_query_cached,
    embed_query_gemma,
)

# 선택적 Kiwi 형태소/띄어쓰기 보정기 (kiwipiepy)
# - Heavy init은 모듈 import 시점이 아닌 최초 접근 시점에만 수행한다.
try:  # pragma: no cover - 환경별 의존성
    from kiwipiepy import Kiwi  # type: ignore
except Exception:  # pragma: no cover
    Kiwi = None  # type: ignore

_KIWI = None  # 지연 초기화용 싱글톤 인스턴스


def _ensure_kiwi():
    """Kiwi 인스턴스를 지연 생성한다. 실패 시 None을 반환한다."""
    global _KIWI
    if _KIWI is not None:
        return _KIWI
    try:
        if "Kiwi" in globals() and Kiwi is not None:  # type: ignore[name-defined]
            _KIWI = Kiwi()  # type: ignore[call-arg]
        else:
            _KIWI = None
    except Exception:
        _KIWI = None
    return _KIWI


# STWM(Short-Term Working Memory): 사용자 발화 전용 휘발성 슬롯 메모리
# - LLM 금지, 규칙/NER 기반 처리
# - TTL=10분, 최대 30개, 초과 시 오래된 10개를 플러시 대상으로 표시


@dataclass
class STWMSnapshot:
    schema_ver: int
    last_person: Optional[str]
    last_loc: Optional[str]
    last_time: Optional[str]
    last_act: Optional[str]
    last_target: Optional[str]
    last_emotion: Optional[str]
    last_topic: Optional[str]
    last_item: Optional[str]
    last_goal: Optional[str]
    entities: List[str]
    ts: float
    extras: Optional[Dict[str, Optional[str]]] = None


class STWMStore:
    """
    세션별 STWM 저장소.

    내부 구조
    - self._by_session: session_id -> List[dict]
      각 원소는 { ts: float, data: STWMSnapshot }
    """

    def __init__(self, ttl_min: int = 10, max_slots: int = 30, flush_batch: int = 10):
        # 기본 TTL을 환경변수로 재정의 가능(기본 60분)
        try:
            ttl_env = int(get_settings().STWM_TTL_MIN)
        except Exception:
            ttl_env = 60
        ttl_min = ttl_env if ttl_min == 10 else ttl_min
        self.ttl_sec = ttl_min * 60
        self.max_slots = max_slots
        self.flush_batch = flush_batch

        # 복합 키 사용: "{user_id}:{session_id}"
        self._by_composite: Dict[str, List[Dict[str, Any]]] = {}
        # 필드별 신뢰도(0~1) 추적: composite_key -> { field: confidence }
        self._conf_by_composite: Dict[str, Dict[str, float]] = {}

        # 주제/위치/인물 스택도 복합 키 기반
        self._topic_stack: Dict[str, Deque[tuple[str, float]]] = {}
        self._loc_stack: Dict[str, Deque[tuple[str, float]]] = {}
        self._person_stack: Dict[str, Deque[tuple[str, float]]] = {}

    # -----------------------------
    # 공개 API
    # -----------------------------
    def update(
        self, session_id: str, user_text: str, user_id: Optional[str] = None
    ) -> STWMSnapshot:
        """
        사용자 발화를 입력받아 STWM 슬롯을 생성/추가한다.

        Args:
            session_id: 세션 ID
            user_text: 사용자 발화 텍스트
            user_id: 사용자 ID (미제공 시 router_context에서 추론)

        Returns:
            STWMSnapshot: 현재 STWM 상태 스냅샷
        """
        from backend.routing.router_context import user_for_session

        now = time.time()
        text_in = (user_text or "").strip()

        # user_id 추론
        uid = user_id or user_for_session(session_id) or session_id

        # 복합 키 생성
        composite_key = f"{uid}:{session_id}"

        # 한국어 띄어쓰기/형태소 보정: Kiwi가 있으면 우선 적용
        # - 엔티티/지명 추출 정확도 향상을 위해 전처리 텍스트에만 적용
        def _normalize_ko(txt: str) -> str:
            """한국어 전처리: Kiwi가 있으면 띄어쓰기 보정, 없으면 원문 유지."""
            if not txt:
                return txt
            try:
                kiwi = _ensure_kiwi()
                if not kiwi:
                    return txt
                norm = kiwi.space(txt)  # 띄어쓰기 보정
                return re.sub(r"\s+", " ", norm).strip()
            except Exception:
                return txt

        user_text_norm = _normalize_ko(text_in)
        slots = self._by_composite.setdefault(composite_key, [])

        # TTL 정리
        cutoff = now - self.ttl_sec
        slots[:] = [s for s in slots if s.get("ts", 0.0) >= cutoff]

        greeting_only = _is_greeting_only(user_text_norm)
        spans = _ner_spans(user_text_norm)
        try:
            _LOGGER.info(
                "[stwm] ner_loaded=%s mecab_enabled=%s text='%s'",
                bool(_NER_MODEL is not None),
                str(bool(get_settings().STWM_USE_MECAB)),
                (user_text[:120] if user_text else ""),
            )
        except Exception:
            pass
        try:
            if bool(get_settings().STWM_NER_DEBUG):
                _LOGGER.info(
                    "[stwm:ner] sid=%s text='%s' spans=%d",
                    session_id,
                    (user_text[:120] if user_text else ""),
                    len(spans or []),
                )
        except Exception:
            pass

        # 직전 스냅샷 (carry-over에 사용)
        try:
            prev = self.get_latest(session_id, user_id=user_id)
        except Exception:
            prev = None

        # 앵커 추출
        last_person = _extract_last_person(user_text_norm, spans)
        last_loc = _extract_last_loc(user_text_norm, spans)
        last_time = _extract_last_time(user_text_norm)
        # 휴리스틱 기반 함수 제거(안티패턴)
        last_act = None
        last_target = None
        last_emotion = None
        last_topic = None
        last_item = None

        # LOC 보정 후보/가제티어/토큰 근사 제거: Ko-NER/Mecab 1차 추출만 유지(오염 방지)

        # 임베딩 유사도 기반 보정 로직
        def _cos_sim(a: Optional[str], b: Optional[str]) -> float:
            try:
                ta = (a or "").strip()
                tb = (b or "").strip()
                if not ta or not tb:
                    return 0.0
                # STWM 신뢰도 계산: 의미적 일관성 검사용 → Gemma 전용
                va = embed_query_gemma(ta)
                vb = embed_query_gemma(tb)
                denom = float((_np.linalg.norm(va) * _np.linalg.norm(vb)) or 1.0)
                s = float(_np.dot(va, vb) / denom)
                return max(0.0, min(1.0, s))
            except Exception:
                return 0.0

        conf = self._conf_by_composite.setdefault(composite_key, {})
        # 기본 신뢰도(초기값)
        for k in (
            "last_person",
            "last_loc",
            "last_time",
            "last_act",
            "last_target",
            "last_emotion",
            "last_topic",
            "last_item",
        ):
            conf.setdefault(k, 0.5)

        # carry-over: 이번 턴 미탐지 → 직전 값 유지 (검증된 유지: 시간+의미 일치 기반)
        s = get_settings()
        STRICT_TAU = float(getattr(s, "STWM_CARRY_STRICT", 0.70))
        MID_TAU = float(getattr(s, "STWM_CARRY_MED", 0.50))
        RET_TTL = float(getattr(s, "STWM_RET_TTL_SEC", 600))
        RET_TAU = float(getattr(s, "STWM_RET_TAU", 0.55))

        def _co(field: str, cur: Optional[str], get_prev):
            pv = get_prev() if prev else None
            cur_norm = (cur or "").strip()
            prev_norm = (pv or "").strip()
            if cur_norm and prev_norm:
                sim = _cos_sim(cur_norm, prev_norm)
                if sim >= STRICT_TAU:
                    conf[field] = conf[field] * 0.8 + 0.2
                    return cur_norm
                if sim >= MID_TAU:
                    conf[field] = conf[field] * 0.9 + 0.1
                    return prev_norm
                conf[field] = conf[field] * 0.95
                return cur_norm if conf[field] < 0.5 else prev_norm
            if cur_norm and not prev_norm:
                conf[field] = conf[field] * 0.85 + 0.15
                return cur_norm
            if not cur_norm and prev_norm:
                conf[field] = conf[field] * 0.98  # 더 천천히 감쇠(맥락 단절 방지)
                return prev_norm
            conf[field] = conf[field] * 0.98
            return None

        def _co_stack(
            field: str,
            cur: Optional[str],
            stack_dict: Dict[str, Deque[tuple[str, float]]],
        ):
            """
            스택 기반 carry-over: 주제/위치가 바뀌어도 스택에 보관하고 재언급 시 복원
            """
            cur_norm = (cur or "").strip()
            stack_key = composite_key
            stack = stack_dict.setdefault(stack_key, deque(maxlen=3))

            # 현재 값이 있으면
            if cur_norm:
                # 스택 최상단과 다르면 푸시
                if not stack or stack[-1][0] != cur_norm:
                    stack.append((cur_norm, now))
                return cur_norm

            # 현재 값이 없으면 스택에서 복원 시도
            if not stack:
                return None

            # 사용자 발화와 의미적으로 가장 유사한 스택 항목 찾기
            best_match = None
            best_sim = 0.0
            user_emb = None

            try:
                for stacked_val, stacked_ts in reversed(list(stack)):
                    # TTL 체크
                    if (now - stacked_ts) > RET_TTL:
                        continue

                    # 의미 유사도 계산 (현재 발화와 스택 항목)
                    if user_emb is None:
                        user_emb = (user_text or "").strip()

                    sim = _cos_sim(user_emb, stacked_val)
                    if sim >= RET_TAU and sim > best_sim:
                        best_sim = sim
                        best_match = stacked_val

                return best_match
            except Exception:
                # 폴백: 스택 최상단 반환 (TTL 체크만)
                for stacked_val, stacked_ts in reversed(list(stack)):
                    if (now - stacked_ts) <= RET_TTL:
                        return stacked_val
                return None

        last_person = _co_stack("last_person", last_person, self._person_stack)
        # 위치/주제는 스택 기반 carry-over 유지, 나머지는 제거
        last_loc = _co_stack("last_loc", last_loc, self._loc_stack)
        last_time = _co(
            "last_time", last_time, lambda: getattr(prev, "last_time", None)
        )
        # 제거된 필드는 carry-over 미적용
        last_topic = _co_stack("last_topic", last_topic, self._topic_stack)

        # MeCab/Ko-NER 보조 추출(추가 QA/가제티어/후보풀 사용 금지)
        try:
            if bool(get_settings().STWM_USE_KONER):
                k = extract_koner_spans(user_text_norm)
                last_loc = last_loc or k.get("last_loc") or last_loc
                last_person = last_person or k.get("last_person") or last_person
            last_goal = None
            if bool(get_settings().STWM_USE_MECAB):
                m = extract_mecab_intent(user_text_norm)
                # MeCab/Ko-NER 전용 추출만 유지. 휴리스틱 미사용.
                last_goal = m.get("last_goal")
            _LOGGER.debug(
                "[stwm] merge fields person=%s loc=%s time=%s act=%s target=%s goal=%s topic=%s",
                str(last_person or ""),
                str(last_loc or ""),
                str(last_time or ""),
                str(last_act or ""),
                str(last_target or ""),
                str(last_goal or ""),
                str(last_topic or ""),
            )
        except Exception:
            last_goal = None

        snap = STWMSnapshot(
            schema_ver=2,
            last_person=last_person,
            last_loc=last_loc,
            last_time=last_time,
            last_act=last_act,
            last_target=last_target,
            last_emotion=last_emotion,
            last_topic=last_topic,
            last_item=last_item,
            last_goal=last_goal,
            entities=_extract_entities(user_text_norm, spans),
            ts=now,
            extras=None,
        )
        slots.append({"ts": now, "data": snap})
        try:
            _LOGGER.info(
                "[stwm] entities=%s last_loc=%s last_topic=%s",
                ", ".join((snap.entities or [])[:8]),
                str(snap.last_loc or ""),
                str(snap.last_topic or ""),
            )
        except Exception:
            pass
        try:
            if bool(get_settings().STWM_NER_DEBUG):
                _LOGGER.info(
                    "[stwm:entities] sid=%s entities=%s loc=%s topic=%s",
                    session_id,
                    ", ".join((snap.entities or [])[:8]),
                    str(snap.last_loc or ""),
                    str(snap.last_topic or ""),
                )
        except Exception:
            pass

        # 용량 초과 시 오래된 일부를 제거
        flushed: List[STWMSnapshot] = []
        if len(slots) > self.max_slots:
            # 오래된 순으로 flush_batch만큼 분리
            overflow = len(slots) - self.max_slots
            n_flush = max(self.flush_batch, overflow)
            old = slots[:n_flush]
            flushed = [o["data"] for o in old]
            del slots[:n_flush]

        return snap

    def get_latest(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Optional[STWMSnapshot]:
        """
        최신 STWM 스냅샷 조회

        Args:
            session_id: 세션 ID
            user_id: 사용자 ID (선택적)

        Returns:
            최신 STWM 스냅샷 (없으면 None)
        """
        from backend.routing.router_context import user_for_session

        uid = user_id or user_for_session(session_id) or session_id
        composite_key = f"{uid}:{session_id}"

        slots = self._by_composite.get(composite_key) or []
        if not slots:
            return None
        return slots[-1]["data"]

    def get_all(
        self, session_id: str, user_id: Optional[str] = None
    ) -> List[STWMSnapshot]:
        """
        모든 STWM 스냅샷 조회

        Args:
            session_id: 세션 ID
            user_id: 사용자 ID (선택적)

        Returns:
            STWM 스냅샷 리스트
        """
        from backend.routing.router_context import user_for_session

        uid = user_id or user_for_session(session_id) or session_id
        composite_key = f"{uid}:{session_id}"

        return [s["data"] for s in (self._by_composite.get(composite_key) or [])]

    def export_latest_as_dict(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        최신 STWM 스냅샷을 딕셔너리로 반환

        Args:
            session_id: 세션 ID
            user_id: 사용자 ID (선택적)

        Returns:
            STWM 상태 딕셔너리
        """
        snap = self.get_latest(session_id, user_id=user_id)
        if snap:
            # 최신 스키마(v2)로 내보내되, 과거 호환을 위해 last_subject 별칭을 포함한다.
            data = asdict(snap)
            data["schema_ver"] = 2
            # 과거 코드 호환성: last_subject → 인물 중심 맥락으로 매핑(없으면 None)
            data["last_subject"] = data.get("last_person") or None
            return data
        # 스냅샷이 없을 때도 v2 스키마로 기본값을 반환하고, last_subject 별칭 포함
        return {
            "schema_ver": 2,
            "last_person": None,
            "last_loc": None,
            "last_time": None,
            "last_act": None,
            "last_target": None,
            "last_emotion": None,
            "last_topic": None,
            "last_item": None,
            "last_goal": None,
            "entities": [],
            "ts": 0.0,
            "extras": None,
            "last_subject": None,
        }


# --------------------------------------
# 규칙/NER 보조 추출기 (LLM 금지)
# --------------------------------------


_TIME_PATTS = [
    r"오늘",
    r"내일",
    r"모레",
    r"글피",
    r"어제",
    r"그제",
    r"그저께",
    r"이번주",
    r"다음주",
    r"지난주",
    r"이번달",
    r"다음달",
    r"지난달",
    r"오전\s*\d{1,2}시",
    r"오후\s*\d{1,2}시",
    r"\d{1,2}시\s*\d{0,2}분?",
]


def _extract_last_time(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    for pat in _TIME_PATTS:
        m = re.search(pat, t)
        if m:
            return m.group(0)
    return None


def _extract_last_loc(
    text: str, spans: List[tuple[str, str]] | None = None
) -> Optional[str]:
    # 간단 장소 힌트 추출: '역/동/구/시/군/도' 접미 또는 '근처/가까운'
    t = (text or "").strip()
    if not t:
        return None
    # NER 우선: 단일 지명 입력("강남역") 처리를 위해 전체 텍스트도 후보로 비교
    if spans:
        # 스팬 표면형 길이/정규식 매칭 점수로 소팅하여 가장 강한 LOC 선택
        loc_cands = []
        for s, lab in spans:
            if lab == "LOC":
                score = len(s)
                # 지명 접미사 가중(‘도’는 광역/도 단위에만 한정)
                if re.search(r"(역|동|구|시|군|읍|면|리|로|길)\b", s) or re.search(
                    r"(경기|강원|충남|충북|전남|전북|경남|경북|제주|세종)도\b", s
                ):
                    score += 3
                loc_cands.append((score, s))
        if loc_cands:
            loc_cands.sort(key=lambda x: -x[0])
            return loc_cands[0][1]
    # 스팬이 비어도 텍스트 자체가 지명 패턴이면 사용
    # 단일 토큰 전체가 지명 패턴일 때만 허용 (접미사 제한)
    if re.fullmatch(r"[가-힣A-Za-z0-9]{2,10}(역|동|구|시|군|읍|면|리|로|길)", t):
        return t
    # 광역/도 단위는 화이트리스트만 허용 (술집도 오탐 방지)
    m_do = re.search(r"(경기|강원|충남|충북|전남|전북|경남|경북|제주|세종)도\b", t)
    if m_do:
        return m_do.group(0)
    m = re.search(r"([가-힣A-Za-z0-9]{2,10})(역|동|구|시|군|읍|면|리|로|길)\b", t)
    if m:
        return m.group(0)
    if re.search(r"근처|가까운", t):
        # 모호하므로 주제 추출에 위임하고 여기선 None
        return None
    # Gazetteer 정확 일치 지름길: 입력 전체가 인명사전 항목과 정확히 일치하면 채택
    try:
        if _ensure_gazetteer():
            nm = t.strip()
            if nm and _GAZ_NAMES and nm in _GAZ_NAMES:
                return nm
    except Exception:
        pass
    return None


def _extract_last_act(text: str) -> Optional[str]:
    ACTS = [
        "데이트",
        "회의",
        "미팅",
        "브리핑",
        "보고",
        "학습",
        "공부",
        "연구",
        "여행",
        "출근",
        "퇴근",
        "운동",
        "식사",
        "장보기",
        "쇼핑",
        "산책",
        "영화보기",
        "독서",
    ]
    t = (text or "").strip()
    if not t:
        return None
    for kw in ACTS:
        if kw in t:
            return kw
    return None


def _extract_last_target(text: str) -> Optional[str]:
    TARGETS = [
        "여친",
        "여자친구",
        "남친",
        "남자친구",
        "친구",
        "지인",
        "고객",
        "동료",
        "팀원",
        "팀장",
        "매니저",
        "상사",
        "대표",
        "엄마",
        "아빠",
        "부모님",
        "가족",
        "아이",
        "선생님",
        "교수님",
    ]
    t = (text or "").strip()
    if not t:
        return None
    for kw in TARGETS:
        if kw in t:
            return kw
    return None


def _extract_last_emotion(text: str) -> Optional[str]:
    EMOTIONS = [
        "기대",
        "설렘",
        "피곤",
        "지침",
        "걱정",
        "불안",
        "기쁨",
        "행복",
        "슬픔",
        "우울",
        "긴장",
    ]
    t = (text or "").strip()
    if not t:
        return None
    for kw in EMOTIONS:
        if kw in t:
            return kw
    return None


def _extract_last_topic(text: str) -> Optional[str]:
    TOPICS = [
        "맛집",
        "카페",
        "논문",
        "일정",
        "여행",
        "공부",
        "운동",
        "영화",
        "음악",
        "주식",
        "뉴스",
        "금융",
        "부동산",
        "정책",
        "개발",
        "프로그래밍",
        "클라우드",
        "헬스케어",
        "의학",
        "식단",
        "다이어트",
        "스포츠",
        "야구",
        "축구",
        "테니스",
        "날씨",
    ]
    t = (text or "").strip()
    if not t:
        return None
    for kw in TOPICS:
        if kw in t:
            return kw
    return None


def _extract_last_item(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    # 따옴표/인용부호로 둘러싼 구절 추출 (ASCII/유니코드 인용부호 지원)
    m = re.search(r'["\'“”‘’](.+?)["\'“”‘’]', t)
    if m:
        cand = m.group(1).strip()
        if 1 <= len(cand) <= 30:
            return cand
    m2 = re.search(r"\b([A-Za-z][A-Za-z0-9\-]{2,20})\b", t)
    if m2:
        return m2.group(1)
    return None


def _extract_entities(
    text: str, spans: List[tuple[str, str]] | None = None
) -> List[str]:
    # STWM 엔티티는 NER 전용. 화이트리스트 레이블만 수집.
    if not spans:
        spans = _ner_spans(text)
    out: List[str] = []
    seen = set()
    for s, lab in spans:
        if lab in {"LOC", "ORG", "PERSON"}:
            k = s.strip()
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        if len(out) >= 8:
            break
    return out


# -------------------------------
# LOC 임베딩 재랭킹 / Gazetteer
# -------------------------------
def _pick_loc_by_tokens(
    user_text: str, loc_cands: list[str], tau: float = 0.65
) -> Optional[str]:
    """OpenAI 임베딩 없이 MeCab 토큰/문자 n-gram 기반 근사 재랭킹."""
    try:
        if not loc_cands:
            return None
        utoks = set(mecab_tokens(user_text))
        if not utoks:
            utoks = set(list(user_text))

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            uni = len(a | b)
            return float(inter) / float(uni or 1)

        best = None
        best_s = 0.0
        for s in set(loc_cands):
            stoks = set(mecab_tokens(s)) or set(list(s))
            score = _jaccard(utoks, stoks)
            if score > best_s:
                best, best_s = s, score
        return best if best_s >= tau else None
    except Exception:
        return None


_GAZ_NAMES: Optional[list] = None
_GAZ_VECS = None


def _ensure_gazetteer():
    global _GAZ_NAMES, _GAZ_VECS
    if _GAZ_NAMES is not None and _GAZ_VECS is not None:
        return True
    try:
        # 간단한 인메모리 로드: 존재 시 파일에서, 없으면 비활성
        import json as _json
        from pathlib import Path

        base_dir = Path(__file__).resolve().parents[2] / "data"
        base = base_dir / "gazetteer_kor.json"
        built = base_dir / "gazetteer_kor_built.json"
        if not base.exists() and not built.exists():
            return False
        # 빌드된 파일이 없으면 즉시 빌드(임베딩 채우기)
        try:

            def _load_items():
                with open(built, "r", encoding="utf-8") as f:
                    return _json.load(f)

            if built.exists():
                items = _load_items()
            else:
                res = __import__(
                    "backend.rag.utils", fromlist=["build_gazetteer_embeddings"]
                ).build_gazetteer_embeddings(str(base), str(built), overwrite=True)
                items = _load_items()

            # 임베딩 차원 불일치 시 Gemma로 재빌드
            try:
                gemma_dim = int(embed_query_gemma("서울").shape[0])
                file_dim = None
                for it in items:
                    emb = it.get("emb") if isinstance(it, dict) else None
                    if isinstance(emb, list) and emb:
                        file_dim = len(emb)
                        break
                if file_dim is None or file_dim != gemma_dim:
                    res = __import__(
                        "backend.rag.utils", fromlist=["build_gazetteer_embeddings"]
                    ).build_gazetteer_embeddings(str(base), str(built), overwrite=True)
                    items = _load_items()
            except Exception:
                pass
        except Exception:
            return False

        # 이름/벡터 분리 후 메모리 적재
        import numpy as np

        names = []
        vecs = []
        for it in items:
            nm = (it.get("name") or "").strip() if isinstance(it, dict) else str(it)
            emb = it.get("emb") if isinstance(it, dict) else None
            if nm and isinstance(emb, list) and emb:
                names.append(nm)
                vecs.append(np.array(emb, dtype=np.float32))
        if not names or not vecs:
            return False
        _GAZ_NAMES = names
        _GAZ_VECS = np.stack(vecs, axis=0)
        return True
    except Exception:
        _GAZ_NAMES, _GAZ_VECS = None, None
        return False


def _gazetteer_lookup(
    user_text: str, topk: int = 15, tau: float = 0.68
) -> Optional[str]:
    try:
        import numpy as np

        if not _ensure_gazetteer():
            return None
        # Gazetteer 지명 검색: 대규모 지명 검색 → Gemma 전용
        qv = embed_query_gemma(user_text)
        den_q = float(np.linalg.norm(qv) or 1.0)
        sims = (_GAZ_VECS @ qv) / (np.linalg.norm(_GAZ_VECS, axis=1) * den_q)
        idx = np.argsort(-sims)[: max(1, min(topk, sims.shape[0]))]

        # 하이브리드 재랭킹: 간단 n-gram 겹침 가중(자모/Levenshtein은 옵션 확장)
        def _ngram_overlap(a: str, b: str, n: int = 2) -> float:
            aa = set(a[i : i + n] for i in range(max(0, len(a) - n + 1)))
            bb = set(b[i : i + n] for i in range(max(0, len(b) - n + 1)))
            if not aa or not bb:
                return 0.0
            return len(aa & bb) / float(len(aa | bb))

        scored = []
        for i in idx:
            name = _GAZ_NAMES[int(i)]
            sim = float(sims[int(i)])
            ng = _ngram_overlap(user_text, name, 2)
            # 가중 결합(환경변수로 조정 가능)
            from backend.config import get_settings as _gs

            w_emb = float(getattr(_gs(), "GAZ_EMB_WEIGHT", 0.8))
            w_ng = 1.0 - w_emb
            score = w_emb * sim + w_ng * ng
            scored.append((score, name, sim))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return None
        best_score, best_name, best_sim = scored[0]
        return best_name if best_sim >= tau else None
    except Exception:
        return None


def _extract_extras(text: str) -> Dict[str, Optional[str]]:
    # 유지: update()에서 last_person을 계산하여 주입
    return {
        "last_person": None,
        "last_event": None,
        "last_geo_hint": None,
    }


_NER_MODEL = None
_NER_TOKENIZER = None


def _lazy_load_ner():
    global _NER_MODEL, _NER_TOKENIZER
    if _NER_MODEL is not None:
        return True
    try:
        from pathlib import Path

        from transformers import AutoModelForTokenClassification, AutoTokenizer

        base = Path(__file__).resolve().parents[2] / "models" / "ko-ner"
        if not base.exists():
            _LOGGER.warning("[stwm:ner] ko-ner path missing: %s", str(base))
            return False
        # 필수 파일 점검(없어도 로드 시도는 진행)
        required = [
            base / "config.json",
            base / "pytorch_model.bin",
            base / "tokenizer.json",
            base / "tokenizer_config.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            _LOGGER.warning("[stwm:ner] ko-ner missing files: %s", ", ".join(missing))
        _NER_TOKENIZER = AutoTokenizer.from_pretrained(str(base))
        _NER_MODEL = AutoModelForTokenClassification.from_pretrained(str(base))
        _NER_MODEL.eval()
        _LOGGER.info("[stwm:ner] loaded ko-ner from %s", str(base))
        return True
    except Exception:
        _LOGGER.exception("[stwm:ner] load failed")
        return False


def _ner_spans(text: str) -> List[tuple[str, str]]:
    """
    토큰 분류 모델에서 연속 스팬을 재구성하여 (표면형, 레이블) 리스트 반환.
    화이트리스트 레이블은 LOC/ORG/PERSON. 없으면 빈 리스트.
    """
    try:
        if not _lazy_load_ner():
            return []
        import torch

        tok = _NER_TOKENIZER
        mdl = _NER_MODEL
        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            out = mdl(**enc)
            pred = out.logits.argmax(-1)[0].tolist()
            ids = enc["input_ids"][0].tolist()
            tokens = tok.convert_ids_to_tokens(ids)
        id2label = getattr(getattr(mdl, "config", None), "id2label", {}) or {}
        # 라벨 표준화: 공개 ko-ner의 축약(PS/LC/OG) → 표준(PERSON/LOC/ORG)
        _LABEL_ALIAS = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "PS": "PERSON",
            "LOC": "LOC",
            "LC": "LOC",
            "ORG": "ORG",
            "OG": "ORG",
        }
        spans: List[tuple[str, str]] = []
        buf_tokens: List[str] = []
        buf_lab: Optional[str] = None
        for tk, lb_id in zip(tokens, pred):
            if tk in ("[CLS]", "[SEP]"):
                continue
            piece = tk.replace("##", "")
            lab_raw = str(id2label.get(int(lb_id), "O"))
            lab_type = lab_raw.split("-")[-1] if lab_raw != "O" else "O"
            lab_norm = _LABEL_ALIAS.get(lab_type, "O")
            if lab_norm in {"LOC", "ORG", "PERSON"}:
                if buf_lab == lab_norm:
                    buf_tokens.append(piece)
                else:
                    if buf_tokens and buf_lab:
                        spans.append(("".join(buf_tokens), buf_lab))
                    buf_tokens = [piece]
                    buf_lab = lab_norm
            else:
                if buf_tokens and buf_lab:
                    spans.append(("".join(buf_tokens), buf_lab))
                buf_tokens = []
                buf_lab = None
        if buf_tokens and buf_lab:
            spans.append(("".join(buf_tokens), buf_lab))
        try:
            if bool(int(os.getenv("STWM_NER_DEBUG", "0"))):
                _LOGGER.info(
                    "[stwm:ner:spans] text='%s' spans=%s",
                    (text or "")[:120],
                    ", ".join([f"{s}:{lab}" for s, lab in spans[:8]]),
                )
        except Exception:
            pass
        return spans
    except Exception:
        _LOGGER.exception("[stwm:ner] inference failed")
        return []


_PERSON_PAT = re.compile(
    r"(?:저는|난|나는|내\s*이름은)\s*([가-힣A-Za-z]{1,20})\s*(?:입니다|이에요|라고\s*해|라고\s*불러|이야|야)",
)


def _extract_last_person(
    text: str, spans: List[tuple[str, str]] | None = None
) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    # 조사/어미 제거를 위한 약한 정규화: '제이야' -> '제이'
    t = re.sub(r"\b([가-힣A-Za-z]{1,20})(?:[은는이가의]|[이-힣]{0,2}야)\b", r"\1", t)
    m = _PERSON_PAT.search(t)
    if m:
        name = m.group(1).strip()
        return name
    if spans:
        for s, lab in spans:
            if lab == "PERSON":
                # 어미/조사 탈락 정규화
                s2 = re.sub(r"(씨|님|군|양)$", "", s)
                return s2.strip() or s
    return None


_GREET_PAT = re.compile(r"(안녕|반갑|잘\s*부탁|고마워|감사|헬로|hello|hi)")
_DOMAIN_PAT = re.compile(r"(카페|식당|병원|영화관|약국|호텔|주차|날씨|가격|영업시간)")


def _is_greeting_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    has_greet = _GREET_PAT.search(t) is not None
    has_domain = _DOMAIN_PAT.search(t) is not None
    # 인사만 있고 도메인 키워드가 없을 때만 greeting-only로 본다
    return has_greet and (not has_domain)


# 전역 스토어 싱글톤
_STWM_STORE = STWMStore()
_LOGGER = logging.getLogger("stwm")


def update_stwm(
    session_id: str, user_text: str, user_id: Optional[str] = None
) -> STWMSnapshot:
    """
    STWM 업데이트 헬퍼 (싱글톤 스토어 사용)

    Args:
        session_id: 세션 ID
        user_text: 사용자 발화
        user_id: 사용자 ID (선택적)

    Returns:
        STWMSnapshot
    """
    return _STWM_STORE.update(session_id, user_text, user_id=user_id)


def get_stwm_snapshot(session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    현재 STWM 상태를 딕셔너리로 반환 (Behavior 분류용)

    Args:
        session_id: 세션 ID
        user_id: 사용자 ID (선택적)

    Returns:
        Dict: STWM 상태 (last_person, last_loc, entities 등)
    """
    return _STWM_STORE.export_latest_as_dict(session_id, user_id=user_id)
