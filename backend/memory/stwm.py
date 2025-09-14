import time
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

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
        self.ttl_sec = ttl_min * 60
        self.max_slots = max_slots
        self.flush_batch = flush_batch
        self._by_session: Dict[str, List[Dict[str, Any]]] = {}

    # -----------------------------
    # 공개 API
    # -----------------------------
    def update(self, session_id: str, user_text: str) -> STWMSnapshot:
        """
        사용자 발화를 입력받아 STWM 슬롯을 생성/추가한다.
        - NER/규칙 기반으로 last_loc/last_time/last_subject/entities 추출
        - TTL 초과 항목은 즉시 제거 대상으로 표시
        - 용량 초과 시 오래된 flush_batch 개수만큼 플러시 대상으로 반환 가능
        """
        now = time.time()
        slots = self._by_session.setdefault(session_id, [])

        # TTL 정리
        cutoff = now - self.ttl_sec
        slots[:] = [s for s in slots if s.get("ts", 0.0) >= cutoff]

        greeting_only = _is_greeting_only(user_text)
        spans = _ner_spans(user_text)

        # 직전 스냅샷 (carry-over에 사용)
        try:
            prev = self.get_latest(session_id)
        except Exception:
            prev = None

        # 앵커 추출
        last_person = _extract_last_person(user_text, spans)
        last_loc = _extract_last_loc(user_text, spans)
        last_time = _extract_last_time(user_text)
        last_act = _extract_last_act(user_text)
        last_target = _extract_last_target(user_text)
        last_emotion = _extract_last_emotion(user_text)
        last_topic = _extract_last_topic(user_text)
        last_item = _extract_last_item(user_text)

        # carry-over: 이번 턴 미탐지 → 직전 값 유지
        def _co(cur: Optional[str], get_prev):
            return cur if (cur and cur.strip()) else (get_prev() if prev else None)

        last_person = _co(last_person, lambda: getattr(prev, "last_person", None))
        last_loc = _co(last_loc, lambda: getattr(prev, "last_loc", None))
        last_time = _co(last_time, lambda: getattr(prev, "last_time", None))
        last_act = _co(last_act, lambda: getattr(prev, "last_act", None))
        last_target = _co(last_target, lambda: getattr(prev, "last_target", None))
        last_emotion = _co(last_emotion, lambda: getattr(prev, "last_emotion", None))
        last_topic = _co(last_topic, lambda: getattr(prev, "last_topic", None))
        last_item = _co(last_item, lambda: getattr(prev, "last_item", None))

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
            entities=_extract_entities(user_text, spans),
            ts=now,
            extras=None,
        )
        slots.append({"ts": now, "data": snap})

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

    def get_latest(self, session_id: str) -> Optional[STWMSnapshot]:
        slots = self._by_session.get(session_id) or []
        if not slots:
            return None
        return slots[-1]["data"]

    def get_all(self, session_id: str) -> List[STWMSnapshot]:
        return [s["data"] for s in (self._by_session.get(session_id) or [])]

    def export_latest_as_dict(self, session_id: str) -> Dict[str, Any]:
        snap = self.get_latest(session_id)
        return (
            asdict(snap)
            if snap
            else {
                "schema_ver": 1,
                "last_loc": None,
                "last_time": None,
                "last_subject": None,
                "entities": [],
                "ts": 0.0,
                "extras": None,
            }
        )


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
                if re.search(r"(역|동|구|시|군|도)\b", s):
                    score += 3
                loc_cands.append((score, s))
        if loc_cands:
            loc_cands.sort(key=lambda x: -x[0])
            return loc_cands[0][1]
    # 스팬이 비어도 텍스트 자체가 지명 패턴이면 사용
    if re.fullmatch(r"[가-힣A-Za-z0-9]{2,10}(역|동|구|시|군|도)", t):
        return t
    m = re.search(r"([가-힣A-Za-z0-9]{2,10})(역|동|구|시|군|도)\b", t)
    if m:
        return m.group(0)
    if re.search(r"근처|가까운", t):
        # 모호하므로 주제 추출에 위임하고 여기선 None
        return None
    return None


def _extract_last_act(text: str) -> Optional[str]:
    ACTS = [
        "데이트",
        "회의",
        "미팅",
        "공부",
        "여행",
        "운동",
        "식사",
        "쇼핑",
        "산책",
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
        "동료",
        "팀장",
        "상사",
        "엄마",
        "아빠",
        "부모님",
        "가족",
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
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from pathlib import Path

        base = Path(__file__).resolve().parents[2] / "models" / "ko-ner"
        if not base.exists():
            return False
        _NER_TOKENIZER = AutoTokenizer.from_pretrained(str(base))
        _NER_MODEL = AutoModelForTokenClassification.from_pretrained(str(base))
        _NER_MODEL.eval()
        return True
    except Exception:
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
        spans: List[tuple[str, str]] = []
        buf_tokens: List[str] = []
        buf_lab: Optional[str] = None
        for tk, lb_id in zip(tokens, pred):
            if tk in ("[CLS]", "[SEP]"):
                continue
            piece = tk.replace("##", "")
            lab_raw = str(id2label.get(int(lb_id), "O"))
            lab_type = lab_raw.split("-")[-1] if lab_raw != "O" else "O"
            if lab_type in {"LOC", "ORG", "PER", "PERSON"}:
                # 통합: PER → PERSON
                lab_norm = "PERSON" if lab_type in {"PER", "PERSON"} else lab_type
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
        return spans
    except Exception:
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


def update_stwm(session_id: str, user_text: str) -> STWMSnapshot:
    """세션의 STWM을 갱신하고 최신 스냅샷을 반환."""
    return _STWM_STORE.update(session_id, user_text)


def get_stwm_snapshot(session_id: str) -> Dict[str, Any]:
    """최신 STWM 스냅샷을 dict로 반환(없으면 빈 스냅샷)."""
    return _STWM_STORE.export_latest_as_dict(session_id)
