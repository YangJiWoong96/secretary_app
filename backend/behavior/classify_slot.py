# -*- coding: utf-8 -*-
"""
backend.behavior.classify_slot - 100개 행동 슬롯 규칙 기반 분류기

요구사항:
- C:\My_Business\행동기반.md 의 100개 슬롯을 전부 그대로 사용
- slot_key, norm_key, 규칙 기반 분류 로직, Evidence 스니펫 구성
- STWM/MeCab/KoNER 등 기존 유틸을 보조 신호로 사용(LLM 금지)

출력 형식:
List[Dict] = [
  {
    "slot_key": str,           # 예: "정보 탐색 패턴/질문 명확도"
    "norm_key": str,           # 예: "behavior.info_seeking.question_clarity"
    "value": str | float | bool,
    "confidence": float,       # 0.0 ~ 1.0
    "evidence": str            # 사용자 텍스트의 근거 스니펫(최대 140자)
  }, ...
]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.memory.stwm import mecab_tokens  # type: ignore
except Exception:

    def mecab_tokens(text: str) -> List[str]:  # pragma: no cover
        return list(text or "")


def _evidence_snippet(text: str, head: int = 140) -> str:
    t = (text or "").strip()
    return t[:head].replace("\n", " ")


@dataclass(frozen=True)
class SlotDef:
    category_key: str  # 예: "info_seeking"
    category_label: str  # 예: "정보 탐색 패턴"
    slot_key: str  # 예: "question_clarity"
    slot_label: str  # 예: "질문 명확도(명확 / 모호)"

    @property
    def norm_key(self) -> str:
        return f"behavior.{self.category_key}.{self.slot_key}"

    @property
    def slot_path(self) -> str:
        return f"{self.category_label}/{self.slot_label}"


# 1) 100개 슬롯 정의(행동기반.md를 1:1 반영 - 이름/순서 고정)
# 카테고리 키 매핑
CAT = {
    1: ("info_seeking", "정보 탐색 패턴"),
    2: ("decision_making", "의사결정 스타일"),
    3: ("affect", "감정 패턴"),
    4: ("interpersonal", "관계/대인 행동"),
    5: ("learning", "학습/지식 습득 성향"),
    6: ("motivation", "목표·동기 패턴"),
    7: ("preferences", "선호/취향 패턴"),
    8: ("problem_solving", "문제해결/트러블 패턴"),
    9: ("trust_safety", "신뢰·안전 행동 패턴"),
    10: ("interaction_style", "상호작용 스타일"),
}


def _slots_v1() -> List[SlotDef]:
    # 카테고리1: 정보 탐색 패턴(1~10)
    c1 = CAT[1]
    s1 = [
        SlotDef(c1[0], c1[1], "question_clarity", "질문 명확도(명확 / 모호)"),
        SlotDef(c1[0], c1[1], "info_demand", "정보 요구량(많이 요구 / 최소 요구)"),
        SlotDef(c1[0], c1[1], "uncertainty_tolerance", "정보 불확실성 허용도"),
        SlotDef(c1[0], c1[1], "evidence_strength", "근거/증거 요구 강도"),
        SlotDef(c1[0], c1[1], "expertise_expectation", "전문성 기대치"),
        SlotDef(c1[0], c1[1], "comparison_request_freq", "비교 요청 빈도"),
        SlotDef(c1[0], c1[1], "step_by_step_preference", "Step-by-step 요구 성향"),
        SlotDef(c1[0], c1[1], "summary_vs_detail", "요약 vs 상세 설명 선호"),
        SlotDef(c1[0], c1[1], "example_preference", "예시 기반 이해 선호"),
        SlotDef(c1[0], c1[1], "hypothesis_vs_acceptance", "본인의 가설 제시 vs 수용형"),
    ]
    # 카테고리2: 의사결정 스타일(11~20)
    c2 = CAT[2]
    s2 = [
        SlotDef(c2[0], c2[1], "impulsive_vs_careful", "즉흥적 결정 vs 신중함"),
        SlotDef(c2[0], c2[1], "alternatives_depth", "대안 탐색 깊이"),
        SlotDef(c2[0], c2[1], "optimization_desire", "최적화 욕구(최적해 vs 충분해)"),
        SlotDef(c2[0], c2[1], "risk_taking", "리스크 감수 성향"),
        SlotDef(
            c2[0],
            c2[1],
            "cost_efficiency_vs_emotion",
            "비용·효율 중심 vs 감정 중심 의사결정",
        ),
        SlotDef(c2[0], c2[1], "fast_conclusion", "빠른 결론 선호"),
        SlotDef(c2[0], c2[1], "regret_reconsideration", "결정 후 후회/재검토 경향"),
        SlotDef(c2[0], c2[1], "recommendation_acceptance", "추천 수용성"),
        SlotDef(c2[0], c2[1], "opinion_request_freq", "의견 요청 빈도"),
        SlotDef(c2[0], c2[1], "criteria_explicitness", "선택 기준 명시 여부"),
    ]
    # 카테고리3: 감정 패턴(21~30)
    c3 = CAT[3]
    s3 = [
        SlotDef(c3[0], c3[1], "affect_variability", "감정 기복 정도"),
        SlotDef(c3[0], c3[1], "anger_expression", "분노 표출 방식"),
        SlotDef(c3[0], c3[1], "anxiety_expression", "불안 표현 패턴"),
        SlotDef(c3[0], c3[1], "humor_usage", "유머 사용 패턴"),
        SlotDef(c3[0], c3[1], "emotion_word_freq", "감정 단어 사용 빈도"),
        SlotDef(c3[0], c3[1], "blame_tendency", "상대 비난 경향"),
        SlotDef(
            c3[0],
            c3[1],
            "stress_avoid_vs_problem_solve",
            "스트레스 시 회피 / 문제 해결 선호",
        ),
        SlotDef(c3[0], c3[1], "praise_responsivity", "칭찬 반응성"),
        SlotDef(c3[0], c3[1], "sensitive_triggers", "예민 트리거(지연, 불확실성 등)"),
        SlotDef(c3[0], c3[1], "disappointment_expression", "실망감 표현 방식"),
    ]
    # 카테고리4: 관계/대인 행동(31~40)
    c4 = CAT[4]
    s4 = [
        SlotDef(c4[0], c4[1], "dominance_vs_compliance", "우위 확보 경향(지배/순응)"),
        SlotDef(c4[0], c4[1], "rebuttal_style", "반박 스타일"),
        SlotDef(c4[0], c4[1], "cooperativeness", "협력성"),
        SlotDef(c4[0], c4[1], "direct_aggressive_speech", "공격적/직설적 말투"),
        SlotDef(c4[0], c4[1], "empathy_need", "공감 요구도"),
        SlotDef(c4[0], c4[1], "responsibility_avoidance", "책임 회피 경향"),
        SlotDef(c4[0], c4[1], "criticism_acceptance", "비판 수용성"),
        SlotDef(c4[0], c4[1], "apology_pattern", "사과 패턴"),
        SlotDef(c4[0], c4[1], "request_expression", "부탁 표현 방식"),
        SlotDef(c4[0], c4[1], "guardedness_level", "경계심 수준"),
    ]
    # 카테고리5: 학습/지식 습득 성향(41~50)
    c5 = CAT[5]
    s5 = [
        SlotDef(c5[0], c5[1], "meta_question_freq", "메타 질문 빈도(왜? 어떻게?)"),
        SlotDef(
            c5[0],
            c5[1],
            "response_when_not_understood",
            "이해도 부족 시 반응(다시 질문/포기/무시)",
        ),
        SlotDef(
            c5[0], c5[1], "generalization_vs_cases", "개념 일반화 vs 사례 중심 이해"
        ),
        SlotDef(c5[0], c5[1], "preference_latest_info", "최신 정보 선호 여부"),
        SlotDef(c5[0], c5[1], "self_censorship", "자기 검열(나는 모른다) 패턴"),
        SlotDef(c5[0], c5[1], "repeat_learning_need", "반복 학습 요구도"),
        SlotDef(c5[0], c5[1], "learning_speed_feedback", "학습 속도 피드백"),
        SlotDef(c5[0], c5[1], "analogy_preference", "비유·유추 선호"),
        SlotDef(c5[0], c5[1], "deep_discussion_preference", "깊이 있는 토론 선호"),
        SlotDef(c5[0], c5[1], "learning_fatigue_expression", "학습 피로도 표현"),
    ]
    # 카테고리6: 목표·동기 패턴(51~60)
    c6 = CAT[6]
    s6 = [
        SlotDef(
            c6[0], c6[1], "short_vs_long_term_focus", "단기 목표 집중 vs 장기목표 지향"
        ),
        SlotDef(c6[0], c6[1], "achievement_desire", "성취 욕구"),
        SlotDef(c6[0], c6[1], "persistence", "꾸준함/지속성"),
        SlotDef(c6[0], c6[1], "realism_vs_idealism", "현실주의 vs 이상주의"),
        SlotDef(c6[0], c6[1], "goal_revision_freq", "목표 수정 빈도"),
        SlotDef(c6[0], c6[1], "resilience", "회복 탄력성(실패 후 반응)"),
        SlotDef(c6[0], c6[1], "help_request_freq", "도움 요청 빈도"),
        SlotDef(c6[0], c6[1], "self_criticism", "자기비판 성향"),
        SlotDef(c6[0], c6[1], "feedback_response", "피드백 반응(수용/방어)"),
        SlotDef(c6[0], c6[1], "self_efficacy_expression", "자기효능감 표현"),
    ]
    # 카테고리7: 선호/취향 패턴(61~70)
    c7 = CAT[7]
    s7 = [
        SlotDef(c7[0], c7[1], "aesthetic_weight", "미적·감성적 요소 비중"),
        SlotDef(c7[0], c7[1], "consistency_of_choice", "선택의 일관성"),
        SlotDef(c7[0], c7[1], "brand_loyalty", "브랜드 충성도"),
        SlotDef(c7[0], c7[1], "experience_based_choice", "경험 기반 선택 선호"),
        SlotDef(c7[0], c7[1], "novelty_seeking", "새로움 탐색 성향"),
        SlotDef(c7[0], c7[1], "comfort_seeking", "편안함·안정성 추구 정도"),
        SlotDef(c7[0], c7[1], "time_efficiency", "시간 효율 중시"),
        SlotDef(c7[0], c7[1], "cost_sensitivity", "비용 민감도"),
        SlotDef(c7[0], c7[1], "hassle_tolerance", "번거로움 허용도"),
        SlotDef(
            c7[0], c7[1], "personalized_message_pref", "개인화 메시지 선호(섬세/간결)"
        ),
    ]
    # 카테고리8: 문제해결/트러블 패턴(71~80)
    c8 = CAT[8]
    s8 = [
        SlotDef(c8[0], c8[1], "problem_definition_skill", "문제 정의 능력"),
        SlotDef(c8[0], c8[1], "root_cause_depth", "원인 탐색 깊이"),
        SlotDef(c8[0], c8[1], "solution_requirement_level", "해결책 요구 수준"),
        SlotDef(c8[0], c8[1], "guidance_preference", "지시/가이드 선호도"),
        SlotDef(c8[0], c8[1], "reaction_to_frustration", "좌절 시 반응"),
        SlotDef(c8[0], c8[1], "error_reproduction_skill", "에러 재현·설명 능력"),
        SlotDef(c8[0], c8[1], "routine_approach", "루틴/체계적 접근 선호"),
        SlotDef(c8[0], c8[1], "problem_solving_speed", "문제 해결 속도"),
        SlotDef(c8[0], c8[1], "logical_structuring", "논리적 구조화"),
        SlotDef(c8[0], c8[1], "priority_judgment_accuracy", "우선순위 판단 정확도"),
    ]
    # 카테고리9: 신뢰·안전(81~90)
    c9 = CAT[9]
    s9 = [
        SlotDef(c9[0], c9[1], "pii_share_tendency", "개인정보 공유 경향"),
        SlotDef(c9[0], c9[1], "system_skepticism", "시스템 의심/검증 강도"),
        SlotDef(c9[0], c9[1], "authority_trust", "권위/전문가 신뢰도"),
        SlotDef(c9[0], c9[1], "reliance_on_past_dialog", "과거 대화 기억 의존성"),
        SlotDef(c9[0], c9[1], "critical_thinking_strength", "비판적 사고 강도"),
        SlotDef(c9[0], c9[1], "anthropomorphism_llm", "LLM 지각(기계 vs 인간화)"),
        SlotDef(c9[0], c9[1], "error_tolerance", "오류 허용도"),
        SlotDef(c9[0], c9[1], "opacity_resistance", "불투명성(why?)에 대한 반발"),
        SlotDef(c9[0], c9[1], "clear_responsibility_demand", "명확한 책임 소재 요구"),
        SlotDef(c9[0], c9[1], "info_trust_criteria", "정보 신뢰 판단 기준"),
    ]
    # 카테고리10: 상호작용 스타일(91~100)
    c10 = CAT[10]
    s10 = [
        SlotDef(c10[0], c10[1], "dialog_tempo", "대화 템포"),
        SlotDef(c10[0], c10[1], "indirect_vs_direct", "돌려 말하기 vs 직설적 말투"),
        SlotDef(c10[0], c10[1], "intent_parsing", "의도 파악 방식(문맥/표면)"),
        SlotDef(c10[0], c10[1], "imperative_vs_request", "명령형 vs 요청형"),
        SlotDef(c10[0], c10[1], "joke_tolerance", "농담·가벼운 톤 수용성"),
        SlotDef(c10[0], c10[1], "repeat_request_pattern", "반복 요청 패턴"),
        SlotDef(c10[0], c10[1], "purposefulness", "대화의 목적성(실용 vs 감정해소)"),
        SlotDef(c10[0], c10[1], "flow_control", "진행 제어 여부(“잠깐”, “이제 이걸”)"),
        SlotDef(c10[0], c10[1], "leadership_preference", "주도권 선호도"),
        SlotDef(c10[0], c10[1], "closing_transition_pattern", "대화 종료/전환 패턴"),
    ]
    return s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10


SLOT_DEFS: List[SlotDef] = _slots_v1()


# -----------------------------
# 규칙 기반 특징 추출기
# -----------------------------
def _extract_features(user_text: str) -> Dict[str, Any]:
    """
    한국어 사용자 입력으로부터 단순/결정적 특징을 추출한다.
    - 키워드 기반 + 형태소 토큰 기반(Jamo/띄어쓰기 취약 대비)
    - 복잡한 휴리스틱 최소화(패턴만 사용)
    """
    txt = (user_text or "").strip()
    toks = set(mecab_tokens(txt))
    low = txt.lower()
    has = lambda s: (s in txt) or (s in low)

    # 신호 그룹
    feat: Dict[str, Any] = {}
    # 정보 탐색
    feat["ask_detail"] = any(
        has(k) for k in ["자세히", "상세", "디테일", "구체", "깊게", "스텝", "단계"]
    )
    feat["ask_summary"] = any(
        has(k) for k in ["간단히", "간단", "짧게", "요약", "핵심만"]
    )
    feat["ask_compare"] = any(has(k) for k in ["비교", "vs", "장단점", "차이", "대안"])
    feat["ask_evidence"] = any(
        has(k) for k in ["근거", "출처", "증거", "레퍼런스", "링크"]
    )
    feat["ask_expert"] = any(
        has(k) for k in ["전문가", "전문적", "공식", "논문", "학술", "peer"]
    )
    feat["ask_step"] = any(has(k) for k in ["step", "스텝", "단계", "순서"])
    feat["ask_example"] = any(has(k) for k in ["예시", "샘플", "사례"])
    feat["hypothesis"] = any(has(k) for k in ["내 생각", "가설", "추정", "내가 보기엔"])
    feat["uncertain"] = any(has(k) for k in ["아마", "일듯", "모르겠", "불확실"])
    feat["clear"] = any(has(k) for k in ["정확히", "명확히", "분명히"])

    # 의사결정
    feat["fast"] = any(has(k) for k in ["빨리", "바로", "즉시", "즉흥"])
    feat["careful"] = any(has(k) for k in ["신중", "천천히", "검토", "재검토"])
    feat["optimize"] = any(has(k) for k in ["최적", "최대화", "최소화", "효율"])
    feat["satisficing"] = any(has(k) for k in ["대충", "적당", "충분", "괜찮"])
    feat["risk"] = any(has(k) for k in ["위험", "리스크", "감수"])
    feat["cost"] = any(has(k) for k in ["비용", "가격", "가성비"])
    feat["emotion"] = any(has(k) for k in ["느낌", "감정", "기분", "감성"])

    # 감정/대인
    feat["humor"] = any(has(k) for k in ["ㅋㅋ", "ㅎㅎ", "하하", "농담", "유머"])
    feat["anger"] = any(has(k) for k in ["화나", "빡", "짜증", "분노"])
    feat["anxiety"] = any(has(k) for k in ["불안", "걱정", "초조"])
    feat["praise"] = any(has(k) for k in ["좋아요", "최고", "굿", "잘했"])
    feat["blame"] = any(has(k) for k in ["잘못", "네 탓", "너 때문", "비난"])
    feat["direct"] = any(has(k) for k in ["직설", "단도직입", "솔직히"])
    feat["empathy_need"] = any(has(k) for k in ["공감", "이해해줘", "위로"])

    # 학습/선호/문제해결
    feat["meta_q"] = any(has(k) for k in ["왜", "어떻게"]) and not has("왜곡")
    feat["latest"] = any(has(k) for k in ["최신", "업데이트", "최근"])
    feat["analogy"] = any(has(k) for k in ["비유", "유추", "닮"])
    feat["routine"] = any(has(k) for k in ["루틴", "체계", "절차"])
    feat["speed"] = any(has(k) for k in ["빨라", "느려", "속도"])
    feat["logical"] = any(has(k) for k in ["논리", "구조", "근거"])
    feat["priority"] = any(has(k) for k in ["우선순위", "먼저", "중요"])

    # 신뢰/안전/상호작용
    feat["pii_share"] = any(has(k) for k in ["전화번호", "주소", "메일", "이메일", "@"])
    feat["skeptic"] = any(has(k) for k in ["검증", "증명", "확인해", "의심"])
    feat["authority"] = any(has(k) for k in ["의사", "변호사", "교수", "전문가"])
    feat["why_tag"] = any(has(k) for k in ["왜", "이유"])
    feat["imperative"] = any(
        has(k) for k in ["해라", "해줘", "해", "하세요"]
    ) and not has("해주세요?")
    feat["request"] = any(
        has(k) for k in ["부탁", "가능할까요", "해주세요", "해주시겠"]
    )
    feat["joke_tolerance"] = any(has(k) for k in ["ㅋㅋ", "ㅎㅎ", "농담", "가볍게"])
    feat["flow_control"] = any(has(k) for k in ["잠깐", "잠시만", "이제", "다음으로"])
    feat["repeat"] = any(has(k) for k in ["다시", "반복", "한 번 더"])
    feat["closing"] = any(has(k) for k in ["그만", "끝", "마무리", "종료"])

    return feat


def _score(
    b: bool, conf_true: float = 0.7, conf_false: float = 0.6
) -> Tuple[bool, float]:
    return (True, conf_true) if b else (False, conf_false)


def classify_slot(
    user_text: str, stwm_snapshot: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    100개 슬롯 전수에 대해 규칙 기반 라벨을 결정한다.
    - 신뢰도가 충분치 않으면 해당 슬롯은 결과에 포함하지 않는다.
    - 각 슬롯의 value는 행동기반.md의 표기(한글/쌍대 구분)를 최대한 유지한다.
    """
    feats = _extract_features(user_text or "")
    out: List[Dict[str, Any]] = []
    ev = _evidence_snippet(user_text or "")

    def add(sd: SlotDef, value: Optional[str], conf: float) -> None:
        if value is None:
            return
        # 최소 신뢰도 0.55 미만은 버림
        if conf < 0.55:
            return
        out.append(
            {
                "slot_key": sd.slot_path,
                "norm_key": sd.norm_key,
                "value": value,
                "confidence": float(conf),
                "evidence": ev,
            }
        )

    # 카테고리 1) 정보 탐색 패턴
    # 1. 질문 명확도(명확 / 모호)
    for sd in SLOT_DEFS:
        ck = sd.category_key
        sk = sd.slot_key
        if ck == "info_seeking" and sk == "question_clarity":
            if feats["clear"] and not feats["uncertain"]:
                add(sd, "명확", 0.75)
            elif feats["uncertain"] and not feats["clear"]:
                add(sd, "모호", 0.7)
            else:
                pass
        elif ck == "info_seeking" and sk == "info_demand":
            if feats["ask_detail"] or feats["ask_evidence"] or feats["ask_expert"]:
                add(sd, "많이 요구", 0.72)
            elif feats["ask_summary"]:
                add(sd, "최소 요구", 0.7)
        elif ck == "info_seeking" and sk == "uncertainty_tolerance":
            if feats["uncertain"]:
                add(sd, "허용", 0.65)
        elif ck == "info_seeking" and sk == "evidence_strength":
            if feats["ask_evidence"]:
                add(sd, "강함", 0.75)
        elif ck == "info_seeking" and sk == "expertise_expectation":
            if feats["ask_expert"]:
                add(sd, "높음", 0.75)
        elif ck == "info_seeking" and sk == "comparison_request_freq":
            if feats["ask_compare"]:
                add(sd, "높음", 0.72)
        elif ck == "info_seeking" and sk == "step_by_step_preference":
            if feats["ask_step"]:
                add(sd, "있음", 0.72)
        elif ck == "info_seeking" and sk == "summary_vs_detail":
            if feats["ask_summary"] and not feats["ask_detail"]:
                add(sd, "요약 선호", 0.72)
            elif feats["ask_detail"] and not feats["ask_summary"]:
                add(sd, "상세 선호", 0.72)
        elif ck == "info_seeking" and sk == "example_preference":
            if feats["ask_example"]:
                add(sd, "예시 선호", 0.7)
        elif ck == "info_seeking" and sk == "hypothesis_vs_acceptance":
            if feats["hypothesis"]:
                add(sd, "가설 제시", 0.7)

        # 카테고리 2) 의사결정
        if ck == "decision_making" and sk == "impulsive_vs_careful":
            if feats["fast"] and not feats["careful"]:
                add(sd, "즉흥적", 0.72)
            elif feats["careful"]:
                add(sd, "신중함", 0.7)
        if ck == "decision_making" and sk == "alternatives_depth":
            if feats["ask_compare"]:
                add(sd, "깊음", 0.7)
        if ck == "decision_making" and sk == "optimization_desire":
            if feats["optimize"] and not feats["satisficing"]:
                add(sd, "최적해 지향", 0.72)
            elif feats["satisficing"] and not feats["optimize"]:
                add(sd, "충분해 지향", 0.7)
        if ck == "decision_making" and sk == "risk_taking":
            if feats["risk"]:
                add(sd, "감수", 0.65)
        if ck == "decision_making" and sk == "cost_efficiency_vs_emotion":
            if feats["cost"] and not feats["emotion"]:
                add(sd, "비용·효율 중심", 0.72)
            elif feats["emotion"] and not feats["cost"]:
                add(sd, "감정 중심", 0.7)
        if ck == "decision_making" and sk == "fast_conclusion":
            if feats["fast"]:
                add(sd, "선호", 0.68)
        if ck == "decision_making" and sk == "regret_reconsideration":
            if feats["careful"] and "재검토" in (user_text or ""):
                add(sd, "있음", 0.62)
        if ck == "decision_making" and sk == "recommendation_acceptance":
            if "추천" in (user_text or "") and "거절" not in (user_text or ""):
                add(sd, "수용", 0.62)
        if ck == "decision_making" and sk == "opinion_request_freq":
            if "의견" in (user_text or "") or "어떻게 생각" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "decision_making" and sk == "criteria_explicitness":
            if "기준" in (user_text or "") or "조건" in (user_text or ""):
                add(sd, "명시", 0.62)

        # 카테고리 3) 감정
        if ck == "affect" and sk == "affect_variability":
            if feats["anger"] or feats["anxiety"]:
                add(sd, "높음", 0.6)
        if ck == "affect" and sk == "anger_expression":
            if feats["anger"]:
                add(sd, "표출", 0.65)
        if ck == "affect" and sk == "anxiety_expression":
            if feats["anxiety"]:
                add(sd, "표현", 0.62)
        if ck == "affect" and sk == "humor_usage":
            if feats["humor"]:
                add(sd, "있음", 0.7)
        if ck == "affect" and sk == "emotion_word_freq":
            if feats["emotion"]:
                add(sd, "높음", 0.6)
        if ck == "affect" and sk == "blame_tendency":
            if feats["blame"]:
                add(sd, "있음", 0.62)
        if ck == "affect" and sk == "stress_avoid_vs_problem_solve":
            if "회피" in (user_text or ""):
                add(sd, "회피", 0.6)
            elif "해결" in (user_text or ""):
                add(sd, "문제 해결", 0.6)
        if ck == "affect" and sk == "praise_responsivity":
            if feats["praise"]:
                add(sd, "높음", 0.62)
        if ck == "affect" and sk == "sensitive_triggers":
            if "지연" in (user_text or "") or "불확실" in (user_text or ""):
                add(sd, "있음", 0.6)
        if ck == "affect" and sk == "disappointment_expression":
            if "실망" in (user_text or ""):
                add(sd, "있음", 0.62)

        # 카테고리 4) 관계/대인
        if ck == "interpersonal" and sk == "dominance_vs_compliance":
            if "내가 하라" in (user_text or "") or feats["imperative"]:
                add(sd, "지배", 0.62)
            elif "알겠어요" in (user_text or "") or "따를게" in (user_text or ""):
                add(sd, "순응", 0.6)
        if ck == "interpersonal" and sk == "rebuttal_style":
            if "반박" in (user_text or "") or "하지만" in (user_text or ""):
                add(sd, "강함", 0.6)
        if ck == "interpersonal" and sk == "cooperativeness":
            if "함께" in (user_text or "") or "협력" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "interpersonal" and sk == "direct_aggressive_speech":
            if feats["direct"]:
                add(sd, "직설", 0.65)
        if ck == "interpersonal" and sk == "empathy_need":
            if feats["empathy_need"]:
                add(sd, "높음", 0.65)
        if ck == "interpersonal" and sk == "responsibility_avoidance":
            if "내 탓 아님" in (user_text or "") or "책임지기 싫" in (user_text or ""):
                add(sd, "있음", 0.6)
        if ck == "interpersonal" and sk == "criticism_acceptance":
            if "지적 감사" in (user_text or "") or "피드백 고맙" in (user_text or ""):
                add(sd, "수용", 0.6)
        if ck == "interpersonal" and sk == "apology_pattern":
            if "미안" in (user_text or "") or "죄송" in (user_text or ""):
                add(sd, "있음", 0.62)
        if ck == "interpersonal" and sk == "request_expression":
            if "부탁" in (user_text or "") or feats["request"]:
                add(sd, "있음", 0.62)
        if ck == "interpersonal" and sk == "guardedness_level":
            if "개인정보" in (user_text or "") or "조심" in (user_text or ""):
                add(sd, "높음", 0.62)

        # 카테고리 5) 학습
        if ck == "learning" and sk == "meta_question_freq":
            if feats["meta_q"]:
                add(sd, "높음", 0.65)
        if ck == "learning" and sk == "response_when_not_understood":
            if "다시" in (user_text or ""):
                add(sd, "다시 질문", 0.6)
        if ck == "learning" and sk == "generalization_vs_cases":
            if "사례" in (user_text or "") or feats["analogy"]:
                add(sd, "사례 중심", 0.62)
        if ck == "learning" and sk == "preference_latest_info":
            if feats["latest"]:
                add(sd, "선호", 0.65)
        if ck == "learning" and sk == "self_censorship":
            if "모르겠" in (user_text or ""):
                add(sd, "있음", 0.62)
        if ck == "learning" and sk == "repeat_learning_need":
            if "반복" in (user_text or "") or "복습" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "learning" and sk == "learning_speed_feedback":
            if feats["speed"]:
                add(sd, "있음", 0.6)
        if ck == "learning" and sk == "analogy_preference":
            if feats["analogy"]:
                add(sd, "선호", 0.62)
        if ck == "learning" and sk == "deep_discussion_preference":
            if "깊이" in (user_text or "") or "토론" in (user_text or ""):
                add(sd, "선호", 0.62)
        if ck == "learning" and sk == "learning_fatigue_expression":
            if "피곤" in (user_text or "") or "지침" in (user_text or ""):
                add(sd, "있음", 0.62)

        # 카테고리 6) 목표·동기
        if ck == "motivation" and sk == "short_vs_long_term_focus":
            if "이번 주" in (user_text or "") or "오늘" in (user_text or ""):
                add(sd, "단기", 0.6)
            if "내년" in (user_text or "") or "장기" in (user_text or ""):
                add(sd, "장기", 0.6)
        if ck == "motivation" and sk == "achievement_desire":
            if "달성" in (user_text or "") or "성취" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "motivation" and sk == "persistence":
            if "꾸준" in (user_text or "") or "매일" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "motivation" and sk == "realism_vs_idealism":
            if "현실적" in (user_text or ""):
                add(sd, "현실주의", 0.6)
            elif "이상적" in (user_text or "") or "이상주의" in (user_text or ""):
                add(sd, "이상주의", 0.6)
        if ck == "motivation" and sk == "goal_revision_freq":
            if "수정" in (user_text or "") or "바꾸" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "motivation" and sk == "resilience":
            if "다시 시도" in (user_text or "") or "회복" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "motivation" and sk == "help_request_freq":
            if "도와줘" in (user_text or "") or "부탁" in (user_text or ""):
                add(sd, "높음", 0.62)
        if ck == "motivation" and sk == "self_criticism":
            if "내 탓" in (user_text or "") or "자책" in (user_text or ""):
                add(sd, "있음", 0.6)
        if ck == "motivation" and sk == "feedback_response":
            if "피드백" in (user_text or "") and "감사" in (user_text or ""):
                add(sd, "수용", 0.6)
        if ck == "motivation" and sk == "self_efficacy_expression":
            if "할 수 있" in (user_text or "") or "자신" in (user_text or ""):
                add(sd, "높음", 0.6)

        # 카테고리 7) 선호/취향
        if ck == "preferences" and sk == "aesthetic_weight":
            if (
                "감성" in (user_text or "")
                or "예쁜" in (user_text or "")
                or "미적" in (user_text or "")
            ):
                add(sd, "높음", 0.6)
        if ck == "preferences" and sk == "consistency_of_choice":
            if "항상" in (user_text or "") or "늘" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "preferences" and sk == "brand_loyalty":
            if (
                "애플" in (user_text or "")
                or "삼성" in (user_text or "")
                or "브랜드" in (user_text or "")
            ):
                add(sd, "있음", 0.6)
        if ck == "preferences" and sk == "experience_based_choice":
            if "경험" in (user_text or "") or "써봤" in (user_text or ""):
                add(sd, "선호", 0.6)
        if ck == "preferences" and sk == "novelty_seeking":
            if "새로운" in (user_text or "") or "신상" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "preferences" and sk == "comfort_seeking":
            if "편안" in (user_text or "") or "안정" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "preferences" and sk == "time_efficiency":
            if "시간" in (user_text or "") and "아끼" in (user_text or ""):
                add(sd, "중시", 0.6)
        if ck == "preferences" and sk == "cost_sensitivity":
            if feats["cost"]:
                add(sd, "높음", 0.62)
        if ck == "preferences" and sk == "hassle_tolerance":
            if "번거롭" in (user_text or "") or "귀찮" in (user_text or ""):
                add(sd, "낮음", 0.6)
        if ck == "preferences" and sk == "personalized_message_pref":
            if "섬세" in (user_text or "") or "간결" in (user_text or ""):
                v = "섬세" if "섬세" in (user_text or "") else "간결"
                add(sd, v, 0.6)

        # 카테고리 8) 문제해결
        if ck == "problem_solving" and sk == "problem_definition_skill":
            if "문제는" in (user_text or "") or "정의" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "problem_solving" and sk == "root_cause_depth":
            if "원인" in (user_text or "") or "왜" in (user_text or ""):
                add(sd, "깊음", 0.6)
        if ck == "problem_solving" and sk == "solution_requirement_level":
            if "해결책" in (user_text or "") or "솔루션" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "problem_solving" and sk == "guidance_preference":
            if "가이드" in (user_text or "") or "지시" in (user_text or ""):
                add(sd, "선호", 0.6)
        if ck == "problem_solving" and sk == "reaction_to_frustration":
            if "포기" in (user_text or "") or "짜증" in (user_text or ""):
                add(sd, "좌절 표출", 0.6)
        if ck == "problem_solving" and sk == "error_reproduction_skill":
            if "재현" in (user_text or "") or "다시 해보" in (user_text or ""):
                add(sd, "있음", 0.6)
        if ck == "problem_solving" and sk == "routine_approach":
            if feats["routine"]:
                add(sd, "선호", 0.6)
        if ck == "problem_solving" and sk == "problem_solving_speed":
            if feats["speed"]:
                add(sd, "빠름", 0.6)
        if ck == "problem_solving" and sk == "logical_structuring":
            if feats["logical"]:
                add(sd, "높음", 0.62)
        if ck == "problem_solving" and sk == "priority_judgment_accuracy":
            if feats["priority"]:
                add(sd, "높음", 0.6)

        # 카테고리 9) 신뢰·안전
        if ck == "trust_safety" and sk == "pii_share_tendency":
            if feats["pii_share"]:
                add(sd, "공유 경향", 0.62)
        if ck == "trust_safety" and sk == "system_skepticism":
            if feats["skeptic"]:
                add(sd, "강함", 0.62)
        if ck == "trust_safety" and sk == "authority_trust":
            if feats["authority"]:
                add(sd, "높음", 0.62)
        if ck == "trust_safety" and sk == "reliance_on_past_dialog":
            if "전에" in (user_text or "") or "지난번" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "trust_safety" and sk == "critical_thinking_strength":
            if feats["skeptic"] and feats["ask_evidence"]:
                add(sd, "강함", 0.65)
        if ck == "trust_safety" and sk == "anthropomorphism_llm":
            if "너" in (user_text or "") and "느낌" in (user_text or ""):
                add(sd, "인간화 경향", 0.6)
        if ck == "trust_safety" and sk == "error_tolerance":
            if "괜찮" in (user_text or "") or "실수" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "trust_safety" and sk == "opacity_resistance":
            if "왜" in (user_text or "") and "투명" not in (user_text or ""):
                add(sd, "반발", 0.6)
        if ck == "trust_safety" and sk == "clear_responsibility_demand":
            if "책임" in (user_text or ""):
                add(sd, "요구", 0.6)
        if ck == "trust_safety" and sk == "info_trust_criteria":
            if "공식" in (user_text or "") or "출처" in (user_text or ""):
                add(sd, "근거 위주", 0.6)

        # 카테고리 10) 상호작용 스타일
        if ck == "interaction_style" and sk == "dialog_tempo":
            if "천천히" in (user_text or ""):
                add(sd, "느림", 0.6)
            elif "빨리" in (user_text or ""):
                add(sd, "빠름", 0.6)
        if ck == "interaction_style" and sk == "indirect_vs_direct":
            if feats["direct"]:
                add(sd, "직설", 0.65)
            elif "돌려" in (user_text or ""):
                add(sd, "돌려 말하기", 0.6)
        if ck == "interaction_style" and sk == "intent_parsing":
            if "문맥" in (user_text or ""):
                add(sd, "문맥", 0.6)
            elif "표면" in (user_text or ""):
                add(sd, "표면", 0.6)
        if ck == "interaction_style" and sk == "imperative_vs_request":
            if feats["imperative"] and not feats["request"]:
                add(sd, "명령형", 0.65)
            elif feats["request"] and not feats["imperative"]:
                add(sd, "요청형", 0.65)
        if ck == "interaction_style" and sk == "joke_tolerance":
            if feats["joke_tolerance"]:
                add(sd, "수용", 0.65)
        if ck == "interaction_style" and sk == "repeat_request_pattern":
            if feats["repeat"]:
                add(sd, "있음", 0.6)
        if ck == "interaction_style" and sk == "purposefulness":
            if "도와줘" in (user_text or "") or "정리" in (user_text or ""):
                add(sd, "실용", 0.6)
            elif "속상" in (user_text or "") or "위로" in (user_text or ""):
                add(sd, "감정해소", 0.6)
        if ck == "interaction_style" and sk == "flow_control":
            if feats["flow_control"]:
                add(sd, "있음", 0.62)
        if ck == "interaction_style" and sk == "leadership_preference":
            if "내가 할게" in (user_text or "") or "내가 주도" in (user_text or ""):
                add(sd, "높음", 0.6)
        if ck == "interaction_style" and sk == "closing_transition_pattern":
            if feats["closing"]:
                add(sd, "있음", 0.62)

    return out


__all__ = [
    "SLOT_DEFS",
    "classify_slot",
]
