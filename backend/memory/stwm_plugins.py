import logging
import os
from typing import Any, Dict, List, Optional, Tuple

_MECAB = None
_NER_PIPE = None
_QA_PIPE = None
_LOGGER = logging.getLogger("stwm.plugins")


def _ensure_mecab():
    global _MECAB
    if _MECAB is None:
        try:
            import MeCab  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"MeCab import failed: {repr(e)}")
        args = os.getenv("MECAB_ARGS", "")
        _MECAB = MeCab.Tagger(args)
        _LOGGER.info("[stwm.plugins] MeCab initialized args='%s'", args)
        try:
            import platform

            _LOGGER.info(
                "[stwm.plugins] MeCab runtime platform=%s", platform.platform()
            )
        except Exception:
            pass
    return _MECAB


def _ensure_ner():
    global _NER_PIPE
    if _NER_PIPE is None:
        try:
            from transformers import (  # type: ignore
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Transformers import failed: {repr(e)}")

        # 1순위: 리모트 기본 모델
        primary = os.getenv("KONER_MODEL", "monologg/koelectra-base-v3-naver-ner")
        # 2순위: 로컬 경로 폴백
        local_path = os.getenv(
            "KONER_MODEL_PATH", "C:/My_Business/models/kor_electra_base_origin"
        )
        name_used = None
        try:
            tok = AutoTokenizer.from_pretrained(primary)
            mdl = AutoModelForTokenClassification.from_pretrained(primary)
            name_used = primary
        except Exception:
            tok = AutoTokenizer.from_pretrained(local_path)
            mdl = AutoModelForTokenClassification.from_pretrained(local_path)
            name_used = local_path

        _NER_PIPE = pipeline(
            "ner",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
            tokenizer_kwargs={"max_length": 512, "truncation": True},
        )
        _LOGGER.info("[stwm.plugins] NER pipeline ready model='%s'", str(name_used))
        try:
            _LOGGER.info(
                "[stwm.plugins] NER primary='%s' fallback_local='%s'",
                str(primary),
                str(local_path),
            )
        except Exception:
            pass
    return _NER_PIPE


def _ensure_qa():
    global _QA_PIPE
    if _QA_PIPE is None:
        try:
            from transformers import (  # type: ignore
                AutoModelForQuestionAnswering,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Transformers import failed: {repr(e)}")
        name = os.getenv(
            "KO_QA_MODEL",
            os.getenv(
                "KO_QA_MODEL_PATH", "monologg/koelectra-base-v3-finetuned-korquad"
            ),
        )
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForQuestionAnswering.from_pretrained(name)
        _QA_PIPE = pipeline(
            "question-answering",
            model=mdl,
            tokenizer=tok,
            tokenizer_kwargs={"max_length": 512, "truncation": True},
        )
        _LOGGER.info("[stwm.plugins] QA pipeline ready model='%s'", str(name))
        try:
            base_local = os.getenv("KO_QA_MODEL_PATH")
            _LOGGER.info("[stwm.plugins] QA model_path_env='%s'", str(base_local))
        except Exception:
            pass
    return _QA_PIPE


def extract_mecab_intent(text: str) -> Dict[str, Optional[str]]:
    """
    형태소 기반 행위/대상/목적 후보 추출(간결 규칙).
    - 마지막 VV(동사) → last_act
    - JKO(목적격 조사) 직전 명사 → last_target
    - 보조/연결 어미 단서(하려고/위해/싶/원하/해줘 등) → last_goal
    """
    tagger = _ensure_mecab()
    node = tagger.parseToNode(text or "")
    verbs: List[str] = []
    last_target: Optional[str] = None
    prev_surface: Optional[str] = None
    while node:
        surf = node.surface
        if surf:
            feats = (node.feature or "").split(",")
            pos = feats[0] if feats else ""
            if pos == "VV":
                verbs.append(surf)
            if pos == "JKO" and prev_surface:
                last_target = prev_surface
            prev_surface = surf
        node = node.next
    last_act = verbs[-1] if verbs else None

    # 목적 단서(간결): 필요 시 도메인 확장
    goal = None
    t = text or ""
    for key in (
        "하려고",
        "위해",
        "싶",
        "원하",
        "해줘",
        "해 주",
        "부탁",
        "찾아",
        "알려",
    ):
        if key in t:
            goal = key
            break

    out = {"last_act": last_act, "last_target": last_target, "last_goal": goal}
    try:
        _LOGGER.debug(
            "[stwm.plugins] mecab_intent act=%s target=%s goal=%s",
            last_act,
            last_target,
            goal,
        )
    except Exception:
        pass
    return out


def extract_koner_spans(text: str) -> Dict[str, Optional[str]]:
    pipe = _ensure_ner()
    out = pipe(text or "")
    loc = None
    per = None
    org = None
    for e in out:
        lab = (e.get("entity_group") or "").upper()
        w = (e.get("word") or "").strip()
        if not w:
            continue
        if lab.endswith("LOC") and loc is None:
            loc = w
        elif lab.endswith("PER") and per is None:
            per = w
        elif lab.endswith("ORG") and org is None:
            org = w
    out = {"last_loc": loc, "last_person": per, "last_org": org}
    try:
        _LOGGER.debug("[stwm.plugins] ko-ner loc=%s person=%s org=%s", loc, per, org)
    except Exception:
        pass
    return out


def extract_goal_with_qa(text: str) -> Tuple[Optional[str], float]:
    qa = _ensure_qa()
    q = "사용자가 하려는 목적은 무엇인가?"
    r = qa(question=q, context=text or "")
    ans = (r.get("answer") or "").strip()
    score = float(r.get("score") or 0.0)
    try:
        _LOGGER.debug("[stwm.plugins] qa_goal score=%.3f ans='%s'", score, ans)
    except Exception:
        pass
    return (ans if ans else None, score)


def mecab_tokens(text: str) -> List[str]:
    """간단 토큰화: 명사/고유명사/동사/형용사 표면형만 수집."""
    tagger = _ensure_mecab()
    node = tagger.parseToNode(text or "")
    toks: List[str] = []
    while node:
        surf = node.surface
        if surf:
            feats = (node.feature or "").split(",")
            pos = feats[0] if feats else ""
            if pos in ("NNG", "NNP", "VV", "VA"):
                toks.append(surf)
        node = node.next
    return toks
