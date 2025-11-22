import re
from typing import Dict, Tuple

# 문장 경계 분리 정규식: 영문 문장부호 + 한국어 종결 조사 패턴 일부
_SENT_SPLIT = re.compile(r"(?<=[.!?…]|[다요]\)|[다요]\])\s+")


def mask_protected(text: str) -> Tuple[str, Dict[str, str]]:
    """
    보호 대상 토큰을 마스킹한다.

    - 코드블록, [출처:…] 메타, URL, 역할 토큰(system/assistant/user:)
    반환: (마스킹된 텍스트, 마스크 매핑)
    """
    pats = [
        r"```.*?```",
        r"\[출처:.*?\]",
        r"https?://\S+",
        r"(?i)(system|assistant|user):",
    ]
    repl: str = text
    mp: Dict[str, str] = {}
    for i, p in enumerate(pats, 1):
        for m in re.finditer(p, repl, flags=re.S):
            tok = f"<<MASK_{i}_{m.start()}>>"
            mp[tok] = m.group(0)
            repl = repl.replace(m.group(0), tok)
    return repl, mp


def unmask(text: str, mp: Dict[str, str]) -> str:
    """마스크를 원문으로 복원한다."""
    for k, v in mp.items():
        text = text.replace(k, v)
    return text


def trim_to_tokens_sentence(text: str, enc, cap: int) -> str:
    """
    토큰 기준 컷 후, 마지막 문장 경계까지 후퇴하여 자연스러운 절단을 보장한다.
    초과 시 말줄임표를 덧붙인다.
    """
    toks = enc.encode(text or "")
    if len(toks) <= max(0, int(cap)):
        return text
    cut = enc.decode(toks[: max(0, int(cap))])
    sents = _SENT_SPLIT.split(cut)
    out = " ".join(sents[:-1]) if len(sents) > 1 else cut
    return (out.strip() + " …").strip()


def must_keep_entities(txt: str) -> set[str]:
    """
    보존해야 할 엔티티(숫자, 날짜, URL)를 추출한다.
    너무 긴 매치는 64자로 잘라 비교 안정성을 높인다.
    """
    pats = [r"\d[\d,./:-]*", r"\b20\d{2}-\d{2}-\d{2}\b", r"https?://\S+"]
    ents: set[str] = set()
    for p in pats:
        for m in re.finditer(p, txt or ""):
            ents.add((m.group(0) or "")[:64])
    return ents


def verify_preservation(src: str, dst: str) -> bool:
    """압축 전후 엔티티 보존 여부 확인."""
    need = must_keep_entities(src)
    have = must_keep_entities(dst)
    return need.issubset(have)
