from __future__ import annotations

"""
사용자 취향 → 키워드 그룹 변환

역할:
- 대화/신호에서 추출된 양/음성 시그널을 바탕으로 required/normal/denied 리스트를 생성한다.
- 최소 구현: 간단 키워드 추출/승격 규칙만 제공(형태소 분석기 미사용).
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9가-힣#@._-]{2,}")


def _extract_keywords(texts: Iterable[str]) -> List[str]:
    kws: List[str] = []
    for t in texts or []:
        for m in _WORD_RE.findall(str(t) or ""):
            s = m.strip().lower()
            if s:
                kws.append(s)
    return kws


def build_interest_from_signals(
    positive_utterances: Iterable[str] | None = None,
    negative_utterances: Iterable[str] | None = None,
    entities_preferred: Iterable[str] | None = None,
    domains_blocked: Iterable[str] | None = None,
    required_promotion_threshold: int = 3,
) -> Dict[str, List[str]]:
    """
    간단한 규칙:
    - positive에서 추출된 키워드 빈도 누적 → 빈도 높으면 required 승격, 나머지는 normal
    - entities_preferred는 무조건 required로 병합
    - negative/blocked는 denied에 합산
    """
    pos_words = _extract_keywords(list(positive_utterances or []))
    neg_words = _extract_keywords(list(negative_utterances or []))
    ent_words = [
        str(x).strip().lower() for x in (entities_preferred or []) if str(x).strip()
    ]
    block_domains = [
        str(x).strip().lower() for x in (domains_blocked or []) if str(x).strip()
    ]

    cnt = Counter(pos_words)
    required: List[str] = []
    normal: List[str] = []
    for w, c in cnt.most_common():
        if c >= int(required_promotion_threshold):
            required.append(w)
        else:
            normal.append(w)

    # 엔티티는 강제 required
    for w in ent_words:
        if w and w not in required:
            required.append(w)

    # denied는 네거티브 시그널 + 차단 도메인
    denied = list(dict.fromkeys(neg_words + block_domains))

    # 중복 제거 및 정리
    def _uniq(xs: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "required": _uniq(required),
        "normal": _uniq(normal),
        "denied": _uniq(denied),
    }


__all__ = ["build_interest_from_signals"]
