from __future__ import annotations

"""
세렌디피티 재랭커

설명:
- 관련성 × (1 - 친숙도) × 신규성 보너스로 후보를 재정렬한다.
- 입력 후보는 {title, body, relevance, source_domain} 키를 가정하며,
  결측 시 보수 기본값을 사용한다.
"""

from typing import Dict, List


def rerank_with_serendipity(candidates: List[Dict], user_profile: Dict) -> List[Dict]:
    """
    후보 리스트를 세렌디피티 스코어로 재정렬한다.
    - known_entities: 사용자가 익숙한 엔터티(제목/본문 포함 시 친숙도=1)
    - recent_domains: 최근 소비 도메인(동일 도메인에는 신규성 보너스 0)
    - recent_tags: 최근 소비 토픽 태그(겹치지 않으면 신규성 보너스 추가)
    """
    known = set(user_profile.get("known_entities", []) or [])
    recent_domains = set(user_profile.get("recent_domains", []) or [])
    recent_tags = set(user_profile.get("recent_tags", []) or [])
    out: List[Dict] = []
    for c in candidates:
        rel = float(c.get("relevance", 0.6))
        text = (c.get("title", "") or "") + "\n" + (c.get("body", "") or "")
        fam = 1.0 if any((k and k in text) for k in known) else 0.0
        dom = (c.get("source_domain") or "").lower()
        dom_bonus = 0.1 if (dom and dom not in recent_domains) else 0.0
        tags = set(c.get("tags", []) or [])
        tag_bonus = 0.1 if (tags and tags.isdisjoint(recent_tags)) else 0.0
        # 세렌디피티: 친숙도 억제 × 신규성 보너스(도메인/태그)
        s = rel * (1 - fam) * (1 + dom_bonus + tag_bonus)
        c = {**c, "serendipity_score": s}
        out.append(c)
    out.sort(key=lambda x: float(x.get("serendipity_score", 0.0)), reverse=True)
    return out


__all__ = ["rerank_with_serendipity"]
