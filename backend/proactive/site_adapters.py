from __future__ import annotations

"""
사이트 어댑터 레지스트리

역할:
- 일반 추출(trafilatura/readability)로는 누락되기 쉬운 구조화 콘텐츠(JSON-LD 등)를 보완한다.
- 입력: (url, html) → 출력: 텍스트(없으면 빈 문자열)

최소 구현:
- JSON-LD(Article/NewsArticle/BlogPosting)의 articleBody/description을 우선 추출
"""

import json
import re
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

Extractor = Callable[[str, str], str]


def _json_ld_article_extractor(_url: str, html: str) -> str:
    """
    JSON-LD 스니펫에서 Article류의 본문을 추출한다.
    - 여러 블록이 있을 수 있으므로 첫 번째 유효 블록을 우선 사용.
    """
    try:
        blocks = re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>([\s\S]*?)</script>',
            html or "",
            flags=re.I,
        )
        for raw in blocks:
            raw = raw.strip()
            if not raw:
                continue
            # JSON 파싱(리스트/객체 모두 수용)
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
            for it in candidates:
                t = it.get("@type") or ""
                if isinstance(t, list):
                    types = [str(x).lower() for x in t]
                else:
                    types = [str(t).lower()]
                if any(x in types for x in ["article", "newsarticle", "blogposting"]):
                    body = (
                        it.get("articleBody") or it.get("description") or ""
                    ).strip()
                    if body:
                        return body
    except Exception:
        return ""
    return ""


def _opengraph_extractor(_url: str, html: str) -> str:
    """
    OpenGraph/메타 태그 기반 간단 본문(설명) 추출.
    - og:description, twitter:description, meta[name=description]
    """
    try:
        # og:description
        m = re.search(
            r'<meta\s+property=["\']og:description["\']\s+content=["\']([\s\S]*?)["\']',
            html or "",
            flags=re.I,
        )
        if m and m.group(1).strip():
            return m.group(1).strip()
        # twitter:description
        m = re.search(
            r'<meta\s+name=["\']twitter:description["\']\s+content=["\']([\s\S]*?)["\']',
            html or "",
            flags=re.I,
        )
        if m and m.group(1).strip():
            return m.group(1).strip()
        # meta description
        m = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\']([\s\S]*?)["\']',
            html or "",
            flags=re.I,
        )
        if m and m.group(1).strip():
            return m.group(1).strip()
    except Exception:
        return ""
    return ""


def _microdata_article_extractor(_url: str, html: str) -> str:
    """
    Microdata 항목에서 itemprop=articleBody/description 추출.
    - 정확도는 도메인에 따라 다르므로, JSON-LD 실패 시 보조로만 사용.
    """
    try:
        # itemprop="articleBody"
        m = re.search(
            r'itemprop=["\']articleBody["\'][^>]*>([\s\S]*?)</',
            html or "",
            flags=re.I,
        )
        if m and m.group(1).strip():
            txt = re.sub(r"<[^>]+>", " ", m.group(1))
            return re.sub(r"\s+", " ", txt).strip()
        # itemprop="description"
        m = re.search(
            r'itemprop=["\']description["\'][^>]*>([\s\S]*?)</',
            html or "",
            flags=re.I,
        )
        if m and m.group(1).strip():
            txt = re.sub(r"<[^>]+>", " ", m.group(1))
            return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""
    return ""


_REGISTRY: Dict[str, Extractor] = {
    # 기본 JSON-LD 어댑터는 모든 도메인에 시도할 가치가 있어 'default'로 제공
    "default_jsonld": _json_ld_article_extractor,
    "opengraph": _opengraph_extractor,
    "microdata": _microdata_article_extractor,
}


def extract_from_html(url: str, html: str) -> Optional[str]:
    """
    레지스트리 기반 추출을 순차 적용한다.
    - 현재는 JSON-LD 어댑터만 기본 적용한다.
    - 실패/미해당 시 None.
    """
    try:
        # 1) JSON-LD(도메인 무관)
        body = _REGISTRY["default_jsonld"](url, html)
        if body:
            return body
        # 2) OpenGraph/메타
        body = _REGISTRY["opengraph"](url, html)
        if body:
            return body
        # 3) Microdata
        body = _REGISTRY["microdata"](url, html)
        if body:
            return body
    except Exception:
        return None
    return None


__all__ = ["extract_from_html"]
