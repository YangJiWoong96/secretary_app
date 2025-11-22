import re
from typing import Dict, List

from .formatter_guard import ensure_block_shape


def _strip_bold(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    return html_text.replace("<b>", "").replace("</b>", "").strip()


def format_items_to_blocks(
    items: List[Dict], kind: str, *, llm_desc: bool = False
) -> str:
    """네이버 응답 items를 3줄 블록들로 변환.
    - 각 블록: 이름, 간단한 설명, 주소 또는 링크
    - 설명은 길면 140자 내로 절단
    """
    if not items:
        return ""
    blocks = []

    def _sanitize_url(url: str) -> str:
        """URL 후보를 간단 정제한다.
        - 양끝의 괄호/따옴표 제거
        - 끝의 구두점(.,;:) 제거
        - 공백 제거
        """
        try:
            u = str(url or "").strip()
            # 괄호/따옴표 래핑 제거
            if (
                (u.startswith("(") and u.endswith(")"))
                or (u.startswith("[") and u.endswith("]"))
                or (
                    (u.startswith('"') and u.endswith('"'))
                    or (u.startswith("'") and u.endswith("'"))
                )
            ):
                u = u[1:-1].strip()
            # 끝 구두점 제거
            while u and u[-1] in ")].,;:":
                u = u[:-1]
            return u.strip()
        except Exception:
            return str(url or "").strip()

    def _fallback_search_url(query_text: str) -> str:
        """타이틀 기반 범용 검색 URL 폴백(항상 https 포함).
        - 벤더 종속 최소화를 위해 Google 검색을 기본값으로 사용
        - 네트워크 의존 없이 문자열만 합성
        """
        try:
            from urllib.parse import quote as _quote

            q = _quote((query_text or "").strip())
        except Exception:
            q = (query_text or "").strip()
        return f"https://www.google.com/search?q={q}"

    def _naver_map_search_url(query_text: str) -> str:
        """네이버 지도 검색 URL을 합성한다.
        - 네트워크 호출 없이 문자열만 합성
        - 로컬 장소 폴백 시에는 주소를 포함하지 않고 '장소명'만 사용한다
        """
        try:
            from urllib.parse import quote as _quote

            q = _quote((query_text or "").strip())
        except Exception:
            q = (query_text or "").strip()
        return f"https://map.naver.com/v5/search/{q}"

    def _extract_place_id(item: Dict) -> str:
        """네이버/카카오 place id를 추출한다."""
        try:
            for url in (item.get("link"), item.get("originallink"), item.get("url")):
                if not url:
                    continue
                m = re.search(r"place\.map\.(kakao|naver)\.com/(\d+)", url)
                if m:
                    return m.group(2)
                m = re.search(r"/entry/place/(\d+)", url)
                if m:
                    return m.group(1)
            for key in ("id", "placeId", "place_id"):
                if key in item and str(item.get(key)).strip():
                    return str(item.get(key)).strip()
        except Exception:
            pass
        return ""

    for it in items[:5]:
        title = _strip_bold(it.get("title", "")) or "(이름 없음)"
        # LLM이 설명을 한번에 쓰도록
        desc = "-" if llm_desc else ""
        # 뉴스/웹 문서에서 발행일(또는 컨텐츠상의 날짜)을 추출하여 설명 앞에 표시
        date_prefix = ""

        def _extract_date_ymd(s: str) -> str:
            if not s:
                return ""
            s = str(s)
            # YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD
            m = re.search(
                r"(20\d{2})[./-](0?[1-9]|1[0-2])[./-](0?[1-9]|[12]\d|3[01])", s
            )
            if m:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return f"{y:04d}-{mo:02d}-{d:02d}"
            # YYYYMMDD
            m = re.search(r"(20\d{2})(0?[1-9]|1[0-2])(0?[1-9]|[12]\d|3[01])", s)
            if m:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return f"{y:04d}-{mo:02d}-{d:02d}"
            # YYYY년 MM월 DD일
            m = re.search(
                r"(20\d{2})\s*년\s*(0?[1-9]|1[0-2])\s*월\s*(0?[1-9]|[12]\d|3[01])\s*일",
                s,
            )
            if m:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return f"{y:04d}-{mo:02d}-{d:02d}"
            return ""

        # 후보 필드에서 먼저 찾고, 없으면 텍스트에서 추출
        for key in ("pubDate", "pubdate", "published", "datetime", "date"):
            ds = _extract_date_ymd(it.get(key, ""))
            if ds:
                date_prefix = ds
                break
        if not date_prefix:
            # 설명/스니펫, 제목에서도 추출 시도
            raw_desc = _strip_bold(it.get("description", "")) or _strip_bold(
                it.get("snippet", "")
            )
            date_prefix = _extract_date_ymd(raw_desc) or _extract_date_ymd(title)
        third_line = it.get("originallink") or it.get("link") or it.get("url") or ""

        if kind == "local":
            # local: 지도/플레이스 링크 우선. 없으면 개별 장소명/주소 기반 지도 검색 URL 합성
            pid = _extract_place_id(it)
            if pid:
                third_line = f"https://map.naver.com/v5/entry/place/{pid}"
            elif not third_line:
                # 허용 도메인 링크가 없으면 '장소명'만으로 지도 검색 URL 생성 (주소는 제외)
                # 주소를 포함하면 검색창에 과도한 문자열이 들어가 UX가 저하됨
                name = _strip_bold(it.get("title", ""))
                q = (name or "").strip()
                third_line = _naver_map_search_url(q)

            # 설명은 카테고리/주소/전화번호 등을 포함해 LLM이 즉시 활용 가능하도록 구성
            try:
                parts = []
                cat = _strip_bold(it.get("category", ""))
                if cat:
                    parts.append(cat)
                addr = _strip_bold(it.get("roadAddress", "")) or _strip_bold(
                    it.get("address", "")
                )
                if addr:
                    parts.append(addr)
                tel = _strip_bold(it.get("telephone", ""))
                if tel:
                    parts.append(tel)

                # 영업/거리 정보 보강
                open_now = ""
                if "open_now" in it:
                    is_open = bool(it.get("open_now"))
                    open_now = " 🟢 영업중" if is_open else " 🔴 영업종료"
                dist_info = ""
                if "distance_km" in it:
                    try:
                        dist_km = float(it.get("distance_km", 0))
                        if dist_km < 1.0:
                            dist_info = f" · {int(dist_km * 1000)}m"
                        else:
                            dist_info = f" · {dist_km:.1f}km"
                    except Exception:
                        dist_info = ""

                base_desc = " · ".join([p for p in parts if p]) or _strip_bold(
                    it.get("description", "")
                )
                if not base_desc:
                    base_desc = "(설명 없음)"

                desc = base_desc + (open_now or "") + (dist_info or "")
            except Exception:
                # 실패 시 기존 단순 규칙 유지
                if not desc or desc == "-":
                    desc = (
                        _strip_bold(it.get("category", ""))
                        or _strip_bold(it.get("description", ""))
                        or "(설명 없음)"
                    )

        elif kind == "blog":
            # blog: 설명 + 블로거명 표시
            bloggername = it.get("bloggername", "")
            if bloggername:
                title = f"{title} (by {bloggername})"
            if not llm_desc:
                desc = _strip_bold(it.get("description", ""))
            third_line = it.get("link") or third_line

        elif kind == "cafearticle":
            # cafearticle: 설명 + 카페명 표시
            cafename = it.get("cafename", "")
            if cafename:
                title = f"{title} [{cafename}]"
            if not llm_desc:
                desc = _strip_bold(it.get("description", ""))
            third_line = it.get("link") or third_line

        elif kind == "shop":
            # shop: 최저가 정보
            lprice = it.get("lprice", "")
            if not llm_desc:
                desc = f"최저가: {int(lprice):,}원" if lprice else "(가격 정보 없음)"
            third_line = it.get("link") or third_line

        elif kind == "image":
            # image: 썸네일 + 원본 링크
            thumbnail = it.get("thumbnail", "")
            if not llm_desc:
                desc = thumbnail or "(썸네일 없음)"
            third_line = it.get("link") or third_line

        elif kind == "kin":
            # kin: 지식iN 질문/답변
            if not llm_desc:
                desc = _strip_bold(it.get("description", ""))
            third_line = it.get("link") or third_line

        elif kind == "book":
            # book: 저자 정보
            author = it.get("author", "")
            if not llm_desc:
                author = it.get("author", "")
                desc = f"저자: {author}" if author else "(저자 정보 없음)"
            third_line = it.get("link") or third_line

        elif kind == "encyc":
            # encyc: 백과사전 설명
            if not llm_desc:
                desc = _strip_bold(it.get("description", ""))
            third_line = it.get("link") or third_line

        elif kind == "academic":
            # academic: 학술 논문 설명
            if not llm_desc:
                desc = _strip_bold(it.get("description", ""))
            third_line = it.get("link") or third_line

        else:
            # news/webkr 및 기타 공통 처리
            if not llm_desc:
                desc = _strip_bold(it.get("description", "")) or _strip_bold(
                    it.get("snippet", "")
                )
            third_line = (
                it.get("originallink") or it.get("link") or it.get("url") or third_line
            )

        # 공통: URL 정제 및 최종 폴백(검색 URL 합성)
        third_line = _sanitize_url(third_line)
        if not third_line or not re.match(r"^https?://", third_line, re.I):
            # kind==local은 위에서 지도/검색 URL을 이미 최대한 합성했으므로
            # 그 외 케이스에 한해 제목/카테고리/설명 기반 범용 검색 링크를 폴백으로 생성
            if kind != "local":
                q_text = (
                    _strip_bold(it.get("title", ""))
                    or _strip_bold(it.get("category", ""))
                    or _strip_bold(it.get("description", ""))
                )
                if q_text:
                    third_line = _fallback_search_url(q_text)
                else:
                    third_line = ""

        # 설명 길이 제한
        if desc and date_prefix:
            desc = f"{date_prefix} · {desc}" if desc != "-" else date_prefix
        if len(desc) > 140:
            desc = desc[:137].rstrip() + "..."

        # 주소나 링크 등 세 번째 줄 정보가 없으면 해당 아이템은 건너뜀
        if not third_line:
            continue

        blocks.append("\n".join([title, desc or "-", third_line]))

    return "\n\n".join(blocks)


def blocks_to_items(text: str) -> List[Dict[str, str]]:
    """
    3줄 블록 문자열을 구조화 리스트로 변환한다.
    - 각 블록: 제목/설명/URL
    - 무효 블록은 제외
    반환: [{"title": str, "desc": str, "url": str}, ...]
    """
    if not text:
        return []
    sanitized = ensure_block_shape(text)
    if not sanitized:
        return []
    out: List[Dict[str, str]] = []
    for blk in sanitized.split("\n\n"):
        lines = [ln.strip() for ln in blk.split("\n") if ln.strip()]
        if len(lines) < 3:
            continue
        title, desc, url = lines[0], lines[1], lines[2]
        out.append({"title": title, "desc": desc, "url": url})
    return out
