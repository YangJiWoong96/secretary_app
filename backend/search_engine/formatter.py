from typing import List, Dict


def _strip_bold(html_text: str) -> str:
    if not isinstance(html_text, str):
        return ""
    return html_text.replace("<b>", "").replace("</b>", "").strip()


def format_items_to_blocks(items: List[Dict], kind: str) -> str:
    """네이버 응답 items를 3줄 블록들로 변환.

    - 각 블록: 이름, 간단한 설명, 링크
    - 설명은 길면 140자 내로 절단
    """
    if not items:
        return ""
    blocks = []
    for it in items[:5]:
        title = _strip_bold(it.get("title", "")) or "(이름 없음)"
        if kind == "local":
            desc = (
                _strip_bold(it.get("description", ""))
                or it.get("category", "")
                or "(설명 없음)"
            )
            link = it.get("link") or it.get("mapx") or ""
        else:
            # news/webkr 공통 처리
            desc = _strip_bold(it.get("description", "")) or _strip_bold(
                it.get("snippet", "")
            )
            link = it.get("originallink") or it.get("link") or it.get("url") or ""

        if len(desc) > 140:
            desc = desc[:137].rstrip() + "..."
        if not link:
            continue
        blocks.append("\n".join([title, desc or "(설명 없음)", link]))
    return "\n\n".join(blocks)
