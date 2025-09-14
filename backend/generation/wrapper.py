import re
from typing import List


def _sanitize_no_json_echo(text: str) -> str:
    if not text:
        return text
    # 사용자 발화 에코 방지 및 JSON/브래킷 쿨다운
    # - 선행/후행의 과도한 괄호 제거
    t = text.strip()
    # 코드/JSON 같은 블록을 발견하면 간단히 문장형으로 전개
    if t.startswith("{") or t.startswith("["):
        return "요청하신 내용을 반영해 정리했어요. 자세한 내용은 요약으로 전환해 드렸습니다."
    # 각종 브래킷을 둥근 괄호로 치환하여 시각적 중립화
    t = t.replace("[", "(").replace("]", ")")
    return t


def _parse_web_blocks(web_ctx_blocks: str) -> List[tuple[str, str, str]]:
    items: List[tuple[str, str, str]] = []
    if not web_ctx_blocks:
        return items
    for block in web_ctx_blocks.split("\n\n"):
        lines = block.strip().split("\n")
        title = lines[0].strip() if len(lines) > 0 else ""
        desc = lines[1].strip() if len(lines) > 1 else ""
        url = lines[2].strip() if len(lines) > 2 else ""
        if url:
            items.append((title, desc, url))
    return items


def wrap_web_reply(question: str, web_ctx_blocks: str, persona10: str = "") -> str:
    items = _parse_web_blocks(web_ctx_blocks)
    if not items:
        return "관련 결과를 찾지 못했습니다. 다른 키워드로 다시 요청해 주세요."
    # 두 문장 정도의 대화체 문장화
    # 1문장: 상위 결과 안내, 2문장: 대표 2개 하이라이트(+링크)
    top = items[:2]
    highlights = []
    for title, desc, url in top:
        if title:
            highlights.append(f"{title} — {desc} ({url})")
        else:
            highlights.append(f"{desc} ({url})")
    sent1 = "요청하신 주제와 맞는 상위 결과를 요약해 드릴게요."
    sent2 = " ; ".join(highlights)
    out = f"{sent1} {sent2}"
    return _sanitize_no_json_echo(out)


def wrap_greeting_reply(question: str, persona10: str = "") -> str:
    return _sanitize_no_json_echo(
        "반가워요! 편하게 말씀해 주세요. 필요하신 걸 빠르게 도와드릴게요."
    )


def wrap_clarify_reply(short_q: str) -> str:
    return _sanitize_no_json_echo(short_q)


def wrap_generic_reply(text: str) -> str:
    return _sanitize_no_json_echo(text)
