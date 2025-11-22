import re

_URL = re.compile(r"^https?://[^\s)]+$", re.I)


def ensure_block_shape(text: str) -> str:
    """
    블록 문자열을 검사/정제한다.
    - 각 블록은 공백 줄로 구분
    - 각 블록은 3줄(제목/설명/URL) 형식
    - URL은 http/https만 허용, 공백/닫힘괄호 금지
    유효 블록만 재조합하여 반환. 없으면 빈 문자열.
    """
    if not text:
        return ""
    out: list[str] = []
    for blk in text.split("\n\n"):
        lines = [ln.strip() for ln in blk.split("\n") if ln.strip()]
        if len(lines) < 3:
            continue
        title, desc, url = lines[0], lines[1], lines[2]
        if not _URL.match(url):
            continue
        out.append("\n".join([title, desc, url]))
    return "\n\n".join(out)
