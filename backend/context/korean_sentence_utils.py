"""
한국어 문장 경계 인식 및 규칙 기반 경량 압축 유틸리티

- split_korean_sentences: 한국어 종결어미/문장부호 기반 문장 분리
- extract_head_tail_sentences: 앞/뒤 N개 문장 보존 + 중간 분리
- compress_middle_sentences_rule_based: 중복축소 및 보호패턴 복원

주의: 본 유틸은 경량 규칙 기반으로 설계되었으며, 외부 라이브러리 의존 없이
운영 환경에서 동작하도록 구성되었다.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def split_korean_sentences(text: str) -> List[str]:
    """
    한국어 문장 경계를 확장 인식하여 안전하게 분리한다.

    인식 패턴(휴리스틱 최소화, 규칙 기반):
    - 종결어미 + 문장부호: '다./요./네./죠./까?/니?/어!/지!' 등
    - 일반 문장부호(./?/!) 뒤 공백 후 한글/영문 대문자 시작
    """
    if not text or not text.strip():
        return []

    # 종결어미 + 문장부호 경계에 마커 삽입
    # 예: "합니다." "해요!" 등 뒤에서 분리
    pattern = r"([다요네죠까니어지]\s*[.!?])\s+(?=[가-힣A-Z])"
    marked = re.sub(pattern, r"\1<SENT_BREAK>", text)

    # 일반 문장부호 뒤 공백 후 한글/영문 대문자 시작도 보조 분할
    marked = re.sub(r"([.!?])\s+(?=[가-힣A-Z])", r"\1<SENT_BREAK>", marked)

    raw_sentences = marked.split("<SENT_BREAK>")

    sentences: List[str] = []
    for sent in raw_sentences:
        s = (sent or "").strip()
        if s:
            sentences.append(s)
    return sentences


def extract_head_tail_sentences(
    text: str, head_count: int = 2, tail_count: int = 2
) -> Tuple[str, str, str]:
    """
    텍스트의 앞/뒤 문장을 보존하고 중간을 분리한다.

    Returns:
        tuple[str, str, str]: (head_text, middle_text, tail_text)
    """
    sentences = split_korean_sentences(text)
    if len(sentences) <= (head_count + tail_count):
        return text, "", ""

    head = " ".join(sentences[:head_count])
    tail = " ".join(sentences[-tail_count:])
    middle = " ".join(sentences[head_count:-tail_count])
    return head, middle, tail


def compress_middle_sentences_rule_based(middle: str) -> str:
    """
    중간 문장의 규칙 기반 경량화.

    전략:
    - 숫자/날짜/URL/이메일 등의 보호 패턴 임시 치환 후 복원
    - 중복 단어 제거(조사/격조사는 일부 허용)
    """
    if not middle or len(middle) < 20:
        return middle

    preserve_patterns = [
        (r"\d+[년월일]", "<DATE>"),
        (r"\d{4}-\d{2}-\d{2}", "<DATEISO>"),
        (r"\d+[.,]?\d*%", "<PERCENT>"),
        (r"\d+원", "<WON>"),
        (r"https?://\S+", "<URL>"),
        (r"\S+@\S+\.\S+", "<EMAIL>"),
    ]

    protected = middle
    replacements: list[tuple[str, str]] = []

    for pattern, token in preserve_patterns:
        matches = re.findall(pattern, protected)
        for i, match in enumerate(matches):
            placeholder = f"{token}{i}"
            protected = protected.replace(match, placeholder, 1)
            replacements.append((placeholder, match))

    words = protected.split()
    unique_words: list[str] = []
    seen: set[str] = set()

    allowed_particles = {
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "에서",
        "로",
        "와",
        "과",
    }

    for word in words:
        lower = word.lower()
        if word in allowed_particles:
            unique_words.append(word)
            continue
        if lower not in seen:
            unique_words.append(word)
            seen.add(lower)

    compressed = " ".join(unique_words)

    for placeholder, original in replacements:
        compressed = compressed.replace(placeholder, original)

    return compressed
