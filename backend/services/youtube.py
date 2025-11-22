from __future__ import annotations

"""
backend.services.youtube - YouTube 텍스트 추출 서비스

기능 개요:
- YouTube 동영상 URL로부터 텍스트를 추출하여 3줄 블록(title/desc/url) 형태로 반환
- 우선순위: 자막(Transcript) → 실패 시 오디오 추출 후 STT(옵션)
- 성능/안전:
  - 간단한 TTL 메모리 캐시로 반복 호출 비용 절감
  - 외부 라이브러리 부재/키 미제공 시 보수적 폴백(빈 문자열 또는 최소 메타)

환경 변수 / 설정 (선택):
- FEATURE_YOUTUBE_TRANSCRIPT: "1"일 때 활성화(기본 1로 가정)
- FEATURE_YOUTUBE_STT: "1"일 때 자막 부재 시 STT 시도(기본 0)
- OPENAI_API_KEY: STT 수행 시 필요
"""

import asyncio
import os
import re
import time
from typing import Dict, Optional, Tuple

import httpx

# ─────────────────────────────────────────────────────────────
# 간단 TTL 캐시
# ─────────────────────────────────────────────────────────────
_TTL_SEC = 1800.0  # 30분
_CACHE: Dict[str, Tuple[float, str]] = {}
_CACHE_TEXT: Dict[str, Tuple[float, str]] = {}


def _get_flag(name: str, default: str = "1") -> bool:
    try:
        from backend.config import get_settings  # 지연 임포트

        val = getattr(get_settings(), name, None)
        if val is None:
            return os.getenv(name, default) in ("1", "true", "True", "YES", "yes")
        return bool(val)
    except Exception:
        return os.getenv(name, default) in ("1", "true", "True", "YES", "yes")


def _video_id_from_url(url: str) -> Optional[str]:
    """
    다양한 YouTube URL 포맷에서 videoId를 추출한다.
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/embed/VIDEOID
    """
    try:
        # youtu.be/VIDEOID
        m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
        if m:
            return m.group(1)
        # watch?v=VIDEOID
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
        if m:
            return m.group(1)
        # /embed/VIDEOID
        m = re.search(r"/embed/([A-Za-z0-9_-]{6,})", url)
        if m:
            return m.group(1)
        return None
    except Exception:
        return None


async def _fetch_title_with_og(url: str) -> Optional[str]:
    """
    OG 메타를 통해 제목을 보수적으로 추출한다(의존 라이브러리 최소화).
    """
    try:
        from backend.utils.http_client import get_async_client

        client = get_async_client()
        r = await client.get(url, timeout=httpx.Timeout(10.0))
        if r.status_code != 200:
            return None
        html = r.text or ""
        # og:title 우선
        m = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
            html,
            flags=re.I,
        )
        if m:
            return m.group(1).strip()
        # <title> 폴백
        m = re.search(r"<title>([^<]+)</title>", html, flags=re.I)
        if m:
            return m.group(1).strip()
        return None
    except Exception:
        return None


def _get_transcript_sync(video_id: str) -> Optional[str]:
    """
    YouTube Transcript API를 사용해 자막 텍스트를 동기 방식으로 가져온다.
    - 한국어/영어 우선 순위로 시도
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except Exception:
        return None
    lang_prio = ["ko", "en"]
    try:
        # 우선 직접 지정한 언어 우선
        for lang in lang_prio:
            try:
                segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])  # type: ignore
                if segments:
                    txt = " ".join(
                        [s.get("text", "") for s in segments if s.get("text")]
                    )
                    return txt.strip()
            except Exception:
                continue
        # 자동 생성 자막 포함 전체에서 첫 성공을 사용
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore
        for t in transcripts:  # type: ignore
            try:
                segs = t.fetch()
                txt = " ".join([s.get("text", "") for s in segs if s.get("text")])
                if txt:
                    return txt.strip()
            except Exception:
                continue
        return None
    except Exception:
        return None


def _stt_sync(url: str) -> Optional[str]:
    """
    yt-dlp로 오디오를 임시 파일로 추출한 뒤 OpenAI STT(Whisper API)로 텍스트화.
    - FEATURE_YOUTUBE_STT 플래그가 켜져 있고, OPENAI_API_KEY가 있어야 동작
    - 실패 시 None
    """
    if not _get_flag("FEATURE_YOUTUBE_STT", default="0"):
        return None
    # 설정에서 API 키 확보(환경변수 보강)
    try:
        from backend.config import get_settings  # 지연 임포트

        key = (get_settings().OPENAI_API_KEY or "").strip()
        if key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        import tempfile
        import yt_dlp  # type: ignore
    except Exception:
        return None
    audio_path = None
    try:
        # 1) 오디오 다운로드
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "noprogress": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
                ydl.download([url])
            # 추정 파일명 검색
            for fn in os.listdir(tmpdir):
                if fn.lower().endswith(".mp3"):
                    audio_path = os.path.join(tmpdir, fn)
                    break
            if not audio_path:
                return None

            # 2) OpenAI STT
            try:
                from openai import OpenAI  # type: ignore
            except Exception:
                return None
            client = OpenAI()
            with open(audio_path, "rb") as f:
                # 최신 Audio Transcriptions 엔드포인트 사용
                resp = client.audio.transcriptions.create(  # type: ignore[attr-defined]
                    model="gpt-4o-transcribe",  # Whisper large 계열 또는 gpt-4o-mini-transcribe로 교체 가능
                    file=f,
                    response_format="text",
                    language="ko",
                )
            txt = str(resp or "").strip()
            return txt or None
    except Exception:
        return None


async def _build_block_from_url(url: str) -> str:
    """
    URL에서 제목/텍스트를 추출하여 3줄 블록(title/desc/url)로 구성.
    - 자막 성공: 자막 앞부분 350자
    - 자막 실패 + STT 성공: STT 텍스트 앞부분 350자
    - 모두 실패: 최소 메타(title 또는 '-')
    """
    vid = _video_id_from_url(url or "")
    # 제목은 우선 OG 메타로 시도
    title = await _fetch_title_with_og(url) or "YouTube 영상"
    desc = "-"
    if vid and _get_flag("FEATURE_YOUTUBE_TRANSCRIPT", default="1"):
        # 동기 자막을 별도 스레드로 실행
        transcript = await asyncio.to_thread(_get_transcript_sync, vid)
        if transcript and len(transcript) >= 30:
            desc = transcript[:350].strip()
        else:
            stt_txt = await asyncio.to_thread(_stt_sync, url)
            if stt_txt and len(stt_txt) >= 30:
                desc = stt_txt[:350].strip()
    return "\n".join([title, desc, url])


async def get_youtube_block(url: str) -> str:
    """
    공개 API:
    - 입력: YouTube URL
    - 출력: 'title\\ndesc\\nurl' 3줄 블록 문자열(실패 시 최소 블록 또는 빈 문자열)
    - TTL 캐시를 적용하여 동일 URL 요청 반복시 부하를 줄인다.
    """
    if not url or "youtu" not in url:
        return ""
    now = time.time()
    hit = _CACHE.get(url)
    if hit and (now - hit[0]) <= _TTL_SEC:
        return hit[1]
    try:
        block = await _build_block_from_url(url)
    except Exception:
        block = ""
    _CACHE[url] = (now, block or "")
    return block or ""


async def get_youtube_text(url: str) -> str:
    """
    멀티에이전트용: 3줄 제한 없이 최대한 긴 텍스트(자막/폴백 STT)를 반환.
    - 캐시 TTL 동일 적용
    """
    if not url or "youtu" not in url:
        return ""
    now = time.time()
    key = f"text:{url}"
    hit = _CACHE_TEXT.get(key)
    if hit and (now - hit[0]) <= _TTL_SEC:
        return hit[1]
    vid = _video_id_from_url(url or "")
    text = ""
    if vid and _get_flag("FEATURE_YOUTUBE_TRANSCRIPT", default="1"):
        try:
            text = await asyncio.to_thread(_get_transcript_sync, vid) or ""
        except Exception:
            text = ""
    if not text:
        try:
            text = await asyncio.to_thread(_stt_sync, url) or ""
        except Exception:
            text = ""
    # 멀티에이전트 상한 적용(설정값)
    try:
        from backend.config import get_settings

        cap = int(get_settings().MA_YT_CONTENT_MAX_CHARS)
    except Exception:
        cap = 20000
    if text and cap > 0:
        text = text[:cap]
    _CACHE_TEXT[key] = (now, text or "")
    return text or ""


__all__ = ["get_youtube_block", "get_youtube_text"]
