from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict

logger = logging.getLogger("compressor")

_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        try:
            from backend.config import get_settings as _gs

            cap = int(getattr(_gs(), "COMPRESS_MAX_CONCURRENCY", 4))
        except Exception:
            cap = 4
        _SEM = asyncio.Semaphore(max(1, cap))
    return _SEM


async def compress_with_llm(
    text: str,
    target_tokens: int,
    strategy: str = "summary",
) -> str:
    """
    LLM 기반 텍스트 압축.

    - summary: 요약 위주 압축 (구조 보존 지침 포함)
    - selective: 선택적 유지(향후 확장)
    """
    from backend.config import get_settings
    from backend.context.compressor_utils import (
        mask_protected,
        must_keep_entities,
        trim_to_tokens_sentence,
        unmask,
        verify_preservation,
    )
    from backend.memory.summarizer import get_tokenizer
    from backend.utils.retry import openai_chat_with_retry

    settings = get_settings()
    enc = get_tokenizer()

    try:
        current_tokens = len(enc.encode(text or ""))
        if current_tokens <= max(0, int(target_tokens)):
            return text

        # 보호 토큰 마스킹
        masked_text, mp = mask_protected(text or "")

        compression_rate = max(0.05, float(target_tokens) / float(current_tokens or 1))
        headroom = int(max(8, target_tokens * 0.15))  # 15% 여유
        max_out = int(max(1, target_tokens - headroom))

        if strategy == "summary":
            sys_msg = {
                "role": "system",
                "content": (
                    "너는 텍스트 요약 전문가다. 출력은 반드시 'key_path: value' 스타일을 최대한 유지하라. "
                    "숫자/날짜/단위를 보존하고, 문체는 한국어를 유지한다. "
                    "추측 금지, 원문에 존재하는 정보만 남겨라."
                ),
            }
            user_msg = {
                "role": "user",
                "content": (
                    f"[원문]\n{masked_text}\n\n"
                    f"[목표]\n토큰 {target_tokens} 이내. 압축률 목표: {compression_rate:.0%}. "
                    "핵심 항목 우선(가드/정책/선호/동적 힌트 순)."
                ),
            }

            try:
                t0 = time.time()
                async with _get_sem():
                    resp = await asyncio.wait_for(
                        openai_chat_with_retry(
                            model=settings.LLM_MODEL,
                            messages=[sys_msg, user_msg],
                            temperature=0.0,
                            max_tokens=min(2048, int(target_tokens * 1.3)),
                        ),
                        timeout=float(getattr(settings, "COMPRESS_TIMEOUT_S", 0.8)),
                    )
                out = (resp.choices[0].message.content or "").strip()
                out = unmask(out, mp)
                # 토큰 초과 시 문장 경계 트림
                if len(enc.encode(out)) > target_tokens:
                    out = trim_to_tokens_sentence(out, enc, target_tokens)

                # 엔티티 보존 검증 실패 시 트림 폴백
                if not verify_preservation(text or "", out or ""):
                    try:
                        need = must_keep_entities(text or "")
                        have = must_keep_entities(out or "")
                        lost = list(sorted(need - have))
                        logger.warning(
                            f"[compressor] entity_loss count={len(lost)} examples={lost[:3]}"
                        )
                    except Exception:
                        pass
                    out = trim_to_tokens_sentence(text or "", enc, target_tokens)

                took_ms = int((time.time() - t0) * 1000)
                logger.info(
                    f"[compressor] LLM summary: {current_tokens}→{len(enc.encode(out))} tokens in {took_ms}ms"
                )
                return out
            except Exception as e:
                logger.exception(f"[compressor] LLM failed: {repr(e)}")
                return trim_to_tokens_sentence(text or "", enc, target_tokens)

        # selective: 현재는 summary와 동일 트림 폴백
        return trim_to_tokens_sentence(text or "", enc, target_tokens)
    except Exception:
        try:
            # 최후 폴백: 토큰 컷
            toks = enc.encode(text or "")[: max(0, int(target_tokens))]
            return enc.decode(toks)
        except Exception:
            return (text or "")[: max(0, int(target_tokens))]


def compress_by_tier(
    guard_text: str,
    core_text: str,
    dynamic_text: str,
    total_budget: int,
) -> Dict[str, str]:
    """
    계층별 압축 전략 적용.

    Guard 20% (압축 금지, 초과 시 문장경계 트림)
    Core 40% (LLM 요약)
    Dynamic 40% (문장경계 트림)
    """
    import asyncio

    from backend.context.compressor_utils import trim_to_tokens_sentence
    from backend.memory.summarizer import get_tokenizer

    enc = get_tokenizer()

    guard_text = guard_text or ""
    core_text = core_text or ""
    dynamic_text = dynamic_text or ""

    g_tok = len(enc.encode(guard_text))
    c_tok = len(enc.encode(core_text))
    d_tok = len(enc.encode(dynamic_text))
    total = g_tok + c_tok + d_tok

    if total <= total_budget:
        return {"guard": guard_text, "core": core_text, "dynamic": dynamic_text}

    # 비율 할당 + Headroom 10%
    headroom = max(8, int(total_budget * 0.1))
    work_budget = max(0, total_budget - headroom)
    g_budget = min(g_tok, int(work_budget * 0.2))
    c_budget = min(c_tok, int(work_budget * 0.4))
    d_budget = max(0, work_budget - g_budget - c_budget)

    # Guard: 압축 금지, 초과 시 문장 경계 트림
    if g_tok > g_budget:
        guard_text = trim_to_tokens_sentence(guard_text, enc, g_budget)

    # Core: LLM 요약, 이벤트 루프 상태 고려
    if c_tok > c_budget:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 동기 컨텍스트면 폴백 트림 (상위에서 비동기로 호출 권장)
                core_text = trim_to_tokens_sentence(core_text, enc, c_budget)
            else:
                core_text = loop.run_until_complete(
                    compress_with_llm(core_text, c_budget, strategy="summary")
                )
        except Exception:
            core_text = trim_to_tokens_sentence(core_text, enc, c_budget)

    # Dynamic: 문장 경계 트림(추후 LLMLingua 등 대체 가능)
    if d_tok > d_budget:
        dynamic_text = trim_to_tokens_sentence(dynamic_text, enc, d_budget)

    return {"guard": guard_text, "core": core_text, "dynamic": dynamic_text}


async def compress_by_tier_async(
    guard_text: str,
    core_text: str,
    dynamic_text: str,
    total_budget: int,
) -> Dict[str, str]:
    """
    비동기 버전: 이벤트 루프 내에서 Core를 LLM 요약으로 압축한다.
    """
    from backend.context.compressor_utils import trim_to_tokens_sentence
    from backend.memory.summarizer import get_tokenizer

    enc = get_tokenizer()

    guard_text = guard_text or ""
    core_text = core_text or ""
    dynamic_text = dynamic_text or ""

    g_tok = len(enc.encode(guard_text))
    c_tok = len(enc.encode(core_text))
    d_tok = len(enc.encode(dynamic_text))
    total = g_tok + c_tok + d_tok

    if total <= total_budget:
        return {"guard": guard_text, "core": core_text, "dynamic": dynamic_text}

    headroom = max(8, int(total_budget * 0.1))
    work_budget = max(0, total_budget - headroom)
    g_budget = min(g_tok, int(work_budget * 0.2))
    c_budget = min(c_tok, int(work_budget * 0.4))
    d_budget = max(0, work_budget - g_budget - c_budget)

    if g_tok > g_budget:
        guard_text = trim_to_tokens_sentence(guard_text, enc, g_budget)

    if c_tok > c_budget:
        try:
            core_text = await compress_with_llm(core_text, c_budget, strategy="summary")
        except Exception:
            core_text = trim_to_tokens_sentence(core_text, enc, c_budget)

    if d_tok > d_budget:
        dynamic_text = trim_to_tokens_sentence(dynamic_text, enc, d_budget)

    return {"guard": guard_text, "core": core_text, "dynamic": dynamic_text}
