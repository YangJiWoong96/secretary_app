"""
backend.generation.validators - 답변 검증

생성된 답변의 적합성을 사전/사후 검증합니다.
"""

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger("validators")

from backend.utils.tracing import traceable

# 안전도 등급: 사전/사후 검증 중 safety-critical 허용 리스트
_SAFETY_CRITICAL_STAGES = {"temporal_consistency"}


class AnswerValidator:
    """답변 검증 클래스"""

    def __init__(self):
        self._settings = None
        # SLO 기반 fail-closed 전환(임시) 제어
        self._slo_window: list[bool] = []  # True=timeout, False=ok
        self._slo_window_cap = 50
        self._fail_closed_until_ts: float = 0.0

    @property
    def settings(self):
        if self._settings is None:
            from backend.config import get_settings

            self._settings = get_settings()
        return self._settings

    def _validator_timeout_s(self) -> float:
        """
        VALIDATOR 타임아웃 우선순위:
        VALIDATOR_TIMEOUT_S > VALIDATION_TIMEOUT_S > default(1.8s)
        """
        try:
            if self.settings.VALIDATOR_TIMEOUT_S is not None:
                return float(self.settings.VALIDATOR_TIMEOUT_S)
            if self.settings.VALIDATION_TIMEOUT_S is not None:
                return float(self.settings.VALIDATION_TIMEOUT_S)
        except Exception:
            pass
        return 1.8

    def _record_timeout_and_maybe_trip(
        self, stage: str, meta: dict | None = None
    ) -> None:
        """타임아웃 SLO 집계 및 서킷 전환 판단"""
        from backend.utils.logger import log_event as _le

        try:
            self._slo_window.append(True)
            if len(self._slo_window) > self._slo_window_cap:
                self._slo_window = self._slo_window[-self._slo_window_cap :]
            total = len(self._slo_window)
            timeouts = sum(1 for x in self._slo_window if x)
            ratio = (timeouts / total) if total > 0 else 0.0
            data = {"stage": stage, "ratio": ratio, "count": total}
            if meta:
                data.update(meta)
            _le("validator.timeout", data, level=logging.WARNING)
            # SLO 위반 시 일시적 fail-closed 모드 전환(60s)
            if total >= 20 and ratio >= 0.30:
                import time as _time

                self._fail_closed_until_ts = _time.time() + 60.0
                _le(
                    "validator.slo_trip",
                    {
                        "stage": stage,
                        "ratio": ratio,
                        "count": total,
                        "fail_closed_sec": 60,
                    },
                    level=logging.WARNING,
                )
        except Exception:
            pass

    def _record_ok(self) -> None:
        try:
            self._slo_window.append(False)
            if len(self._slo_window) > self._slo_window_cap:
                self._slo_window = self._slo_window[-self._slo_window_cap :]
        except Exception:
            pass

    def _should_fail_closed(self) -> bool:
        try:
            import time as _time

            return _time.time() < self._fail_closed_until_ts
        except Exception:
            return False

    @traceable(
        name="Validator: post_verify_answer",
        run_type="chain",
        tags=["validator", "post_verify"],
    )
    async def post_verify_answer(
        self,
        user_input: str,
        rag_ctx: str,
        web_ctx: str,
        answer: str,
        websocket: WebSocket,
    ) -> None:
        """
        답변 사후 검토 (비차단)

        RAG/WEB 컨텍스트가 존재할 때 답변의 적합성을 경량 검토합니다.
        문제 발견 시 경고 메시지를 WebSocket으로 전송합니다.
        """
        if not (rag_ctx.strip() or web_ctx.strip()):
            return

        from backend.memory import model_supports_response_format as _msrf
        from backend.utils.retry import openai_chat_with_retry
        from backend.utils.schema_builder import build_json_schema as _bjs

        ctx_cut = (f"{rag_ctx}\n{web_ctx}")[:1200]
        ans_cut = (answer or "")[:800]

        msgs = [
            {
                "role": "system",
                "content": (
                    "너는 답변 검토자다. [컨텍스트]와 [질문] 대비 [답변]의 적합성을 확인하고,"
                    " 무관/상충/추측/과장을 감지하면 경고 메시지 1줄을 한국어로 출력하라."
                    " 문제가 없으면 빈 문자열만 돌려라."
                ),
            },
            {
                "role": "user",
                "content": f"[질문]\n{user_input}\n\n[컨텍스트]\n{ctx_cut}\n\n[답변]\n{ans_cut}",
            },
        ]

        try:
            # JSON 스키마(엄격) - 레지스트리 참조
            from backend.utils.schema_registry import get_post_verify_tip_schema as _gpt

            rf = None
            try:
                rf = _bjs("PostVerifyTipV1", _gpt(), strict=True)
            except Exception:
                rf = None

            resp = await asyncio.wait_for(
                openai_chat_with_retry(
                    model=self.settings.LLM_MODEL,
                    messages=msgs,
                    temperature=0.0,
                    max_tokens=80,
                    **(
                        {"response_format": rf}
                        if (rf and _msrf(self.settings.LLM_MODEL))
                        else {}
                    ),
                ),
                timeout=self._validator_timeout_s(),
            )

            raw = (resp.choices[0].message.content or "").strip()
            tip = ""
            # JSON 파싱 우선, 실패 시 문자열로 강제 캐스팅
            try:
                if raw.startswith("{"):
                    data = json.loads(raw)
                    tip = str(data.get("tip") or "").strip()
                else:
                    tip = raw
            except Exception:
                tip = raw

            try:
                from backend.utils.logger import log_event

                log_event(
                    "post_verify_result",
                    {
                        "user_input": user_input,
                        "rag_ctx": rag_ctx,
                        "web_ctx": web_ctx,
                        "answer": answer,
                        "tip": tip,
                    },
                )
            except Exception:
                pass
            if tip:
                try:
                    await websocket.send_text(f"\n[검토] {tip}")
                except Exception:
                    pass
        except asyncio.TimeoutError:
            # fail-open: 경고 이벤트만
            try:
                from backend.utils.logger import log_event as _le

                self._record_timeout_and_maybe_trip(
                    "post_verify", {"len_ctx": len((rag_ctx or "") + (web_ctx or ""))}
                )
                _le(
                    "post_verify_timeout",
                    {"len_ctx": len((rag_ctx or "") + (web_ctx or ""))},
                    level=logging.WARNING,
                )
            except Exception:
                pass
            return
        except Exception as e:
            try:
                from backend.utils.logger import log_event as _le

                _le(
                    "validator.error",
                    {"stage": "post_verify", "error": repr(e)},
                    level=logging.WARNING,
                )
            except Exception:
                pass
            return

    @traceable(
        name="Validator: validate_final_answer", run_type="chain", tags=["validator"]
    )
    async def validate_final_answer(
        self, user_input: str, rag_ctx: str, web_ctx: str, answer: str
    ) -> bool:
        """
        최종 답변 적합성 사전 검증

        질문/컨텍스트 대비 부적합하면 False 반환.

        Returns:
            bool: True면 적합, False면 부적합
        """
        try:
            if not answer:
                return False

            from backend.memory import model_supports_response_format
            from backend.utils.retry import openai_chat_with_retry
            from backend.utils.schema_builder import build_json_schema
            from backend.utils.logger import log_event as _le

            # 1) 기본 적합성 검증(JSON Schema 표준)
            from backend.utils.schema_registry import (
                get_answer_fit_schema,
                get_date_consensus_schema,
            )
            from backend.validation.temporal_validator import get_temporal_validator

            fit_rf = build_json_schema(
                "AnswerFit", get_answer_fit_schema(), strict=True
            )

            ctx_short = (rag_ctx or "") + "\n" + (web_ctx or "")
            ctx_short = ctx_short[:1200]
            ans_short = (answer or "")[:800]

            msgs = [
                {
                    "role": "system",
                    "content": (
                        "너는 최종 검증자다. [질문]과 [컨텍스트] 대비 [답변]이 의도에 적합하면 keep=true,"
                        " 부적합/무관/노이즈 과다면 keep=false. JSON만."
                    ),
                },
                {
                    "role": "user",
                    "content": f"[질문]\n{user_input}\n\n[컨텍스트]\n{ctx_short}\n\n[답변]\n{ans_short}",
                },
            ]

            kwargs = {
                "model": self.settings.LLM_MODEL,
                "messages": msgs,
                "max_tokens": 20,
                "temperature": 0.0,
            }
            if model_supports_response_format(self.settings.LLM_MODEL):
                kwargs["response_format"] = fit_rf

            # 1-a) 적합성 검증 호출(저위험) — 타임아웃/에러 시 fail-open
            keep_basic = True
            try:
                resp = await asyncio.wait_for(
                    openai_chat_with_retry(**kwargs),
                    timeout=self._validator_timeout_s(),
                )
                txt = (resp.choices[0].message.content or "").strip()
                data = json.loads(txt) if txt.startswith("{") else {}
                keep_basic = bool(data.get("keep", False))
                self._record_ok()
            except asyncio.TimeoutError:
                self._record_timeout_and_maybe_trip(
                    "answer_fit", {"ctx_len": len(ctx_short), "ans_len": len(ans_short)}
                )
                # fail-open 유지
            except Exception as e:
                try:
                    _le(
                        "validator.error",
                        {"stage": "answer_fit", "error": repr(e)},
                        level=logging.WARNING,
                    )
                except Exception:
                    pass
                # fail-open 유지

            # 2) 시간 정합성 검증(증거가 있을 때 강제, safety-critical → fail-closed 유지)
            if (web_ctx or rag_ctx).strip():
                validator = get_temporal_validator()
                ok, _reason = validator.validate_date_consistency(
                    answer, web_ctx or rag_ctx, strict=True
                )
                if not ok:
                    return False

            # 3) 날짜 합의 기반 강화 검증 (web_ctx가 있을 때 추가, 비핵심 → timeout 시 fail-open)
            if (web_ctx or "").strip():
                try:
                    from backend.memory import model_supports_response_format as _msrf
                    from backend.utils.retry import openai_chat_with_retry as _ocwr
                    from backend.utils.schema_builder import build_json_schema as _bjs

                    date_rf = _bjs(
                        "DateConsensus", get_date_consensus_schema(), strict=True
                    )

                    msgs_dc = [
                        {
                            "role": "system",
                            "content": (
                                "너는 날짜 정합성 검증기다. [WEB 컨텍스트]에서 날짜 후보를 교차검증해"
                                " 합의된 날짜(consensus)를 YYYYMMDD로 추정하고, [답변]에서 언급된 날짜를 추출하여"
                                " 일치 여부를 판단해라. JSON만 출력."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"[WEB 컨텍스트]\n{web_ctx[:1200]}\n\n[답변]\n{(answer or '')[:600]}"
                            ),
                        },
                    ]

                    kwargs_dc = {
                        "model": self.settings.LLM_MODEL,
                        "messages": msgs_dc,
                        "temperature": 0.0,
                        "max_tokens": 80,
                    }
                    if _msrf(self.settings.LLM_MODEL):
                        kwargs_dc["response_format"] = date_rf

                    resp_dc = await asyncio.wait_for(
                        _ocwr(**kwargs_dc), timeout=self._validator_timeout_s()
                    )
                    txt_dc = (resp_dc.choices[0].message.content or "").strip()
                    data_dc = json.loads(txt_dc) if txt_dc.startswith("{") else {}

                    cons_conf = float(data_dc.get("consensus_confidence", 0.0))
                    match = bool(data_dc.get("match", True))
                    thr = float(
                        getattr(self.settings, "CONSENSUS_DATE_STRICT_THR", 0.75)
                    )
                    if cons_conf >= thr and not match:
                        return False
                    ans_date = int(data_dc.get("answer_date_ymd") or 0)
                    cons_date = int(data_dc.get("consensus_date_ymd") or 0)
                    if cons_date and ans_date and (ans_date != cons_date):
                        return False
                except asyncio.TimeoutError:
                    self._record_timeout_and_maybe_trip(
                        "date_consensus", {"web_len": len(web_ctx or "")}
                    )
                    # 강화 검증 실패는 fail-open
                except Exception:
                    # 강화 검증 오류는 무시(기본 검증으로 충분)
                    pass

            return keep_basic
        except Exception as e:
            # 최상위 예외: 기본적으로 fail-open, 단 서킷 트립 시 fail-closed
            try:
                from backend.utils.logger import log_event as _le

                _le(
                    "validator.error",
                    {"stage": "final", "error": repr(e)},
                    level=logging.WARNING,
                )
            except Exception:
                pass
            return False if self._should_fail_closed() else True


_validator_instance = None


def get_answer_validator():
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = AnswerValidator()
    return _validator_instance


async def post_verify_answer(user_input, rag_ctx, web_ctx, answer, websocket):
    """호환성 래퍼"""
    validator = get_answer_validator()
    return await validator.post_verify_answer(
        user_input, rag_ctx, web_ctx, answer, websocket
    )


async def validate_final_answer(user_input, rag_ctx, web_ctx, answer):
    """호환성 래퍼"""
    validator = get_answer_validator()
    return await validator.validate_final_answer(user_input, rag_ctx, web_ctx, answer)
