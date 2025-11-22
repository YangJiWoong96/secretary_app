"""
backend.utils.schema_registry - 구조화 출력(JSON Schema) 단일 소스

모든 LLM response_format용 JSON Schema를 한 곳에서 관리한다.
원칙:
- type: "object" 고정
- properties/required 명시
- additionalProperties: False

부트타임에 validate_schema_registry()를 호출하여 스키마의 최소 요건을 검증한다.
"""

from __future__ import annotations

from typing import Any, Dict


def _base(schema: Dict[str, Any]) -> Dict[str, Any]:
    """스키마 공통 보정: type/properties/required/additionalProperties 강제 점검.

    - 호출 전 schema는 최상위에 type/properties가 포함되어야 한다.
    - 누락 시 KeyError를 유발하여 초기화 단계에서 바로 잡는다.
    """
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dict")
    # 필수 키 존재성 검사 (최소 유효성)
    if schema.get("type") != "object":
        raise KeyError("schema.type must be 'object'")
    if "properties" not in schema or not isinstance(schema["properties"], dict):
        raise KeyError("schema.properties must exist and be an object")
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    return schema


def get_web_filter_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "filtered": {"type": "string"},
            },
            "required": ["keep", "filtered"],
            "additionalProperties": False,
        }
    )


def get_rag_filter_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "filtered": {"type": "string"},
            },
            "required": ["keep", "filtered"],
            "additionalProperties": False,
        }
    )


def get_archive_decision_schema() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "should_archive": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "fact_density": {"type": "number"},
            "novelty": {"type": "number"},
            "user_interest": {"type": "number"},
            "date_candidates": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "consensus_date_ymd": {"type": ["integer", "null"]},
            "consensus_confidence": {"type": "number"},
        },
        "required": [],
        "additionalProperties": False,
    }
    # strict 모드 호환: 모든 키를 required에 포함
    schema["required"] = list(schema["properties"].keys())
    return _base(schema)


def get_feedback_analysis_schema() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "has_feedback": {"type": "boolean"},
            "feedbacks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence_type": {"type": "string", "enum": ["rag", "web"]},
                        "evidence_snippet": {"type": "string"},
                        "feedback_type": {
                            "type": "string",
                            "enum": [
                                "positive",
                                "negative",
                                "correction",
                                "elaboration",
                            ],
                        },
                        "user_comment": {"type": "string"},
                        "confidence_adjustment": {"type": "number"},
                    },
                    # 엄격 모드 호환: 모든 속성을 required로 고정
                    "required": [
                        "evidence_type",
                        "evidence_snippet",
                        "feedback_type",
                        "user_comment",
                        "confidence_adjustment",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        # 최상위도 모든 키 required
        "required": ["has_feedback", "feedbacks"],
        "additionalProperties": False,
    }
    return _base(schema)


def get_explicit_facts_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "explicit_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key_path": {"type": "string"},
                            # OpenAI Structured Outputs 제약 상 모든 필드는 명시적 타입 필요
                            # 값은 스칼라로 제한 (복합 타입은 문자열로 전달)
                            "value": {"type": ["string", "number", "boolean", "null"]},
                            "evidence": {"type": "string"},
                        },
                        # 400 방지: items.required에 evidence를 포함하여 스키마 완전성 보장
                        "required": ["key_path", "value", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["explicit_facts"],
            "additionalProperties": False,
        }
    )


def get_keyword_query_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "2~6개 핵심 키워드 공백 구분",
                    "maxLength": 120,
                }
            },
            "required": ["q"],
            "additionalProperties": False,
        }
    )


def get_answer_fit_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {"keep": {"type": "boolean"}},
            "required": ["keep"],
            "additionalProperties": False,
        }
    )


def get_date_consensus_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "consensus_date_ymd": {"type": ["integer", "null"]},
                "consensus_confidence": {"type": "number"},
                "answer_date_ymd": {"type": ["integer", "null"]},
                "match": {"type": "boolean"},
            },
            # 합의 날짜 누락으로 400 오류가 발생하지 않도록 필수 키를 확정한다
            "required": [
                "consensus_date_ymd",
                "consensus_confidence",
                "answer_date_ymd",
                "match",
            ],
            "additionalProperties": False,
        }
    )


def get_post_verify_tip_schema() -> Dict[str, Any]:
    """
    사후 검증(post-verify)에서 사용하는 Tip 스키마 (버전 고정).
    - 모든 속성은 명시적 type 지정
    - strict 모드 대응
    - $id로 버전 명시: post_verify_tip:v1
    """
    return _base(
        {
            "$id": "post_verify_tip:v1",
            "type": "object",
            "properties": {
                "tip": {"type": "string"},
            },
            "required": ["tip"],
            "additionalProperties": False,
        }
    )


def get_analysis_v1_schema() -> Dict[str, Any]:
    """
    AnalysisV1 - 라우팅/컨텍스트 일치성 감사용 구조화 출력 스키마
    - route: 최종 라우트
    - ctx_used: LLM이 판단한 실제 사용 컨텍스트(allowed_ctx ∩ ctx_present의 부분집합)
    - reasons: 선택적 사유 코드 목록
    - notes: 짧은 메모(최대 160자)
    """
    return _base(
        {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["conv", "web", "rag"]},
                "ctx_used": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["conv_only", "web_ctx", "rag_ctx"],
                    },
                },
                "reasons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "symptom_general",
                            "needs_evidence",
                            "location_intent",
                            "no_high_quality_evidence",
                            "ambiguous_margin",
                        ],
                    },
                },
                "notes": {"type": "string", "maxLength": 160},
            },
            # OpenAI Structured Output 400 방지: notes를 필수로 포함
            "required": ["route", "ctx_used", "reasons", "notes"],
            "additionalProperties": False,
        }
    )


def get_finance_intent_schema() -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "is_finance": {"type": "boolean"},
            "intent": {
                "type": "string",
                "enum": ["realtime_price", "historical_price", "news", "unknown"],
            },
            "entity": {"type": "string"},
            "ticker_hint": {"type": "string"},
            "exchange_hint": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }
    schema["required"] = list(schema["properties"].keys())
    return _base(schema)


def get_ticker_resolve_schema() -> Dict[str, Any]:
    return _base(
        {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "exchange": {"type": "string"},
                "is_crypto": {"type": "boolean"},
                "name": {"type": "string"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        }
    )


def validate_schema_registry() -> None:
    """레지스트리 스키마 최소 요건 검증. KeyError/TypeError 발생 시 초기화 실패로 조기 노출.
    런타임에서는 예외를 삼켜 품질 저하를 방지한다.
    """
    validators = [
        get_web_filter_schema,
        get_rag_filter_schema,
        get_archive_decision_schema,
        get_feedback_analysis_schema,
        get_explicit_facts_schema,
        get_keyword_query_schema,
        get_answer_fit_schema,
        get_date_consensus_schema,
        get_post_verify_tip_schema,
        get_finance_intent_schema,
        get_ticker_resolve_schema,
    ]
    for f in validators:
        try:
            _ = f()
        except Exception as e:
            # 검증 실패 시 즉시 예외 발생 (부트 중단)
            raise RuntimeError(f"Schema validation failed for {f.__name__}: {e}")
