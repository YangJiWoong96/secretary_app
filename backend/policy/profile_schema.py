# -*- coding: utf-8 -*-
"""
프로필 JSON 스키마 정의 및 검증 유틸.
- 사용자(User) 프로필과 챗봇(Bot) 프로필을 분리 정의
- 확장 가능하고 필수 필드가 명확한 스키마 유지
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import jsonschema  # type: ignore
except Exception as _e:
    jsonschema = None  # 런타임 환경에 없을 수 있음. 호출부에서 가드.


USER_PROFILE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "UserProfile",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "location": {"type": "string"},
        "occupation": {"type": "string"},
        "interests": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "preferences": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "food": {"type": "string"},
                "cafe": {"type": "string"},
            },
            "default": {},
        },
        "constraints": {
            "type": "object",
            "additionalProperties": True,
            "default": {},
        },
    },
    "required": [
        "name",
        "location",
        "occupation",
        "interests",
        "preferences",
        "constraints",
    ],
}


def validate_user_profile(data: Dict[str, Any]) -> None:
    if jsonschema is None:
        return
    jsonschema.validate(instance=data, schema=USER_PROFILE_SCHEMA)
