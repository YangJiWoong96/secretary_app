"""
backend.utils.schema_builder - Pydantic → OpenAI JSON Schema 변환

목적:
- Pydantic 스키마를 OpenAI JSON Schema 형식으로 변환
- strict=True 옵션 지원
"""

from typing import Any, Dict


def build_json_schema(
    name: str, schema: Dict[str, Any], strict: bool = True
) -> Dict[str, Any]:
    """
    Pydantic 스키마를 OpenAI JSON Schema로 변환

    Args:
        name: 스키마 이름
        schema: Pydantic schema (model.schema())
        strict: strict mode 활성화 여부

    Returns:
        Dict: OpenAI JSON Schema
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": strict,
        },
    }
