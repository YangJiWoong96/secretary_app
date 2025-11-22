from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    from backend.policy.profile_schema import validate_bot_profile  # type: ignore
except Exception:
    # 세션2 이후 BotProfile 스키마 제거와의 호환성: 검증을 생략한다.
    def validate_bot_profile(data):  # type: ignore
        return None


@dataclass
class BotProfile:
    persona: str = ""
    style: str = "neutral"
    abilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


def load_static_bot_profile(path: str | None = None) -> BotProfile:
    try:
        import os
        from pathlib import Path

        # 파일 우선 → 없으면 ENV → 기본값
        if path is None:
            base = (
                Path(__file__).resolve().parents[1]
                / "directives"
                / "bot_profile_static.json"
            )
        else:
            base = Path(path)
        if base.exists():
            with open(base, "r", encoding="utf-8") as f:
                data = json.load(f)
            validate_bot_profile(data)
            return BotProfile(
                persona=str(data.get("persona") or ""),
                style=str(data.get("style") or "neutral"),
                abilities=list(data.get("abilities") or []),
                constraints=dict(data.get("constraints") or {}),
            )
        # ENV/설정 기반(간략)
        try:
            from backend.config import get_settings as _gs

            _s = _gs()
            persona = str(getattr(_s, "BOT_PERSONA", "한국어 개인 비서"))
            style = str(getattr(_s, "BOT_STYLE", "neutral"))
            _abilities_raw = str(getattr(_s, "BOT_ABILITIES", ""))
        except Exception:
            persona = os.getenv("BOT_PERSONA", "한국어 개인 비서")
            style = os.getenv("BOT_STYLE", "neutral")
            _abilities_raw = os.getenv("BOT_ABILITIES", "")
        abilities = [s.strip() for s in _abilities_raw.split(",") if s.strip()]
        constraints = {}
        data = {
            "persona": persona,
            "style": style,
            "abilities": abilities,
            "constraints": constraints,
        }
        validate_bot_profile(data)
        return BotProfile(**data)
    except Exception:
        return BotProfile()


def merge_bot_profiles(static_bp: BotProfile, dynamic_bp: BotProfile) -> BotProfile:
    out = BotProfile(
        persona=(dynamic_bp.persona or static_bp.persona),
        style=(dynamic_bp.style or static_bp.style),
        abilities=list(
            dict.fromkeys((static_bp.abilities or []) + (dynamic_bp.abilities or []))
        ),
        constraints={**(static_bp.constraints or {}), **(dynamic_bp.constraints or {})},
    )
    try:
        validate_bot_profile(out.__dict__)
    except Exception:
        # 정적만이라도 유효하면 복구
        out = static_bp
    return out
