from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

EventKind = Literal["preference", "constraint", "style", "ability"]
EventSource = Literal["dialog", "action", "feedback", "external"]


@dataclass
class PersonaEvent:
    kind: EventKind
    key: str
    value: str
    weight: float
    ttl_sec: int
    ts: int
    source: EventSource
    extras: Optional[Dict[str, Any]] = None


def serialize_event(ev: PersonaEvent) -> Dict[str, Any]:
    return {
        "type": "bot_event",
        "k": ev.key,
        "v": ev.value,
        "kind": ev.kind,
        "weight": float(ev.weight),
        "ttl_sec": int(ev.ttl_sec),
        "ts": int(ev.ts),
        "source": ev.source,
        "extras": ev.extras or {},
        "text": f"[{ev.kind}] {ev.key}={ev.value} w={ev.weight}",
    }
