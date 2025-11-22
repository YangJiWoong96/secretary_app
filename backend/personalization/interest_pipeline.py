from __future__ import annotations

"""
대화 이벤트 훅 → 관심사(interest) 업데이트 파이프라인

역할:
- 사용자의 최신 대화(positive/negative 시그널)로부터 required/normal/denied를 재계산하고 저장한다.
- 저장 위치는 Firestore(users/{uid}/personalization/interest)로 구성하며, Firestore 가용 불가 시 로컬 파일 백업도 수행한다.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import json

from backend.personalization.interest_profile import build_interest_from_signals


def _fs():
    try:
        from backend.config import get_firestore_client

        return get_firestore_client()
    except Exception:
        return None


def _local_interest_path(user_id: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "evidence" / "interest"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{user_id}.json"


def load_interest(user_id: str) -> Dict[str, List[str]]:
    """
    Firestore → 로컬 파일 순으로 로드.
    """
    db = _fs()
    if db:
        try:
            ref = (
                db.collection("users")
                .document(user_id)
                .collection("personalization")
                .document("interest")
            )
            snap = ref.get()
            if snap.exists:
                data = snap.to_dict() or {}
                return {
                    "required": list(data.get("required", [])),
                    "normal": list(data.get("normal", [])),
                    "denied": list(data.get("denied", [])),
                }
        except Exception:
            pass
    # local fallback
    try:
        fp = _local_interest_path(user_id)
        if fp.exists():
            return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"required": [], "normal": [], "denied": []}


def save_interest(user_id: str, interest: Dict[str, List[str]]) -> None:
    db = _fs()
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "required": list(interest.get("required", [])),
        "normal": list(interest.get("normal", [])),
        "denied": list(interest.get("denied", [])),
        "last_updated": now,
    }
    ok = False
    if db:
        try:
            ref = (
                db.collection("users")
                .document(user_id)
                .collection("personalization")
                .document("interest")
            )
            ref.set(payload)
            ok = True
        except Exception:
            ok = False
    # local backup
    try:
        fp = _local_interest_path(user_id)
        fp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        if not ok:
            raise


def update_interest_from_conversation(
    user_id: str,
    utterance: str,
    is_positive: bool = True,
    entities_preferred: Optional[List[str]] = None,
    domains_blocked: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    대화 1건(positive/negative) 기반으로 interest를 재계산하고 저장.
    - 누적 적용: 기존 interest를 불러와 positive/negative를 합성하여 승격/거부 업데이트
    """
    cur = load_interest(user_id)
    pos = [utterance] if is_positive else []
    neg = [utterance] if (not is_positive) else []

    # 기존 normal/required도 positive로 간주하여 승격 기회 부여
    seeds_positive = pos + cur.get("required", []) + cur.get("normal", [])
    seeds_negative = neg + cur.get("denied", [])

    new_interest = build_interest_from_signals(
        positive_utterances=seeds_positive,
        negative_utterances=seeds_negative,
        entities_preferred=entities_preferred or [],
        domains_blocked=domains_blocked or [],
        required_promotion_threshold=3,
    )
    save_interest(user_id, new_interest)
    return new_interest


__all__ = ["load_interest", "save_interest", "update_interest_from_conversation"]
