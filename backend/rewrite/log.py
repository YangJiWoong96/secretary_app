from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RewriteRecord:
    raw_query: str
    query_rewritten: str
    applied_slots: List[str]
    unresolved_refs: Optional[List[str]] = None
    stwm_snapshot_id: Optional[str] = None


_REWRITE_LOGS: Dict[str, List[RewriteRecord]] = {}


def add_rewrite(session_id: str, rec: RewriteRecord):
    _REWRITE_LOGS.setdefault(session_id, []).append(rec)


def get_rewrites(session_id: str) -> List[RewriteRecord]:
    return list(_REWRITE_LOGS.get(session_id, []))
