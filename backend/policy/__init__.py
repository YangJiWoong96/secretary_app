"""
backend.policy - 정책 및 상태 관리 모듈

프로필, 세션 상태, 스냅샷 관리, 개인정보 정책 등을 담당합니다.
"""

from .profile_schema import (
    USER_PROFILE_SCHEMA,
    validate_user_profile,
)
from .profile_utils import (
    get_pinned_facts,
    pinned_facts_of,
)
from .snapshot_manager import (
    IDEMPOTENCY_CACHE,
    SESSION_STATE,
    SnapshotManager,
    edge_and_debounce,
    enqueue_snapshot,
    ensure_workers,
    flatten_profile_items,
    get_snapshot_manager,
    near_duplicate_log,
    sha256,
)
from .state import (
    GlobalState,
    OpsConfig,
    PolicyState,
    get_global_state,
    get_ops_config,
    get_policy_state,
    redact_text,
)

__all__ = [
    # State
    "PolicyState",
    "OpsConfig",
    "GlobalState",
    "get_policy_state",
    "get_ops_config",
    "get_global_state",
    "redact_text",
    # Snapshot
    "SnapshotManager",
    "get_snapshot_manager",
    "SESSION_STATE",
    "IDEMPOTENCY_CACHE",
    "sha256",
    "flatten_profile_items",
    "near_duplicate_log",
    "ensure_workers",
    "enqueue_snapshot",
    "edge_and_debounce",
    # Profile Utils
    "pinned_facts_of",
    "get_pinned_facts",
    # Schema
    "USER_PROFILE_SCHEMA",
    "validate_user_profile",
]
