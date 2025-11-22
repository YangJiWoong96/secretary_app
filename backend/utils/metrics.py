"""
backend.utils.metrics - Prometheus 메트릭 정의

목적:
- Behavior 파이프라인 모니터링
- Profile 업데이트 추적
- Proactive 푸시 성능 측정

사용법:
    from backend.utils.metrics import behavior_slots_detected

    behavior_slots_detected.labels(user_id="user123", slot_key="food.spice").set(3)
"""

from prometheus_client import Counter, Histogram, Gauge


# === Behavior Metrics ===

behavior_slots_detected = Gauge(
    "behavior_slots_detected",
    "Number of behavior slots classified per turn",
    ["user_id", "slot_key"],
)

behavior_scoreboard_updates = Counter(
    "behavior_scoreboard_updates_total",
    "Total Scoreboard updates",
    ["user_id", "status"],  # status: active, pending, retired
)

behavior_rag_upserts = Counter(
    "behavior_rag_upserts_total",
    "Total RAG behavior_slots upserts",
    ["user_id", "status"],  # status: success, failure
)


# === Profile Metrics ===

profile_updates_total = Counter(
    "profile_updates_total",
    "Profile chunk upserts",
    [
        "user_id",
        "source",
        "status",
    ],  # source: inferred, explicit / status: active, pending, contradiction
)

profile_key_path_rejected = Counter(
    "profile_key_path_rejected_total",
    "Rejected key_path (not in whitelist)",
    ["key_path"],
)


# === Proactive Metrics ===

proactive_push_sent = Counter(
    "proactive_push_sent_total", "Total proactive pushes sent", ["user_id", "topic"]
)

proactive_context_build_latency = Histogram(
    "proactive_context_build_latency_seconds",
    "Proactive context build latency",
    ["user_id"],
)


# === Memory Metrics ===

memory_scope_errors = Counter(
    "memory_scope_errors_total",
    "STM/MTM/LTM scoping errors",
    [
        "layer",
        "error_type",
    ],  # layer: STM, STWM, MTM, LTM / error_type: user_id_missing, session_not_mapped
)


# === RAG Metrics ===

rag_retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds", "RAG retrieval latency", ["route", "user_id"]
)

rag_entity_expansion_count = Gauge(
    "rag_entity_expansion_count",
    "Number of entities added by query expansion",
    ["user_id"],
)
