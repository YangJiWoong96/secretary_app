"""
backend.memory - 메모리 관리 모듈

단기/장기 메모리, 요약, STWM(Short-Term Working Memory), 턴 관리를 담당합니다.
"""

from .coordinator import (
    MemoryCoordinator,
    TurnResult,
    get_memory_coordinator,
)
from .redis_memory import (
    HybridSummaryMemory,
    get_short_term_memory,
)
from .selector import (
    select_summaries,
)
from .stwm import (
    get_stwm_snapshot,
    update_stwm,
)
from .summarizer import (
    DialogueSummarizer,
    build_structured_and_summary,
    count_tokens_msgs,
    count_tokens_text,
    get_structure_schema,
    get_summarizer,
    get_tokenizer,
    get_verify_rule,
    messages_to_text,
    model_supports_response_format,
    split_for_summary,
)
from .turns import (
    add_turn,
    get_summaries,
    maybe_summarize,
)

__all__ = [
    # Summarizer
    "DialogueSummarizer",
    "get_summarizer",
    "count_tokens_text",
    "count_tokens_msgs",
    "messages_to_text",
    "model_supports_response_format",
    "split_for_summary",
    "build_structured_and_summary",
    "get_structure_schema",
    "get_verify_rule",
    "get_tokenizer",
    # Redis Memory
    "HybridSummaryMemory",
    "get_short_term_memory",
    # STWM
    "update_stwm",
    "get_stwm_snapshot",
    # Turns
    "add_turn",
    "maybe_summarize",
    "get_summaries",
    # Selector
    "select_summaries",
    # Coordinator (re-export only; do NOT redefine singleton here)
    "get_memory_coordinator",
    "MemoryCoordinator",
    "TurnResult",
]
