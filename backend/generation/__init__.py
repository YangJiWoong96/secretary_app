"""
backend.generation - 응답 생성 모듈
"""

from .conversation import (
    ConversationGenerator,
    conversation_chain,
    get_conversation_generator,
)
from .evidence_handlers import (
    detect_evidence_detail_request,
    detect_unused_evidence_reload,
    handle_evidence_detail_request,
    reload_unused_evidence_eids,
)
from .filters import (
    ContextFilter,
    filter_semantic_mismatch,
    filter_web_ctx,
    get_context_filter,
)
from .schema import (
    AssistantResponse,
    TurnSummary,
)
from .selector import (
    build_evidence_meta,
    select_blocks_by_ids,
)
from .tagger import (
    extract_tags,
)
from .validators import (
    AnswerValidator,
    get_answer_validator,
    post_verify_answer,
    validate_final_answer,
)

__all__ = [
    # Filters
    "ContextFilter",
    "get_context_filter",
    "filter_semantic_mismatch",
    "filter_web_ctx",
    # Validators
    "AnswerValidator",
    "get_answer_validator",
    "post_verify_answer",
    "validate_final_answer",
    # Conversation
    "ConversationGenerator",
    "get_conversation_generator",
    "conversation_chain",
    # Selector
    "build_evidence_meta",
    "select_blocks_by_ids",
    # Tagger
    "extract_tags",
    # Schema
    "AssistantResponse",
    "TurnSummary",
    # Evidence Handlers
    "detect_evidence_detail_request",
    "detect_unused_evidence_reload",
    "handle_evidence_detail_request",
    "reload_unused_evidence_eids",
]
