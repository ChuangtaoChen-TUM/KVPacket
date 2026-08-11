"""Optional structured-request codecs and serving-state helpers."""

from .codecs import (
    ChatTokenizer,
    HFChatCodec,
    ModelCodec,
)
from .completion import (
    CompletionEvent,
    CompletionFinished,
    CompletionPrefilled,
    CompletionResult,
    CompletionToken,
    PacketCompletionEngine,
)
from .manager import PacketStateManager, SessionStateSnapshot
from .types import (
    ChatMessage,
    InferenceRequest,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolKind,
)

__all__ = [
    "ChatMessage",
    "InferenceRequest",
    "ModelRequest",
    "ModelResponse",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolKind",
    "ChatTokenizer",
    "ModelCodec",
    "HFChatCodec",
    "CompletionResult",
    "CompletionEvent",
    "CompletionPrefilled",
    "CompletionToken",
    "CompletionFinished",
    "PacketCompletionEngine",
    "PacketStateManager",
    "SessionStateSnapshot",
]
