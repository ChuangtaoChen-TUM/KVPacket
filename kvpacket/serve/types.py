"""Define provider-neutral chat, tool, and generation request records."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from ..runtime import SamplingConfig

__all__ = [
    "ChatMessage",
    "InferenceRequest",
    "ModelRequest",
    "ModelResponse",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolKind",
]

MessageRole = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["stop", "length", "tool_use"]
ToolKind = Literal["function", "custom"]


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral function or free-form tool exposed to the model.

    Attributes:
        name: Function name emitted in tool calls.
        description: Human-readable usage guidance.
        input_schema: JSON Schema for function arguments.
        kind: Responses tool-call representation expected by the caller.
        custom_format: Optional provider grammar metadata for a free-form tool.
        namespace: Optional Responses namespace containing the tool.
    """

    name: str
    description: str
    input_schema: Mapping[str, object]
    kind: ToolKind = "function"
    custom_format: Mapping[str, object] | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate the externally visible function name."""
        if not self.name:
            raise ValueError("tool name must be non-empty")
        if self.kind == "function" and self.custom_format is not None:
            raise ValueError("custom_format is valid only for custom tools")
        if self.namespace == "":
            raise ValueError("tool namespace must be non-empty or None")
        object.__setattr__(self, "input_schema", MappingProxyType(dict(self.input_schema)))
        if self.custom_format is not None:
            object.__setattr__(
                self,
                "custom_format",
                MappingProxyType(dict(self.custom_format)),
            )

    @property
    def model_name(self) -> str:
        """Chat-template-safe unique name, including a namespace when present."""
        return self.name if self.namespace is None else f"{self.namespace}__{self.name}"


@dataclass(frozen=True)
class ToolCall:
    """One model-requested function or free-form invocation."""

    call_id: str
    name: str
    arguments: Mapping[str, object]
    kind: ToolKind = "function"
    raw_input: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers required for a tool round trip."""
        if not self.call_id or not self.name:
            raise ValueError("tool call ID and name must be non-empty")
        if self.kind == "function" and self.raw_input is not None:
            raise ValueError("raw_input is valid only for custom tool calls")
        if self.kind == "custom" and self.raw_input is None:
            raise ValueError("custom tool calls require raw_input")
        if self.namespace == "":
            raise ValueError("tool namespace must be non-empty or None")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    @property
    def model_name(self) -> str:
        """Chat-template-safe unique name, including a namespace when present."""
        return self.name if self.namespace is None else f"{self.namespace}__{self.name}"


@dataclass(frozen=True)
class ChatMessage:
    """One provider-neutral conversation message.

    ``tool_calls`` is meaningful on assistant messages. ``tool_call_id`` and ``name``
    identify a tool-result message.
    """

    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Reject role-specific fields on incompatible messages."""
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are valid only on assistant messages")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and (self.tool_call_id is not None or self.name is not None):
            raise ValueError("tool_call_id and name are valid only on tool messages")


@dataclass(frozen=True)
class ModelRequest:
    """Structured model input independent of an external API schema."""

    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        """Require at least one conversation message."""
        if not self.messages:
            raise ValueError("model requests require at least one message")
        model_names = [tool.model_name for tool in self.tools]
        if len(model_names) != len(set(model_names)):
            raise ValueError("tool names must be unique after namespace rendering")


@dataclass(frozen=True)
class InferenceRequest:
    """Provider-neutral generation request produced by a wire adapter.

    Attributes:
        model: External model identifier requested by the caller.
        input: Structured messages and tools for the model codec.
        max_output_tokens: Maximum number of generated tokens.
        sampling: Framework-owned token-selection settings.
        stream: Whether the transport should emit streaming events.
        stop_sequences: Decoded strings that terminate the response.
    """

    model: str
    input: ModelRequest
    max_output_tokens: int
    sampling: SamplingConfig = SamplingConfig(do_sample=True)
    stream: bool = False
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate transport-independent generation limits."""
        if not self.model:
            raise ValueError("model must be non-empty")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if any(not stop for stop in self.stop_sequences):
            raise ValueError("stop sequences must be non-empty")


@dataclass(frozen=True)
class ModelResponse:
    """Structured assistant output after model-specific parsing."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = "stop"
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Detach token IDs from mutable caller sequences."""
        object.__setattr__(self, "token_ids", tuple(int(token_id) for token_id in self.token_ids))


@dataclass(frozen=True)
class TokenUsage:
    """Input and output token counts reported to an external API."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject negative accounting values."""
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def total_tokens(self) -> int:
        """Combined input and output token count."""
        return self.input_tokens + self.output_tokens
