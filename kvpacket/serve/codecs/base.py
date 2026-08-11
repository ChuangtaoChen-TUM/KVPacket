"""Protocols for model-family request rendering and assistant-output parsing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..types import ModelRequest, ModelResponse, ToolCall

__all__ = ["ModelCodec", "StreamingToolCallParser", "ToolCallParser"]


@runtime_checkable
class ToolCallParser(Protocol):
    """Model-family parser for tool calls embedded in decoded assistant text."""

    def parse(self, text: str) -> tuple[str, Sequence[ToolCall]]:
        """Split decoded text into assistant content and tool calls."""
        ...


@runtime_checkable
class StreamingToolCallParser(ToolCallParser, Protocol):
    """Tool parser that identifies cumulative text safe to expose while decoding."""

    def visible_text_prefix(self, text: str) -> str:
        """Cumulative assistant text known not to be model-specific tool syntax."""
        ...


@runtime_checkable
class ModelCodec(Protocol):
    """Bidirectional model-family boundary above token-level packet execution.

    A codec validates model-facing features while rendering. In particular, a
    tool-bearing request must be rejected when its model family has no configured
    tool syntax. Compute backends remain responsible only for token and cache
    operations and do not interpret messages or tools.
    """

    def render(self, request: ModelRequest) -> list[int]:
        """Validate and render a structured request into generation-ready token IDs."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated tokens without applying a tool-call parser."""
        ...

    def stream_text(self, token_ids: Sequence[int]) -> str:
        """Cumulative decoded text safe to expose before generation finishes."""
        ...

    def parse(self, token_ids: Sequence[int], *, reached_limit: bool = False) -> ModelResponse:
        """Decode and parse one generated continuation."""
        ...
