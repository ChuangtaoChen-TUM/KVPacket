"""Hugging Face chat-template codec for structured model requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from torch import Tensor

from ..types import ChatMessage, ModelRequest, ModelResponse, ToolCall, ToolDefinition
from .base import StreamingToolCallParser, ToolCallParser

__all__ = ["ChatTokenizer", "HFChatCodec"]


class ChatTokenizer(Protocol):
    """Tokenizer operations used by the default Hugging Face chat codec."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> object:
        """Render chat messages using the tokenizer's model template."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str | list[str]:
        """Decode one generated token sequence."""
        ...


class HFChatCodec:
    """Default codec backed by a Hugging Face tokenizer chat template.

    Tool-call syntax is model-specific, so callers inject a ``ToolCallParser`` for
    models that support tools. A request containing tool definitions or a prior tool
    transcript is rejected when no parser is configured. The codec, rather than
    ``PacketSession`` or ``PacketExecutor``, owns the chat template and parser so
    lower layers remain token-only.

    Supplying a parser declares that the selected model and tokenizer understand that
    parser's tool syntax. The framework can validate this configuration boundary, but
    it cannot prove that arbitrary model weights were trained to call tools reliably.

    Args:
        tokenizer: Tokenizer with ``apply_chat_template`` and ``decode``.
        tool_call_parser: Optional model-family tool-call parser.
    """

    def __init__(
        self,
        tokenizer: ChatTokenizer,
        *,
        tool_call_parser: ToolCallParser | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser

    def _validate(self, request: ModelRequest) -> None:
        """Reject tool-bearing requests when no model-family tool syntax is configured."""
        uses_tools = bool(request.tools) or any(
            message.role == "tool" or bool(message.tool_calls) for message in request.messages
        )
        if uses_tools and self.tool_call_parser is None:
            raise ValueError(
                "this model codec has no tool-call support; configure a model-family "
                "ToolCallParser or send a request without tools"
            )

    def render(self, request: ModelRequest) -> list[int]:
        """Apply the model's chat template and request an assistant continuation."""
        self._validate(request)
        messages = [self._message_payload(message) for message in request.messages]
        tools = [self._tool_payload(tool) for tool in request.tools] or None
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        return self._token_ids(rendered)

    def parse(self, token_ids: Sequence[int], *, reached_limit: bool = False) -> ModelResponse:
        """Decode generated tokens and apply the configured tool-call parser."""
        ids = [int(token_id) for token_id in token_ids]
        decoded = self.decode(ids)
        content = decoded
        tool_calls: tuple[ToolCall, ...] = ()
        if self.tool_call_parser is not None:
            content, parsed = self.tool_call_parser.parse(decoded)
            tool_calls = tuple(parsed)
        if reached_limit:
            stop_reason = "length"
        elif tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "stop"
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            token_ids=tuple(ids),
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated tokens without interpreting model-specific tool syntax."""
        decoded = self.tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=True,
        )
        if not isinstance(decoded, str):
            raise TypeError("tokenizer must decode one token sequence to a string")
        return decoded

    def stream_text(self, token_ids: Sequence[int]) -> str:
        """Decode a stable visible prefix while withholding tool syntax and split Unicode."""
        decoded = self.decode(token_ids)
        decoded = decoded.rstrip("\ufffd")
        if self.tool_call_parser is None:
            return decoded
        if isinstance(self.tool_call_parser, StreamingToolCallParser):
            return self.tool_call_parser.visible_text_prefix(decoded)
        # A final-only parser cannot prove that a partial prefix is ordinary text.
        return ""

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict[str, object]:
        """Convert one normalized message to chat-template conventions."""
        payload: dict[str, object] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call.model_name,
                        "arguments": dict(call.arguments),
                    },
                    "id": call.call_id,
                }
                for call in message.tool_calls
            ]
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                payload["name"] = message.name
        return payload

    @staticmethod
    def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
        """Convert one normalized tool to the chat template's function schema.

        Hugging Face chat templates generally understand JSON-schema functions, not
        Responses free-form tools. A custom tool is therefore rendered as a function
        with one string ``input`` argument. The completion engine restores its custom
        identity before the wire adapter emits the call.
        """
        schema = dict(tool.input_schema)
        description = tool.description
        if tool.kind == "custom":
            description = (
                f"{description}\nReturn the free-form tool input in the `input` string."
            ).strip()
        return {
            "type": "function",
            "function": {
                "name": tool.model_name,
                "description": description,
                "parameters": schema,
            },
        }

    @staticmethod
    def _token_ids(rendered: object) -> list[int]:
        """Normalize tokenizer list, mapping, or tensor output."""
        if isinstance(rendered, Tensor):
            values = rendered.tolist()
        elif isinstance(rendered, Mapping):
            values = rendered.get("input_ids")
            if isinstance(values, Tensor):
                values = values.tolist()
        else:
            values = rendered
        if isinstance(values, list) and values and isinstance(values[0], list):
            if len(values) != 1:
                raise ValueError("chat codec supports one request at a time")
            values = values[0]
        if not isinstance(values, list):
            raise TypeError("chat template must return a token-ID list or tensor")
        token_ids: list[int] = []
        for token_id in values:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("chat template must return a token-ID list or tensor")
            token_ids.append(token_id)
        return token_ids
