"""Translate Anthropic Messages payloads to the provider-neutral serving API."""

from __future__ import annotations

from collections.abc import Mapping

from ...runtime import SamplingConfig
from ..types import (
    ChatMessage,
    InferenceRequest,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from ._parsing import (
    boolean,
    integer,
    mapping,
    number,
    sequence,
    string,
    tool_definition,
)
from .base import WireFormatError

__all__ = ["AnthropicMessagesAdapter"]


class AnthropicMessagesAdapter:
    """Pure dictionary adapter for Anthropic Messages requests and SSE data objects."""

    def parse_request(self, payload: Mapping[str, object]) -> InferenceRequest:
        """Normalize text/tool blocks, system instructions, and generation options."""
        model = string(payload.get("model"), where="model")
        messages: list[ChatMessage] = []
        system = payload.get("system")
        if system is not None:
            messages.append(ChatMessage("system", self._text_content(system, "system")))
        raw_messages = sequence(payload.get("messages"), where="messages")
        for index, raw in enumerate(raw_messages):
            messages.extend(self._message(raw, where=f"messages[{index}]"))

        tools = tuple(
            tool_definition(item, where=f"tools[{index}]", nested_function=False)
            for index, item in enumerate(sequence(payload.get("tools", []), where="tools"))
        )
        stop_value = payload.get("stop_sequences", [])
        stops = tuple(
            string(item, where=f"stop_sequences[{index}]")
            for index, item in enumerate(sequence(stop_value, where="stop_sequences"))
        )
        return InferenceRequest(
            model=model,
            input=ModelRequest(tuple(messages), tools),
            max_output_tokens=integer(payload.get("max_tokens"), where="max_tokens"),
            sampling=self._sampling(payload),
            stream=boolean(payload.get("stream", False), where="stream"),
            stop_sequences=stops,
        )

    def format_response(
        self,
        response: ModelResponse,
        *,
        response_id: str,
        model: str,
        usage: TokenUsage = TokenUsage(),
    ) -> dict[str, object]:
        """Build one completed Anthropic message."""
        content: list[dict[str, object]] = []
        if response.content or not response.tool_calls:
            content.append({"type": "text", "text": response.content})
        content.extend(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": dict(call.arguments),
            }
            for call in response.tool_calls
        )
        return {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": self._stop_reason(response),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }

    def _message(self, value: object, *, where: str) -> list[ChatMessage]:
        """Expand one Anthropic message into ordered normalized messages."""
        item = mapping(value, where=where)
        role = string(item.get("role"), where=f"{where}.role")
        if role not in ("user", "assistant"):
            raise WireFormatError(f"{where}.role is unsupported: {role!r}")
        content = item.get("content")
        if isinstance(content, str):
            return [ChatMessage(role, content)]  # type: ignore[arg-type]

        result: list[ChatMessage] = []
        pending_text: list[str] = []
        pending_calls: list[ToolCall] = []

        def flush() -> None:
            if not pending_text and not pending_calls:
                return
            result.append(
                ChatMessage(
                    role,  # type: ignore[arg-type]
                    "".join(pending_text),
                    tool_calls=tuple(pending_calls),
                )
            )
            pending_text.clear()
            pending_calls.clear()

        for index, raw in enumerate(sequence(content, where=f"{where}.content")):
            block_where = f"{where}.content[{index}]"
            block = mapping(raw, where=block_where)
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(string(block.get("text"), where=f"{block_where}.text"))
            elif block_type == "tool_use" and role == "assistant":
                pending_calls.append(
                    ToolCall(
                        string(block.get("id"), where=f"{block_where}.id"),
                        string(block.get("name"), where=f"{block_where}.name"),
                        {
                            str(key): item
                            for key, item in mapping(
                                block.get("input", {}), where=f"{block_where}.input"
                            ).items()
                        },
                    )
                )
            elif block_type == "tool_result" and role == "user":
                flush()
                result.append(
                    ChatMessage(
                        "tool",
                        self._text_content(block.get("content", ""), f"{block_where}.content"),
                        tool_call_id=string(
                            block.get("tool_use_id"), where=f"{block_where}.tool_use_id"
                        ),
                    )
                )
            else:
                raise WireFormatError(
                    f"{block_where}.type is unsupported for role {role!r}: {block_type!r}"
                )
        flush()
        return result

    @staticmethod
    def _text_content(value: object, where: str) -> str:
        """Join Anthropic text blocks and reject non-text system/tool results."""
        if isinstance(value, str):
            return value
        texts: list[str] = []
        for index, raw in enumerate(sequence(value, where=where)):
            block = mapping(raw, where=f"{where}[{index}]")
            if block.get("type") != "text":
                raise WireFormatError(f"{where}[{index}] must be a text block")
            texts.append(string(block.get("text"), where=f"{where}[{index}].text"))
        return "".join(texts)

    @staticmethod
    def _sampling(payload: Mapping[str, object]) -> SamplingConfig:
        """Map Anthropic sampling fields to framework token selection."""
        temperature = number(payload.get("temperature", 1.0), where="temperature")
        top_p = number(payload.get("top_p", 1.0), where="top_p")
        top_k = integer(payload.get("top_k", 0), where="top_k")
        if temperature < 0:
            raise WireFormatError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise WireFormatError("top_p must be in (0, 1]")
        return SamplingConfig(
            do_sample=temperature > 0,
            temperature=temperature or 1.0,
            top_p=top_p,
            top_k=top_k,
        )

    @staticmethod
    def _stop_reason(response: ModelResponse) -> str:
        """Map normalized stop reasons to Anthropic values."""
        return {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_use": "tool_use",
        }[response.stop_reason]
