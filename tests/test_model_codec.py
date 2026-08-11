"""Provider-neutral model-codec rendering and parsing tests."""

from __future__ import annotations

from typing import cast

import pytest

from kvpacket.serve import (
    ChatMessage,
    HFChatCodec,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered: tuple[list[dict[str, object]], dict[str, object]] | None = None

    def apply_chat_template(self, conversation, **kwargs):
        self.rendered = (conversation, kwargs)
        return [[10, 11, 12]]

    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens
        return "call calculator"


class _Parser:
    def parse(self, text):
        assert text == "call calculator"
        return "", [ToolCall("call-1", "calculator", {"expression": "2+2"})]


def test_hf_chat_codec_owns_rendering_and_tool_parsing() -> None:
    tokenizer = _Tokenizer()
    codec = HFChatCodec(tokenizer, tool_call_parser=_Parser())
    request = ModelRequest(
        messages=(ChatMessage("system", "Be useful."), ChatMessage("user", "2+2?")),
        tools=(
            ToolDefinition(
                "calculator",
                "Evaluate an expression.",
                {"type": "object", "properties": {"expression": {"type": "string"}}},
            ),
        ),
    )

    assert codec.render(request) == [10, 11, 12]
    assert tokenizer.rendered is not None
    messages, options = tokenizer.rendered
    assert messages == [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "2+2?"},
    ]
    tools = options["tools"]
    assert isinstance(tools, list)
    tool_payload = cast(dict[str, object], tools[0])
    function = tool_payload["function"]
    function = cast(dict[str, object], function)
    assert function["name"] == "calculator"
    assert options["add_generation_prompt"] is True

    response = codec.parse([20, 21])
    assert response.content == ""
    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].arguments == {"expression": "2+2"}
    assert codec.stream_text([20]) == ""


def test_hf_chat_codec_round_trips_assistant_and_tool_messages() -> None:
    tokenizer = _Tokenizer()
    codec = HFChatCodec(tokenizer, tool_call_parser=_Parser())
    call = ToolCall("call-1", "lookup", {"query": "x"})
    request = ModelRequest(
        messages=(
            ChatMessage("assistant", tool_calls=(call,)),
            ChatMessage("tool", "result", tool_call_id="call-1", name="lookup"),
        )
    )

    codec.render(request)
    assert tokenizer.rendered is not None
    messages, _ = tokenizer.rendered
    tool_calls = messages[0]["tool_calls"]
    assert isinstance(tool_calls, list)
    call_payload = cast(dict[str, object], tool_calls[0])
    function = call_payload["function"]
    function = cast(dict[str, object], function)
    assert function["arguments"] == {"query": "x"}
    assert messages[1] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-1",
        "name": "lookup",
    }
    assert codec.parse([20], reached_limit=True).stop_reason == "length"
    assert codec.stream_text([20]) == ""


def test_hf_chat_codec_rejects_tools_without_a_model_family_parser() -> None:
    codec = HFChatCodec(_Tokenizer())
    request = ModelRequest(
        messages=(ChatMessage("user", "2+2?"),),
        tools=(
            ToolDefinition(
                "calculator",
                "Evaluate an expression.",
                {"type": "object", "properties": {}},
            ),
        ),
    )

    with pytest.raises(ValueError, match="no tool-call support"):
        codec.render(request)
