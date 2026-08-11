"""Verify provider payload normalization independently of model execution."""

from __future__ import annotations

import pytest
from typing import cast

from kvpacket.serve import ModelResponse, TokenUsage, ToolCall
from kvpacket.serve.adapters import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
    WireFormatError,
)


def test_openai_responses_request_normalization() -> None:
    adapter = OpenAIResponsesAdapter()
    request = adapter.parse_request(
        {
            "model": "local-model",
            "instructions": "Be concise.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Weather?"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": '{"city":"Berlin"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Sunny",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
                {"type": "web_search"},
            ],
            "max_output_tokens": 32,
            "temperature": 0,
            "stream": True,
        }
    )

    assert request.model == "local-model"
    assert request.max_output_tokens == 32
    assert request.stream is True
    assert request.sampling.do_sample is False
    assert [message.role for message in request.input.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert request.input.messages[2].tool_calls[0].arguments == {"city": "Berlin"}
    assert request.input.messages[3].tool_call_id == "call_1"
    assert request.input.tools[0].input_schema["type"] == "object"


def test_openai_responses_formats_function_calls() -> None:
    adapter = OpenAIResponsesAdapter()
    response = ModelResponse(
        content="Checking.",
        tool_calls=(ToolCall("call_1", "weather", {"city": "Berlin"}),),
        stop_reason="tool_use",
    )
    payload = adapter.format_response(
        response,
        response_id="resp_1",
        model="local-model",
        usage=TokenUsage(12, 4),
    )

    assert payload["object"] == "response"
    assert payload["usage"] == {
        "input_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 16,
    }
    output = cast(list[dict[str, object]], payload["output"])
    assert [item["type"] for item in output] == ["message", "function_call"]

def test_openai_responses_preserves_custom_tools_and_transcript_items() -> None:
    adapter = OpenAIResponsesAdapter()
    request = adapter.parse_request(
        {
            "model": "local-model",
            "input": [
                {"role": "user", "content": "Edit the file."},
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "output": "Done!",
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
            ],
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/",
                    },
                }
            ],
        }
    )

    tool = request.input.tools[0]
    assert tool.kind == "custom"
    assert tool.input_schema["required"] == ["input"]
    assert tool.custom_format == {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/",
    }
    assert len(request.input.tools) == 1
    assert [message.role for message in request.input.messages] == ["user", "assistant", "tool"]
    call = request.input.messages[1].tool_calls[0]
    assert call.kind == "custom"
    assert call.raw_input == "*** Begin Patch\n*** End Patch"

    response = ModelResponse(
        tool_calls=(
            ToolCall(
                "call_patch_2",
                "apply_patch",
                {"input": "*** Begin Patch\n*** End Patch"},
                kind="custom",
                raw_input="*** Begin Patch\n*** End Patch",
            ),
        ),
        stop_reason="tool_use",
    )
    payload = adapter.format_response(
        response,
        response_id="resp_custom",
        model="local-model",
    )
    output = cast(list[dict[str, object]], payload["output"])
    assert output[0]["type"] == "custom_tool_call"
    assert output[0]["input"] == "*** Begin Patch\n*** End Patch"

def test_openai_responses_flattens_and_restores_tool_namespaces() -> None:
    adapter = OpenAIResponsesAdapter()
    request = adapter.parse_request(
        {
            "model": "local-model",
            "input": "Inspect the repository.",
            "tools": [
                {
                    "type": "namespace",
                    "name": "workspace",
                    "description": "Workspace tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read_file",
                            "description": "Read a file.",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        }
                    ],
                }
            ],
        }
    )

    tool = request.input.tools[0]
    assert tool.name == "read_file"
    assert tool.namespace == "workspace"
    assert tool.model_name == "workspace__read_file"

    response = ModelResponse(
        tool_calls=(
            ToolCall(
                "call_read",
                "read_file",
                {"path": "README.md"},
                namespace="workspace",
            ),
        ),
        stop_reason="tool_use",
    )
    payload = adapter.format_response(
        response,
        response_id="resp_namespace",
        model="local-model",
    )
    output = cast(list[dict[str, object]], payload["output"])
    assert output[0]["name"] == "read_file"
    assert output[0]["namespace"] == "workspace"


def test_openai_responses_rejects_implicit_history_and_images() -> None:
    adapter = OpenAIResponsesAdapter()
    with pytest.raises(WireFormatError, match="response-history store"):
        adapter.parse_request({"model": "local", "previous_response_id": "resp_0", "input": "next"})
    with pytest.raises(WireFormatError, match="only text"):
        adapter.parse_request(
            {
                "model": "local",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "data:..."}],
                    }
                ],
            }
        )


def test_anthropic_messages_request_normalization() -> None:
    adapter = AnthropicMessagesAdapter()
    request = adapter.parse_request(
        {
            "model": "local-model",
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "weather",
                            "input": {"city": "Berlin"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "Sunny"}],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                }
            ],
            "max_tokens": 64,
            "temperature": 0.5,
            "top_k": 20,
            "stop_sequences": ["END"],
        }
    )

    assert [message.role for message in request.input.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert request.input.messages[2].tool_calls[0].name == "weather"
    assert request.input.messages[3].content == "Sunny"
    assert request.max_output_tokens == 64
    assert request.sampling.top_k == 20
    assert request.stop_sequences == ("END",)


def test_anthropic_messages_formats_response() -> None:
    adapter = AnthropicMessagesAdapter()
    response = ModelResponse(
        content="Checking.",
        tool_calls=(ToolCall("toolu_1", "weather", {"city": "Berlin"}),),
        stop_reason="tool_use",
    )
    payload = adapter.format_response(
        response,
        response_id="msg_1",
        model="local-model",
        usage=TokenUsage(10, 3),
    )

    assert payload["type"] == "message"
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"] == [
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "weather",
            "input": {"city": "Berlin"},
        },
    ]
