"""Qwen tool-call parsing and streaming visibility tests."""

from __future__ import annotations

from itertools import pairwise

import pytest

from kvpacket.serve.codecs import QwenToolCallParser


def test_qwen_parser_extracts_multiple_calls_and_keeps_text() -> None:
    parser = QwenToolCallParser()
    text = (
        "I will inspect both.\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>\n'
        '<tool_call>{"id":"given","name":"read_file",'
        '"arguments":"{\\"path\\":\\"b.py\\"}"}</tool_call>'
    )

    content, calls = parser.parse(text)

    assert content == "I will inspect both."
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert calls[0].call_id.startswith("call_")
    assert calls[1].call_id == "given"
    assert calls[1].arguments == {"path": "b.py"}


def test_qwen_parser_withholds_partial_and_complete_tool_markers() -> None:
    parser = QwenToolCallParser()

    assert parser.visible_text_prefix("Answer<tool_") == "Answer"
    assert parser.visible_text_prefix("Answer<tool_call>{") == "Answer"
    assert parser.visible_text_prefix("ordinary text") == "ordinary text"


def test_qwen_parser_keeps_streamed_prefix_monotonic_around_whitespace() -> None:
    parser = QwenToolCallParser()
    prefixes = (
        "\n",
        "\nI will inspect ",
        "\nI will inspect now.\n<tool_",
        "\nI will inspect now.\n<tool_call>{",
    )
    visible = [parser.visible_text_prefix(prefix) for prefix in prefixes]

    assert visible == ["", "I will inspect", "I will inspect now.", "I will inspect now."]
    assert all(current.startswith(previous) for previous, current in pairwise(visible))

    text = (
        "\nI will inspect now.\n"
        '<tool_call>{"name":"search_text","arguments":{"query":"x"}}</tool_call>\n'
    )
    content, calls = parser.parse(text)
    assert content == visible[-1]
    assert calls[0].name == "search_text"


def test_qwen_parser_rejects_invalid_complete_calls() -> None:
    parser = QwenToolCallParser()

    with pytest.raises(ValueError, match="invalid tool-call JSON"):
        parser.parse("<tool_call>{bad}</tool_call>")
    with pytest.raises(ValueError, match="non-empty name"):
        parser.parse('<tool_call>{"arguments":{}}</tool_call>')
