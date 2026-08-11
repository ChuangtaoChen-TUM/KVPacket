"""Tokenizer-driven prompt templates for multi-document evaluation.

Chat scaffolding is split around a placeholder: everything before the reusable
documents becomes ``preamble`` and everything after them becomes ``task_prompt``.
The documents themselves remain untouched, so each can still be cached as an
independent packet.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import RetrievalSample

__all__ = [
    "TokenizerType",
    "apply_chat_template",
]

TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast
_DOCUMENT_PLACEHOLDER = "{DOCUMENT_PLACEHOLDER}"


def apply_chat_template(
    sample: RetrievalSample,
    tokenizer: TokenizerType,
    system_prompt: str = "",
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> RetrievalSample:
    """Apply the tokenizer's native chat template without joining the documents.

    The placeholder occupies the documents' position in the user message. Splitting
    the rendered prompt at that placeholder preserves the engine layout
    ``[preamble][document packets...][task_prompt]`` while following the exact chat
    format shipped with the selected model/tokenizer.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _DOCUMENT_PLACEHOLDER})

    extra_kwargs = dict(chat_template_kwargs or {})
    reserved = {"tokenize", "add_generation_prompt"} & extra_kwargs.keys()
    if reserved:
        raise ValueError(
            "chat_template_kwargs must not override reserved arguments: "
            f"{sorted(reserved)}"
        )
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra_kwargs,
    )
    if not isinstance(formatted, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return str")

    try:
        pivot = formatted.index(_DOCUMENT_PLACEHOLDER)
    except ValueError as exc:
        raise ValueError(
            "the tokenizer chat template did not preserve the document placeholder"
        ) from exc

    preamble = formatted[:pivot] + sample.preamble
    task_prompt = sample.task_prompt + formatted[pivot + len(_DOCUMENT_PLACEHOLDER) :]
    return replace(sample, preamble=preamble, task_prompt=task_prompt)
