"""Resolve reusable sources into framework-owned model input representations.

Source resolution and tokenization are independent of model execution. This module
therefore owns file reading, media-type validation, tokenization, matching metadata,
and source and encoded-input fingerprints. Custom encoders return an ``EncodedBody``;
the framework derives persistent identity from that value before materialization.
"""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from ..errors import SourceConfigError, UnsupportedModalityError
from .sources import Tokenizable

__all__ = [
    "EncodedBody",
    "SourceEncoder",
    "TextSourceEncoder",
    "encode_pretokenized",
]

_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_TEXT_APPLICATION_TYPES = {
    "application/json",
    "application/toml",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
}


@dataclass(frozen=True)
class EncodedBody:
    """Source content and its model-input representation.

    Exactly one of ``input_ids`` and ``inputs_embeds`` is present. ``token_ids`` and
    ``match_text`` are optional matching representations retained for online sessions.
    The framework fingerprints ``content`` and the selected input tensor; custom
    encoders do not construct persistence metadata.

    Attributes:
        content: Exact source bytes represented by this encoding.
        media_type: Resolved source content type.
        input_ids: Optional token IDs shaped ``(1, sequence)``.
        inputs_embeds: Optional embeddings shaped ``(1, sequence, hidden_size)``.
        token_ids: Optional immutable token sequence used for token matching.
        match_text: Optional decoded source text used for string matching.
    """

    content: bytes
    media_type: str
    input_ids: Tensor | None = None
    inputs_embeds: Tensor | None = None
    token_ids: tuple[int, ...] | None = None
    match_text: str | None = None

    def __post_init__(self) -> None:
        """Validate source bytes and the single-example encoded representation."""
        if not isinstance(self.content, bytes):
            raise TypeError("encoded body content must be bytes")
        if not self.media_type:
            raise ValueError("media_type must be non-empty")
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        body = self.input_ids if self.input_ids is not None else self.inputs_embeds
        assert body is not None
        if body.dim() not in (2, 3) or body.shape[0] != 1:
            raise ValueError("encoded body tensor must have batch size one")
        if int(body.shape[1]) <= 0:
            raise ValueError("encoded body sequence must not be empty")
        if self.token_ids is not None and len(self.token_ids) != int(body.shape[1]):
            raise ValueError("token_ids length must match encoded body")
        if self.match_text is not None and not isinstance(self.match_text, str):
            raise TypeError("match_text must be a string or None")

    @property
    def body_len(self) -> int:
        """Sequence length before packet wrapper tokens are inserted."""
        body = self.input_ids if self.input_ids is not None else self.inputs_embeds
        assert body is not None
        return int(body.shape[1])

    @property
    def encoded_input(self) -> Tensor:
        """Exact model-input tensor used to materialize the packet."""
        body = self.input_ids if self.input_ids is not None else self.inputs_embeds
        assert body is not None
        return body


@runtime_checkable
class SourceEncoder(Protocol):
    """Framework extension point for converting declared sources into packet bodies."""

    def encode(self, source: Tokenizable) -> EncodedBody:
        """Resolve and encode one reusable source."""
        ...


class TextSourceEncoder:
    """Encode raw text and local UTF-8 files with a callable tokenizer.

    Args:
        tokenizer: Callable returning an ``input_ids`` field for one input string.

    Raises:
        TypeError: If the tokenizer is not callable.
    """

    def __init__(self, tokenizer: object) -> None:
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        self.tokenizer = tokenizer

    @staticmethod
    def _resolve_text(source: Tokenizable) -> tuple[bytes, str, str]:
        """Resolve source bytes, decoded text, and media type."""
        if source.kind == "raw-text":
            content = source.value.encode("utf-8")
            text = source.value
            media_type = source.media_type or "text/plain"
        elif source.kind == "file":
            if _URI_RE.match(source.value):
                raise UnsupportedModalityError(
                    f"no URI resolver is configured for source {source.source_id!r}"
                )
            path = Path(source.value)
            try:
                content = path.read_bytes()
            except OSError as error:
                raise SourceConfigError(
                    f"cannot read source {source.source_id!r} at {path}: {error}"
                ) from error
            guessed_type = mimetypes.guess_type(path.name)[0]
            media_type = source.media_type or guessed_type or "application/octet-stream"
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise SourceConfigError(
                    f"text source {source.source_id!r} is not valid UTF-8"
                ) from error
        else:
            raise UnsupportedModalityError(
                f"TextSourceEncoder does not support source kind {source.kind!r}"
            )

        if not (media_type.startswith("text/") or media_type in _TEXT_APPLICATION_TYPES):
            raise UnsupportedModalityError(
                f"TextSourceEncoder does not support media type {media_type!r} for "
                f"source {source.source_id!r}"
            )
        return content, text, media_type

    def encode(self, source: Tokenizable) -> EncodedBody:
        """Tokenize one raw-text or local UTF-8 file source.

        The framework later fingerprints source bytes and token IDs separately so
        changes before and after tokenization remain distinguishable.

        Args:
            source: Typed reusable source declaration.

        Returns:
            Single-example token IDs with matching and identity metadata.

        Raises:
            TypeError: If ``source`` is not a ``Tokenizable``.
            SourceConfigError: If a local source cannot be read or encodes to no tokens.
            UnsupportedModalityError: If the source is remote, non-text, or the tokenizer
                does not return token IDs.
        """
        if not isinstance(source, Tokenizable):
            raise TypeError("source must be a Tokenizable")
        content, text, media_type = self._resolve_text(source)
        tokenizer_call = cast(Callable[..., object], self.tokenizer)
        raw_encoded = tokenizer_call(text, return_tensors="pt", add_special_tokens=False)
        if not isinstance(raw_encoded, Mapping) or "input_ids" not in raw_encoded:
            raise UnsupportedModalityError(
                f"tokenizer did not return input_ids for source {source.source_id!r}"
            )
        encoded = cast(Mapping[object, object], raw_encoded)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"tokenizer must return one input_ids sequence for source {source.source_id!r}"
            )
        if input_ids.shape[1] == 0:
            raise SourceConfigError(
                f"source {source.source_id!r} encodes to an empty token sequence"
            )
        token_ids = tuple(int(token) for token in input_ids[0].tolist())
        return EncodedBody(
            content=content,
            media_type=media_type,
            input_ids=input_ids,
            token_ids=token_ids,
            match_text=text,
        )


def encode_pretokenized(source: Tokenizable, token_ids: list[int]) -> EncodedBody:
    """Pair original text content with caller-provided token IDs.

    Args:
        source: Original raw-text or local-file source.
        token_ids: Non-empty tokenization of ``source``.

    Returns:
        Pretokenized source body with original-content and encoded-input identities.

    Raises:
        TypeError: If ``source`` is not a ``Tokenizable``.
        ValueError: If the token sequence is empty.
        SourceConfigError: If a local source cannot be read.
        UnsupportedModalityError: If the source is remote or non-text.
    """
    if not isinstance(source, Tokenizable):
        raise TypeError("source must be a Tokenizable")
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    content, text, media_type = TextSourceEncoder._resolve_text(source)
    ids = torch.tensor([token_ids], dtype=torch.long)
    return EncodedBody(
        content=content,
        media_type=media_type,
        input_ids=ids,
        token_ids=tuple(token_ids),
        match_text=text,
    )
