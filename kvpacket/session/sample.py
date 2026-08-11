"""Represent and persist teacher trajectories recorded by ``PacketSession``.

A training sample stores the complete prompt-and-continuation token trajectory, marks
the generated targets, and records every reusable-source occurrence. Wrapper routing is
trainer configuration and is deliberately absent from the teacher data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

from .._internal.persistence import atomic_torch_save
from .._internal.validation import string_mapping

__all__ = ["TRAINING_SAMPLE_FORMAT", "TrainingSample"]

TRAINING_SAMPLE_FORMAT = "kvpacket.training_sample/1"


@dataclass
class TrainingSample:
    """Recorded teacher trajectory and packet reconstruction metadata.

    Attributes:
        token_ids: Complete prompt and continuation shaped ``(trajectory_tokens,)``.
        generated_mask: Boolean mask selecting generated target positions.
        chunk_spans: Half-open token spans for each matched reusable chunk.
        logits: Optional teacher logits shaped ``(generated_tokens, vocabulary)``.
    """

    token_ids: Tensor
    generated_mask: Tensor
    chunk_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    logits: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate trajectory tensors, source spans, and teacher logits."""
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a tensor")
        if not isinstance(self.generated_mask, Tensor):
            raise TypeError("generated_mask must be a tensor")
        if self.token_ids.dim() != 1:
            raise ValueError("token_ids must be 1-D [T]")
        if self.token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")
        if self.generated_mask.shape != self.token_ids.shape:
            raise ValueError("generated_mask must match token_ids shape")
        if self.generated_mask.dtype != torch.bool:
            raise ValueError("generated_mask must have dtype torch.bool")

        spans: dict[str, list[tuple[int, int]]] = {}
        for chunk_id, occurrences in self.chunk_spans.items():
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("chunk IDs must be non-empty strings")
            checked: list[tuple[int, int]] = []
            for span in occurrences:
                if not (
                    isinstance(span, (tuple, list))
                    and len(span) == 2
                    and all(isinstance(position, int) for position in span)
                ):
                    raise ValueError(f"chunk {chunk_id!r} contains a malformed span")
                start, end = int(span[0]), int(span[1])
                if not 0 <= start < end <= self.length:
                    raise ValueError(f"chunk {chunk_id!r} span {(start, end)} is out of range")
                checked.append((start, end))
            if not checked:
                raise ValueError(f"chunk {chunk_id!r} must contain at least one span")
            spans[chunk_id] = checked
        self.chunk_spans = spans

        if self.logits is not None:
            if self.logits.dim() != 2:
                raise ValueError("logits must be 2-D [G, V]")
            if self.logits.shape[0] != self.num_generated:
                raise ValueError("logits first dimension must equal generated token count")

    @property
    def length(self) -> int:
        """Total number of prompt and continuation tokens."""
        return int(self.token_ids.shape[0])

    @property
    def num_generated(self) -> int:
        """Number of positions selected as training targets."""
        return int(self.generated_mask.sum())

    def to(self, device: torch.device | str) -> "TrainingSample":
        """Copy trajectory and teacher tensors to a device."""
        return TrainingSample(
            token_ids=self.token_ids.to(device),
            generated_mask=self.generated_mask.to(device),
            chunk_spans={key: list(value) for key, value in self.chunk_spans.items()},
            logits=None if self.logits is None else self.logits.to(device),
        )

    def save(self, path: str) -> None:
        """Atomically save a CPU copy of the versioned training sample."""
        atomic_torch_save(
            {
                "format": TRAINING_SAMPLE_FORMAT,
                "token_ids": self.token_ids.cpu(),
                "generated_mask": self.generated_mask.cpu(),
                "chunk_spans": self.chunk_spans,
                "logits": None if self.logits is None else self.logits.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TrainingSample":
        """Load and validate a training sample on CPU.

        Args:
            path: Checkpoint path produced by ``save``.

        Returns:
            Validated CPU-resident sample.

        Raises:
            ValueError: If the format, fields, tensors, spans, or wrapper metadata are
                invalid.
        """
        raw = torch.load(path, map_location="cpu", weights_only=True)
        data = string_mapping(raw, where="training sample")
        if data.get("format") != TRAINING_SAMPLE_FORMAT:
            raise ValueError(
                f"unrecognized training sample format {data.get('format')!r}; "
                f"expected {TRAINING_SAMPLE_FORMAT!r}"
            )

        fields = {"format", "token_ids", "generated_mask", "chunk_spans", "logits"}
        missing = fields - set(data)
        unknown = set(data) - fields
        if missing or unknown:
            raise ValueError(
                f"invalid training sample keys: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(
            token_ids=cast(Tensor, data["token_ids"]),
            generated_mask=cast(Tensor, data["generated_mask"]),
            chunk_spans=cast(dict[str, list[tuple[int, int]]], data["chunk_spans"]),
            logits=cast(Tensor | None, data["logits"]),
        )

    def __repr__(self) -> str:
        return (
            f"TrainingSample(len={self.length}, generated={self.num_generated}, "
            f"chunks={{{', '.join(f'{key}:{len(value)}' for key, value in self.chunk_spans.items())}}}, "
            f"has_logits={self.logits is not None})"
        )
