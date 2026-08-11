"""Provide framework-owned token selection and incremental decoding state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .types import PacketCacheState

__all__ = [
    "SamplingConfig",
    "PacketDecodeState",
    "PacketDecodeStep",
    "select_next_token",
]


@dataclass(frozen=True)
class SamplingConfig:
    """Greedy or stochastic first-token and continuation selection.

    Attributes:
        do_sample: Whether to sample instead of selecting the maximum logit.
        temperature: Positive sampling temperature.
        top_p: Cumulative-probability threshold in ``(0, 1]``.
        top_k: Maximum number of candidates; zero disables this filter.
        min_p: Minimum probability relative to the most likely token.
    """

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    def __post_init__(self) -> None:
        """Validate sampling bounds before any model work begins."""
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if not 0 < self.top_p <= 1 or not math.isfinite(self.top_p):
            raise ValueError("top_p must be finite and in (0, 1]")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0 <= self.min_p <= 1 or not math.isfinite(self.min_p):
            raise ValueError("min_p must be finite and in [0, 1]")


@dataclass(frozen=True)
class PacketDecodeState:
    """Caller-owned cache and logits preceding the next generated token."""

    cache_state: PacketCacheState
    next_token_logits: Tensor


@dataclass(frozen=True)
class PacketDecodeStep:
    """One selected token and the state after committing it to the cache."""

    token_id: int
    state: PacketDecodeState


def select_next_token(logits: Tensor, sampling: SamplingConfig) -> int:
    """Select one token using greedy decoding or standard sampling filters."""
    if not sampling.do_sample:
        return int(logits.argmax(dim=-1))

    scores = logits.float() / sampling.temperature
    vocab_size = int(scores.shape[-1])
    if 0 < sampling.top_k < vocab_size:
        threshold = torch.topk(scores, sampling.top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -float("inf"))
    if sampling.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative > sampling.top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, -float("inf"))
    if sampling.min_p > 0.0:
        threshold = scores.amax(dim=-1, keepdim=True) + math.log(sampling.min_p)
        scores = scores.masked_fill(scores < threshold, -float("inf"))

    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())
