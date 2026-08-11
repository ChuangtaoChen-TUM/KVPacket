"""Define caller-owned state and results for packet-aware model forwards."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor

from ..kvcache.cache import KVCache
from .matching import Match

__all__ = [
    "MixedPrefillPolicy",
    "PacketCacheState",
    "PacketProcessResult",
    "PacketPrefillResult",
    "PacketTokenForwardResult",
]


class MixedPrefillPolicy(Enum):
    """Strategy for an uncached span followed by another cached packet.

    ``SEQUENTIAL`` forwards each uncached span before appending a logically later
    packet. It minimizes transformer work but may require multiple serial forwards.
    ``FUSED`` appends packet caches once and forwards all online tokens together under
    an explicit logical-causality mask. It trades extra dense attention work for fewer
    online forwards.
    """

    SEQUENTIAL = "sequential"
    FUSED = "fused"


@dataclass(frozen=True)
class PacketCacheState:
    """Caller-owned cache and the next logical model position.

    ``next_position`` is intentionally independent of the cache's physical KV length.
    Compressed packets retain fewer KV slots while still occupying their original
    logical span.

    Attributes:
        cache: Committed backend-independent cache, or ``None`` for an empty prefix.
        next_position: Logical position assigned to the next forwarded token.
    """

    cache: KVCache | None = None
    next_position: int = 0

    def __post_init__(self) -> None:
        """Validate the logical position counter."""
        if isinstance(self.next_position, bool) or not isinstance(self.next_position, int):
            raise TypeError("next_position must be an integer")
        if self.next_position < 0:
            raise ValueError("next_position must be non-negative")


@dataclass(frozen=True)
class PacketProcessResult:
    """Result of committing packet-aware input without requesting logits.

    Attributes:
        cache_state: Extended caller-owned cache state.
        cover: Packet matches substituted during the operation.
        rerotation_flops: Analytic FLOPs spent shifting cached packet keys.
    """

    cache_state: PacketCacheState
    cover: list[Match] = field(default_factory=list)
    rerotation_flops: int = 0


@dataclass(frozen=True)
class PacketPrefillResult:
    """Result of packet-aware prompt prefill.

    Attributes:
        cache_state: Cache including the supplied prompt.
        first_token_logits: Logits used to select the first continuation token.
        prompt_logits: Optional logits for every trailing prompt position.
        prompt_len: Number of trailing unmatched prompt tokens.
        cover: Packet matches substituted during the operation.
        rerotation_flops: Analytic FLOPs spent shifting cached packet keys.
    """

    cache_state: PacketCacheState
    first_token_logits: Tensor
    prompt_logits: Tensor | None
    prompt_len: int
    cover: list[Match] = field(default_factory=list)
    rerotation_flops: int = 0


@dataclass(frozen=True)
class PacketTokenForwardResult:
    """Cache state and logits after forwarding explicit online tokens.

    Attributes:
        cache_state: Cache including the supplied tokens.
        logits: Model logits for each supplied token position.
    """

    cache_state: PacketCacheState
    logits: Tensor
