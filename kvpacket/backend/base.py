"""Define framework-neutral model execution and packet-storage interfaces.

The session engine works exclusively with KV Packet types rather than framework-native
cache objects. ``ComputeBackend`` owns online forwards and cache repositioning,
``PacketBuildBackend`` adds embedding and offline prefill, ``GenerationBackend`` adds
optional backend-native continuation, and ``TrainingBackend`` adds differentiable
adapter training. ``CacheStore`` exposes materialized reusable packets. Compatibility
metadata lets these components reject caches and wrappers produced for a different
model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from .._internal.validation import string_mapping
from ..kvcache.cache import KVCache
from ..fingerprint import fingerprint_json, is_fingerprint

__all__ = [
    "RerotationError",
    "BackendCompatibility",
    "ForwardOutput",
    "BackendGenerationOutput",
    "GenerationBackend",
    "ComputeBackend",
    "PacketBuildBackend",
    "TrainingBackend",
    "Packet",
    "CacheStore",
]


class RerotationError(RuntimeError):
    """Raised when a model's positional encoding cannot be re-rotated safely."""


@dataclass(frozen=True)
class BackendCompatibility:
    """Stable identity and geometry of a model execution backend.

    Attributes:
        backend: Backend implementation family, such as ``"hf"``.
        model_id: Human-readable model identifier used in diagnostics.
        model_config_fingerprint: Content fingerprint of the effective model config.
        hidden_size: Input-embedding width expected by packet wrappers.
        model_artifact_fingerprint: Identity of the exact model weights or revision.
    """

    backend: str
    model_id: str
    model_config_fingerprint: str
    hidden_size: int
    model_artifact_fingerprint: str

    def __post_init__(self) -> None:
        """Validate required identifiers, dimensions, and fingerprints."""
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not is_fingerprint(self.model_config_fingerprint):
            raise ValueError("model_config_fingerprint must be a sha256 fingerprint")
        if not is_fingerprint(self.model_artifact_fingerprint):
            raise ValueError("model_artifact_fingerprint must be a sha256 fingerprint")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

    @property
    def fingerprint(self) -> str:
        """Canonical fingerprint of the complete compatibility record."""
        return fingerprint_json(self.to_dict(), domain="kvpacket.backend_compatibility/1")

    def to_dict(self) -> dict[str, object]:
        """Serialize the compatibility record into primitive values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "BackendCompatibility":
        """Parse a compatibility record from untrusted persisted data.

        Args:
            value: Candidate serialized mapping.

        Returns:
            A validated compatibility record.

        Raises:
            ValueError: If keys, field types, dimensions, or fingerprints are invalid.
        """
        data = string_mapping(value, where="compatibility")
        expected = {
            "backend",
            "model_id",
            "model_config_fingerprint",
            "hidden_size",
            "model_artifact_fingerprint",
        }
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise ValueError(
                f"invalid compatibility keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        string_fields = (
            "backend",
            "model_id",
            "model_config_fingerprint",
            "model_artifact_fingerprint",
        )
        if not all(isinstance(data[field], str) for field in string_fields):
            raise ValueError("backend, model_id, and model_config_fingerprint must be strings")
        hidden_size = data["hidden_size"]
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
            raise ValueError("hidden_size must be an integer")
        return cls(
            backend=cast(str, data["backend"]),
            model_id=cast(str, data["model_id"]),
            model_config_fingerprint=cast(str, data["model_config_fingerprint"]),
            hidden_size=hidden_size,
            model_artifact_fingerprint=cast(str, data["model_artifact_fingerprint"]),
        )


@dataclass
class ForwardOutput:
    """Backend-neutral result of one causal model forward.

    Attributes:
        logits: Model logits shaped ``(batch, sequence, vocabulary)``.
        cache: Cache including the supplied prefix and newly forwarded tokens.
    """

    logits: Tensor
    cache: KVCache


@dataclass
class BackendGenerationOutput:
    """Backend-native continuation and persistent cache.

    Attributes:
        token_ids: Generated continuation IDs excluding the supplied prompt.
        logits: Raw model logits preceding each generated token when requested.
        cache: Cache containing the prompt and committed continuation tokens.
        final_offset: Logical position assigned to the next committed token.
    """

    token_ids: list[int]
    logits: list[Tensor]
    cache: KVCache
    final_offset: int


@runtime_checkable
class ComputeBackend(Protocol):
    """Inference capabilities required by the framework-agnostic session engine."""

    @property
    def compatibility(self) -> BackendCompatibility:
        """Stable identity used to validate wrappers, samples, and stored packets."""
        ...

    @property
    def device(self) -> torch.device:
        """Device on which model inputs and active caches must reside."""
        ...

    def assert_rerotatable(self) -> None:
        """Reject a positional encoding that cannot be shifted safely."""
        ...

    def shift_cache(self, cache: KVCache, offset: int) -> KVCache:
        """Move a reusable cache to a new logical starting position."""
        ...

    def shift_cache_flops(self, cache: KVCache) -> int:
        """Analytic cost of repositioning one cache with this backend."""
        ...

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ForwardOutput:
        """Run a causal forward over an optional KV Packet prefix.

        Implementations convert ``cache`` to and from their native representation
        internally.

        Args:
            input_ids: Token IDs for the newly forwarded sequence.
            inputs_embeds: Precomputed embeddings used instead of ``input_ids``.
            position_ids: Logical positions of the new sequence.
            attention_mask: Backend-supported attention mask.
            cache: Optional framework-neutral prefix cache.

        Returns:
            New-token logits and an extended cache; the input cache remains unchanged.
        """
        ...


@runtime_checkable
class PacketBuildBackend(ComputeBackend, Protocol):
    """Offline packet-construction capabilities layered on inference."""

    def embed(self, input_ids: Tensor) -> Tensor:
        """Map token IDs to embeddings before packet wrapper insertion."""
        ...

    def prefill_caches(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> list[KVCache]:
        """Build one independent, padding-free cache per batch item.

        Unlike ``forward``, offline prefill does not extend an existing cache and
        discards logits. Implementations own native cache extraction and padding
        removal.

        Args:
            input_ids: Batched token IDs to prefill.
            inputs_embeds: Batched embeddings used instead of ``input_ids``.
            attention_mask: Valid-token mask for padded inputs.
            position_ids: Optional explicit positions for the input batch.

        Returns:
            Independent caches in batch order.
        """
        ...


@runtime_checkable
class GenerationBackend(ComputeBackend, Protocol):
    """Optional capability for backend-native continuation.

    The session framework owns prompt prefill, first-token selection, and TTFT. An
    implementation may continue from that selected token using its native decoding
    loop, including its logits processing, later token selection, and stopping.
    """

    def generate(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
        cache: KVCache | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | list[int] | None = None,
        capture_logits: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> BackendGenerationOutput:
        """Generate natively and return raw per-token logits when requested."""
        ...


@runtime_checkable
class TrainingBackend(PacketBuildBackend, Protocol):
    """Differentiable capabilities required by ``PacketTrainer``.

    Keeping training separate allows inference-only serving backends to implement
    ``ComputeBackend`` without exposing model parameters or autograd.
    """

    @property
    def sliding_window(self) -> int | None:
        """Maximum causal attention window, or ``None`` for global attention."""
        ...

    def prepare_for_training(self) -> None:
        """Freeze the base model and put it in deterministic inference mode."""
        ...

    def embedding_statistics(self) -> tuple[float, float]:
        """Measure the base input-embedding mean and standard deviation."""
        ...

    def training_forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run the differentiable student forward used for wrapper optimization.

        Args:
            inputs_embeds: Wrapped student inputs.
            attention_mask: Four-dimensional packet attention topology.

        Returns:
            Logits shaped ``(batch, sequence, vocabulary)``.
        """
        ...


@runtime_checkable
class Packet(Protocol):
    """Minimal stored-packet view consumed by a session.

    Attributes:
        packet: Materialized cache for the wrapped source.
        wrapped_len: Original logical span including wrapper tokens; compressed
            physical caches may be shorter.
        token_ids: Optional original body tokens used for matching.
        match_text: Optional original text used for matching.
    """

    packet: KVCache
    wrapped_len: int
    token_ids: list[int] | None
    match_text: str | None


@runtime_checkable
class CacheStore(Protocol):
    """Read-only packet lookup interface used by online sessions."""

    def get(self, chunk_id: str) -> Packet:
        """Fetch an existing packet by registration ID."""
        ...

    def __contains__(self, chunk_id: str) -> bool:
        """Whether a packet is registered under ``chunk_id``."""
        ...

    def ids(self) -> list[str]:
        """Registered packet IDs in store-defined order."""
        ...
