"""Create packet wrappers with backend-compatible dimensions and initialization.

The framework owns wrapper construction while the training backend supplies only
model-specific metadata and, when requested, input-embedding statistics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import torch

from ..backend import TrainingBackend
from ..packet.wrapper import PacketWrapper

__all__ = ["InitializationStrategy", "create_wrapper"]

InitializationStrategy = Literal["fixed", "model-embedding"]


def _number(value: object, *, name: str) -> float:
    """Validate and normalize one numeric initialization parameter."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"initialization {name} must be a number")
    return float(value)


def create_wrapper(
    backend: TrainingBackend,
    *,
    header_len: int,
    trailer_len: int,
    dtype: torch.dtype,
    initialization: Mapping[str, object] | None = None,
    requires_grad: bool = True,
) -> PacketWrapper:
    """Create a wrapper using fixed or model-embedding normal statistics.

    Fixed initialization defaults to ``mean=0`` and ``std=0.02``. The
    ``model-embedding`` strategy delegates model-specific weight access to the
    training backend. The returned wrapper uses the backend's hidden size and
    device.

    Args:
        backend: Backend that defines model compatibility, device placement, and
            optional embedding statistics.
        header_len: Number of learned tokens prepended to each packet body.
        trailer_len: Number of learned tokens appended to each packet body.
        dtype: Data type used by the learned header and trailer parameters.
        initialization: Strategy configuration. ``{"strategy": "fixed"}`` accepts
            optional ``mean`` and ``std`` values; ``{"strategy":
            "model-embedding"}`` accepts no additional values.
        requires_grad: Whether the new wrapper parameters initially track gradients.

    Returns:
        A wrapper placed on the backend device and sized for its model.

    Raises:
        TypeError: If the backend lacks the training capability.
        ValueError: If the strategy, its options, or a numeric value is invalid.
    """
    if not isinstance(backend, TrainingBackend):
        raise TypeError("backend does not implement the TrainingBackend capability")
    config = dict(initialization or {"strategy": "fixed"})
    strategy = config.get("strategy", "fixed")
    if strategy not in ("fixed", "model-embedding"):
        raise ValueError("initialization strategy must be 'fixed' or 'model-embedding'")
    strategy = cast(InitializationStrategy, strategy)
    if strategy == "fixed":
        unknown = set(config) - {"strategy", "mean", "std"}
        if unknown:
            raise ValueError(f"unknown fixed initialization options: {sorted(unknown)}")
        mean = _number(config.get("mean", 0.0), name="mean")
        std = _number(config.get("std", 0.02), name="std")
    else:
        unknown = set(config) - {"strategy"}
        if unknown:
            raise ValueError("model-embedding initialization does not accept explicit mean/std")
        mean, std = backend.embedding_statistics()
    if std < 0:
        raise ValueError("initialization std must be non-negative")
    return PacketWrapper(
        header_len,
        trailer_len,
        backend.compatibility.hidden_size,
        mean=mean,
        std=std,
        dtype=dtype,
        device=backend.device,
        requires_grad=requires_grad,
    )
