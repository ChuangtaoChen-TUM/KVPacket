"""Reposition cached keys with rotary embeddings instead of model recomputation.

A key cached at position ``old`` can be moved to position ``new`` by applying the
incremental rotation for ``new - old``. KV Packet uses this identity after concatenating
independently encoded caches, making re-rotation the only additional tensor arithmetic
needed to align their keys with the assembled context.

The functions accept a duck-typed rotary module that produces cosine and sine tensors
from position IDs. An optional leading non-rotary slice supports partial-RoPE models.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "rotate_half",
    "apply_rotary",
    "rerotate_keys",
    "rerotate_flops",
]


def rotate_half(x: Tensor) -> Tensor:
    """Apply the RoPE quarter-turn ``[a, b] -> [-b, a]`` on the last axis."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> Tensor:
    """Apply precomputed rotary factors to a tensor.

    Args:
        x: States to rotate, conventionally shaped
            ``(batch, heads, sequence, head_dim)``.
        cos: Cosine factors with the head axis omitted.
        sin: Sine factors matching ``cos``.
        unsqueeze_dim: Axis inserted into ``cos`` and ``sin`` for broadcasting.

    Returns:
        Rotated states with the same shape as ``x``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def rerotate_keys(
    keys: Tensor,
    rotary_emb,
    old_pos: Tensor,
    new_pos: Tensor,
    nope_dim: int | None = None,
    unsqueeze_dim: int = 1,
) -> Tensor:
    """Move already-rotated keys from old positions to new positions.

    The operation applies ``R(new - old)`` to satisfy
    ``R(new - old) R(old) = R(new)``. Frequency-remapped RoPE variants remain exact
    because the model's own rotary module produces the delta rotation. When the module
    exposes constant ``attention_scaling``, that amplitude is divided out of the delta
    factors because it is already present in the cached keys.

    Args:
        keys: Key states shaped ``(batch, heads, sequence, head_dim)`` and already
            rotated to ``old_pos``.
        rotary_emb: Model rotary module called with the position delta.
        old_pos: Existing positions shaped ``(batch, sequence)``.
        new_pos: Desired positions shaped ``(batch, sequence)``.
        nope_dim: Size of an optional leading non-rotary key slice.
        unsqueeze_dim: Axis inserted into rotary factors for broadcasting over keys.

    Returns:
        Keys rotated to ``new_pos`` with the same shape as ``keys``.

    Raises:
        ValueError: If either position tensor does not match the key batch and sequence
            dimensions.

    Warning:
        Dynamic NTK variants that change their frequencies with runtime sequence length
        cannot in general be shifted exactly by a position delta.
    """
    if old_pos.shape != (keys.size(0), keys.size(2)):
        raise ValueError(
            f"old_pos shape {tuple(old_pos.shape)} != (B, S)={(keys.size(0), keys.size(2))}"
        )
    if new_pos.shape != (keys.size(0), keys.size(2)):
        raise ValueError(
            f"new_pos shape {tuple(new_pos.shape)} != (B, S)={(keys.size(0), keys.size(2))}"
        )

    scaling = float(getattr(rotary_emb, "attention_scaling", 1.0))
    delta = new_pos - old_pos

    def _delta_rotate(x: Tensor) -> Tensor:
        cos, sin = rotary_emb(x, position_ids=delta)
        if abs(scaling - 1.0) > 1e-6:
            # Divide out the constant amplitude so we apply a pure rotation R(Δ);
            # the scalar already present in `x` is preserved.
            cos = cos / scaling
            sin = sin / scaling
        return apply_rotary(x, cos, sin, unsqueeze_dim=unsqueeze_dim)

    if nope_dim is not None:
        k_nope = keys[..., :nope_dim]
        k_rope = _delta_rotate(keys[..., nope_dim:])
        return torch.cat([k_nope, k_rope], dim=-1)

    return _delta_rotate(keys)


def rerotate_flops(key_shape: tuple[int, int, int, int], nope_dim: int | None = None) -> int:
    """Estimate the arithmetic cost of re-rotating one layer's keys.

    The reference accounting assigns ``4 + 3 * heads`` operations to every batch,
    position, and rotated dimension: four for rotary-factor construction and three per
    head for ``x * cos + rotate_half(x) * sin``.

    Args:
        key_shape: Key dimensions ordered as ``(batch, heads, sequence, head_dim)``.
        nope_dim: Size of an optional leading non-rotary key slice.

    Returns:
        Estimated floating-point operation count.
    """
    batch, num_heads, seq_len, head_dim = key_shape
    rope_dim = head_dim if nope_dim is None else head_dim - nope_dim
    return batch * (4 + 3 * num_heads) * seq_len * rope_dim
