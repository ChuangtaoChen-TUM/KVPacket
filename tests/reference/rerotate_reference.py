"""Vendored reference re-rotation — the golden implementation for parity tests.

Copied verbatim (general RoPE path) from the original paper code
``kv_packet/cache/rotate.py`` so the parity test does not depend on the external
``kv_packet_clean/`` reference folder. This is intentionally a standalone copy: it
uses only ``torch`` + a duck-typed rotary module, no project imports.

The one difference from our engine's ``rerotate_keys`` is that this reference
version does not divide out ``attention_scaling`` (it raises instead) — so parity
is asserted on the scaling==1 case, which is exactly what both compute.
"""
from __future__ import annotations

import torch


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def rerotate_embeddings(x_p_orig, rotary_emb, p_orig, p_new, nope_dim=None, unsqueeze_dim=1):
    """Transform key/query tensors rotated by ``p_orig`` to being rotated by ``p_new``."""
    p_delta = p_new - p_orig

    if hasattr(rotary_emb, "attention_scaling") and abs(rotary_emb.attention_scaling - 1.0) > 1e-6:
        raise NotImplementedError(
            "reference re-rotation does not support attention_scaling != 1.0"
        )

    if nope_dim is not None:
        x_nope = x_p_orig[..., :nope_dim]
        x_rope = x_p_orig[..., nope_dim:]
        cos_delta, sin_delta = rotary_emb.forward(x_rope, position_ids=p_delta)
        x_rope_new = apply_rotary_pos_emb(x_rope, cos_delta, sin_delta, unsqueeze_dim=unsqueeze_dim)
        return torch.cat([x_nope, x_rope_new], dim=-1)

    cos_delta, sin_delta = rotary_emb.forward(x_p_orig, position_ids=p_delta)
    return apply_rotary_pos_emb(x_p_orig, cos_delta, sin_delta, unsqueeze_dim=unsqueeze_dim)
