"""Represent, move, concatenate, and reposition backend-independent KV caches.

Each reusable packet is stored as a ``KVCache`` containing one ``KeyValue`` per model
layer. Alongside the key and value tensors, every layer retains the complete logical
position span occupied by the source. For an uncompressed cache this maps one-to-one to
KV slots; for a compressed cache it can be longer than the stored token axis. Serving
can concatenate physical tensors while translating packets to their logical positions.

The cache representation depends only on PyTorch tensors. Framework-specific cache
conversion belongs to the backend rather than this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .rope import rerotate_flops, rerotate_keys

__all__ = ["KeyValue", "KVCache", "concat_caches"]


@dataclass
class KeyValue:
    """Cached key, value, and key-position state for one model layer.

    Attributes:
        key: Key states shaped ``(batch, heads, cached_tokens, key_dim)``.
        value: Value states shaped ``(batch, heads, cached_tokens, value_dim)``.
        logical_positions: Complete logical position span represented by this layer,
            shaped ``(batch, logical_tokens)``. For an uncompressed cache these are the
            key positions. A compressed cache need not retain one KV slot per entry.
    """

    key: Tensor
    value: Tensor
    logical_positions: Tensor

    def __post_init__(self) -> None:
        """Validate the tensor ranks required by cache operations."""
        if self.key.dim() != 4 or self.value.dim() != 4:
            raise ValueError("key/value must be 4-D [B, H, S, D]")
        if self.logical_positions.dim() != 2:
            raise ValueError("logical_positions must be 2-D [B, S]")
        if self.key.shape[:3] != self.value.shape[:3]:
            raise ValueError("key/value batch, head, and sequence dimensions must match")
        if self.logical_positions.shape[0] != self.key.shape[0]:
            raise ValueError("logical_positions batch size must match key/value tensors")
        if not self.key.dtype.is_floating_point or not self.value.dtype.is_floating_point:
            raise ValueError("key/value tensors must be floating point")
        if (
            self.logical_positions.dtype == torch.bool
            or self.logical_positions.dtype.is_floating_point
        ):
            raise ValueError("logical_positions must use an integer data type")


class KVCache:
    """Ordered per-layer state for one complete model KV cache.

    Cache operations return new ``KVCache`` containers and do not mutate the source
    cache. The layer list must be non-empty for shape properties and serving operations.

    Args:
        layers: Layer states in model execution order.
    """

    def __init__(self, layers: list[KeyValue]):
        if not layers:
            raise ValueError("KVCache requires at least one layer")
        self.layers: list[KeyValue] = layers
        first = layers[0]
        expected = (
            first.key.shape[0],
            first.key.shape[2],
            first.logical_positions.shape[1],
        )
        for layer in layers[1:]:
            observed = (
                layer.key.shape[0],
                layer.key.shape[2],
                layer.logical_positions.shape[1],
            )
            if observed != expected:
                raise ValueError(
                    "cache layers must share batch, physical length, and logical length"
                )

    @property
    def num_layers(self) -> int:
        """Number of model layers represented by the cache."""
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        """Number of physically stored key/value positions."""
        return self.layers[0].key.shape[2]

    @property
    def doc_len(self) -> int:
        """Number of logical positions represented by the cache."""
        return self.layers[0].logical_positions.shape[1]

    @property
    def batch_size(self) -> int:
        """Number of sequences represented by each layer."""
        return self.layers[0].key.shape[0]

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, layer: int) -> KeyValue:
        return self.layers[layer]

    def __repr__(self) -> str:
        kv = self.layers[0]
        b, h, s, d = kv.key.shape
        return f"KVCache(layers={self.num_layers}, batch={b}, heads={h}, seq={s}, head_dim={d})"

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "KVCache":
        """Copy every cache tensor to a device.

        Args:
            device: Destination accepted by ``Tensor.to``.
            non_blocking: Whether eligible transfers may proceed asynchronously.

        Returns:
            A new cache containing the transferred tensors.
        """
        return KVCache(
            [
                KeyValue(
                    kv.key.to(device, non_blocking=non_blocking),
                    kv.value.to(device, non_blocking=non_blocking),
                    kv.logical_positions.to(device, non_blocking=non_blocking),
                )
                for kv in self.layers
            ]
        )

    def clone(self) -> "KVCache":
        """Deep-copy every tensor in the cache."""
        return KVCache(
            [
                KeyValue(kv.key.clone(), kv.value.clone(), kv.logical_positions.clone())
                for kv in self.layers
            ]
        )

    def rerotate(
        self,
        rotary_emb,
        old_pos: Tensor,
        new_pos: Tensor,
        nope_dim: int | None = None,
    ) -> "KVCache":
        """Reposition every layer's keys without recomputing packet states.

        Values are position-independent and are reused in the returned cache. Keys are
        newly re-rotated from ``old_pos`` to ``new_pos``; the source cache is unchanged.

        Args:
            rotary_emb: Model rotary embedding used to produce delta rotations.
            old_pos: Existing key positions shaped ``(batch, cached_tokens)``.
            new_pos: Desired key positions with the same shape as ``old_pos``.
            nope_dim: Size of an optional leading non-rotary key slice.

        Returns:
            A new cache whose keys and recorded position IDs use ``new_pos``.

        Raises:
            ValueError: If the cache is compressed. Per-position remapping requires
                retained source positions, which token-pruning compressors need not
                expose; use ``shift`` for a uniform translation instead.
        """
        if self.seq_len != self.doc_len:
            raise ValueError("compressed caches support uniform shift(), not tokenwise rerotate()")
        new_layers: list[KeyValue] = []
        for kv in self.layers:
            new_key = rerotate_keys(kv.key, rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
            new_layers.append(KeyValue(new_key, kv.value, new_pos))
        return KVCache(new_layers)

    def shift(
        self,
        rotary_emb,
        delta: int,
        nope_dim: int | None = None,
    ) -> "KVCache":
        """Translate all stored keys and the complete logical span by one offset.

        Token-pruning compressors may retain different source positions per layer or
        head without exposing those indices. A uniform translation needs only
        ``R(delta)`` for every stored key, so it remains exact without the indices.
        Physical key/value length is unchanged, while logical position metadata keeps
        the original uncompressed length.

        Args:
            rotary_emb: Model rotary embedding used to produce the delta rotation.
            delta: Signed logical-position offset applied to the packet.
            nope_dim: Size of an optional leading non-rotary key slice.

        Returns:
            A translated cache with its physical and logical lengths preserved.
        """
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise TypeError("delta must be an integer")
        new_layers: list[KeyValue] = []
        for kv in self.layers:
            shape = (kv.key.shape[0], kv.key.shape[2])
            old_pos = torch.zeros(shape, dtype=torch.long, device=kv.key.device)
            new_pos = torch.full(shape, delta, dtype=torch.long, device=kv.key.device)
            new_key = rerotate_keys(
                kv.key,
                rotary_emb,
                old_pos,
                new_pos,
                nope_dim=nope_dim,
            )
            new_layers.append(KeyValue(new_key, kv.value, kv.logical_positions + delta))
        return KVCache(new_layers)

    def rerotate_flops(self, nope_dim: int | None = None) -> int:
        """Analytic re-rotation FLOPs summed across all cache layers.

        Args:
            nope_dim: Size of an optional leading non-rotary key slice.
        """
        return sum(
            rerotate_flops(
                (kv.key.shape[0], kv.key.shape[1], kv.key.shape[2], kv.key.shape[3]),
                nope_dim=nope_dim,
            )
            for kv in self.layers
        )


def concat_caches(caches: list[KVCache]) -> KVCache:
    """Stitch caches in order along their stored and logical sequence axes.

    Concatenation is performed independently for every layer. Keys and values are
    joined along their cached-token axis, while position IDs are joined along their
    logical-token axis. Input caches are not modified.

    Args:
        caches: Non-empty caches with the same number of layers and compatible tensor
            geometry.

    Returns:
        One cache containing the inputs in list order.

    Raises:
        ValueError: If no caches are supplied or their layer counts differ.
    """
    if not caches:
        raise ValueError("concat_caches got an empty list")
    n_layers = caches[0].num_layers
    if any(c.num_layers != n_layers for c in caches):
        raise ValueError("all caches must have the same number of layers")

    layers: list[KeyValue] = []
    for i in range(n_layers):
        key = torch.cat([c.layers[i].key for c in caches], dim=2)
        value = torch.cat([c.layers[i].value for c in caches], dim=2)
        pos = torch.cat([c.layers[i].logical_positions for c in caches], dim=1)
        layers.append(KeyValue(key, value, pos))
    return KVCache(layers)
