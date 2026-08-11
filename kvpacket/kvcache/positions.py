"""Construct source and destination positions for independently cached packets.

Offline packet construction normally assigns every packet positions beginning at zero.
After several caches are stitched, their keys must instead use positions in the combined
context. This module builds the flattened position sequences used as the source and
destination of that re-rotation.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cumsum_pos", "shifted_cumsum_pos"]


def cumsum_pos(cache_lens: list[int], device: torch.device | str = "cpu") -> Tensor:
    """Build concatenated positions that restart at zero for every cache.

    Args:
        cache_lens: Stored token count of each independently encoded cache.
        device: Device on which to create the positions.

    Returns:
        One-dimensional positions of length ``sum(cache_lens)``.

    Examples:
        ``cumsum_pos([3, 5, 2]).tolist()`` produces
        ``[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]``.
    """
    lens = torch.tensor(cache_lens, dtype=torch.long, device=device)
    offset = torch.cumsum(lens, dim=0) - lens
    repeated = torch.repeat_interleave(offset, lens)
    return torch.arange(int(lens.sum()), device=device) - repeated


def shifted_cumsum_pos(
    cache_lens: list[int],
    doc_lens: list[int],
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build stitched positions while preserving each segment's logical span.

    ``cache_lens[i]`` is the stored (possibly compressed) length of segment ``i``;
    ``doc_lens[i]`` is the logical length it occupies in the assembled context. Each
    segment starts after the preceding logical spans and advances by one for every
    stored token.

    Args:
        cache_lens: Physically stored token count of each cache.
        doc_lens: Logical context span allocated to each corresponding cache.
        device: Device on which to create the positions.

    Returns:
        One-dimensional stitched positions of length ``sum(cache_lens)``.

    Raises:
        ValueError: If a preceding document span is shorter than its stored cache.

    Examples:
        ``shifted_cumsum_pos([3, 5, 2], [4, 6, 3]).tolist()`` produces
        ``[0, 1, 2, 4, 5, 6, 7, 8, 10, 11]``.
    """
    cache_t = torch.tensor(cache_lens, dtype=torch.long, device=device)
    doc_t = torch.tensor(doc_lens, dtype=torch.long, device=device)

    offset = torch.zeros_like(cache_t)
    offset[1:] = doc_t[:-1] - cache_t[:-1]
    if not torch.all(offset >= 0):
        raise ValueError("Each doc_len must be >= the corresponding cache_len.")

    cum_offset = torch.cumsum(offset, dim=0)
    repeated = torch.repeat_interleave(cum_offset, cache_t)
    return torch.arange(int(cache_t.sum()), device=device) + repeated
