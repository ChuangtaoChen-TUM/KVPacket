"""Attention masks for reconstructing independently prefetched packets in training.

Serving concatenates packet caches that were built independently. During adapter
training, the student forward reproduces that topology by preventing rows inside each
materialized packet from attending to earlier context. Ordinary rows remain globally
causal and can integrate all packets that precede them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["packet_attention_mask", "batch_attention_masks"]


def packet_attention_mask(
    sequence_length: int,
    packet_spans: Sequence[tuple[int, int]],
    *,
    sliding_window: int | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build the attention topology for one reconstructed student sequence.

    Packet rows attend causally only within their own span. All rows outside packets
    retain ordinary global causal attention. A configured sliding window is applied
    after packet isolation.

    Args:
        sequence_length: Total wrapped student-sequence length.
        packet_spans: Non-overlapping half-open packet intervals in student positions.
        sliding_window: Maximum number of causal positions visible to each row.
        device: Device on which to allocate the mask.

    Returns:
        Boolean attention mask shaped ``(1, 1, sequence_length, sequence_length)``.

    Raises:
        ValueError: If the sequence length, sliding window, or packet spans are invalid.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")

    ordered_spans = sorted(packet_spans)
    previous_end = 0
    for start, end in ordered_spans:
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError(
                f"invalid packet span ({start}, {end}) for sequence length {sequence_length}"
            )
        if start < previous_end:
            raise ValueError("packet spans must not overlap")
        previous_end = end

    mask = torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device)
    )
    # A packet was prefetched without earlier context, so its rows cannot recover those
    # edges. Later ordinary rows still integrate every causal predecessor.
    for start, end in ordered_spans:
        mask[start:end, :start] = False

    if sliding_window is not None:
        window = torch.triu(torch.ones_like(mask), diagonal=1 - sliding_window)
        mask &= window

    return mask.unsqueeze(0).unsqueeze(0)


def batch_attention_masks(
    masks: Sequence[Tensor],
    *,
    padding_side: str = "right",
) -> Tensor:
    """Pad per-sample 4-D boolean masks into one training microbatch.

    Padding rows receive only a self edge. They are never selected for the loss, but
    avoiding fully masked rows prevents undefined behavior in some attention kernels.

    Args:
        masks: Non-empty sequence of masks shaped ``(1, 1, S, S)`` on one device.
        padding_side: Side on which shorter masks are padded: ``"left"`` or ``"right"``.

    Returns:
        Boolean attention mask shaped ``(B, 1, max_S, max_S)``.

    Raises:
        ValueError: If no masks are supplied, padding is unsupported, or a mask has an
            invalid shape, data type, or device.
    """
    if not masks:
        raise ValueError("masks must not be empty")
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")

    device = masks[0].device
    for mask in masks:
        if (
            mask.dim() != 4
            or mask.shape[0] != 1
            or mask.shape[1] != 1
            or mask.shape[2] != mask.shape[3]
        ):
            raise ValueError("each mask must have shape (1, 1, S, S)")
        if mask.dtype != torch.bool:
            raise ValueError("attention masks must be boolean")
        if mask.device != device:
            raise ValueError("attention masks must share one device")

    max_length = max(int(mask.shape[-1]) for mask in masks)
    batched = torch.zeros(
        (len(masks), 1, max_length, max_length),
        dtype=torch.bool,
        device=device,
    )
    for index, mask in enumerate(masks):
        length = int(mask.shape[-1])
        offset = max_length - length if padding_side == "left" else 0
        batched[index, :, offset : offset + length, offset : offset + length] = mask[0]
        padding_positions = (
            torch.arange(0, offset, device=device)
            if padding_side == "left"
            else torch.arange(length, max_length, device=device)
        )
        batched[index, 0, padding_positions, padding_positions] = True
    return batched
