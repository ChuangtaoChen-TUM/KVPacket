"""Training attention topology for independently prefetched packets."""

import pytest
import torch

from kvpacket.training.mask import batch_attention_masks, packet_attention_mask


def test_contiguous_packets_and_trailing_query_topology() -> None:
    mask = packet_attention_mask(14, [(0, 3), (3, 8), (8, 10)])[0, 0]

    assert mask[0].int().tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert mask[2, :3].all()
    assert not mask[3, :3].any()
    assert mask[10, :11].all()  # The first query row integrates all preceding packets.
    assert mask[13].all()


def test_interleaved_packet_rows_are_isolated_but_later_plain_rows_are_global() -> None:
    mask = packet_attention_mask(8, [(2, 5)])[0, 0]

    assert mask[1, :2].all()
    assert not mask[2:5, :2].any()
    assert mask[4, 2:5].all()
    assert mask[5, :6].all()


def test_sliding_window_has_exact_width() -> None:
    mask = packet_attention_mask(6, [(0, 2)], sliding_window=3)[0, 0]

    for row in range(mask.shape[0]):
        allowed = mask[row].nonzero(as_tuple=False).flatten().tolist()
        assert all(column >= row - 2 for column in allowed)
        assert row in allowed


def test_batch_padding_preserves_masks_and_gives_padding_rows_self_edges() -> None:
    short = packet_attention_mask(3, [(0, 2)])
    long = packet_attention_mask(5, [(1, 3)])
    batched = batch_attention_masks([short, long])

    assert batched.shape == (2, 1, 5, 5)
    assert torch.equal(batched[0, :, :3, :3], short[0])
    assert not batched[0, :, :3, 3:].any()
    assert batched[0, 0, 3, 3] and batched[0, 0, 4, 4]
    assert torch.equal(batched[1], long[0])


def test_batch_left_padding_offsets_the_complete_mask() -> None:
    short = packet_attention_mask(3, [(0, 2)])
    long = packet_attention_mask(5, [(1, 3)])
    batched = batch_attention_masks([short, long], padding_side="left")

    assert torch.equal(batched[0, :, 2:, 2:], short[0])
    assert batched[0, 0, 0, 0] and batched[0, 0, 1, 1]
    assert not batched[0, :, 2:, :2].any()
    assert torch.equal(batched[1], long[0])


@pytest.mark.parametrize(
    "spans",
    [
        [(-1, 2)],
        [(2, 2)],
        [(1, 5)],
        [(0, 3), (2, 4)],
    ],
)
def test_invalid_packet_spans_are_rejected(spans: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        packet_attention_mask(4, spans)
