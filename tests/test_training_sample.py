"""Tests for wrapper-agnostic recorded teacher samples."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kvpacket.session import TrainingSample


def _sample() -> TrainingSample:
    return TrainingSample(
        token_ids=torch.arange(8, dtype=torch.long),
        generated_mask=torch.tensor([False, False, False, False, False, False, True, True]),
        chunk_spans={"doc": [(0, 2)], "skill-block": [(2, 5)]},
        logits=torch.randn(2, 16),
    )


def test_sample_contains_teacher_data_without_wrapper_state():
    sample = _sample()
    assert sample.chunk_spans == {"doc": [(0, 2)], "skill-block": [(2, 5)]}
    assert not hasattr(sample, "chunk_wrappers")
    assert not hasattr(sample, "default_wrapper")


def test_versioned_sample_roundtrip_preserves_teacher_data(tmp_path: Path):
    sample = _sample()
    path = tmp_path / "nested" / "sample.pt"
    sample.save(str(path))
    loaded = TrainingSample.load(str(path))
    assert loaded.chunk_spans == sample.chunk_spans
    assert torch.equal(loaded.token_ids, sample.token_ids)
    assert torch.equal(loaded.generated_mask, sample.generated_mask)
    assert loaded.logits is not None and sample.logits is not None
    assert torch.equal(loaded.logits, sample.logits)


def test_chunks_need_no_wrapper_assignment_during_recording():
    sample = TrainingSample(
        token_ids=torch.arange(4, dtype=torch.long),
        generated_mask=torch.tensor([False, False, False, True]),
        chunk_spans={"unrouted-source": [(0, 2)]},
        logits=torch.randn(1, 8),
    )
    assert "unrouted-source" in sample.chunk_spans


def test_malformed_or_unknown_persisted_fields_are_rejected(tmp_path: Path):
    sample = _sample()
    path = tmp_path / "sample.pt"
    sample.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["wrapper_snapshot"] = {}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="unknown"):
        TrainingSample.load(str(path))


def test_unversioned_sample_is_rejected(tmp_path: Path):
    unversioned = {
        "token_ids": torch.arange(4, dtype=torch.long),
        "generated_mask": torch.tensor([False, False, False, True]),
        "chunk_spans": {"doc": [(0, 2)]},
        "logits": torch.randn(1, 8),
    }
    path = tmp_path / "unversioned.pt"
    torch.save(unversioned, path)
    with pytest.raises(ValueError, match="unrecognized training sample format"):
        TrainingSample.load(str(path))
