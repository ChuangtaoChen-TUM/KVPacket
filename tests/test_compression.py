"""Cache-compression identity, persistence, position, and serving tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from kvpacket.compression import CompressionSpec
from kvpacket.kvcache import KeyValue, KVCache, apply_rotary
from kvpacket.packet import PacketWrapper, RawText, WrapperEntry, WrapperRegistry
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import PacketStore, PacketSession
from tests.test_session_preprocessed import (
    _ServingBackend,
    _ServingSourceEncoder,
)


class _EveryOtherCompressor:
    """Test extension that makes the fake backend retain alternating positions."""

    spec = CompressionSpec(
        provider="test",
        method="every-other",
        implementation_version="1",
        config={"stride": 2},
    )

    @contextmanager
    def activate(self, backend):
        backend.compress_every_other = True
        try:
            yield
        finally:
            backend.compress_every_other = False


class _CompressedServingBackend(_ServingBackend):
    compress_every_other = False

    def prefill_caches(self, **kwargs) -> list[KVCache]:
        caches = super().prefill_caches(**kwargs)
        if not self.compress_every_other:
            return caches
        compressed: list[KVCache] = []
        for cache in caches:
            layers = [
                KeyValue(
                    layer.key[:, :, ::2, :],
                    layer.value[:, :, ::2, :],
                    layer.logical_positions,
                )
                for layer in cache.layers
            ]
            compressed.append(KVCache(layers))
        return compressed


def _compressed_store(backend: _CompressedServingBackend) -> PacketStore:
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"document": WrapperEntry(PacketWrapper(1, 1, backend.hidden_size))},
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        source_encoder=_ServingSourceEncoder(backend.tokenizer),
        wrapper_registry=registry,
        cache_compressor=_EveryOtherCompressor(),
    )
    preprocessor.preprocess({RawText("DOC", source_id="doc"): "document"})
    return preprocessor.store


def _small_rotary(head_dim: int = 16):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=head_dim * 2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=head_dim,
    )
    return LlamaRotaryEmbedding(config)


def test_compressed_cache_shift_uses_logical_span_and_uniform_delta():
    torch.manual_seed(3)
    logical_len = 5
    retained = torch.tensor([0, 2, 4])
    raw = torch.randn(1, 2, logical_len, 16)
    rotary = _small_rotary()
    source_positions = retained.unsqueeze(0)
    retained_raw = raw[:, :, retained, :]
    source_keys = apply_rotary(
        retained_raw,
        *rotary(retained_raw, position_ids=source_positions),
    )
    cache = KVCache(
        [
            KeyValue(
                source_keys,
                torch.randn_like(source_keys),
                torch.arange(logical_len).unsqueeze(0),
            )
        ]
    )

    shifted = cache.shift(rotary, 7)
    target_positions = source_positions + 7
    expected_keys = apply_rotary(
        retained_raw,
        *rotary(retained_raw, position_ids=target_positions),
    )

    assert shifted.seq_len == 3
    assert shifted.doc_len == logical_len
    assert shifted.layers[0].logical_positions.tolist() == [[7, 8, 9, 10, 11]]
    torch.testing.assert_close(shifted.layers[0].key, expected_keys, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="uniform shift"):
        cache.rerotate(rotary, source_positions, target_positions)


def test_preprocessor_persists_compression_and_both_lengths(tmp_path: Path):
    backend = _CompressedServingBackend()
    store = _compressed_store(backend)
    packet = store.get("doc")

    assert packet.wrapped_len == 5
    assert packet.stored_len == 3
    assert packet.packet.doc_len == 5
    assert packet.compression == _EveryOtherCompressor.spec
    assert packet.identity.cache_compression_fingerprint == _EveryOtherCompressor.spec.fingerprint

    path = tmp_path / "compressed-packets.pt"
    store.save(path)
    loaded = PacketStore.load(path, backend=backend).get("doc")
    assert loaded.wrapped_len == 5
    assert loaded.stored_len == 3
    assert loaded.compression == packet.compression
    assert loaded.identity == packet.identity


def test_sequential_and_fused_serving_keep_logical_offsets_for_compressed_packets():
    for policy in ("sequential", "fused"):
        backend = _CompressedServingBackend()
        store = _compressed_store(backend)
        session = PacketSession(
            backend=backend,
            store=store,
            tokenizer=backend.tokenizer,
            match="token",
            mixed_prefill_policy=policy,
        )
        session.register_store_chunks()
        session.push("X", match=False)
        session.push("DOC")
        session.push("?", match=False)
        result = session.generate(max_new_tokens=1, eos_token_id=0)

        assert result.matched_chunk_ids == ["doc"]
        assert session._offset == 8
        assert session._cache is not None
        assert session._cache.seq_len == 6
        assert session._cache.doc_len == 8


def test_hf_backend_extends_cache_with_shorter_physical_than_logical_length():
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket.backend import HFBackend

    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    backend = HFBackend(LlamaForCausalLM(config).float().eval())
    full = backend.prefill_caches(input_ids=torch.tensor([[4, 5, 6, 7, 8]]))[0]
    retained = torch.tensor([0, 2, 4])
    compressed = KVCache(
        [
            KeyValue(
                layer.key[:, :, retained, :],
                layer.value[:, :, retained, :],
                layer.logical_positions,
            )
            for layer in full.layers
        ]
    )

    output = backend.forward(
        input_ids=torch.tensor([[9, 10]]),
        position_ids=torch.tensor([[5, 6]]),
        cache=compressed,
    )

    assert output.cache.seq_len == 5
    assert output.cache.doc_len == 7
    assert output.cache.layers[0].logical_positions.tolist() == [list(range(7))]
