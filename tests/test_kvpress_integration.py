"""Optional real-KVPress integration tests on a local tiny model."""

from __future__ import annotations

import pytest
import torch

from kvpacket.compression import KVPressCompressor
from kvpacket.packet import PacketWrapper, RawText, WrapperEntry, WrapperRegistry
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import PacketSession


@pytest.mark.parametrize("method", ["random", "knorm", "expected_attention"])
def test_kvpress_materializes_and_serves_tiny_llama(method: str):
    pytest.importorskip("kvpress")
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket.backend import HFBackend

    torch.manual_seed(13)
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
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"document": WrapperEntry(PacketWrapper(1, 1, backend.hidden_size))},
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        wrapper_registry=registry,
        cache_compressor=KVPressCompressor(method, 0.5),
    )
    packet = preprocessor.preprocess_pretokenized(
        RawText("synthetic document", source_id="doc"),
        [4, 5, 6, 7, 8, 9, 10, 11],
        wrapper_id="document",
    )

    assert packet.wrapped_len == 10
    assert packet.stored_len == 5
    assert packet.packet.doc_len == 10
    session = PacketSession(backend=backend, store=preprocessor.store, match="token")
    session.register_store_chunks()
    session.push([4, 5, 6, 7, 8, 9, 10, 11])
    session.push([12], match=False)
    result = session.generate(max_new_tokens=1, eos_token_id=[])
    assert len(result.token_ids) == 1
    assert session._offset == 12
