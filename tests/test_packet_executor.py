"""Direct packet-executor state ownership and PacketSession parity tests."""

from __future__ import annotations

import torch

from kvpacket.runtime import PacketCacheState, PacketDecodeState, PacketExecutor, SamplingConfig
from kvpacket.session import PacketSession
from tests.test_compression import _CompressedServingBackend, _compressed_store
from tests.test_phase2 import _FakeBackend, _packet_store


def _assert_cache_equal(left, right) -> None:
    assert left is not None and right is not None
    assert left.num_layers == right.num_layers
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        torch.testing.assert_close(left_layer.key, right_layer.key)
        torch.testing.assert_close(left_layer.value, right_layer.value)
        assert torch.equal(left_layer.logical_positions, right_layer.logical_positions)


def test_direct_executor_process_matches_stateful_session() -> None:
    backend = _FakeBackend()
    store = _packet_store(backend, {"doc": [10, 11, 12]})
    executor = PacketExecutor(backend=backend, store=store)

    direct = executor.process(PacketCacheState(), [7, 10, 11, 12, 8])

    session = PacketSession(backend=backend, store=store)
    session.register_store_chunks()
    session.push([7, 10, 11, 12, 8])
    cover = session.process()

    assert direct.cover == cover
    assert direct.cache_state.next_position == session._offset == 5
    _assert_cache_equal(direct.cache_state.cache, session._cache)


def test_direct_executor_prefill_returns_first_token_state() -> None:
    backend = _FakeBackend()
    store = _packet_store(backend, {"doc": [10, 11]})
    executor = PacketExecutor(backend=backend, store=store)

    committed = executor.process(PacketCacheState(), [10, 11])
    result = executor.prefill(
        committed.cache_state,
        [20, 21],
        return_prompt_logits=True,
    )

    assert result.prompt_len == 2
    assert result.prompt_logits is not None
    assert result.prompt_logits.shape == (1, 2, backend.V)
    assert result.first_token_logits.argmax(dim=-1).tolist() == [1]
    assert result.cache_state.next_position == 4
    assert committed.cache_state.next_position == 2


def test_direct_executor_supports_incremental_generation() -> None:
    backend = _FakeBackend()
    store = _packet_store(backend, {})
    executor = PacketExecutor(backend=backend, store=store)
    prefill = executor.prefill(PacketCacheState(), [4, 5])
    state = PacketDecodeState(prefill.cache_state, prefill.first_token_logits)

    first = executor.decode_step(state, SamplingConfig())
    second = executor.decode_step(first.state, SamplingConfig())

    assert [first.token_id, second.token_id] == [1, 1]
    assert state.cache_state.next_position == 2
    assert first.state.cache_state.next_position == 3
    assert second.state.cache_state.next_position == 4


def test_executor_does_not_mix_caller_owned_conversations() -> None:
    backend = _FakeBackend()
    store = _packet_store(backend, {"doc": [10, 11]})
    executor = PacketExecutor(backend=backend, store=store)
    empty = PacketCacheState()

    first = executor.process(empty, [10, 11])
    second = executor.process(empty, [7, 8, 9])

    assert empty.cache is None and empty.next_position == 0
    assert first.cache_state.next_position == 2
    assert second.cache_state.next_position == 3
    assert first.cache_state.cache is not second.cache_state.cache


def test_executor_preserves_compressed_packet_logical_span() -> None:
    backend = _CompressedServingBackend()
    store = _compressed_store(backend)
    executor = PacketExecutor(
        backend=backend,
        store=store,
        mixed_prefill_policy="fused",
    )
    tokenizer = backend.tokenizer
    input_ids = tokenizer("X", add_special_tokens=False)["input_ids"]
    input_ids += tokenizer("DOC", add_special_tokens=False)["input_ids"]
    input_ids += tokenizer("?", add_special_tokens=False)["input_ids"]

    result = executor.prefill(PacketCacheState(), input_ids)

    assert result.cover[0].chunk_id == "doc"
    assert result.cache_state.next_position == 7
    assert result.cache_state.cache is not None
    assert result.cache_state.cache.seq_len == 5
    assert result.cache_state.cache.doc_len == 7
