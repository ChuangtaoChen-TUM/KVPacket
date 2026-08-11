"""Phase 2 verification for the ``PacketSession`` engine.

Runnable two ways:
  * ``uv run pytest tests/test_phase2.py``     (light tests; no model download)
  * ``python tests/test_phase2.py``            (also runs the live-model checks)

Light tests exercise the pure matching / DP / sample logic. The ``__main__`` block
loads the locally cached ``Qwen/Qwen2.5-1.5B`` (in float32, for tight parity) and
checks the five Deliverable-1 acceptance criteria end to end.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpacket.runtime.matching import (
    Match,
    find_occurrences,
    string_cover,
    token_cover,
    weighted_interval_cover,
)
from kvpacket.session.sample import TrainingSample


# --------------------------------------------------------------------- light
def test_find_occurrences():
    assert find_occurrences([1, 2, 3, 2, 3, 4], [2, 3]) == [(1, 3), (3, 5)]
    assert find_occurrences([1, 2, 3], [9]) == []
    assert find_occurrences([1, 2], []) == []


def test_token_cover_basic():
    # stream: A=[10,11], B=[20,21,22]; query tail [99]
    tokens = [10, 11, 20, 21, 22, 99]
    chunks = {"A": [10, 11], "B": [20, 21, 22]}
    cover = token_cover(tokens, chunks)
    assert cover == [Match("A", 0, 2), Match("B", 2, 5)]


def test_token_cover_overlap_prefers_more_coverage():
    # Two candidates overlap; the longer (more covered tokens) wins.
    tokens = [1, 2, 3, 4, 5]
    chunks = {"short": [2, 3], "long": [2, 3, 4, 5]}
    cover = token_cover(tokens, chunks)
    assert cover == [Match("long", 1, 5)]


def test_weighted_interval_cover_tie_prefers_earliest():
    # Equal total coverage (2) achievable two ways; earliest-cover wins.
    a = Match("a", 0, 2)
    b = Match("b", 2, 4)
    c = Match("c", 1, 3)  # overlaps both; alone gives coverage 2
    chosen = weighted_interval_cover([a, b, c])
    # {a,b} covers 4 > {c} covers 2, so both a and b selected.
    assert chosen == [a, b]


def test_string_cover():
    text = "use calc now"
    #          0123456789...
    # tokens:  "use"(0,3) " calc"(3,8) " now"(8,12)
    offsets = [(0, 3), (3, 8), (8, 12)]
    chunks = {"calc": "calc"}
    cover = string_cover(text, chunks, offsets)
    assert cover == [Match("calc", 1, 2)]


def test_training_sample_roundtrip():
    s = TrainingSample(
        token_ids=torch.arange(6),
        generated_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        chunk_spans={"skill": [(1, 3)]},
        logits=torch.randn(2, 5),
    )
    assert s.num_generated == 2 and s.length == 6
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.pt")
        s.save(p)
        s2 = TrainingSample.load(p)
    assert torch.equal(s.token_ids, s2.token_ids)
    assert torch.equal(s.generated_mask, s2.generated_mask)
    assert s2.chunk_spans == {"skill": [(1, 3)]}
    assert s.logits is not None and s2.logits is not None
    assert torch.allclose(s.logits, s2.logits)


def _small_rotary(head_dim: int = 4, device: str = "cpu"):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=head_dim * 4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        head_dim=head_dim,
    )
    return LlamaRotaryEmbedding(config).to(device)


class _FakeBackend:
    """A minimal, model-free ComputeBackend for the decoupling test.

    ``forward`` speaks ``KVCache`` directly (no framework-native handle, no
    ``to_kvcache``/``from_kvcache``) — exactly what the protocol now requires.
    """

    hidden, H, D, V = 8, 2, 4, 10

    def __init__(self):
        self._rotary = _small_rotary(head_dim=self.D)

    @property
    def compatibility(self):
        from kvpacket.backend import BackendCompatibility
        from kvpacket.fingerprint import fingerprint_json

        return BackendCompatibility(
            backend="fake",
            model_id="test-model",
            model_config_fingerprint=fingerprint_json(
                {"hidden_size": self.hidden}, domain="tests.fake_model/1"
            ),
            hidden_size=self.hidden,
            model_artifact_fingerprint=fingerprint_json(
                "test-model", domain="tests.fake_artifact/1"
            ),
        )

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def rotary_emb(self):
        return self._rotary

    @property
    def nope_dim(self):
        return None

    @property
    def is_rerotatable(self):
        return True

    def assert_rerotatable(self):
        return None

    def shift_cache(self, cache, offset):
        return cache.shift(self.rotary_emb, offset, nope_dim=self.nope_dim)

    def shift_cache_flops(self, cache) -> int:
        return cache.rerotate_flops(nope_dim=self.nope_dim)

    def embed(self, input_ids):
        b, s = input_ids.shape
        return torch.zeros(b, s, self.hidden)

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        cache=None,
    ):
        from kvpacket.backend import ForwardOutput
        from kvpacket.kvcache.cache import KeyValue, KVCache

        src = inputs_embeds if inputs_embeds is not None else input_ids
        assert src is not None
        b, s = src.shape[0], src.shape[1]
        pos = (
            position_ids if position_ids is not None else torch.arange(s).unsqueeze(0).expand(b, s)
        )
        new_k = pos.float().reshape(b, 1, s, 1).expand(b, self.H, s, self.D).contiguous()
        new_v = new_k.clone()
        if cache is None:
            keys, values = new_k, new_v
        else:
            layer = cache.layers[0]
            keys = torch.cat([layer.key, new_k], dim=2)
            values = torch.cat([layer.value, new_v], dim=2)
        logits = torch.zeros(b, s, self.V)
        logits[..., 1] = 1.0  # argmax -> token 1 (never the EOS below)
        out_pos = torch.arange(keys.shape[2]).unsqueeze(0)
        return ForwardOutput(logits=logits, cache=KVCache([KeyValue(keys, values, out_pos)]))

    def prefill_caches(
        self, *, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None
    ):
        """The fake backend only needs the single-item path used by the store test."""
        output = self.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return [output.cache]


def _packet_store(backend, chunks: dict[str, list[int]]):
    """Preprocess explicit token sequences into a zero-filler packet store."""
    from kvpacket.packet import PacketWrapper, RawText, WrapperEntry, WrapperRegistry
    from kvpacket.preprocessing import PacketPreprocessor

    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"default": WrapperEntry(PacketWrapper(0, 0, backend.compatibility.hidden_size))},
    )
    preprocessor = PacketPreprocessor(backend=backend, wrapper_registry=registry)
    for chunk_id, token_ids in chunks.items():
        preprocessor.preprocess_pretokenized(
            RawText(f"pretokenized test source {chunk_id}", source_id=chunk_id),
            token_ids,
            wrapper_id="default",
        )
    return preprocessor.store


class _NonRerotatableBackend(_FakeBackend):
    @property
    def is_rerotatable(self):
        return False

    def assert_rerotatable(self):
        from kvpacket.backend import RerotationError

        raise RerotationError("test backend does not support packet re-rotation")


def test_engine_runs_on_custom_backend():
    """The engine drives a non-HF ComputeBackend end-to-end (SERVE + substitution)."""
    from kvpacket.backend import ComputeBackend
    from kvpacket.session import Mode, PacketSession

    backend = _FakeBackend()
    assert isinstance(backend, ComputeBackend)  # structural protocol conformance

    sess = PacketSession(
        backend=backend,
        store=_packet_store(backend, {"c": [1, 2, 3]}),
        mode=Mode.SERVE,
        match="token",
    )
    sess.register_store_chunks()
    sess.push([1, 2, 3])  # matched -> served from packet (re-rotated)
    sess.push([4, 5])  # trailing prompt
    res = sess.generate(max_new_tokens=3, eos_token_id=0)  # emits token 1, never EOS

    assert res.matched_chunk_ids == ["c"]
    assert res.cached_tokens == 3
    assert len(res.token_ids) == 3
    assert sess._offset == 3 + 2 + 3  # packet + prompt + generated


def test_session_sampling_uses_configured_warpers(monkeypatch):
    """Sampling is explicit; greedy remains the unchanged default."""
    from kvpacket.session import Mode, PacketSession

    sampled = []

    def fake_multinomial(probabilities, num_samples):
        sampled.append(probabilities.detach().clone())
        return torch.tensor([[2]], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    sess = PacketSession(backend=_FakeBackend(), mode=Mode.SERVE, match="token")
    sess.push([4, 5])
    result = sess.generate(
        max_new_tokens=1,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    assert result.token_ids == [2]
    assert len(sampled) == 1
    assert torch.isclose(sampled[0].sum(), torch.tensor(1.0))


def test_score_continuation_uses_fixed_tokens():
    from kvpacket.session import Mode, PacketSession

    sess = PacketSession(backend=_FakeBackend(), mode=Mode.SERVE, match="token")
    sess.push([4, 5])
    score = sess.score_continuation([1, 2])

    normalizer = torch.exp(torch.tensor(1.0)) + 9
    expected = [
        float(torch.log(normalizer) - 1.0),
        float(torch.log(normalizer)),
    ]
    assert score.token_ids == [1, 2]
    assert torch.allclose(torch.tensor(score.token_nll), torch.tensor(expected))
    assert score.mean_nll == sum(score.token_nll) / 2
    assert score.perplexity > 1
    assert sess._cache is None
    assert sess._offset == 0
    assert len(sess._pending) == 1
    repeated = sess.score_continuation([1, 2])
    assert repeated.token_nll == score.token_nll
    assert len(sess._pending) == 1


def test_generate_can_record_compact_teacher_tokens_without_logits():
    from kvpacket.session import Mode, PacketSession

    session = PacketSession(backend=_FakeBackend(), mode=Mode.RECORD, match="token")
    session.push([4, 5])
    result = session.generate(max_new_tokens=2, capture_logits=False, eos_token_id=[])
    sample = session.export_sample()

    assert sample is not None
    assert sample.token_ids.tolist() == [4, 5, *result.token_ids]
    assert sample.generated_mask.tolist() == [False, False, True, True]
    assert sample.logits is None


def test_add_forced_continuation_captures_fixed_trajectory_logits():
    import pytest

    from kvpacket.session import Mode, PacketSession

    with pytest.raises(RuntimeError, match="RECORD mode"):
        PacketSession(
            backend=_FakeBackend(),
            mode=Mode.SERVE,
        ).add_forced_continuation([1])

    session = PacketSession(
        backend=_FakeBackend(),
        mode=Mode.RECORD,
        match="token",
    )
    session.register_chunk("doc", token_ids=[2, 3])
    session.push([2, 3])
    session.push([4])
    result = session.add_forced_continuation([7, 8, 9])
    sample = session.export_sample()

    assert result.token_ids == [7, 8, 9]
    assert sample is not None and sample.logits is not None
    assert sample.token_ids.tolist() == [2, 3, 4, 7, 8, 9]
    assert sample.generated_mask.tolist() == [False, False, False, True, True, True]
    assert sample.chunk_spans == {"doc": [(0, 2)]}
    assert sample.logits.shape == (3, _FakeBackend.V)
    assert sample.logits.argmax(dim=-1).tolist() == [1, 1, 1]
    assert session._pending == []
    assert session._trajectory_ids == sample.token_ids.tolist()
    assert session._offset == len(sample.token_ids)


def test_add_forced_continuation_can_record_token_ids_only():
    from kvpacket.session import Mode, PacketSession

    session = PacketSession(backend=_FakeBackend(), mode=Mode.RECORD, match="token")
    session.register_chunk("doc", token_ids=[2, 3])
    session.push([2, 3])
    session.push([4])
    session.add_forced_continuation([7, 8], capture_logits=False)
    sample = session.export_sample()

    assert sample is not None
    assert sample.token_ids.tolist() == [2, 3, 4, 7, 8]
    assert sample.generated_mask.tolist() == [False, False, False, True, True]
    assert sample.logits is None
    assert session._pending == []
    assert session._offset == len(sample.token_ids)


def test_record_samples_include_all_committed_history():
    """Process, forced continuation, and generation share one logical trajectory."""
    from kvpacket.session import Mode, PacketSession

    session = PacketSession(backend=_FakeBackend(), mode=Mode.RECORD, match="token")
    session.register_chunk("doc", token_ids=[2, 3])
    session.push([2, 3])
    session.process()
    assert session.export_sample() is None

    session.push([4])
    session.add_forced_continuation([7, 8])
    first = session.export_sample()
    assert first is not None
    assert first.token_ids.tolist() == [2, 3, 4, 7, 8]
    assert first.generated_mask.tolist() == [False, False, False, True, True]
    assert first.chunk_spans == {"doc": [(0, 2)]}

    session.push([5])
    result = session.generate(max_new_tokens=1, eos_token_id=[])
    second = session.export_sample()
    assert second is not None
    assert result.token_ids == [1]
    assert second.token_ids.tolist() == [2, 3, 4, 7, 8, 5, 1]
    assert second.generated_mask.tolist() == [False] * 6 + [True]
    assert second.chunk_spans == {"doc": [(0, 2)]}
    assert session._trajectory_ids == second.token_ids.tolist()
    assert session._offset == len(second.token_ids)


def test_session_requires_explicit_supported_backend():
    """Backend selection is explicit; support policy is delegated to the backend."""
    import inspect
    import pytest

    from kvpacket.backend import RerotationError
    from kvpacket.session import Mode, PacketSession

    backend_param = inspect.signature(PacketSession).parameters["backend"]
    assert backend_param.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(RerotationError, match="test backend"):
        PacketSession(backend=_NonRerotatableBackend(), mode=Mode.SERVE)

    # RECORD never substitutes/re-rotates packets, so the same backend remains valid
    # for teacher sample generation.
    PacketSession(backend=_NonRerotatableBackend(), mode=Mode.RECORD)


def test_record_prefill_chunking_preserves_cache_and_teacher_logits():
    """Chunked RECORD prefill is causally equivalent on the backend seam."""
    import pytest

    from kvpacket.session import Mode, PacketSession

    with pytest.raises(ValueError, match="prefill_chunk_size"):
        PacketSession(
            backend=_FakeBackend(),
            mode=Mode.RECORD,
            prefill_chunk_size=0,
        )

    def run(chunk_size):
        session = PacketSession(
            backend=_FakeBackend(),
            mode=Mode.RECORD,
            match="token",
            prefill_chunk_size=chunk_size,
        )
        session.register_chunk("doc", token_ids=[2, 3, 4, 5, 6])
        session.push([2, 3, 4, 5, 6])
        session.push([7])
        result = session.generate(max_new_tokens=1, eos_token_id=1)
        sample = session.export_sample()
        assert sample is not None and sample.logits is not None
        return session._cache, sample.logits, result

    full_cache, full_logits, full_result = run(None)
    chunked_cache, chunked_logits, chunked_result = run(2)
    assert full_cache is not None and chunked_cache is not None
    assert full_cache.seq_len == chunked_cache.seq_len
    assert torch.equal(full_cache.layers[0].key, chunked_cache.layers[0].key)
    assert torch.equal(full_cache.layers[0].value, chunked_cache.layers[0].value)
    assert torch.equal(full_logits, chunked_logits)
    assert full_result.token_ids == chunked_result.token_ids


def test_serve_batches_cached_packets_into_one_concat(monkeypatch):
    """Lazy pushes stay free, and one trigger stitches a cached run only once."""
    import kvpacket.runtime.executor as executor_module
    from kvpacket.session import PacketSession

    backend = _FakeBackend()
    chunks = ([10, 11], [20, 21], [30, 31])
    packet_inputs = {f"c{index}": list(chunk) for index, chunk in enumerate(chunks)}
    session = PacketSession(
        backend=backend, store=_packet_store(backend, packet_inputs), match="token"
    )
    session.register_store_chunks()

    concat_calls = 0
    original_concat = executor_module.concat_caches

    def counted_concat(caches):
        nonlocal concat_calls
        concat_calls += 1
        return original_concat(caches)

    monkeypatch.setattr(executor_module, "concat_caches", counted_concat)
    for chunk in chunks:
        session.push(list(chunk))
    session.push([99])

    assert concat_calls == 0
    result = session.generate(
        max_new_tokens=1,
        eos_token_id=list(range(backend.V)),
    )
    assert result.matched_chunk_ids == ["c0", "c1", "c2"]
    assert concat_calls == 1


def test_immediate_process_materializes_uncached_context_before_later_push():
    """An explicit process() remains the boundary for an immediately needed gap."""
    from kvpacket.session import PacketSession

    class RecordingBackend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.online_calls: list[tuple[int, bool]] = []

        def forward(self, **kwargs):
            source = kwargs.get("input_ids")
            if source is None:
                source = kwargs.get("inputs_embeds")
            assert source is not None
            self.online_calls.append(
                (int(source.shape[1]), kwargs.get("attention_mask") is not None)
            )
            return super().forward(**kwargs)

    backend = RecordingBackend()
    chunks = {"c1": [10, 11], "c2": [20, 21], "c4": [40, 41]}
    session = PacketSession(
        backend=backend,
        store=_packet_store(backend, chunks),
        match="token",
        mixed_prefill_policy="fused",
    )
    session.register_store_chunks()
    backend.online_calls.clear()

    session.push([10, 11])
    session.push([20, 21])
    session.push([30, 31], match=False)
    session.process()
    assert backend.online_calls == [(2, False)]

    session.push([40, 41])
    session.push([50, 51], match=False)
    session.generate(
        max_new_tokens=1,
        eos_token_id=list(range(backend.V)),
    )
    assert backend.online_calls == [(2, False), (2, False), (1, False)]


def test_fused_mixed_prefill_matches_sequential_tiny_llama():
    """The lower-TTFT mixed policy preserves the logical attention graph."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket.backend import HFBackend
    from kvpacket.session import PacketSession

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().eval()
    backend = HFBackend(model)

    chunks = {"c1": [10, 11], "c2": [20, 21], "c4": [40, 41]}

    def run(policy: str):
        session = PacketSession(
            backend=backend,
            store=_packet_store(backend, chunks),
            match="token",
            mixed_prefill_policy=policy,
        )
        session.register_store_chunks()
        session.push([10, 11])
        session.push([20, 21])
        session.push([30, 31], match=False)
        session.push([40, 41])
        session.push([50, 51], match=False)
        result = session.generate(
            max_new_tokens=1,
            return_prompt_logits=True,
            eos_token_id=list(range(config.vocab_size)),
        )
        return result, session

    sequential, sequential_session = run("sequential")
    fused, fused_session = run("fused")
    assert sequential.prompt_logits is not None
    assert fused.prompt_logits is not None
    assert sequential.prompt_logits.shape == fused.prompt_logits.shape == (1, 2, 128)
    assert torch.allclose(
        sequential.prompt_logits,
        fused.prompt_logits,
        atol=2e-5,
        rtol=2e-5,
    )
    assert sequential.token_ids == fused.token_ids
    assert sequential_session._offset == fused_session._offset == 11
    assert fused_session._cache is not None
    assert fused_session._cache.layers[0].logical_positions.tolist() == [
        [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10]
    ]


def test_fused_mixed_process_persists_for_a_later_generation():
    """A fused process() result remains valid history for subsequent turns."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket.backend import HFBackend
    from kvpacket.session import PacketSession

    torch.manual_seed(11)
    config = LlamaConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    backend = HFBackend(LlamaForCausalLM(config).float().eval())

    chunks = {"c1": [10, 11], "c2": [20, 21]}

    def run(policy: str):
        session = PacketSession(
            backend=backend,
            store=_packet_store(backend, chunks),
            match="token",
            mixed_prefill_policy=policy,
        )
        session.register_store_chunks()
        session.push([10, 11])
        session.push([30, 31], match=False)
        session.push([20, 21])
        session.process()
        session.push([50, 51], match=False)
        return session.generate(
            max_new_tokens=1,
            return_prompt_logits=True,
            eos_token_id=list(range(config.vocab_size)),
        )

    sequential = run("sequential")
    fused = run("fused")
    assert sequential.prompt_logits is not None
    assert fused.prompt_logits is not None
    assert torch.allclose(
        sequential.prompt_logits,
        fused.prompt_logits,
        atol=2e-5,
        rtol=2e-5,
    )
    assert sequential.token_ids == fused.token_ids


def test_hf_backend_owns_model_support_checks():
    """HF-specific layout and RoPE policy never leak into PacketSession."""
    import pytest

    from kvpacket.backend import HFBackend, RerotationError

    class Config:
        hidden_size = 8
        num_hidden_layers = 1
        model_type = "test"
        rope_scaling = {"rope_type": "dynamic"}

    class Base:
        embed_tokens = object()
        rotary_emb = object()

    class Model:
        config = Config()
        model = Base()
        device = torch.device("cpu")

    backend = HFBackend(Model())
    with pytest.raises(RerotationError, match="dynamic"):
        backend.assert_rerotatable()

    class MissingRotary:
        config = Config()
        model = type("Base", (), {"embed_tokens": object()})()

    with pytest.raises(AttributeError, match="rotary_emb"):
        HFBackend(MissingRotary())


def test_engine_has_no_transformers_dependency():
    """Regression guard: the engine speaks only KVCache + the backend protocol."""
    import inspect

    import kvpacket.session.session as S

    src = inspect.getsource(S)
    assert "transformers" not in src and "DynamicCache" not in src


def _run_light():
    test_engine_runs_on_custom_backend()
    print("[ok] engine runs on a non-HF backend")
    test_session_requires_explicit_supported_backend()
    print("[ok] explicit backend + delegated support check")
    test_hf_backend_owns_model_support_checks()
    print("[ok] HF backend owns model support checks")
    test_engine_has_no_transformers_dependency()
    print("[ok] engine free of transformers")
    test_find_occurrences()
    print("[ok] find_occurrences")
    test_token_cover_basic()
    print("[ok] token cover basic")
    test_token_cover_overlap_prefers_more_coverage()
    print("[ok] token cover overlap")
    test_weighted_interval_cover_tie_prefers_earliest()
    print("[ok] interval cover tie-break")
    test_string_cover()
    print("[ok] string cover")
    test_training_sample_roundtrip()
    print("[ok] training sample roundtrip")


def _tiny_native_generation_backend():
    """Create a deterministic local HF backend without downloading weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

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
    model = LlamaForCausalLM(config).float().eval()

    from kvpacket.backend import HFBackend

    return model, HFBackend(model)


def test_hf_native_generation_matches_generate_and_returns_reusable_cache():
    """Native backend continuation preserves HF tokens and returns a reusable cache."""
    from kvpacket.backend import GenerationBackend

    torch.manual_seed(17)
    model, backend = _tiny_native_generation_backend()
    assert isinstance(backend, GenerationBackend)
    prefix = backend.prefill_caches(input_ids=torch.tensor([[7, 8]]))[0]
    prompt = torch.tensor([[1, 2, 3]])
    positions = torch.arange(2, 5).unsqueeze(0)
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    reference = model.generate(
        inputs=prompt,
        past_key_values=backend._to_dynamic(prefix),
        position_ids=positions,
        attention_mask=attention_mask,
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=False,
    )

    result = backend.generate(
        input_ids=prompt,
        position_ids=positions,
        cache=prefix,
        max_new_tokens=3,
        eos_token_id=[],
    )

    assert result.token_ids == reference[0, prompt.shape[1] :].tolist()
    assert result.cache.seq_len == result.final_offset == 8
    assert result.cache.layers[0].logical_positions.tolist() == [list(range(8))]
    assert result.logits == []
    logit_reference = model.generate(
        inputs=prompt,
        past_key_values=backend._to_dynamic(prefix),
        position_ids=positions,
        attention_mask=attention_mask,
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )
    captured = backend.generate(
        input_ids=prompt,
        position_ids=positions,
        cache=prefix,
        max_new_tokens=3,
        eos_token_id=[],
        capture_logits=True,
    )
    assert captured.token_ids == logit_reference.sequences[0, prompt.shape[1] :].tolist()
    assert len(captured.logits) == len(captured.token_ids)
    for actual, expected in zip(captured.logits, logit_reference.logits, strict=True):
        torch.testing.assert_close(actual, expected[0])
    torch.manual_seed(23)
    sampled_reference = model.generate(
        inputs=prompt,
        past_key_values=backend._to_dynamic(prefix),
        position_ids=positions,
        attention_mask=attention_mask,
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.05,
    )
    torch.manual_seed(23)
    sampled = backend.generate(
        input_ids=prompt,
        position_ids=positions,
        cache=prefix,
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.05,
    )
    assert sampled.token_ids == sampled_reference[0, prompt.shape[1] :].tolist()


def test_packet_session_owns_ttft_and_uses_native_continuation(monkeypatch):
    """The session times its first token before delegating native continuation."""
    from kvpacket.session import PacketSession

    model, backend = _tiny_native_generation_backend()
    native_generate = backend.generate
    calls = 0

    def counted_generate(**kwargs):
        nonlocal calls
        calls += 1
        return native_generate(**kwargs)

    monkeypatch.setattr(backend, "generate", counted_generate)
    prompt = torch.tensor([[1, 2, 3]])
    reference = model.generate(inputs=prompt, max_new_tokens=2, eos_token_id=[], do_sample=False)
    session = PacketSession(backend=backend, match="token")
    session.push([1, 2, 3], match=False)
    result = session.generate(max_new_tokens=2, eos_token_id=[])

    assert result.token_ids == reference[0, prompt.shape[1] :].tolist()
    torch.manual_seed(29)
    sampled_reference = model.generate(
        inputs=prompt,
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.05,
    )
    torch.manual_seed(29)
    sampled_session = PacketSession(backend=backend, match="token")
    sampled_session.push([1, 2, 3], match=False)
    sampled_result = sampled_session.generate(
        max_new_tokens=3,
        eos_token_id=[],
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=10,
        min_p=0.05,
    )
    assert sampled_result.token_ids == sampled_reference[0, prompt.shape[1] :].tolist()
    assert calls == 2
    assert result.ttft > 0.0
    assert session._cache is not None
    assert session._cache.seq_len == session._offset == 5


def test_packet_session_native_continuation_records_all_logits(monkeypatch):
    """RECORD mode combines framework and native raw logits without gaps."""
    from kvpacket.session import Mode, PacketSession

    _, backend = _tiny_native_generation_backend()
    native_generate = backend.generate
    calls = 0

    def counted_generate(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["capture_logits"] is True
        return native_generate(**kwargs)

    monkeypatch.setattr(backend, "generate", counted_generate)
    prompt = torch.tensor([[1, 2, 3]])
    session = PacketSession(backend=backend, mode=Mode.RECORD, match="token")
    session.push(prompt[0].tolist(), match=False)
    result = session.generate(max_new_tokens=3, eos_token_id=[])
    sample = session.export_sample()

    assert calls == 1
    assert sample is not None and sample.logits is not None
    assert sample.logits.shape == (3, 64)

    output = backend.forward(input_ids=prompt, position_ids=torch.arange(3).unsqueeze(0))
    cache = output.cache
    expected = []
    for position, token_id in enumerate(result.token_ids, start=3):
        expected.append(output.logits[0, -1].detach())
        output = backend.forward(
            input_ids=torch.tensor([[token_id]]),
            position_ids=torch.tensor([[position]]),
            cache=cache,
        )
        cache = output.cache
    torch.testing.assert_close(sample.logits, torch.stack(expected))


# ---------------------------------------------------------------------- live
def _load_model(dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpacket.backend import HFBackend

    name = "Qwen/Qwen2.5-1.5B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    assert tok is not None
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    model.eval()
    return HFBackend(model), tok, device


def _check_serve_matches_manual_stitch(adapter, tok, device):
    """D1.1 — SERVE (no wrapper) reproduces the Phase-1 manual stitch exactly."""
    from kvpacket.kvcache.cache import concat_caches
    from kvpacket.kvcache.positions import cumsum_pos, shifted_cumsum_pos
    from kvpacket.session import Mode, PacketSession

    docs = ["The capital of France is Paris.", " Water boils at 100 degrees Celsius."]
    query = " Q: What is the capital of France? A:"

    # ---- manual stitch (exactly the Phase-1 live path) ----
    caches = []
    for d in docs:
        ids = tok([d], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        caches.append(adapter.prefill_caches(input_ids=ids)[0])
    lens = [c.doc_len for c in caches]
    stitched = concat_caches(caches)
    old = cumsum_pos(lens, device).unsqueeze(0)
    new = shifted_cumsum_pos(lens, lens, device).unsqueeze(0)
    rotated = stitched.rerotate(adapter.rotary_emb, old, new, nope_dim=adapter.nope_dim)
    total = sum(lens)
    q_ids = tok([query], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    hf = adapter._to_dynamic(rotated)  # white-box: feed the raw HF model directly
    pos = torch.arange(total, total + q_ids.size(1), device=device).unsqueeze(0)
    with torch.no_grad():  # raw HF model = the reference path the engine must reproduce
        manual_logits = adapter.model(input_ids=q_ids, past_key_values=hf, position_ids=pos).logits

    # ---- SERVE session: zero-filler document packets, query pushed ----
    packet_inputs = {
        f"doc{index}": tok(document, add_special_tokens=False)["input_ids"]
        for index, document in enumerate(docs)
    }
    sess = PacketSession(
        backend=adapter,
        store=_packet_store(adapter, packet_inputs),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tok,
    )
    sess.register_store_chunks()
    for d in docs:
        sess.push(d)
    sess.push(query)
    res = sess.generate(max_new_tokens=1, return_prompt_logits=True)

    assert res.prompt_logits is not None
    assert res.prompt_logits.shape == manual_logits.shape, (
        res.prompt_logits.shape,
        manual_logits.shape,
    )
    max_diff = (res.prompt_logits - manual_logits).abs().max().item()
    argmax_match = torch.equal(res.prompt_logits.argmax(-1), manual_logits.argmax(-1))
    assert len(res.cover) == 2, f"expected both docs matched, got {res.cover}"
    assert argmax_match, "argmax over prompt differs from manual stitch"
    assert max_diff < 1e-3, f"SERVE vs manual-stitch logits differ: max_diff={max_diff}"
    print(
        f"[live] D1.1 SERVE==manual-stitch  max_logit_diff={max_diff:.2e}  cover={[m.chunk_id for m in res.cover]}"
    )


def _check_matching_across_pushes(adapter, tok, device):
    """D1.2 — a chunk split across two pushes is still matched; match=False opts out."""
    from kvpacket.session import Mode, PacketSession

    chunk_ids = tok("The capital of France is Paris.", add_special_tokens=False)["input_ids"]
    half = len(chunk_ids) // 2

    sess = PacketSession(
        backend=adapter,
        store=_packet_store(adapter, {"fact": chunk_ids}),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tok,
    )
    sess.register_store_chunks()

    # chunk spans two pushes (as ids, so concatenation is exact)
    sess.push(chunk_ids[:half])
    sess.push(chunk_ids[half:])
    sess.push(" and more text")
    stream, records = sess._build_stream()
    cover = sess._match(stream, records)
    assert cover == [Match("fact", 0, len(chunk_ids))], cover

    # opt-out: same content, match=False -> no cover
    sess.reset()
    sess.push(chunk_ids, match=False)
    sess.push(" tail")
    stream, records = sess._build_stream()
    assert sess._match(stream, records) == []
    print("[live] D1.2 matching across pushes + match=False opt-out")


def _check_record_sample(adapter, tok, device):
    """D1.3 — RECORD yields a TrainingSample with correct chunk_spans + generated_mask."""
    from kvpacket.session import Mode, PacketSession

    skill_ids = tok("Use the calculator tool.", add_special_tokens=False)["input_ids"]
    pre_ids = tok("System preamble. ", add_special_tokens=False)["input_ids"]

    sess = PacketSession(
        backend=adapter,
        mode=Mode.RECORD,
        match="token",
        tokenizer=tok,
    )
    sess.register_chunk("skill:calc", token_ids=skill_ids)
    sess.push(pre_ids)  # unmatched
    sess.push(skill_ids)  # the skill (matched, but RECORD does not substitute)
    sess.push(" Now: 2+2=")  # trailing prompt
    res = sess.generate(max_new_tokens=4)
    sample = sess.export_sample()

    assert sample is not None
    span_start = len(pre_ids)
    assert sample.chunk_spans == {"skill:calc": [(span_start, span_start + len(skill_ids))]}, (
        sample.chunk_spans
    )
    assert sample.num_generated == len(res.token_ids)
    # generated mask marks exactly the trajectory tail
    assert bool(sample.generated_mask[-1]) and not bool(sample.generated_mask[0])
    assert sample.logits is not None and sample.logits.shape[0] == len(res.token_ids)
    print(f"[live] D1.3 RECORD sample: {sample}")


def _check_multi_turn_offsets(adapter, tok, device):
    """D1.4 — offsets grow correctly across turns; a re-used packet re-rotates."""
    from kvpacket.session import Mode, PacketSession

    doc = "Backpropagation computes gradients via the chain rule."
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"]
    wl = len(doc_ids)  # no wrapper -> wrapped_len == body_len

    sess = PacketSession(
        backend=adapter,
        store=_packet_store(adapter, {"doc": doc_ids}),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tok,
    )
    sess.register_store_chunks()

    sess.push(doc)
    sess.push(" Q1: what rule? A:")
    q1 = tok(" Q1: what rule? A:", add_special_tokens=False)["input_ids"]
    r1 = sess.generate(max_new_tokens=3)
    expected1 = wl + len(q1) + len(r1.token_ids)
    assert sess._offset == expected1, (sess._offset, expected1)

    # turn 2: re-use the same doc packet (re-rotated to the new offset) + a new query
    sess.push(doc)
    sess.push(" Q2: summarize. A:")
    q2 = tok(" Q2: summarize. A:", add_special_tokens=False)["input_ids"]
    r2 = sess.generate(max_new_tokens=3)
    expected2 = expected1 + wl + len(q2) + len(r2.token_ids)
    assert sess._offset == expected2, (sess._offset, expected2)
    assert len(r2.cover) == 1 and r2.cover[0].chunk_id == "doc"
    print(f"[live] D1.4 multi-turn offsets: turn1={expected1}, turn2={expected2}")


def _check_store_roundtrip(adapter, tok, device):
    """D1.5 — PacketStore save/load roundtrip (with a wrapper)."""
    from kvpacket.packet import PacketWrapper, RawText, WrapperEntry, WrapperRegistry
    from kvpacket.session.store import PacketStore

    w = PacketWrapper(
        header_len=2,
        trailer_len=2,
        dim=adapter.hidden_size,
        dtype=next(adapter.model.parameters()).dtype,
        device=device,
    )
    from kvpacket.preprocessing import PacketPreprocessor

    registry = WrapperRegistry(
        compatibility=adapter.compatibility,
        wrappers={"skill": WrapperEntry(w)},
    )
    preprocessor = PacketPreprocessor(backend=adapter, wrapper_registry=registry)
    store = preprocessor.store
    text = "A reusable skill body."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    sp = preprocessor.preprocess_pretokenized(
        RawText(text, source_id="skill"), ids, wrapper_id="skill"
    )
    assert sp.wrapped_len == len(ids) + 4

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "store.pt")
        store.save(p)
        store2 = PacketStore.load(p, backend=adapter, device=device)
    a = store.get("skill").packet.layers[0].key
    b = store2.get("skill").packet.layers[0].key
    assert a.shape == b.shape and torch.allclose(a, b)
    print("[live] D1.5 PacketStore save/load roundtrip")


def _check_observability(adapter, tok, device):
    """Caller-visible matches: preview_matches() (dry run) + result.cover/cached_tokens."""
    from kvpacket.session import Mode, PacketSession

    doc = "The mitochondria is the powerhouse of the cell."
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"]

    sess = PacketSession(
        backend=adapter,
        store=_packet_store(adapter, {"bio": doc_ids}),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tok,
    )
    sess.register_store_chunks()
    sess.push("Context: ")
    sess.push(doc)
    sess.push(" Q: what is it? A:")

    # dry-run: the caller sees the match (and thus the reuse) BEFORE committing/generating
    preview = sess.preview_matches()
    assert [m.chunk_id for m in preview] == ["bio"], preview
    assert preview[0].length == len(doc_ids)

    res = sess.generate(max_new_tokens=1)
    assert res.matched_chunk_ids == ["bio"]
    assert res.cached_tokens == len(doc_ids)  # doc served from packet, not re-prefilled
    print(
        f"[live] observability: preview+result expose match 'bio' "
        f"(cached_tokens={res.cached_tokens}, rerotate_flops={res.flops:,})"
    )


def _run_live():
    print("\n[live] loading Qwen/Qwen2.5-1.5B (float32) ...")
    adapter, tok, device = _load_model(dtype=torch.float32)
    print(f"[live] {adapter}")
    _check_serve_matches_manual_stitch(adapter, tok, device)
    _check_matching_across_pushes(adapter, tok, device)
    _check_record_sample(adapter, tok, device)
    _check_multi_turn_offsets(adapter, tok, device)
    _check_store_roundtrip(adapter, tok, device)
    _check_observability(adapter, tok, device)
    print("[live] all Deliverable-1 checks passed.")


if __name__ == "__main__":
    _run_light()
    if "--no-model" not in sys.argv:
        _run_live()
    print("\nAll Phase 2 checks passed.")
