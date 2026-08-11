"""D5 tests for serving preprocessed mixed-wrapper packets through PacketSession."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from kvpacket.backend import BackendCompatibility, ForwardOutput
from kvpacket.errors import BackendCompatibilityError, UnsupportedModalityError
from kvpacket.fingerprint import fingerprint_json
from kvpacket.kvcache import KeyValue, KVCache
from kvpacket.packet import EncodedBody, PacketWrapper, RawText, WrapperEntry, WrapperRegistry
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import PacketStore, Mode, PacketSession


class _CharacterTokenizer:
    eos_token_id = None
    chat_template = None

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False
    ):
        del add_special_tokens
        result: dict[str, object] = {"input_ids": [(ord(character) % 97) + 1 for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr((token - 1) % 26 + ord("a")) for token in token_ids)


class _ServingBackend:
    hidden_size = 8
    heads = 2
    head_dim = 4
    vocab_size = 128

    def __init__(self) -> None:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        config = LlamaConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.heads,
            num_key_value_heads=self.heads,
            num_hidden_layers=1,
            head_dim=self.head_dim,
        )
        self._rotary = LlamaRotaryEmbedding(config)
        self._compatibility = BackendCompatibility(
            backend="test-serving",
            model_id="test/serving-model",
            model_config_fingerprint=fingerprint_json(
                {"hidden_size": self.hidden_size}, domain="tests.serving_model/1"
            ),
            hidden_size=self.hidden_size,
            model_artifact_fingerprint=fingerprint_json(
                "test/serving-model", domain="tests.serving_artifact/1"
            ),
        )
        self.tokenizer = _CharacterTokenizer()
        self.prefill_calls = 0

    @property
    def compatibility(self) -> BackendCompatibility:
        return self._compatibility

    @property
    def device(self) -> torch.device:
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

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float().unsqueeze(-1).expand(-1, -1, self.hidden_size)

    def _cache(self, source: torch.Tensor, position_ids: torch.Tensor) -> KVCache:
        batch, sequence = source.shape[:2]
        base = position_ids.float().reshape(batch, 1, sequence, 1)
        keys = base.expand(batch, self.heads, sequence, self.head_dim).contiguous()
        values = keys.clone()
        return KVCache([KeyValue(keys, values, position_ids)])

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        cache=None,
    ) -> ForwardOutput:
        del attention_mask
        source = inputs_embeds if inputs_embeds is not None else input_ids
        assert source is not None
        batch, sequence = source.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(sequence).unsqueeze(0).expand(batch, -1)
        new_cache = self._cache(source, position_ids)
        if cache is not None:
            old = cache.layers[0]
            new = new_cache.layers[0]
            new_cache = KVCache(
                [
                    KeyValue(
                        torch.cat([old.key, new.key], dim=2),
                        torch.cat([old.value, new.value], dim=2),
                        torch.cat([old.logical_positions, new.logical_positions], dim=1),
                    )
                ]
            )
        logits = torch.zeros(batch, sequence, self.vocab_size)
        logits[..., 1] = 1.0
        return ForwardOutput(logits=logits, cache=new_cache)

    def prefill_caches(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
    ) -> list[KVCache]:
        del attention_mask
        self.prefill_calls += 1
        source = inputs_embeds if inputs_embeds is not None else input_ids
        assert source is not None
        sequence = source.shape[1]
        positions = (
            position_ids if position_ids is not None else torch.arange(sequence).unsqueeze(0)
        )
        return [self._cache(source, positions)]


class _ServingSourceEncoder:
    def __init__(self, tokenizer: _CharacterTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, source) -> EncodedBody:
        content = source.value.encode("utf-8")
        if source.media_type == "application/x-nonmatchable":
            embeddings = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
            return EncodedBody(
                content=content,
                media_type=source.media_type,
                inputs_embeds=embeddings,
            )
        token_ids = tuple(self.tokenizer(source.value)["input_ids"])
        ids = torch.tensor([token_ids], dtype=torch.long)
        return EncodedBody(
            content=content,
            media_type=source.media_type or "text/plain",
            input_ids=ids,
            token_ids=token_ids,
            match_text=source.value,
        )


def _preprocessed_store(backend: _ServingBackend) -> PacketStore:
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={
            "document": WrapperEntry(PacketWrapper(1, 1, backend.hidden_size)),
            "skill": WrapperEntry(PacketWrapper(2, 1, backend.hidden_size)),
        },
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        source_encoder=_ServingSourceEncoder(backend.tokenizer),
        wrapper_registry=registry,
    )
    preprocessor.preprocess(
        {
            RawText("DOC", source_id="document-1"): "document",
            RawText("SKILL", source_id="skill-1"): "skill",
        },
    )
    return preprocessor.store


def test_mixed_wrapper_packets_serve_in_one_turn_and_reuse_across_turns():
    backend = _ServingBackend()
    store = _preprocessed_store(backend)
    assert backend.prefill_calls == 2
    assert store.get("document-1").identity.wrapper_id == "document"
    assert store.get("skill-1").identity.wrapper_id == "skill"

    session = PacketSession(
        backend=backend,
        store=store,
        tokenizer=backend.tokenizer,
        mode=Mode.SERVE,
        match="token",
    )
    assert session.register_store_chunks() == ["document-1", "skill-1"]
    session.push("DOC")
    session.push("SKILL")
    session.push("?")
    first = session.generate(max_new_tokens=1, eos_token_id=0)
    assert first.matched_chunk_ids == ["document-1", "skill-1"]
    assert first.cached_tokens == len("DOCSKILL")
    assert backend.prefill_calls == 2

    session.push("SKILL")
    session.push("DOC")
    session.push("!")
    second = session.generate(max_new_tokens=1, eos_token_id=0)
    assert second.matched_chunk_ids == ["skill-1", "document-1"]
    assert backend.prefill_calls == 2

    assert "wrapper" not in inspect.signature(PacketSession.push).parameters
    assert "wrapper" not in inspect.signature(PacketSession.generate).parameters


def test_loaded_store_registers_for_token_and_string_matching(tmp_path: Path):
    backend = _ServingBackend()
    original = _preprocessed_store(backend)
    path = tmp_path / "packets.pt"
    original.save(path)
    loaded = PacketStore.load(path, backend=backend)

    token_session = PacketSession(
        backend=backend, store=loaded, tokenizer=backend.tokenizer, match="token"
    )
    token_session.register_store_chunks(["document-1"])
    token_session.push("DOC")
    token_session.push("?")
    assert token_session.generate(max_new_tokens=1).matched_chunk_ids == ["document-1"]

    string_session = PacketSession(
        backend=backend, store=loaded, tokenizer=backend.tokenizer, match="string"
    )
    string_session.register_store_chunks()
    string_session.push("DOCSKILL?")
    result = string_session.generate(max_new_tokens=1)
    assert result.matched_chunk_ids == ["document-1", "skill-1"]


def test_register_chunk_reuses_preprocessed_packet_without_rebuilding():
    backend = _ServingBackend()
    store = _preprocessed_store(backend)
    session = PacketSession(backend=backend, store=store, tokenizer=backend.tokenizer)
    session.register_chunk("document-1", text="DOC")
    assert backend.prefill_calls == 2
    with pytest.raises(ValueError, match="does not match its preprocessed packet"):
        session.register_chunk("document-1", text="different")


def test_nonmatchable_backend_representation_cannot_be_auto_registered():
    backend = _ServingBackend()
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"image": WrapperEntry(PacketWrapper(1, 1, backend.hidden_size))},
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        source_encoder=_ServingSourceEncoder(backend.tokenizer),
        wrapper_registry=registry,
    )
    preprocessor.preprocess(
        {
            RawText(
                "opaque-image",
                source_id="image-1",
                media_type="application/x-nonmatchable",
            ): "image"
        }
    )
    session = PacketSession(
        backend=backend,
        store=preprocessor.store,
        tokenizer=backend.tokenizer,
    )
    with pytest.raises(UnsupportedModalityError, match="no framework-produced token"):
        session.register_store_chunks()


def test_registration_rejects_session_tokenizer_mismatch():
    backend = _ServingBackend()
    store = _preprocessed_store(backend)

    class DifferentTokenizer(_CharacterTokenizer):
        def __call__(
            self,
            text: str,
            add_special_tokens: bool = False,
            return_offsets_mapping: bool = False,
        ):
            result = super().__call__(text, add_special_tokens, return_offsets_mapping)
            original = result["input_ids"]
            assert isinstance(original, list)
            result["input_ids"] = [token + 1 for token in original]
            return result

    session = PacketSession(
        backend=backend,
        store=store,
        tokenizer=DifferentTokenizer(),
    )
    with pytest.raises(BackendCompatibilityError, match="does not reproduce"):
        session.register_store_chunks()


def test_record_sample_is_wrapper_agnostic():
    backend = _ServingBackend()
    store = _preprocessed_store(backend)
    session = PacketSession(
        backend=backend,
        store=store,
        tokenizer=backend.tokenizer,
        mode=Mode.RECORD,
    )
    session.register_store_chunks()
    session.push("DOC")
    session.push("SKILL")
    session.push("?")
    session.generate(max_new_tokens=1)
    sample = session.export_sample()
    assert sample is not None
    assert set(sample.chunk_spans) == {"document-1", "skill-1"}
    assert not hasattr(sample, "default_wrapper")
    assert not hasattr(sample, "chunk_wrappers")


def test_record_session_needs_no_wrapper_configuration():
    backend = _ServingBackend()
    session = PacketSession(
        backend=backend,
        tokenizer=backend.tokenizer,
        mode=Mode.RECORD,
    )
    session.register_chunk("doc", text="DOC")
    session.push("DOC")
    session.push("?")
    session.generate(max_new_tokens=1)
    sample = session.export_sample()
    assert sample is not None
    assert set(sample.chunk_spans) == {"doc"}
