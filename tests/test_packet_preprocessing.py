"""D4 tests for registry-aware preprocessing and identity-safe persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kvpacket.backend import BackendCompatibility, ForwardOutput, HFBackend
from kvpacket.errors import (
    PacketIdentityError,
    SourceConflictError,
    UnsupportedModalityError,
)
from kvpacket.fingerprint import fingerprint_bytes, fingerprint_json, fingerprint_tensor
from kvpacket.kvcache import KeyValue, KVCache
from kvpacket.packet import (
    EncodedBody,
    FileSource,
    PacketWrapper,
    WrapperEntry,
    WrapperRegistry,
    RawText,
    TextSourceEncoder,
)
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import PacketStore


class _PreprocessBackend:
    hidden_size = 6

    def __init__(self, *, model_id: str = "test/model") -> None:
        self._compatibility = BackendCompatibility(
            backend="test",
            model_id=model_id,
            model_config_fingerprint=fingerprint_json(
                {"hidden_size": self.hidden_size}, domain="tests.preprocess_model/1"
            ),
            hidden_size=self.hidden_size,
            model_artifact_fingerprint=fingerprint_json(
                model_id, domain="tests.preprocess_artifact/1"
            ),
        )
        self.prefill_calls = 0

    @property
    def compatibility(self) -> BackendCompatibility:
        return self._compatibility

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def rotary_emb(self):
        return object()

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

    def forward(self, **kwargs) -> ForwardOutput:
        raise NotImplementedError

    def prefill_caches(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
    ) -> list[KVCache]:
        self.prefill_calls += 1
        source = inputs_embeds if inputs_embeds is not None else input_ids
        assert source is not None
        batch, sequence = source.shape[:2]
        positions = torch.arange(sequence).unsqueeze(0).expand(batch, -1)
        values = source[..., :2].unsqueeze(1).contiguous()
        return [KVCache([KeyValue(values, values.clone(), positions)])]


class _PreprocessEncoder:
    def encode(self, source) -> EncodedBody:
        if source.kind == "raw-text":
            content = source.value.encode("utf-8")
            media_type = source.media_type or "text/plain"
        else:
            content = Path(source.value).read_bytes()
            media_type = source.media_type or "text/plain"
        text = content.decode("utf-8")
        token_ids = tuple((ord(character) % 61) + 1 for character in text)
        ids = torch.tensor([token_ids], dtype=torch.long)
        return EncodedBody(
            content=content,
            media_type=media_type,
            input_ids=ids,
            token_ids=token_ids,
        )


def _collection(backend: _PreprocessBackend) -> WrapperRegistry:
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"biography_doc": WrapperEntry(PacketWrapper(1, 2, backend.hidden_size))},
    )
    registry.add("powerpoint_skill", PacketWrapper(2, 2, backend.hidden_size))
    registry.add("unused_wrapper", PacketWrapper(1, 1, backend.hidden_size))
    return registry


def _preprocessor(
    backend: _PreprocessBackend,
    registry: WrapperRegistry,
) -> PacketPreprocessor:
    return PacketPreprocessor(
        backend=backend,
        source_encoder=_PreprocessEncoder(),
        wrapper_registry=registry,
    )


def test_preprocess_selects_per_source_wrappers_and_records_identity(tmp_path: Path):
    backend = _PreprocessBackend()
    registry = _collection(backend)
    skill_path = tmp_path / "skill.md"
    skill_path.write_text("slides", encoding="utf-8")
    assignments = {
        RawText("biography", source_id="bio"): "biography_doc",
        FileSource(
            str(skill_path), source_id="skill", media_type="text/markdown"
        ): "powerpoint_skill",
    }

    preprocessor = _preprocessor(backend, registry)
    packets = preprocessor.preprocess(assignments)

    assert [packet.chunk_id for packet in packets] == ["bio", "skill"]
    assert packets[0].wrapped_len == packets[0].body_len + 3
    assert packets[1].wrapped_len == packets[1].body_len + 4
    assert packets[0].identity.wrapper_id == "biography_doc"
    assert packets[1].identity.wrapper_id == "powerpoint_skill"
    assert (
        packets[0].identity.backend_compatibility_fingerprint == backend.compatibility.fingerprint
    )
    assert packets[0].token_ids is not None
    assert backend.prefill_calls == 2


def test_preprocess_reuses_valid_packet_but_rebuilds_changed_file(tmp_path: Path):
    backend = _PreprocessBackend()
    registry = _collection(backend)
    document = tmp_path / "document.txt"
    document.write_text("first", encoding="utf-8")
    source = FileSource(str(document), source_id="doc", media_type="text/plain")
    assignments = {source: "biography_doc"}
    preprocessor = _preprocessor(backend, registry)

    first = preprocessor.preprocess(assignments)[0]
    assert backend.prefill_calls == 1
    second = preprocessor.preprocess(assignments)[0]
    assert second.identity.cache_key == first.identity.cache_key
    assert backend.prefill_calls == 1

    document.write_text("changed", encoding="utf-8")
    with pytest.raises(PacketIdentityError, match="stale"):
        preprocessor.preprocess(assignments)
    assert backend.prefill_calls == 1
    rebuilt = preprocessor.preprocess(assignments, rebuild=True)[0]
    assert rebuilt.identity.cache_key != first.identity.cache_key
    assert backend.prefill_calls == 2


def test_unrelated_wrapper_change_does_not_invalidate_packet():
    backend = _PreprocessBackend()
    registry = _collection(backend)
    assignments = {RawText("body", source_id="doc"): "biography_doc"}
    preprocessor = _preprocessor(backend, registry)
    first = preprocessor.preprocess(assignments)[0]

    with torch.no_grad():
        registry["unused_wrapper"].wrapper.header[0, 0, 0] += 1
    refreshed = preprocessor.preprocess(assignments)[0]
    assert refreshed.identity.cache_key == first.identity.cache_key
    assert backend.prefill_calls == 1


def test_pretokenized_packet_requires_original_source_content():
    backend = _PreprocessBackend()
    registry = _collection(backend)
    preprocessor = _preprocessor(backend, registry)
    source = RawText("original text", source_id="doc")

    packet = preprocessor.preprocess_pretokenized(source, [1, 2, 3], wrapper_id="biography_doc")

    assert packet.identity.source_content_fingerprint == fingerprint_bytes(
        b"original text", domain="kvpacket.source_content/1"
    )
    assert packet.identity.encoded_input_fingerprint == fingerprint_tensor(
        torch.tensor([[1, 2, 3]], dtype=torch.long)
    )
    with pytest.raises(PacketIdentityError, match="stale"):
        preprocessor.preprocess_pretokenized(
            RawText("changed text", source_id="doc"),
            [1, 2, 3],
            wrapper_id="biography_doc",
        )


def test_same_matchable_tokens_with_different_wrappers_fail_without_partial_build():
    backend = _PreprocessBackend()
    preprocessor = _preprocessor(backend, _collection(backend))
    assignments = {
        RawText("same", source_id="first"): "biography_doc",
        RawText("same", source_id="second"): "powerpoint_skill",
    }
    with pytest.raises(SourceConflictError, match="same matchable tokens"):
        preprocessor.preprocess(assignments)
    assert len(preprocessor.store) == 0
    assert backend.prefill_calls == 0


def test_identity_safe_store_roundtrip(tmp_path: Path):
    backend = _PreprocessBackend()
    registry = _collection(backend)
    preprocessor = _preprocessor(backend, registry)
    preprocessor.preprocess({RawText("persist me", source_id="doc"): "biography_doc"})

    path = tmp_path / "packet-store.pt"
    preprocessor.store.save(path)
    loaded = PacketStore.load(path, backend=backend)
    assert loaded.get("doc").identity == preprocessor.store.get("doc").identity
    assert torch.equal(
        loaded.get("doc").packet.layers[0].key,
        preprocessor.store.get("doc").packet.layers[0].key,
    )

    other_backend = _PreprocessBackend(model_id="other/model")
    with pytest.raises(PacketIdentityError, match="different backend/model"):
        PacketStore.load(path, backend=other_backend)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["packets"]["doc"]["identity"]["wrapper_id"] = "tampered"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(checkpoint, tampered_path)
    with pytest.raises(PacketIdentityError, match="cache identity mismatch"):
        PacketStore.load(tampered_path, backend=backend)


def test_legacy_identityless_store_is_rejected(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    torch.save({"doc": {"token_ids": [1]}}, path)
    with pytest.raises(PacketIdentityError, match="legacy packet store"):
        PacketStore.load(path, backend=_PreprocessBackend())


class _Processor:
    name_or_path = "test/tokenizer"
    init_kwargs: dict[str, object] = {}
    special_tokens_map: dict[str, object] = {}

    def get_vocab(self):
        return {"token": 1}

    def __call__(self, text: str, **kwargs):
        del kwargs
        return {"input_ids": [[(ord(character) % 31) + 1 for character in text]]}


def _hf_backend() -> HFBackend:
    class Config:
        hidden_size = 4
        num_hidden_layers = 1
        model_type = "test"
        rope_scaling = None
        _name_or_path = "test/model"

    class Base:
        embed_tokens = object()
        rotary_emb = object()

    class Model:
        config = Config()
        model = Base()
        device = torch.device("cpu")

    return HFBackend(Model())


def test_text_source_encoder_handles_text_and_rejects_other_sources(tmp_path: Path):
    encoder = TextSourceEncoder(_Processor())
    raw = encoder.encode(RawText("hello", source_id="raw"))
    assert raw.token_ids is not None and raw.body_len == 5
    text_file = tmp_path / "skill.md"
    text_file.write_text("skill", encoding="utf-8")
    encoded_file = encoder.encode(
        FileSource(str(text_file), source_id="file", media_type="text/markdown")
    )
    assert encoded_file.body_len == 5

    image = tmp_path / "image.png"
    image.write_bytes(b"not-an-image")
    with pytest.raises(UnsupportedModalityError, match="image/png"):
        encoder.encode(FileSource(str(image), source_id="image"))
    with pytest.raises(UnsupportedModalityError, match="URI resolver"):
        encoder.encode(FileSource("https://example.com/doc.txt", source_id="remote"))


def test_hf_backend_does_not_own_source_processor():
    backend = _hf_backend()
    assert not hasattr(backend, "processor")
    assert not hasattr(backend, "encode_source")
