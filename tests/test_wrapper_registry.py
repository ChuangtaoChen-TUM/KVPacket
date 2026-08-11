"""D2 tests for named wrapper storage and backend compatibility."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from kvpacket.backend import BackendCompatibility, HFBackend
from kvpacket.errors import BackendCompatibilityError, WrapperFormatError
from kvpacket.fingerprint import fingerprint_json, fingerprint_tensor
from kvpacket.packet import PacketWrapper, WrapperRegistry, WrapperEntry


def _compatibility(*, hidden_size: int = 16, model_id: str = "test/model") -> BackendCompatibility:
    return BackendCompatibility(
        backend="hf",
        model_id=model_id,
        model_config_fingerprint=fingerprint_json(
            {"hidden_size": hidden_size, "layers": 2}, domain="tests.model_config/1"
        ),
        hidden_size=hidden_size,
        model_artifact_fingerprint=fingerprint_json(model_id, domain="tests.model_artifact/1"),
    )


def _registry() -> WrapperRegistry:
    compatibility = _compatibility()
    return WrapperRegistry(
        compatibility=compatibility,
        wrappers={
            "biography_doc": WrapperEntry(
                PacketWrapper(2, 3, compatibility.hidden_size),
                training_config={"lr": 0.01},
                meta={"metric": 0.5},
            ),
            "powerpoint_skill": WrapperEntry(
                PacketWrapper(1, 2, compatibility.hidden_size),
                training_config={"frozen": True},
            ),
        },
        meta={"experiment": "mixed"},
    )


def test_tensor_fingerprint_covers_dtype_shape_and_bytes():
    tensor = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)
    assert fingerprint_tensor(tensor) == fingerprint_tensor(tensor.clone())
    assert fingerprint_tensor(tensor) != fingerprint_tensor(tensor.float())
    changed = tensor.clone()
    changed[0, 0, 0] += 1
    assert fingerprint_tensor(tensor) != fingerprint_tensor(changed)


def test_registry_roundtrip_and_metadata_does_not_change_parameter_identity():
    registry = _registry()
    wrapper_fingerprint = registry["biography_doc"].fingerprint
    registry_fingerprint = registry.fingerprint

    registry["biography_doc"].meta["metric"] = 0.9
    registry["biography_doc"].training_config["epochs"] = 2
    registry.meta["note"] = "metadata is non-functional"
    assert registry["biography_doc"].fingerprint == wrapper_fingerprint
    assert registry.fingerprint == registry_fingerprint

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "registry.pt"
        registry.save(path)
        loaded = WrapperRegistry.load(
            path,
            requires_grad=False,
            runtime=_compatibility(),
        )

    assert loaded.ids() == ["biography_doc", "powerpoint_skill"]
    assert loaded.fingerprint == registry.fingerprint
    assert loaded["biography_doc"].training_config == {"lr": 0.01, "epochs": 2}
    assert not loaded["biography_doc"].wrapper.header.requires_grad
    assert torch.equal(
        loaded["powerpoint_skill"].wrapper.trailer,
        registry["powerpoint_skill"].wrapper.trailer,
    )


def test_parameter_change_updates_wrapper_and_registry_fingerprints():
    registry = _registry()
    old_wrapper = registry["biography_doc"].fingerprint
    old_registry = registry.fingerprint
    with torch.no_grad():
        registry["biography_doc"].wrapper.header[0, 0, 0] += 1
    assert registry["biography_doc"].fingerprint != old_wrapper
    assert registry.fingerprint != old_registry


def test_checkpoint_rejects_tampering_and_bad_geometry():
    registry = _registry()
    checkpoint = registry.state_dict()
    checkpoint["fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(WrapperFormatError, match="registry fingerprint mismatch"):
        WrapperRegistry.from_state_dict(checkpoint)

    with pytest.raises(WrapperFormatError, match="hidden size"):
        WrapperRegistry(
            compatibility=_compatibility(hidden_size=16),
            wrappers={"wrong_dim": WrapperEntry(PacketWrapper(1, 1, 8))},
        )
    with pytest.raises(WrapperFormatError, match="invalid wrapper_id"):
        WrapperRegistry(
            compatibility=_compatibility(),
            wrappers={"PowerPoint Skill": WrapperEntry(PacketWrapper(1, 1, 16))},
        )


def test_compatibility_validation_reports_functional_mismatch():
    registry = _registry()
    incompatible = BackendCompatibility(
        backend="hf",
        model_id="test/model",
        model_config_fingerprint=fingerprint_json(
            {"hidden_size": 16, "layers": 3}, domain="tests.model_config/1"
        ),
        hidden_size=16,
        model_artifact_fingerprint=fingerprint_json(
            "test/model", domain="tests.model_artifact/1"
        ),
    )
    with pytest.raises(BackendCompatibilityError, match="model_config_fingerprint"):
        registry.validate_compatibility(incompatible)


def test_compatibility_validation_rejects_different_model_artifact():
    registry = _registry()
    expected = registry.compatibility
    incompatible = BackendCompatibility(
        backend=expected.backend,
        model_id=expected.model_id,
        model_config_fingerprint=expected.model_config_fingerprint,
        hidden_size=expected.hidden_size,
        model_artifact_fingerprint=fingerprint_json(
            "different-weights", domain="tests.model_artifact/1"
        ),
    )
    with pytest.raises(BackendCompatibilityError, match="model_artifact_fingerprint"):
        registry.validate_compatibility(incompatible)


def test_compatibility_validation_rejects_diagnostic_identity_mismatch():
    registry = _registry()
    other_model = _compatibility(model_id="other/fine-tune")
    assert other_model.fingerprint != registry.compatibility.fingerprint
    with pytest.raises(BackendCompatibilityError, match="model_id"):
        registry.validate_compatibility(other_model)


def test_remove_uses_typed_errors_for_unknown_and_last_wrapper():
    singleton = WrapperRegistry(
        compatibility=_compatibility(),
        wrappers={"only": WrapperEntry(PacketWrapper(1, 1, 16))},
    )
    with pytest.raises(WrapperFormatError, match="unknown wrapper_id"):
        singleton.remove("missing")
    with pytest.raises(WrapperFormatError, match="cannot be empty"):
        singleton.remove("only")

    registry = _registry()
    with pytest.raises(WrapperFormatError, match="unknown wrapper_id"):
        registry.remove("missing")
    removed = registry.remove("powerpoint_skill")
    assert removed.wrapper.dim == 16


def test_backend_compatibility_rejects_unpublished_processor_fields():
    state = _compatibility().to_dict()
    state["processor_id"] = "legacy/tokenizer"
    with pytest.raises(ValueError, match="unknown=.*processor_id"):
        BackendCompatibility.from_dict(state)


def test_hf_backend_owns_compatibility_descriptor():
    class Config:
        hidden_size = 8
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

    backend = HFBackend(Model())
    compatibility = backend.compatibility
    assert compatibility.backend == "hf"
    assert compatibility.model_id == "test/model"
    assert compatibility.hidden_size == 8
