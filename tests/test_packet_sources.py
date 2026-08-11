"""D3 tests for typed sources and assignment config/manifest expansion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvpacket.backend import BackendCompatibility
from kvpacket.errors import SourceConfigError, SourceConflictError
from kvpacket.fingerprint import fingerprint_json
from kvpacket.packet import (
    FileSource,
    PacketWrapper,
    WrapperRegistry,
    WrapperEntry,
    RawText,
    Tokenizable,
    load_wrapper_assignments,
    load_wrapper_registry_and_assignments,
)


def _registry() -> WrapperRegistry:
    compatibility = BackendCompatibility(
        backend="hf",
        model_id="test/model",
        model_config_fingerprint=fingerprint_json(
            {"hidden_size": 8}, domain="tests.sources_model/1"
        ),
        hidden_size=8,
        model_artifact_fingerprint=fingerprint_json("test/model", domain="tests.model/1"),
    )
    registry = WrapperRegistry(
        compatibility=compatibility,
        wrappers={
            "biography_doc": WrapperEntry(PacketWrapper(1, 1, 8)),
            "powerpoint_skill": WrapperEntry(PacketWrapper(2, 1, 8)),
        },
    )
    return registry


def test_tokenizable_identity_is_typed_hashable_and_deterministic():
    text = RawText("same spelling")
    same_text = RawText("same spelling")
    file = FileSource("same spelling")

    assert text == same_text
    assert hash(text) == hash(same_text)
    assert text != file
    assert text.kind == "raw-text" and text.media_type == "text/plain"
    assert text.source_id.startswith("text-")
    assert file.source_id.startswith("file-")
    assert text == Tokenizable("raw-text", text.source_id, text.value, "text/plain")

    first_copy = RawText("duplicate body", source_id="copy-1")
    second_copy = RawText("duplicate body", source_id="copy-2")
    mapping: dict[Tokenizable, str] = {first_copy: "a", second_copy: "b"}
    assert len(mapping) == 2


def _write_complete_config(root: Path, registry: WrapperRegistry) -> Path:
    (root / "configs").mkdir()
    (root / "docs").mkdir()
    (root / "manifests").mkdir()
    (root / "files").mkdir()
    registry.save(root / "registry.pt")
    (root / "docs" / "skill.md").write_text("# PowerPoint skill", encoding="utf-8")
    (root / "files" / "slides.pptx").write_bytes(b"fake-presentation")
    (root / "manifests" / "texts.jsonl").write_text(
        "\n" + json.dumps({"id": "turing", "text": "Alan Turing worked on ..."}) + "\n",
        encoding="utf-8",
    )
    (root / "manifests" / "files.jsonl").write_text(
        json.dumps(
            {
                "id": "slide-deck",
                "path": "../files/slides.pptx",
                "media_type": "application/vnd.test.presentation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = root / "configs" / "assignments.yaml"
    config.write_text(
        """registry_path: ../registry.pt
assignments:
  biography_doc:
    - type: raw-text
      doc:
        - id: ada
          text: Ada Lovelace was born ...
        - literal convenience text
    - type: text-manifest
      doc:
        - ../manifests/texts.jsonl
  powerpoint_skill:
    - type: file
      doc:
        - id: skill-file
          path: ../docs/skill.md
          media_type: text/markdown
        - id: remote-skill
          path: HTTPS://EXAMPLE.COM/skills/tool.md
    - type: file-manifest
      doc:
        - ../manifests/files.jsonl
""",
        encoding="utf-8",
    )
    return config


def test_config_expands_inline_files_and_manifests(tmp_path: Path):
    registry = _registry()
    config = _write_complete_config(tmp_path, registry)

    assignments = load_wrapper_assignments(config, registry)
    assert isinstance(assignments, dict)
    assert len(assignments) == 6
    by_id = {source.source_id: (source, wrapper_id) for source, wrapper_id in assignments.items()}
    assert by_id["ada"][1] == "biography_doc"
    assert by_id["turing"][0].value == "Alan Turing worked on ..."
    assert by_id["skill-file"][0].value == str((tmp_path / "docs" / "skill.md").resolve())
    assert by_id["slide-deck"][0].value == str((tmp_path / "files" / "slides.pptx").resolve())
    assert by_id["slide-deck"][0].media_type == "application/vnd.test.presentation"
    assert by_id["remote-skill"][0].value == "https://example.com/skills/tool.md"

    # The result is deliberately an ordinary mapping users may edit before preprocessing.
    assignments[RawText("runtime addition", source_id="runtime")] = "biography_doc"
    assert len(assignments) == 7


def test_convenience_loader_returns_registry_and_inference_ready_mapping(tmp_path: Path):
    config = _write_complete_config(tmp_path, _registry())
    registry, assignments = load_wrapper_registry_and_assignments(config)
    assert registry.ids() == ["biography_doc", "powerpoint_skill"]
    assert len(assignments) == 6
    assert not registry["biography_doc"].wrapper.header.requires_grad


def test_duplicate_source_ids_are_rejected_across_groups(tmp_path: Path):
    registry = _registry()
    registry.save(tmp_path / "registry.pt")
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        """registry_path: registry.pt
assignments:
  biography_doc:
    - type: raw-text
      doc:
        - {id: shared-id, text: first}
  powerpoint_skill:
    - type: raw-text
      doc:
        - {id: shared-id, text: second}
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceConflictError, match="duplicate source_id 'shared-id'"):
        load_wrapper_assignments(config, registry)


def test_unknown_wrapper_and_missing_file_fail_before_model_work(tmp_path: Path):
    registry = _registry()
    registry.save(tmp_path / "registry.pt")
    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {
                "registry_path": "registry.pt",
                "assignments": {"unknown_wrapper": [{"type": "raw-text", "doc": ["text"]}]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError, match="unknown wrapper ID"):
        load_wrapper_assignments(unknown, registry)

    missing = tmp_path / "missing.yaml"
    missing.write_text(
        """registry_path: registry.pt
assignments:
  biography_doc:
    - type: file
      doc: [does-not-exist.txt]
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError, match="does not exist"):
        load_wrapper_assignments(missing, registry)


def test_manifests_are_strict_and_non_empty(tmp_path: Path):
    registry = _registry()
    registry.save(tmp_path / "registry.pt")
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        json.dumps({"id": "record", "text": "hello", "typo": True}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "manifest.yaml"
    config.write_text(
        """registry_path: registry.pt
assignments:
  biography_doc:
    - type: text-manifest
      doc: [bad.jsonl]
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError, match=r"unknown=\['typo'\]"):
        load_wrapper_assignments(config, registry)

    manifest.write_text("\n", encoding="utf-8")
    with pytest.raises(SourceConfigError, match="contains no source records"):
        load_wrapper_assignments(config, registry)


def test_duplicate_yaml_keys_are_not_silently_overwritten(tmp_path: Path):
    registry = _registry()
    registry.save(tmp_path / "registry.pt")
    config = tmp_path / "duplicate-key.yaml"
    config.write_text(
        """registry_path: registry.pt
assignments:
  biography_doc:
    - type: raw-text
      doc: [first]
  biography_doc:
    - type: raw-text
      doc: [second]
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError, match="duplicate YAML mapping key 'biography_doc'"):
        load_wrapper_assignments(config, registry)
