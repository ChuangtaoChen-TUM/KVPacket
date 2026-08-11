"""Light checks for the public multi-document training example."""

from pathlib import Path

import pytest

from tasks.multi_doc.train import (
    _training_data_configs,
    _training_objective,
    load_training_config,
)


def test_training_example_is_small_and_self_contained():
    root = Path(__file__).parents[1]
    config = load_training_config(root / "tasks/multi_doc/training_configs/synthetic_qwen.json")

    assert config["model"]["model_path"] == "Qwen/Qwen2.5-1.5B"
    assert config["wrapper"]["id"] == "document"
    assert config["objective"] == "teacher_ce"
    assert config["training_data"]["dataset_name"] == "synthetic"
    assert config["training_data"]["num_samples"] == 8
    assert config["evaluation"]["num_samples"] == 4
    assert config["evaluation"]["methods"] == [
        "full_recompute",
        "no_recompute",
        "kv_packet",
    ]
    assert config["artifact_dir"].startswith("outputs/")


def test_training_objective_defaults_to_kl_and_accepts_compact_ce():
    assert _training_objective({}) == "teacher_kl"
    assert _training_objective({"objective": "teacher_ce"}) == "teacher_ce"
    with pytest.raises(ValueError, match="objective"):
        _training_objective({"objective": "cross_entropy"})


def test_converted_training_configs_use_the_public_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("KVP_STORAGE", str(tmp_path))
    root = Path(__file__).parents[1]
    config_dir = root / "tasks/multi_doc/training_configs"
    expected = {
        "biography_qwen2_5_1_5b.json": ("biography", "Qwen/Qwen2.5-1.5B"),
        "biography_llama3_1_8b.json": (
            "biography",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        "hotpot_qa_llama3_2_3b.json": (
            "hotpot_qa",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
        "musique_llama3_2_3b.json": (
            "musique",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
        "gsm8k_dependent_llama3_2_3b.json": (
            "gsm8k_dependent",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
        "multinews_llama3_2_3b.json": (
            "multinews",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
    }

    for filename, (dataset, model) in expected.items():
        config = load_training_config(config_dir / filename)
        training_data = _training_data_configs(config)
        assert {item["dataset_name"] for item in training_data} == {dataset}
        assert config["evaluation"]["dataset_name"] == dataset
        assert config["model"]["model_path"] == model
        assert config["objective"] == "teacher_kl"
        assert config["artifact_dir"] != config["record_artifact_dir"]
        assert config["artifact_dir"].startswith(str(tmp_path / "kvpacket/multi_doc_training"))


def test_storage_backed_config_requires_kvp_storage(monkeypatch):
    monkeypatch.delenv("KVP_STORAGE", raising=False)
    root = Path(__file__).parents[1]
    path = root / "tasks/multi_doc/training_configs/hotpot_qa_llama3_2_3b.json"

    with pytest.raises(ValueError, match="KVP_STORAGE is not set"):
        load_training_config(path)


def test_training_data_accepts_one_or_multiple_dataset_groups():
    one = {"dataset_name": "synthetic"}
    two = [one, {"dataset_name": "biography"}]
    assert _training_data_configs({"training_data": one}) == [one]
    assert _training_data_configs({"training_data": two}) == two
    with pytest.raises(ValueError, match="training_data"):
        _training_data_configs({"training_data": []})
