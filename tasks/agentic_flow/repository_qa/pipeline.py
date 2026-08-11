"""Record, train, and materialize the minimal Codex agent example.

Run from the repository root::

    python -m tasks.agentic_flow.repository_qa.pipeline \
        --config tasks/agentic_flow/repository_qa/training_configs/qwen3_4b.json \
        --phase all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from kvpacket import (
    HFBackend,
    PacketPreprocessor,
    PacketTrainer,
    TrainingSample,
    WrapperRegistry,
    create_wrapper,
)
from kvpacket.packet import WrapperEntry

from .agent import AgentMethod, AgentRun, RepositoryAgent
from .skills import AgentTokenizer, load_skills, skill_token_ids

Phase = Literal["record", "train", "materialize", "all"]
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One stable identifier and repository question used for teacher recording."""

    prompt_id: str
    question: str


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Validated decoding options shared by recorded prompts."""

    max_turns: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the example config and resolve its local and external paths."""
    config_path = Path(path).resolve()
    with config_path.open() as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("training config must be a JSON object")
    config = cast(dict[str, Any], raw)
    for key in ("repository_root", "artifact_dir", "pretrained_wrapper"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        if ("$KVP_STORAGE" in value or "${KVP_STORAGE}" in value) and not os.environ.get(
            "KVP_STORAGE"
        ):
            raise ValueError(f"{key} references KVP_STORAGE, but KVP_STORAGE is not set")
        expanded = Path(os.path.expanduser(os.path.expandvars(value)))
        config[key] = str(expanded if expanded.is_absolute() else Path.cwd() / expanded)
    return config


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Extract one required configuration object."""
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return cast(Mapping[str, Any], value)


def _positive_int(value: object, *, where: str) -> int:
    """Validate one positive integer option."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _prompt_specs(config: Mapping[str, Any]) -> tuple[PromptSpec, ...]:
    """Read a short, unique prompt list directly from the example config."""
    raw = config.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompts must be a non-empty list")
    prompts: list[PromptSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"prompts[{index}] must be an object")
        prompt_id = item.get("id")
        question = item.get("question")
        if (
            not isinstance(prompt_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prompt_id) is None
        ):
            raise ValueError(f"prompts[{index}].id must be a filename-safe identifier")
        if not isinstance(question, str) or not question:
            raise ValueError(f"prompts[{index}].question must be a non-empty string")
        prompts.append(PromptSpec(prompt_id, question))
    if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
        raise ValueError("prompt IDs must be unique")
    return tuple(prompts)


def _wrapper_id(config: Mapping[str, Any]) -> str:
    """Stable wrapper route shared by the reusable skill block."""
    value = _mapping(config, "wrapper").get("id", "agent-instructions")
    if not isinstance(value, str) or not value:
        raise ValueError("wrapper.id must be a non-empty string")
    return value


def _load_model(config: Mapping[str, Any]) -> tuple[HFBackend, object]:
    """Load the configured Hugging Face model and matching tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = _mapping(config, "model")
    model_path = model_config.get("model_path")
    dtype_name = model_config.get("dtype", "bfloat16")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model.model_path must be a non-empty string")
    if dtype_name not in _DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(_DTYPES)}")
    device_map = model_config.get("device_map", "cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(device_map, (str, Mapping)):
        raise ValueError("model.device_map must be a string or object")
    load_options: dict[str, object] = {
        "dtype": _DTYPES[cast(str, dtype_name)],
        "device_map": device_map,
    }
    attention = model_config.get("attention_implementation")
    if attention is not None:
        load_options["attn_implementation"] = attention
    revision = model_config.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("model.revision must be a non-empty string")
    if revision is not None:
        load_options["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_options).eval()
    return HFBackend(model, model_artifact_id=revision), tokenizer


def _generation_settings(config: Mapping[str, Any]) -> GenerationSettings:
    """Validate bounded generation settings for teacher recording."""
    raw = _mapping(config, "generation")
    settings = GenerationSettings(
        max_turns=_positive_int(raw.get("max_turns", 5), where="generation.max_turns"),
        max_new_tokens=_positive_int(
            raw.get("max_new_tokens", 192), where="generation.max_new_tokens"
        ),
        do_sample=bool(raw.get("do_sample", False)),
        temperature=float(raw.get("temperature", 1.0)),
        top_p=float(raw.get("top_p", 1.0)),
    )
    if not 0 < settings.top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    return settings


def _initial_registry(config: Mapping[str, Any], backend: HFBackend) -> WrapperRegistry:
    """Create the seeded wrapper registry used for training."""
    wrapper_config = _mapping(config, "wrapper")
    dtype_name = wrapper_config.get("dtype", _mapping(config, "model").get("dtype", "bfloat16"))
    if dtype_name not in _DTYPES:
        raise ValueError(f"wrapper.dtype must be one of {sorted(_DTYPES)}")
    initialization = wrapper_config.get("initialization", {"strategy": "model-embedding"})
    if not isinstance(initialization, Mapping):
        raise ValueError("wrapper.initialization must be an object")
    torch.manual_seed(int(config.get("seed", 42)))
    wrapper = create_wrapper(
        backend,
        header_len=_positive_int(wrapper_config.get("header_len", 8), where="wrapper.header_len"),
        trailer_len=_positive_int(
            wrapper_config.get("trailer_len", 8), where="wrapper.trailer_len"
        ),
        dtype=_DTYPES[cast(str, dtype_name)],
        initialization=cast(Mapping[str, object], initialization),
    )
    training_config = {
        "objective": "teacher_ce",
        "loss_reduction": "token_mean",
        "optimizer": dict(_mapping(config, "optimizer")),
    }
    return WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={_wrapper_id(config): WrapperEntry(wrapper, training_config=training_config)},
    )


def record_samples(
    config: Mapping[str, Any], backend: HFBackend, tokenizer: object, artifact_dir: Path
) -> list[Path]:
    """Record compact token-ID teachers from the configured repository prompts."""
    repository_root = Path(cast(str, config["repository_root"])).resolve(strict=True)
    if not repository_root.is_dir():
        raise ValueError("repository_root must be a directory")
    sample_dir = artifact_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "recorded_runs.jsonl"
    completed = _load_jsonl_by_id(manifest_path)
    agent = RepositoryAgent(
        backend=backend,
        tokenizer=tokenizer,
        method=AgentMethod.FULL_RECOMPUTE,
    )
    generation = _generation_settings(config)
    rows: list[dict[str, object]] = []
    sample_paths: list[Path] = []
    prompts = _prompt_specs(config)
    for index, prompt in enumerate(prompts, 1):
        previous = completed.get(prompt.prompt_id)
        previous_files = None if previous is None else previous.get("sample_files")
        previous_filenames = (
            _sample_filenames(previous_files, where=prompt.prompt_id)
            if previous is not None and previous.get("question") == prompt.question
            else None
        )
        if (
            previous is not None
            and previous_filenames is not None
            and all((sample_dir / name).is_file() for name in previous_filenames)
        ):
            rows.append(previous)
            sample_paths.extend(sample_dir / name for name in previous_filenames)
            print(f"[record] {index}/{len(prompts)} {prompt.prompt_id} reused")
            continue
        run = agent.run(
            prompt.question,
            str(repository_root),
            run_id=prompt.prompt_id,
            max_turns=generation.max_turns,
            max_new_tokens=generation.max_new_tokens,
            capture_samples=True,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            top_p=generation.top_p,
        )
        filenames: list[str] = []
        for turn_index, sample in enumerate(run.samples):
            path = sample_dir / f"{prompt.prompt_id}-turn-{turn_index:02d}.pt"
            sample.save(str(path))
            sample_paths.append(path)
            filenames.append(path.name)
        rows.append({**_run_record(run), "question": prompt.question, "sample_files": filenames})
        _write_jsonl(manifest_path, rows)
        print(
            f"[record] {index}/{len(prompts)} {prompt.prompt_id} "
            f"turns={len(run.turns)} samples={len(run.samples)}"
        )
    if not sample_paths:
        raise RuntimeError("no prompt activated the skill; no wrapper samples were recorded")
    return sample_paths


def train_wrapper(
    config: Mapping[str, Any], backend: HFBackend, artifact_dir: Path
) -> PacketTrainer:
    """Train and save one shared skill wrapper from recorded token teachers."""
    sample_paths = _recorded_sample_paths(config, artifact_dir)
    if not sample_paths:
        raise FileNotFoundError("no recorded samples; run --phase record first")
    trainer = PacketTrainer(
        backend,
        _initial_registry(config, backend),
        default_wrapper_id=_wrapper_id(config),
        objective="teacher_ce",
        loss_reduction="token_mean",
        optimizer_kwargs=cast(Mapping[str, object], _mapping(config, "optimizer")),
        provenance={
            "example": "codex_agent",
            "teacher_representation": "token_ids",
            "num_samples": len(sample_paths),
        },
    )
    samples = [TrainingSample.load(str(path)) for path in sample_paths]
    training = _mapping(config, "training")
    epochs = _positive_int(training.get("epochs", 5), where="training.epochs")
    target_tokens = training.get("target_tokens_per_step")
    if target_tokens is not None:
        target_tokens = _positive_int(target_tokens, where="training.target_tokens_per_step")
    seed = int(config.get("seed", 42))
    metrics = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(samples)
        epoch_metrics = trainer.fit(
            samples,
            epochs=1,
            batch_size=_positive_int(training.get("batch_size", 1), where="training.batch_size"),
            target_tokens_per_step=target_tokens,
            forward_batch_size=_positive_int(
                training.get("forward_batch_size", 1), where="training.forward_batch_size"
            ),
        )
        metrics.extend(epoch_metrics)
        if epoch_metrics:
            mean_loss = sum(metric.loss for metric in epoch_metrics) / len(epoch_metrics)
            print(f"[train] epoch={epoch + 1}/{epochs} mean_loss={mean_loss:.6f}")
    trainer.user_provenance["step_losses"] = [metric.loss for metric in metrics]
    trainer.save(artifact_dir / "trainer.pt")
    return trainer


@torch.no_grad()
def materialize_store(
    config: Mapping[str, Any],
    backend: HFBackend,
    tokenizer: object,
    artifact_dir: Path,
) -> Path:
    """Build a serving-ready packet store from a local or packaged trainer checkpoint."""
    local_checkpoint = artifact_dir / "trainer.pt"
    packaged_checkpoint = Path(cast(str, config["pretrained_wrapper"]))
    checkpoint = local_checkpoint if local_checkpoint.is_file() else packaged_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "no trained wrapper; run --phase train or restore pretrained_wrapper"
        )
    trainer = PacketTrainer.load(checkpoint, backend=backend)
    preprocessor = PacketPreprocessor(backend=backend, wrapper_registry=trainer.registry)
    skills = load_skills()
    tokens = skill_token_ids(cast(AgentTokenizer, tokenizer), skills)
    for skill in skills:
        preprocessor.preprocess_pretokenized(
            skill.source,
            tokens[skill.skill_id],
            wrapper_id=_wrapper_id(config),
        )
    output = artifact_dir / "kv_packet_store.pt"
    preprocessor.store.save(output)
    print(f"[materialize] {output}")
    return output


def _run_record(run: AgentRun) -> dict[str, object]:
    """Build a compact JSON-compatible record without benchmark metrics."""
    return {
        "run_id": run.run_id,
        "prediction": run.prediction,
        "num_turns": len(run.turns),
        "valid_tool_calls": run.valid_tool_calls,
        "total_tool_calls": run.total_tool_calls,
        "skill_load_calls": run.skill_load_calls,
        "successful_skill_loads": run.successful_skill_loads,
        "error": run.error,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write deterministic JSON Lines output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, object]]:
    """Load resumable prompt records keyed by run ID."""
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or not isinstance(raw.get("run_id"), str):
                raise ValueError(f"invalid recorded run at {path}:{line_number}")
            record = cast(dict[str, object], raw)
            run_id = cast(str, record["run_id"])
            if run_id in records:
                raise ValueError(f"duplicate recorded run {run_id!r} in {path}")
            records[run_id] = record
    return records


def _sample_filenames(value: object, *, where: str) -> list[str] | None:
    """Validate optional manifest filenames without permitting path traversal."""
    if not isinstance(value, list):
        return None
    if not all(isinstance(name, str) and name and Path(name).name == name for name in value):
        raise ValueError(f"invalid sample filename in recorded run {where!r}")
    return cast(list[str], value)


def _recorded_sample_paths(config: Mapping[str, Any], artifact_dir: Path) -> list[Path]:
    """Resolve samples belonging to every completed configured prompt."""
    records = _load_jsonl_by_id(artifact_dir / "recorded_runs.jsonl")
    paths: list[Path] = []
    for prompt in _prompt_specs(config):
        record = records.get(prompt.prompt_id)
        files = None if record is None else record.get("sample_files")
        if record is None or record.get("question") != prompt.question:
            raise FileNotFoundError(f"recorded run is incomplete for {prompt.prompt_id}")
        filenames = _sample_filenames(files, where=prompt.prompt_id)
        if filenames is None:
            raise FileNotFoundError(f"recorded run is incomplete for {prompt.prompt_id}")
        for name in filenames:
            path = artifact_dir / "samples" / name
            if not path.is_file():
                raise FileNotFoundError(f"missing recorded sample: {path}")
            paths.append(path)
    return paths


def main() -> None:
    """Run one resumable stage or the complete minimal training pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=("record", "train", "materialize", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    backend, tokenizer = _load_model(config)

    phase = cast(Phase, arguments.phase)
    if phase in ("record", "all"):
        record_samples(config, backend, tokenizer, artifact_dir)
    if phase in ("train", "all"):
        train_wrapper(config, backend, artifact_dir)
    if phase in ("materialize", "all"):
        materialize_store(config, backend, tokenizer, artifact_dir)


if __name__ == "__main__":
    main()
