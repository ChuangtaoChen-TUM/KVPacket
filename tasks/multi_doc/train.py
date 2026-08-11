"""Minimal RECORD → train → evaluate example for multi-document adapters.

The script uses the task dataset registry and public ``kvpacket`` APIs throughout:
RECORD sessions create full-attention teacher samples, ``PacketTrainer`` optimizes one
document wrapper, and the evaluator compares Full Recompute, No Recompute, and the
trained KV Packet wrapper on a held-out split.

Run from the repository root:

    uv run python tasks/multi_doc/train.py \
        --config tasks/multi_doc/training_configs/synthetic_qwen.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kvpacket import (  # noqa: E402
    HFBackend,
    Mode,
    PacketSession,
    PacketTrainer,
    TrainingSample,
    WrapperRegistry,
    create_wrapper,
)
from kvpacket.packet import WrapperEntry  # noqa: E402
from tasks.common.metrics import MetricAccumulator  # noqa: E402
from tasks.multi_doc.data import get_dataset, get_ret_eval_generator  # noqa: E402
from tasks.multi_doc.data.base import RetrievalSample  # noqa: E402
from tasks.multi_doc.data.templates import TokenizerType  # noqa: E402
from tasks.multi_doc.evaluator import Method, evaluate_sample  # noqa: E402

Objective = Literal["teacher_kl", "teacher_ce"]

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_training_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a standalone multi-document training example config.

    Environment variables and ``~`` are expanded in artifact and result paths.
    Configs that reference ``KVP_STORAGE`` require it to be set explicitly.

    Args:
        path: JSON configuration path.

    Returns:
        Parsed configuration with normalized output paths.

    Raises:
        ValueError: If the JSON root is not an object or a path references an
            unset ``KVP_STORAGE`` variable.
    """
    with Path(path).open() as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("training config must be a JSON object")
    config = cast(dict[str, Any], raw)
    for key in ("artifact_dir", "record_artifact_dir", "result_path"):
        value = config.get(key)
        if isinstance(value, str):
            uses_storage = "$KVP_STORAGE" in value or "${KVP_STORAGE}" in value
            if uses_storage and not os.environ.get("KVP_STORAGE"):
                raise ValueError(f"{key} references KVP_STORAGE, but KVP_STORAGE is not set")
            config[key] = os.path.expanduser(os.path.expandvars(value))
    return config


def _training_objective(config: Mapping[str, Any]) -> Objective:
    """Resolve the full-distribution or compact token-level teacher objective."""
    objective = config.get("objective", "teacher_kl")
    if objective not in ("teacher_kl", "teacher_ce"):
        raise ValueError("objective must be 'teacher_kl' or 'teacher_ce'")
    return cast(Objective, objective)


def _wrapper_id(config: Mapping[str, Any]) -> str:
    """Resolve the stable wrapper ID stored in registries and checkpoints."""
    wrapper_config = config.get("wrapper")
    if not isinstance(wrapper_config, Mapping):
        raise ValueError("wrapper must be an object")
    wrapper_id = wrapper_config.get("id", "document")
    if not isinstance(wrapper_id, str) or not wrapper_id:
        raise ValueError("wrapper.id must be a non-empty string")
    return wrapper_id


def _load_model(config: Mapping[str, Any]) -> tuple[HFBackend, TokenizerType]:
    """Load the configured Hugging Face causal model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("model must be an object")
    model_path = model_config.get("model_path")
    dtype_name = model_config.get("dtype", "bfloat16")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model.model_path must be a non-empty string")
    if dtype_name not in _DTYPES:
        raise ValueError(f"model.dtype must be one of {sorted(_DTYPES)}")
    device_map = model_config.get("device_map", "cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(device_map, (str, Mapping)):
        raise ValueError("model.device_map must be a string or object")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    load_kwargs: dict[str, Any] = {
        "dtype": _DTYPES[cast(str, dtype_name)],
        "device_map": device_map,
    }
    attention_implementation = model_config.get("attention_implementation")
    if attention_implementation is not None:
        load_kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return HFBackend(model), cast(TokenizerType, tokenizer)


def _generation_settings(config: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Separate the token limit from keyword arguments shared by generation paths."""
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("model must be an object")
    raw = model_config.get("generation_kwargs", {})
    if not isinstance(raw, Mapping):
        raise ValueError("model.generation_kwargs must be an object")
    settings = dict(raw)
    max_new_tokens = settings.pop("max_new_tokens", 32)
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("model.generation_kwargs.max_new_tokens must be an integer")
    return max_new_tokens, settings


def _dataset_samples(
    data_config: Mapping[str, Any],
    tokenizer: TokenizerType,
    config: Mapping[str, Any],
) -> list[RetrievalSample]:
    """Materialize one configured dataset split with tokenizer-native formatting."""
    data_kwargs = data_config.get("data_kwargs", {})
    if not isinstance(data_kwargs, Mapping):
        raise ValueError("data_kwargs must be an object")
    return list(
        get_ret_eval_generator(
            name=str(data_config["dataset_name"]),
            num_samples=int(data_config["num_samples"]),
            num_documents=int(data_config.get("num_documents", 0)),
            num_shots=int(data_config.get("num_shots", 0)),
            subset=str(data_config.get("subset", "default")),
            split=str(data_config.get("split", "train")),
            seed=int(data_config.get("seed", 42)),
            tokenizer=tokenizer,
            data_kwargs=cast(Mapping[str, Any], data_kwargs),
            chat_template_kwargs=config.get("chat_template_kwargs"),
            system_prompt=str(config.get("system_prompt", "")),
        )
    )


def _training_data_configs(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Normalize one or several training dataset configurations."""
    value = config.get("training_data")
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
        return cast(list[Mapping[str, Any]], value)
    raise ValueError("training_data must be an object or a non-empty list of objects")


def _create_registry(config: Mapping[str, Any], backend: HFBackend) -> WrapperRegistry:
    """Create the seeded document wrapper used by the example."""
    wrapper_config = config.get("wrapper")
    if not isinstance(wrapper_config, Mapping):
        raise ValueError("wrapper must be an object")
    torch.manual_seed(int(config.get("seed", 42)))
    dtype_name = wrapper_config.get("dtype")
    if dtype_name is None:
        dtype = next(backend.model.parameters()).dtype
    elif dtype_name not in _DTYPES:
        raise ValueError(f"wrapper.dtype must be one of {sorted(_DTYPES)}")
    else:
        dtype = _DTYPES[cast(str, dtype_name)]
    initialization = wrapper_config.get("initialization")
    if initialization is not None and not isinstance(initialization, Mapping):
        raise ValueError("wrapper.initialization must be an object")
    wrapper = create_wrapper(
        backend,
        header_len=int(wrapper_config.get("header_len", 8)),
        trailer_len=int(wrapper_config.get("trailer_len", 8)),
        dtype=dtype,
        initialization=cast(Mapping[str, object] | None, initialization),
    )
    wrapper_id = _wrapper_id(config)
    training_config = {
        "objective": _training_objective(config),
        "loss_reduction": config.get("loss_reduction", "token_mean"),
        "optimizer": config.get("optimizer", {"lr": 1e-3}),
        "training_data": _training_data_configs(config),
    }
    return WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={wrapper_id: WrapperEntry(wrapper, training_config=training_config)},
    )


def record_samples(
    config: Mapping[str, Any],
    backend: HFBackend,
    tokenizer: TokenizerType,
    sample_dir: Path,
) -> list[Path]:
    """Record or validate compact teacher trajectories for the training split.

    Existing samples are reused only when their indices and objective representation
    match the current config.
    """
    entries = [
        sample
        for data_config in _training_data_configs(config)
        for sample in _dataset_samples(data_config, tokenizer, config)
    ]
    sample_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{index:04d}.pt" for index in range(len(entries))}
    unexpected = {path.name for path in sample_dir.glob("*.pt")} - expected_names
    if unexpected:
        raise RuntimeError(f"sample directory contains unexpected files: {sorted(unexpected)}")

    objective = _training_objective(config)
    capture_logits = objective == "teacher_kl"
    max_new_tokens, generation_kwargs = _generation_settings(config)
    paths: list[Path] = []
    for index, entry in enumerate(entries):
        path = sample_dir / f"{index:04d}.pt"
        if path.exists():
            sample = TrainingSample.load(str(path))
            if (sample.logits is not None) != capture_logits:
                raise RuntimeError(
                    f"sample representation does not match objective {objective!r}: {path}"
                )
            paths.append(path)
            continue

        session = PacketSession(
            backend=backend,
            mode=Mode.RECORD,
            tokenizer=tokenizer,
            match="token",
            prefill_chunk_size=config.get("record_prefill_chunk_size"),
        )
        for document_index, document in enumerate(entry.documents):
            source_id = f"doc-{document_index}"
            if isinstance(document, str):
                session.register_chunk(source_id, text=document)
            else:
                session.register_chunk(source_id, token_ids=document)
        if entry.preamble:
            session.push(entry.preamble)
        for document in entry.documents:
            session.push(document)
        session.push(entry.task_prompt)
        session.generate(
            max_new_tokens=max_new_tokens,
            capture_logits=capture_logits,
            **generation_kwargs,
        )
        sample = session.export_sample()
        if sample is None:
            raise RuntimeError(f"RECORD did not produce sample {index}")
        sample.save(str(path))
        paths.append(path)
        print(f"[record] {index + 1}/{len(entries)} {path}")
    return paths


def train_adapter(
    config: Mapping[str, Any],
    backend: HFBackend,
    registry: WrapperRegistry,
    sample_paths: list[Path],
    checkpoint_path: Path,
) -> PacketTrainer:
    """Train one document wrapper and save a complete trainer checkpoint."""
    samples = [TrainingSample.load(str(path)) for path in sample_paths]
    optimizer = config.get("optimizer", {"lr": 1e-3})
    if not isinstance(optimizer, Mapping):
        raise ValueError("optimizer must be an object")
    trainer = PacketTrainer(
        backend,
        registry,
        default_wrapper_id=_wrapper_id(config),
        objective=_training_objective(config),
        loss_reduction=config.get("loss_reduction", "token_mean"),
        optimizer_kwargs=cast(Mapping[str, object], optimizer),
        provenance={"example": "multi_doc_record_train_evaluate"},
    )
    use_gradient_checkpointing = bool(config.get("gradient_checkpointing", False))
    if use_gradient_checkpointing:
        enable = getattr(backend.model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("gradient_checkpointing requires model support")
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Hugging Face activates layer checkpointing only in training mode. Decoder
        # models used by these examples have no dropout, so the objective is unchanged.
        backend.model.train()
    try:
        metrics = []
        rng = random.Random(int(config.get("seed", 42)))
        for _ in range(int(config.get("epochs", 1))):
            rng.shuffle(samples)
            metrics.extend(
                trainer.fit(
                    samples,
                    epochs=1,
                    batch_size=int(config.get("batch_size", 1)),
                    target_tokens_per_step=config.get("target_tokens_per_step"),
                    forward_batch_size=int(config.get("forward_batch_size", 1)),
                )
            )
    finally:
        backend.model.eval()
        if use_gradient_checkpointing:
            disable = getattr(backend.model, "gradient_checkpointing_disable", None)
            if callable(disable):
                disable()
    trainer.user_provenance["step_losses"] = [item.loss for item in metrics]
    trainer.save(checkpoint_path)
    if metrics:
        print(f"[train] steps={len(metrics)} final_loss={metrics[-1].loss:.6f}")
    print(f"[train] checkpoint={checkpoint_path}")
    return trainer


@torch.no_grad()
def evaluate_adapter(
    config: Mapping[str, Any],
    backend: HFBackend,
    tokenizer: TokenizerType,
    trainer: PacketTrainer,
) -> dict[str, dict[str, object]]:
    """Evaluate the trained wrapper and the two reference bounds."""
    eval_config = config.get("evaluation")
    if not isinstance(eval_config, Mapping):
        raise ValueError("evaluation must be an object")
    samples = _dataset_samples(eval_config, tokenizer, config)
    postprocess = get_dataset(str(eval_config["dataset_name"])).answer_postprocess
    methods = [
        Method(value)
        for value in eval_config.get(
            "methods",
            [
                Method.FULL_RECOMPUTE.value,
                Method.NO_RECOMPUTE.value,
                Method.KV_PACKET.value,
            ],
        )
    ]
    _, generation_kwargs = _generation_settings(config)
    max_new_tokens = int(eval_config.get("max_new_tokens", 32))
    wrapper = trainer.registry[_wrapper_id(config)].wrapper
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        accumulator = MetricAccumulator()
        for sample in samples:
            result = evaluate_sample(
                sample,
                backend,
                tokenizer,
                method,
                wrapper=wrapper,
                max_new_tokens=max_new_tokens,
                answer_postprocess=postprocess,
                decoding_kwargs=generation_kwargs,
            )
            accumulator.add(
                result.tp,
                result.fp,
                result.fn,
                ttft=result.ttft,
                flops=result.flops,
                exact=result.exact,
            )
        summary = cast(dict[str, object], accumulator.summary())
        results[method.value] = summary
        print(f"[eval] {method.value}: {summary}")
    return results


def main() -> None:
    """Run one or all stages of the minimal multi-document training example."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=("record", "train", "eval", "all"),
        default="all",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    _training_objective(config)
    artifact_dir = Path(config["artifact_dir"])
    record_artifact_dir = Path(config.get("record_artifact_dir", artifact_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_artifact_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = record_artifact_dir / "samples"
    registry_path = record_artifact_dir / "initial_registry.pt"
    checkpoint_path = artifact_dir / "trainer.pt"

    backend, tokenizer = _load_model(config)
    sample_paths: list[Path] = []
    if args.phase in ("record", "all"):
        if registry_path.exists():
            registry = WrapperRegistry.load(
                registry_path,
                device=backend.device,
                runtime=backend.compatibility,
            )
        else:
            registry = _create_registry(config, backend)
            registry.save(registry_path)
        sample_paths = record_samples(config, backend, tokenizer, sample_dir)

    if args.phase in ("train", "all"):
        if not sample_paths:
            sample_paths = sorted(sample_dir.glob("*.pt"))
        if not sample_paths:
            raise FileNotFoundError("no recorded samples; run --phase record first")
        registry = WrapperRegistry.load(
            registry_path,
            device=backend.device,
            runtime=backend.compatibility,
        )
        trainer = train_adapter(config, backend, registry, sample_paths, checkpoint_path)
        # Exercise the same persisted state that a later evaluation process loads.
        trainer = PacketTrainer.load(checkpoint_path, backend=backend)
    elif args.phase == "eval":
        if not checkpoint_path.exists():
            raise FileNotFoundError("no trainer checkpoint; run --phase train first")
        trainer = PacketTrainer.load(checkpoint_path, backend=backend)
    else:
        trainer = None

    if args.phase in ("eval", "all"):
        assert trainer is not None
        results = evaluate_adapter(config, backend, tokenizer, trainer)
        result_path = Path(config["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as handle:
            json.dump(
                {
                    "config": config,
                    "trainer_step": trainer.step_count,
                    "results": results,
                },
                handle,
                indent=2,
            )
        print(f"[out] {result_path}")


if __name__ == "__main__":
    main()
