"""Gate 3 short old/new convergence parity on exact Qwen3 RECORD samples."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kvpacket import (
    HFBackend,
    Mode,
    PacketSession,
    PacketTrainer,
    WrapperRegistry,
    TrainingSample,
)
from tasks.multi_doc.data import get_ret_eval_generator
from tasks.multi_doc.evaluator import _packet_store
from tasks.multi_doc.data.templates import TokenizerType
from tests.reference.qwen3.replay import MODEL_ID, MODEL_REVISION
from tests.reference.qwen3.training_differential import (
    _build_fixture,
    _legacy_inputs,
    _load_runtime,
    _max_diff,
)

NUM_SAMPLES = 16
BATCH_SIZE = 4
REFERENCE_TOTAL_STEPS = 120


def _record_samples(
    backend: HFBackend,
    tokenizer: TokenizerType,
    sample_dir: Path,
) -> list[Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    expected = [sample_dir / f"{index:04d}.pt" for index in range(NUM_SAMPLES)]
    if all(path.exists() for path in expected):
        return expected
    if any(sample_dir.iterdir()):
        raise RuntimeError(f"incomplete Gate 3 sample directory: {sample_dir}")
    entries = get_ret_eval_generator(
        name="biography",
        num_samples=NUM_SAMPLES,
        num_documents=2,
        num_shots=0,
        subset="10k",
        split="train",
        seed=4,
        tokenizer=tokenizer,
        data_kwargs={"question_type": "QA"},
    )
    for index, entry in enumerate(entries):
        session = PacketSession(
            backend=backend,
            mode=Mode.RECORD,
            match="token",
            tokenizer=tokenizer,
        )
        for doc_index, document in enumerate(entry.documents):
            if isinstance(document, str):
                session.register_chunk(f"doc-{doc_index}", text=document)
            else:
                session.register_chunk(f"doc-{doc_index}", token_ids=list(document))
            session.push(document)
        session.push(entry.task_prompt)
        session.generate(max_new_tokens=32)
        sample = session.export_sample()
        if sample is None or sample.logits is None:
            raise RuntimeError(f"failed to record Gate 3 sample {index}")
        sample.save(str(expected[index]))
        print(f"[record] {index + 1}/{NUM_SAMPLES}")
    return expected


def _greedy_ids(
    backend: HFBackend,
    tokenizer: TokenizerType,
    wrapper,
    entry,
) -> list[int]:
    session = PacketSession(
        backend=backend,
        store=_packet_store(entry, backend, tokenizer, wrapper),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tokenizer,
    )
    session.register_store_chunks()
    for document in entry.documents:
        session.push(document)
    session.push(entry.task_prompt)
    return session.generate(max_new_tokens=32).token_ids


def _legacy_batch_step(
    backend: HFBackend,
    registry: WrapperRegistry,
    samples: list[TrainingSample],
    optimizer: torch.optim.Optimizer,
) -> tuple[float, int]:
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    tokens = 0
    for sample in samples:
        embeds, mask, positions = _legacy_inputs(backend, registry, sample)
        logits = backend.training_forward(inputs_embeds=embeds, attention_mask=mask)[
            0
        ].index_select(0, positions)
        assert sample.logits is not None
        teacher = sample.logits.to(device=logits.device, dtype=logits.dtype)
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(teacher, dim=-1),
            reduction="sum",
            log_target=True,
        )
        loss.backward()
        total_loss += float(loss.detach())
        tokens += sample.num_generated
    optimizer.step()
    return total_loss / tokens, tokens


def main() -> None:
    storage = os.environ.get("KVP_STORAGE")
    if not storage:
        raise RuntimeError("set KVP_STORAGE before running short convergence")
    artifact_dir = Path(storage) / "kvpacket/reference_parity/qwen3"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    backend, tokenizer = _load_runtime()
    initial_path, _ = _build_fixture(backend, tokenizer, artifact_dir)
    initial = WrapperRegistry.load(
        initial_path, device=backend.device, runtime=backend.compatibility
    )
    paths = _record_samples(backend, tokenizer, artifact_dir / "gate3_samples")
    samples = [TrainingSample.load(str(path)).to(backend.device) for path in paths]
    legacy = WrapperRegistry.from_state_dict(
        initial.state_dict(), device=backend.device, requires_grad=True
    )
    current_registry = WrapperRegistry.from_state_dict(
        initial.state_dict(), device=backend.device, requires_grad=True
    )
    optimizer_kwargs = {"lr": 5e-4, "weight_decay": 0.0}
    legacy_optimizer = torch.optim.AdamW(
        legacy["biography_doc"].wrapper.parameters(),
        lr=5e-4,
        weight_decay=0.0,
    )
    current = PacketTrainer(
        backend,
        current_registry,
        default_wrapper_id="biography_doc",
        loss_reduction="legacy_sum",
        optimizer_kwargs=optimizer_kwargs,
        provenance={"goal": "reference-training-parity", "gate": 3},
    )
    total_steps = NUM_SAMPLES // BATCH_SIZE
    legacy_scheduler = torch.optim.lr_scheduler.LinearLR(
        legacy_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=REFERENCE_TOTAL_STEPS,
    )
    current_scheduler = torch.optim.lr_scheduler.LinearLR(
        current.optimizers["biography_doc"],
        start_factor=1.0,
        end_factor=0.0,
        total_iters=REFERENCE_TOTAL_STEPS,
    )
    indices = list(range(NUM_SAMPLES))
    random.Random(42).shuffle(indices)
    steps: list[dict[str, object]] = []
    for offset in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch_indices = indices[offset : offset + BATCH_SIZE]
        batch = [samples[index] for index in batch_indices]
        legacy_loss, tokens = _legacy_batch_step(backend, legacy, batch, legacy_optimizer)
        legacy_scheduler.step()
        current_metrics = current.step_batch(batch)
        current_scheduler.step()
        tensor_diff = max(
            _max_diff(
                getattr(legacy["biography_doc"].wrapper, name),
                getattr(current_registry["biography_doc"].wrapper, name),
            )
            for name in ("header", "trailer")
        )
        steps.append(
            {
                "indices": batch_indices,
                "tokens": tokens,
                "legacy_token_mean_loss": legacy_loss,
                "current_token_mean_loss": current_metrics.loss,
                "loss_abs_diff": abs(legacy_loss - current_metrics.loss),
                "wrapper_max_abs_diff": tensor_diff,
                "legacy_lr": legacy_scheduler.get_last_lr()[0],
                "current_lr": current_scheduler.get_last_lr()[0],
            }
        )
        print(f"[train] step={len(steps)}/{total_steps} diff={tensor_diff:.3e}")

    checkpoint = artifact_dir / "gate3_trainer.pt"
    current.save(checkpoint)
    restored = PacketTrainer.load(checkpoint, backend=backend)
    # AdamW's current LR is part of its state; the next update must survive reload.
    uninterrupted_metrics = current.step(samples[0])
    restored_metrics = restored.step(samples[0])
    resume_diff = max(
        _max_diff(
            getattr(current.registry["biography_doc"].wrapper, name),
            getattr(restored.registry["biography_doc"].wrapper, name),
        )
        for name in ("header", "trailer")
    )

    eval_entries = list(
        get_ret_eval_generator(
            name="biography",
            num_samples=4,
            num_documents=5,
            num_shots=0,
            subset="10k",
            split="test",
            seed=42,
            tokenizer=tokenizer,
            data_kwargs={"question_type": "QA"},
        )
    )
    legacy_ids = [
        _greedy_ids(backend, tokenizer, legacy["biography_doc"].wrapper, entry)
        for entry in eval_entries
    ]
    # Compare the pre-resume current wrapper, which is identical to legacy at Gate 3.
    gate3_current = PacketTrainer.load(checkpoint, backend=backend)
    current_ids = [
        _greedy_ids(
            backend,
            tokenizer,
            gate3_current.registry["biography_doc"].wrapper,
            entry,
        )
        for entry in eval_entries
    ]
    report = {
        "gate": 3,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "num_samples": NUM_SAMPLES,
        "epochs": 1,
        "batch_size": BATCH_SIZE,
        "scheduler_total_steps": REFERENCE_TOTAL_STEPS,
        "sample_order": indices,
        "steps": steps,
        "max_loss_abs_diff": max(cast(float, step["loss_abs_diff"]) for step in steps),
        "max_wrapper_abs_diff": max(cast(float, step["wrapper_max_abs_diff"]) for step in steps),
        "free_generation": {
            "legacy_token_ids": legacy_ids,
            "current_token_ids": current_ids,
            "all_exact": legacy_ids == current_ids,
        },
        "checkpoint_resume": {
            "next_loss_abs_diff": abs(uninterrupted_metrics.loss - restored_metrics.loss),
            "next_wrapper_max_abs_diff": resume_diff,
            "restored_step_count": restored.step_count,
        },
        "artifact_dir": str(artifact_dir),
    }
    output = Path("outputs/reference_parity_qwen3_gate3.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
