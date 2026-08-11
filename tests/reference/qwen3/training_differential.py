"""Gate 0/2 exact-model one-step training differential.

This freezes one Qwen3 RECORD sample and initial registry under ``KVP_STORAGE``,
then compares an independent implementation of the legacy summed-KL step with
``PacketTrainer(loss_reduction="legacy_sum")``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kvpacket import (
    HFBackend,
    Mode,
    PacketSession,
    PacketTrainer,
    PacketWrapper,
    WrapperRegistry,
    TrainingSample,
)
from kvpacket.packet import WrapperEntry
from tasks.multi_doc.data import get_ret_eval_generator
from tasks.multi_doc.data.templates import TokenizerType
from tests.reference.qwen3.replay import MODEL_ID, MODEL_REVISION


def _load_runtime() -> tuple[HFBackend, TokenizerType]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return HFBackend(model), cast(TokenizerType, tokenizer)


def _build_fixture(
    backend: HFBackend,
    tokenizer: TokenizerType,
    artifact_dir: Path,
) -> tuple[Path, Path]:
    registry_path = artifact_dir / "gate0_initial_registry.pt"
    sample_path = artifact_dir / "gate0_training_sample.pt"
    if registry_path.exists() and sample_path.exists():
        return registry_path, sample_path

    torch.manual_seed(42)
    wrapper = PacketWrapper(
        8,
        8,
        backend.hidden_size,
        dtype=torch.bfloat16,
        device=backend.device,
    )
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={
            "biography_doc": WrapperEntry(
                wrapper,
                training_config={
                    "loss_reduction": "legacy_sum",
                    "optimizer": {"lr": 5e-4, "weight_decay": 0.0},
                    "scheduler": {
                        "type": "LinearLR",
                        "start_factor": 1.0,
                        "end_factor": 0.0,
                        "total_iters": 120,
                    },
                },
            )
        },
    )
    entry = next(
        get_ret_eval_generator(
            name="biography",
            num_samples=1,
            num_documents=1,
            num_shots=0,
            subset="10k",
            split="train",
            seed=0,
            tokenizer=tokenizer,
            data_kwargs={"question_type": "QA"},
        )
    )
    session = PacketSession(
        backend=backend,
        mode=Mode.RECORD,
        match="token",
        tokenizer=tokenizer,
    )
    document = entry.documents[0]
    if not isinstance(document, str):
        raise TypeError("the biography parity task requires text documents")
    session.register_chunk("doc-0", text=document)
    if entry.preamble:
        session.push(entry.preamble)
    session.push(entry.documents[0])
    session.push(entry.task_prompt)
    session.generate(max_new_tokens=16)
    sample = session.export_sample()
    if sample is None or sample.logits is None:
        raise RuntimeError("failed to freeze the Gate 0 teacher trajectory")
    registry.save(registry_path)
    sample.save(str(sample_path))
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "preamble": entry.preamble,
        "documents": entry.documents,
        "task_prompt": entry.task_prompt,
        "answer": entry.answer,
        "token_ids": sample.token_ids.tolist(),
        "generated_target_token_ids": sample.token_ids[sample.generated_mask].tolist(),
        "chunk_spans": sample.chunk_spans,
        "optimizer": {"type": "AdamW", "lr": 5e-4, "weight_decay": 0.0},
        "scheduler": {
            "type": "LinearLR",
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": 120,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
    }
    (artifact_dir / "gate0_fixture.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return registry_path, sample_path


def _legacy_inputs(
    backend: HFBackend,
    registry: WrapperRegistry,
    sample: TrainingSample,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct the old ``prepare_sample_input`` and packet mask independently."""
    original = sample.token_ids[:-1].to(backend.device).unsqueeze(0)
    with torch.no_grad():
        base = backend.embed(original).detach()
    occurrences = sorted(
        (start, end, chunk_id)
        for chunk_id, spans in sample.chunk_spans.items()
        for start, end in spans
    )
    pieces: list[Tensor] = []
    independent_sizes: list[int] = []
    cursor = 0
    for start, end, chunk_id in occurrences:
        if start > cursor:
            plain = base[:, cursor:start]
            pieces.append(plain)
            independent_sizes.append(int(plain.shape[1]))
        wrapper_id = "biography_doc"
        wrapped = registry[wrapper_id].wrapper(base[:, start:end])
        pieces.append(wrapped)
        independent_sizes.append(int(wrapped.shape[1]))
        cursor = end
    query = base[:, cursor:]
    pieces.append(query)
    query_len = int(query.shape[1])
    embeds = torch.cat(pieces, dim=1)

    length = int(embeds.shape[1])
    mask = torch.zeros((length, length), dtype=torch.bool, device=backend.device)
    block_start = 0
    for size in independent_sizes:
        block_end = block_start + size
        mask[block_start:block_end, block_start:block_end] = torch.tril(
            torch.ones((size, size), dtype=torch.bool, device=backend.device)
        )
        block_start = block_end
    mask[-query_len:, :] = torch.tril(
        torch.ones((query_len, length), dtype=torch.bool, device=backend.device),
        diagonal=length - query_len,
    )
    positions = torch.arange(
        length - sample.num_generated,
        length,
        dtype=torch.long,
        device=backend.device,
    )
    return embeds, mask.unsqueeze(0).unsqueeze(0), positions


def _max_diff(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def main() -> None:
    storage = os.environ.get("KVP_STORAGE")
    if not storage:
        raise RuntimeError("set KVP_STORAGE before running the parity differential")
    artifact_dir = Path(storage) / "kvpacket/reference_parity/qwen3"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    backend, tokenizer = _load_runtime()
    registry_path, sample_path = _build_fixture(backend, tokenizer, artifact_dir)
    source = WrapperRegistry.load(
        registry_path, device=backend.device, runtime=backend.compatibility
    )
    sample = TrainingSample.load(str(sample_path)).to(backend.device)
    legacy_registry = WrapperRegistry.from_state_dict(
        source.state_dict(), device=backend.device, requires_grad=True
    )
    current_registry = WrapperRegistry.from_state_dict(
        source.state_dict(), device=backend.device, requires_grad=True
    )

    legacy_embeds, legacy_mask, legacy_positions = _legacy_inputs(backend, legacy_registry, sample)
    current = PacketTrainer(
        backend,
        current_registry,
        default_wrapper_id="biography_doc",
        loss_reduction="legacy_sum",
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        provenance={"goal": "reference-training-parity", "gate": 2},
    )
    current_embeds, current_mask, current_positions = current._student_inputs(sample)
    input_report = {
        "wrapped_embeddings_max_abs_diff": _max_diff(legacy_embeds, current_embeds),
        "attention_mask_exact": bool(torch.equal(legacy_mask, current_mask)),
        "student_positions_exact": bool(torch.equal(legacy_positions, current_positions)),
    }
    if not input_report["attention_mask_exact"] or not input_report["student_positions_exact"]:
        raise RuntimeError(f"Gate 2 structural mismatch: {input_report}")

    legacy_optimizer = torch.optim.AdamW(
        legacy_registry["biography_doc"].wrapper.parameters(),
        lr=5e-4,
        weight_decay=0.0,
    )
    legacy_scheduler = torch.optim.lr_scheduler.LinearLR(
        legacy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=120
    )
    current_scheduler = torch.optim.lr_scheduler.LinearLR(
        current.optimizers["biography_doc"],
        start_factor=1.0,
        end_factor=0.0,
        total_iters=120,
    )

    legacy_logits = backend.training_forward(
        inputs_embeds=legacy_embeds, attention_mask=legacy_mask
    )
    legacy_student = legacy_logits[0].index_select(0, legacy_positions)
    assert sample.logits is not None
    teacher = sample.logits.to(device=legacy_student.device, dtype=legacy_student.dtype)
    legacy_loss = F.kl_div(
        F.log_softmax(legacy_student, dim=-1),
        F.log_softmax(teacher, dim=-1),
        reduction="sum",
        log_target=True,
    )
    legacy_loss.backward()
    legacy_gradients = {
        name: getattr(legacy_registry["biography_doc"].wrapper, name).grad.detach().clone()
        for name in ("header", "trailer")
    }
    legacy_optimizer.step()
    legacy_scheduler.step()

    current_metrics = current.step(sample)
    current_scheduler.step()
    wrapper_legacy = legacy_registry["biography_doc"].wrapper
    wrapper_current = current_registry["biography_doc"].wrapper
    report = {
        "gate": 2,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "loss_reduction": "legacy_sum",
        "fixture": str(sample_path),
        "inputs": input_report,
        "selected_logits_max_abs_diff": _max_diff(
            legacy_student,
            backend.training_forward(inputs_embeds=current_embeds, attention_mask=current_mask)[
                0
            ].index_select(0, current_positions),
        ),
        "legacy_sum_loss": float(legacy_loss.detach()),
        "current_reported_token_mean_loss": current_metrics.loss,
        "gradient_max_abs_diff": {
            name: _max_diff(legacy_gradients[name], getattr(wrapper_current, name).grad)
            for name in ("header", "trailer")
        },
        "updated_tensor_max_abs_diff": {
            name: _max_diff(getattr(wrapper_legacy, name), getattr(wrapper_current, name))
            for name in ("header", "trailer")
        },
        "learning_rate": {
            "legacy": legacy_scheduler.get_last_lr()[0],
            "current": current_scheduler.get_last_lr()[0],
        },
    }
    output = Path("outputs/reference_parity_qwen3_gate2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    current.save(artifact_dir / "gate2_trainer.pt")
    print(json.dumps(report, indent=2))
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
