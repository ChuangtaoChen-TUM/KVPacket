"""Gate 0/1 replay for the historical Qwen3 biography experiment.

Large model and fixture artifacts are written below ``KVP_STORAGE``.  The JSON
report is intentionally small enough to keep under ``outputs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kvpacket import HFBackend, Mode, PacketSession
from kvpacket.kvcache import KVCache, KeyValue, concat_caches
from kvpacket.packet.wrapper import PacketWrapper
from tasks.common.metrics import MetricAccumulator, f1_states
from tasks.multi_doc.data import get_dataset, get_ret_eval_generator
from tasks.multi_doc.data.base import RetrievalSample
from tasks.multi_doc.data.templates import TokenizerType
from tasks.multi_doc.evaluator import _packet_store

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
LEGACY_DATASET_SHA256 = "329d6e76ce7c71043be22a6e01f12dc727e9b014ac02da685cdd54e7647188d6"
HISTORICAL_F1 = {"full_recompute": 0.9813953488372092, "kv_packet": 0.8486238532110092}


def _rows(samples: list[RetrievalSample]) -> list[dict[str, object]]:
    return [
        {
            "preamble": sample.preamble,
            "documents": sample.documents,
            "task_prompt": sample.task_prompt,
            "query": sample.query,
            "answer": sample.answer,
        }
        for sample in samples
    ]


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _samples(tokenizer: TokenizerType, count: int) -> list[RetrievalSample]:
    return list(
        get_ret_eval_generator(
            name="biography",
            num_samples=count,
            num_documents=5,
            num_shots=0,
            subset="10k",
            split="test",
            seed=42,
            tokenizer=tokenizer,
            data_kwargs={"question_type": "QA"},
        )
    )


def _raw_samples(count: int) -> list[RetrievalSample]:
    return list(
        get_ret_eval_generator(
            name="biography",
            num_samples=count,
            num_documents=5,
            num_shots=0,
            subset="10k",
            split="test",
            seed=42,
            data_kwargs={"question_type": "QA"},
        )
    )


def _prompt_ids(sample: RetrievalSample, tokenizer: TokenizerType) -> list[int]:
    parts = [sample.preamble] if sample.preamble else []
    parts.extend(sample.documents)
    parts.append(sample.task_prompt)
    token_ids: list[int] = []
    for part in parts:
        token_ids.extend(tokenizer(part, add_special_tokens=False)["input_ids"])
    return token_ids


def _run_sample(
    sample: RetrievalSample,
    backend: HFBackend,
    tokenizer: TokenizerType,
    wrapper: PacketWrapper,
    method: str,
) -> tuple[dict[str, object], torch.Tensor]:
    use_packet = method == "kv_packet"
    store = _packet_store(sample, backend, tokenizer, wrapper) if use_packet else None
    session = PacketSession(
        backend=backend,
        store=store,
        mode=Mode.SERVE,
        match="token",
        tokenizer=tokenizer,
    )
    if store is not None:
        session.register_store_chunks()
    if sample.preamble:
        session.push(sample.preamble)
    for document in sample.documents:
        session.push(document)
    session.push(sample.task_prompt)
    result = session.generate(max_new_tokens=32, return_prompt_logits=True)
    decoded = cast(str, tokenizer.decode(result.token_ids, skip_special_tokens=True))
    postprocess = get_dataset("biography").answer_postprocess
    prediction, gold = postprocess(decoded, sample.answer)
    tp, fp, fn = f1_states(gold.split(), prediction.split())
    assert result.prompt_logits is not None
    return (
        {
            "method": method,
            "token_ids": result.token_ids,
            "decoded_prediction": decoded,
            "postprocessed_prediction": prediction,
            "gold": gold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "match_cover": [
                {"chunk_id": item.chunk_id, "start": item.start, "end": item.end}
                for item in result.cover
            ],
            "prompt_token_ids": _prompt_ids(sample, tokenizer),
        },
        result.prompt_logits,
    )


def _packet_diagnostics(
    sample: RetrievalSample,
    backend: HFBackend,
    tokenizer: TokenizerType,
    wrapper: PacketWrapper,
) -> tuple[dict[str, float], torch.Tensor]:
    """Compare stored packets and re-rotation with the old implementation."""
    session = PacketSession(
        backend=backend,
        store=_packet_store(sample, backend, tokenizer, wrapper),
        mode=Mode.SERVE,
        match="token",
        tokenizer=tokenizer,
    )
    session.register_store_chunks()
    legacy_shifted = []
    prefill_key_diff = 0.0
    prefill_value_diff = 0.0
    rerotated_key_diff = 0.0
    rerotated_value_diff = 0.0
    offset = 0
    if sample.preamble:
        preamble_ids = torch.tensor(
            tokenizer(sample.preamble, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
            device=backend.device,
        ).unsqueeze(0)
        preamble_positions = torch.arange(preamble_ids.shape[1], device=backend.device).unsqueeze(0)
        legacy_shifted.append(
            backend.prefill_caches(input_ids=preamble_ids, position_ids=preamble_positions)[0]
        )
        offset = int(preamble_ids.shape[1])
    for index, document in enumerate(sample.documents):
        chunk_id = f"doc-{index}"
        if not isinstance(document, str):
            raise TypeError("the biography parity task requires text documents")
        session.register_chunk(chunk_id, text=document)
        stored = session.store.get(chunk_id).packet
        ids = torch.tensor(
            tokenizer(document, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
            device=backend.device,
        ).unsqueeze(0)
        embeds = backend.embed(ids)
        wrapped = wrapper(embeds)
        zero = torch.arange(wrapped.shape[1], device=backend.device).unsqueeze(0)
        direct_zero = backend.prefill_caches(inputs_embeds=wrapped, position_ids=zero)[0]
        absolute = torch.arange(offset, offset + wrapped.shape[1], device=backend.device).unsqueeze(
            0
        )
        rerotated = stored.rerotate(
            backend.rotary_emb,
            zero,
            absolute,
            nope_dim=backend.nope_dim,
        )
        legacy_layers = []
        for stored_layer, zero_layer, shifted_layer in zip(
            stored.layers, direct_zero.layers, rerotated.layers
        ):
            prefill_key_diff = max(
                prefill_key_diff,
                float((stored_layer.key - zero_layer.key).abs().max().float()),
            )
            prefill_value_diff = max(
                prefill_value_diff,
                float((stored_layer.value - zero_layer.value).abs().max().float()),
            )
            delta = absolute - zero
            cos, sin = backend.rotary_emb(stored_layer.key, position_ids=delta)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            key = stored_layer.key
            half = key.shape[-1] // 2
            rotated_half = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
            legacy_key = key * cos + rotated_half * sin
            rerotated_key_diff = max(
                rerotated_key_diff,
                float((shifted_layer.key - legacy_key).abs().max().float()),
            )
            rerotated_value_diff = max(
                rerotated_value_diff,
                float((shifted_layer.value - stored_layer.value).abs().max().float()),
            )
            legacy_layers.append(KeyValue(legacy_key, stored_layer.value, absolute))
        legacy_shifted.append(KVCache(legacy_layers))
        offset += int(wrapped.shape[1])
    assembled = concat_caches(legacy_shifted)
    query_ids = torch.tensor(
        tokenizer(sample.task_prompt, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
        device=backend.device,
    ).unsqueeze(0)
    query_positions = torch.arange(
        offset, offset + query_ids.shape[1], device=backend.device
    ).unsqueeze(0)
    manual_logits = backend.forward(
        input_ids=query_ids,
        position_ids=query_positions,
        cache=assembled,
    ).logits
    return (
        {
            "packet_prefill_key_max_abs_diff": prefill_key_diff,
            "packet_prefill_value_max_abs_diff": prefill_value_diff,
            "rerotated_old_new_key_max_abs_diff": rerotated_key_diff,
            "rerotated_old_new_value_max_abs_diff": rerotated_value_diff,
        },
        manual_logits,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--wrapper",
        default="tests/reference/adapters/qwen3_4b_biography_8_8.pt",
    )
    parser.add_argument("--output", default="outputs/reference_parity_qwen3_gate1.json")
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    storage = os.environ.get("KVP_STORAGE")
    if not storage:
        raise RuntimeError("set KVP_STORAGE before running the parity replay")
    artifact_dir = Path(storage) / "kvpacket/reference_parity/qwen3"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw = _raw_samples(100)
    dataset_digest = _canonical_digest(_rows(raw))
    if dataset_digest != LEGACY_DATASET_SHA256:
        raise RuntimeError(f"Gate 0 dataset mismatch: {dataset_digest} != {LEGACY_DATASET_SHA256}")

    backend, tokenizer = _load_runtime()
    samples = _samples(tokenizer, args.num_samples)
    checkpoint = torch.load(args.wrapper, map_location="cpu", weights_only=True)
    header = checkpoint["header"]
    trailer = checkpoint["trailer"]
    wrapper = PacketWrapper(
        header.size(1),
        trailer.size(1),
        header.size(2),
        dtype=header.dtype,
        device=backend.device,
    )
    wrapper.load_state_dict({"header": header, "trailer": trailer})
    wrapper.to(dtype=torch.bfloat16)
    methods: dict[str, Any] = {}
    full_prompt_diffs: list[float] = []
    packet_diagnostics: dict[str, float] = {}
    for method in ("full_recompute", "kv_packet"):
        accumulator = MetricAccumulator()
        sample_results: list[dict[str, object]] = []
        for index, sample in enumerate(samples):
            row, prompt_logits = _run_sample(sample, backend, tokenizer, wrapper, method)
            accumulator.add(
                cast(int, row["tp"]),
                cast(int, row["fp"]),
                cast(int, row["fn"]),
            )
            if method == "full_recompute":
                ids = torch.tensor(
                    row["prompt_token_ids"],
                    dtype=torch.long,
                    device=backend.device,
                ).unsqueeze(0)
                direct = backend.forward(input_ids=ids).logits
                full_prompt_diffs.append(float((direct - prompt_logits).abs().max().float()))
            elif index == 0:
                packet_diagnostics, manual_logits = _packet_diagnostics(
                    sample, backend, tokenizer, wrapper
                )
                packet_diagnostics["packet_query_logit_max_abs_diff"] = float(
                    (manual_logits - prompt_logits).abs().max().float()
                )
            sample_results.append(row)
            if (index + 1) % 10 == 0 or index + 1 == len(samples):
                print(f"[{method}] {index + 1}/{len(samples)}")
        methods[method] = {
            "metrics": accumulator.summary(),
            "historical_f1": HISTORICAL_F1[method],
            "samples": sample_results,
        }

    report = {
        "gate": "0-1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "dataset": {
            "name": "biography",
            "subset": "10k",
            "split": "test",
            "seed": 42,
            "num_documents": 5,
            "reference_raw_sha256": LEGACY_DATASET_SHA256,
            "current_raw_sha256": dataset_digest,
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": 32,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "full_recompute_direct_prompt_max_abs_diff": max(full_prompt_diffs, default=0.0),
        "packet_diagnostics_first_sample": packet_diagnostics,
        "methods": methods,
        "large_artifact_dir": str(artifact_dir),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
