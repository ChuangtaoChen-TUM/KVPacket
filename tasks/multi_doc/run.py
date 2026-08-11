"""Multi-doc (RAG) engine verification + paper-style reproduction.

Drives a :class:`PacketSession` over a retrieval dataset in three modes
(full_recompute / no_recompute / kv_packet) and reports token-F1, exact-match, TTFT,
and FLOPs per method — the well-understood setting that confirms the engine is
correct before agentic work.

    python tasks/multi_doc/run.py --config /path/to/evaluation.json
    python tasks/multi_doc/run.py --config /path/to/evaluation.json --num-samples 20

Pick a GPU with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kvpacket.backend import HFBackend
from kvpacket.packet.wrapper import PacketWrapper
from kvpacket.training import PacketTrainer
from tasks.common.config import load_config
from tasks.common.metrics import MetricAccumulator
from tasks.multi_doc.data import get_dataset, get_ret_eval_generator
from tasks.multi_doc.evaluator import Method, evaluate_sample

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _load_model(name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=_DTYPES[dtype], device_map=device)
    model.eval()
    return HFBackend(model), tok, device


def _make_wrapper(cfg: dict, backend: HFBackend, device: str) -> PacketWrapper:
    pk = cfg.get("packet", {})
    dtype = next(backend.model.parameters()).dtype
    path = pk.get("path")
    if path:
        trainer = PacketTrainer.load(path, backend=backend)
        wrapper_id = str(pk.get("wrapper_id", "document"))
        return trainer.registry[wrapper_id].wrapper.to(device=device, dtype=dtype)
    return PacketWrapper(
        header_len=int(pk.get("header_len", 8)),
        trailer_len=int(pk.get("trailer_len", 8)),
        dim=backend.hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples

    print(f"[cfg] {json.dumps({k: v for k, v in cfg.items() if k != 'packet'})}")
    backend, tok, device = _load_model(cfg["model"], cfg.get("dtype", "bfloat16"))
    print(f"[model] {backend}")

    spec = get_dataset(cfg["dataset"])
    samples = list(
        get_ret_eval_generator(
            name=cfg["dataset"],
            num_samples=cfg["num_samples"],
            num_documents=cfg.get("num_documents", 4),
            num_shots=cfg.get("num_shots", 0),
            subset=cfg.get("subset", "1k" if cfg["dataset"] == "biography" else "default"),
            split=cfg.get("split", "test"),
            seed=cfg.get("seed", 0),
            tokenizer=tok,
            data_kwargs=cfg.get("dataset_kwargs"),
            chat_template_kwargs=cfg.get("chat_template_kwargs"),
            system_prompt=cfg.get("system_prompt", ""),
        )
    )
    print(
        f"[data] {cfg['dataset']}: {len(samples)} samples, {cfg.get('num_documents', 4)} docs each"
    )

    wrapper = None
    methods = [
        Method(m) for m in cfg.get("methods", ["full_recompute", "no_recompute", "kv_packet"])
    ]
    if Method.KV_PACKET in methods:
        wrapper = _make_wrapper(cfg, backend, device)
        print(f"[packet] {wrapper}  (trained={bool(cfg.get('packet', {}).get('path'))})")

    results: dict[str, dict] = {}
    for method in methods:
        acc = MetricAccumulator()
        rows: list[dict] = []
        t0 = time.time()
        for sample in samples:
            r = evaluate_sample(
                sample,
                backend,
                tok,
                method,
                wrapper=wrapper,
                max_new_tokens=cfg.get("max_new_tokens", 16),
                answer_postprocess=spec.answer_postprocess,
                match=cfg.get("match", "token"),
                cache_preamble_before_ttft=bool(cfg.get("cache_preamble_before_ttft", False)),
                mixed_prefill_policy=cfg.get("mixed_prefill_policy", "sequential"),
            )
            acc.add(r.tp, r.fp, r.fn, ttft=r.ttft, flops=r.flops, exact=r.exact)
            if cfg.get("include_rows", False):
                rows.append(
                    {
                        "dataset_index": sample.meta.get("index"),
                        "sample_meta": sample.meta,
                        "raw_response": r.raw_pred,
                        "prediction": r.pred,
                        "gold": r.gold,
                        "tp": r.tp,
                        "fp": r.fp,
                        "fn": r.fn,
                        "exact": r.exact,
                        "ttft_s": r.ttft,
                        "flops": r.flops,
                        "online_total_flops": r.online_total_flops,
                        "num_generated": r.num_generated,
                    }
                )
        summ = acc.summary()
        summ["wall_s"] = round(time.time() - t0, 2)
        if rows:
            summ["rows"] = rows
        results[method.value] = summ

    _print_table(results)
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs",
        f"multi_doc_{cfg['dataset']}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"config": {k: v for k, v in cfg.items() if k != "packet"}, "results": results},
            f,
            indent=2,
        )
    print(f"\n[out] wrote {out}")


def _print_table(results: dict[str, dict]) -> None:
    cols = ["f1", "macro_f1", "exact_match", "ttft_ms", "flops"]
    print("\n" + "=" * 78)
    print(f"{'method':<16}" + "".join(f"{c:>13}" for c in cols))
    print("-" * 78)
    for method, s in results.items():
        row = f"{method:<16}"
        for c in cols:
            v = s[c]
            row += f"{v:>13.4g}" if isinstance(v, float) else f"{v:>13}"
        print(row)
    print("=" * 78)


if __name__ == "__main__":
    main()
