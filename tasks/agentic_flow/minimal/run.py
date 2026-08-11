"""Minimal agentic-flow verification: 3-way compare + RECORD over a controlled suite.

Runs the toy skill-loading suite in full_prefill / naive_concat / kv_packet, reports
task success + TTFT + FLOPs (overall and vs. #activated skills), verifies KV Packet
auto-matches exactly the activated skills, and does a RECORD pass yielding
`TrainingSample`s.

    CUDA_VISIBLE_DEVICES=1 python tasks/agentic_flow/minimal/run.py
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO)

from kvpacket.backend import HFBackend
from kvpacket.packet.wrapper import PacketWrapper
from tasks.agentic_flow.minimal.agent import AgentMode, MinimalAgent
from tasks.agentic_flow.minimal.skills import load_skills
from tasks.agentic_flow.minimal.tasks import build_tasks


def _load_model(name: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    model.eval()
    return HFBackend(model), tok, device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(_REPO, "outputs", "agentic_minimal.json"))
    args = ap.parse_args()

    backend, tok, device = _load_model(args.model, torch.bfloat16)
    print(f"[model] {backend}")
    skills = load_skills()
    tasks = build_tasks()
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]
    print(
        f"[flow] {len(skills)} skills, {len(tasks)} tasks "
        f"(activated-skill counts: {sorted({t.num_skills for t in tasks})})"
    )

    wrapper = PacketWrapper(
        header_len=8,
        trailer_len=8,
        dim=backend.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=False,
    )
    agent = MinimalAgent(backend, tok, skills, wrapper=wrapper)

    modes = [AgentMode.FULL_PREFILL, AgentMode.NAIVE_CONCAT, AgentMode.KV_PACKET]

    # warm up CUDA (kernel compile / allocator) so the first timed task isn't penalized
    for mode in modes:
        agent.run(tasks[-1], mode, max_new_tokens=2)
    summary: dict[str, dict] = {}
    by_nskills: dict[str, dict[int, dict]] = {}
    match_ok = True

    for mode in modes:
        agg = {"success": 0, "f1": 0.0, "ttft": 0.0, "flops": 0, "n": 0}
        buckets: dict[int, dict] = defaultdict(lambda: {"ttft": 0.0, "flops": 0, "n": 0})
        for task in tasks:
            r = agent.run(task, mode, max_new_tokens=args.max_new_tokens)
            agg["success"] += int(r.success)
            agg["f1"] += r.f1
            agg["ttft"] += r.ttft
            agg["flops"] += r.flops
            agg["n"] += 1
            b = buckets[task.num_skills]
            b["ttft"] += r.ttft
            b["flops"] += r.flops
            b["n"] += 1
            if mode is AgentMode.KV_PACKET and sorted(r.matched_skills) != sorted(
                r.activated_skills
            ):
                match_ok = False
                print(
                    f"  [warn] {task.task_id}: matched {r.matched_skills} != activated {r.activated_skills}"
                )
        n = agg["n"]
        summary[mode.value] = {
            "success_rate": agg["success"] / n,
            "macro_f1": agg["f1"] / n,
            "ttft_ms": agg["ttft"] / n * 1e3,
            "flops": agg["flops"] / n,
        }
        by_nskills[mode.value] = {
            k: {"ttft_ms": v["ttft"] / v["n"] * 1e3, "flops": v["flops"] / v["n"]}
            for k, v in sorted(buckets.items())
        }

    # RECORD pass -> training samples
    samples = [agent.record(t, max_new_tokens=args.max_new_tokens) for t in tasks]

    _print(summary, by_nskills, samples, match_ok)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "model": args.model,
                "num_tasks": len(tasks),
                "summary": summary,
                "by_num_skills": by_nskills,
                "auto_match_ok": match_ok,
                "num_record_samples": len(samples),
            },
            f,
            indent=2,
        )
    print(f"\n[out] wrote {args.out}")

    assert match_ok, "KV Packet did not auto-match the activated skills on every task"
    assert all(s.num_generated > 0 for s in samples), "RECORD produced empty trajectories"
    assert summary["full_prefill"]["flops"] > 50 * summary["kv_packet"]["flops"]
    print("[ok] D3 verification: auto-match exact, RECORD samples valid, FLOPs >> KV Packet.")


def _print(summary, by_nskills, samples, match_ok) -> None:
    print("\n" + "=" * 70)
    print(f"{'mode':<16}{'success':>10}{'macro_f1':>10}{'ttft_ms':>12}{'flops':>14}")
    print("-" * 70)
    for mode, s in summary.items():
        print(
            f"{mode:<16}{s['success_rate']:>10.3f}{s['macro_f1']:>10.3f}"
            f"{s['ttft_ms']:>12.1f}{s['flops']:>14.3g}"
        )
    print("=" * 70)
    print(f"auto-match activated skills (kv_packet): {'OK' if match_ok else 'FAILED'}")
    print(f"RECORD training samples: {len(samples)}  e.g. {samples[0]}")

    print("\nTTFT / FLOPs vs. #activated skills (full_prefill -> kv_packet):")
    counts = sorted(by_nskills["kv_packet"].keys())
    print(f"  {'#skills':>8}{'FP ttft':>10}{'KVP ttft':>10}{'FP flops':>14}{'KVP flops':>14}")
    for k in counts:
        fp, kvp = by_nskills["full_prefill"][k], by_nskills["kv_packet"][k]
        print(
            f"  {k:>8}{fp['ttft_ms']:>9.1f}m{kvp['ttft_ms']:>9.1f}m"
            f"{fp['flops']:>14.3g}{kvp['flops']:>14.3g}"
        )


if __name__ == "__main__":
    main()
