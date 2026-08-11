"""Phase 2 Deliverable 3 — minimal agentic framework.

* ``uv run pytest tests/test_phase3_agentic.py``   (light: skills, tasks, answers)
* ``python tests/test_phase3_agentic.py``          (also runs a small 3-mode + RECORD check)
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.agentic_flow.minimal.skills import load_skills
from tasks.agentic_flow.minimal.tasks import (
    _caesar_decode,
    _initials,
    _pig_latin,
    _reverse,
    _roman,
    build_tasks,
)


def test_skill_loading():
    skills = load_skills()
    assert set(skills) >= {"caesar-shift", "word-reverse", "roman-numeral", "initials", "pig-latin"}
    for s in skills.values():
        assert s.description and s.body
        assert s.block.endswith("\n\n")  # trailing blank line -> back-to-back packets


def test_solvers():
    assert _caesar_decode("khoor") == "hello"
    assert _reverse("stressed") == "desserts"
    assert _roman(24) == "XXIV" and _roman(49) == "XLIX"
    assert _initials("john von neumann") == "J.V.N."
    assert _pig_latin("smile") == "ilesmay" and _pig_latin("pig") == "igpay"


def test_task_suite_controlled():
    tasks = build_tasks()
    assert len(tasks) == 15
    assert sorted({t.num_skills for t in tasks}) == [1, 2, 3, 4, 5]
    for t in tasks:
        assert t.primary in t.skills  # the needed skill is activated
        assert t.answer  # gold answer present
        assert len(t.skills) == t.num_skills


def _run_light():
    test_skill_loading()
    print("[ok] skill loading + frontmatter")
    test_solvers()
    print("[ok] reference solvers")
    test_task_suite_controlled()
    print("[ok] controlled task suite (1..5 skills)")


def _run_live():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpacket.backend import HFBackend
    from kvpacket.packet.wrapper import PacketWrapper
    from tasks.agentic_flow.minimal.agent import AgentMode, MinimalAgent

    name = "Qwen/Qwen2.5-1.5B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[live] loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    adapter = HFBackend(model)

    skills = load_skills()
    tasks = build_tasks()
    wrapper = PacketWrapper(
        header_len=8,
        trailer_len=8,
        dim=adapter.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=False,
    )
    agent = MinimalAgent(adapter, tok, skills, wrapper=wrapper)

    # a 3-skill task exercises multiple packets
    task = next(t for t in tasks if t.num_skills == 3)
    fp = agent.run(task, AgentMode.FULL_PREFILL, max_new_tokens=8)
    nc = agent.run(task, AgentMode.NAIVE_CONCAT, max_new_tokens=8)
    kv = agent.run(task, AgentMode.KV_PACKET, max_new_tokens=8)

    # auto-match: KV Packet must detect exactly the activated skills (no chunk_id given)
    assert sorted(kv.matched_skills) == sorted(task.skills), (kv.matched_skills, task.skills)
    assert sorted(nc.matched_skills) == sorted(task.skills)
    # fundamental efficiency: full prefill spends orders of magnitude more FLOPs
    assert fp.flops > 100 * kv.flops, (fp.flops, kv.flops)
    print(f"[live] task {task.task_id}: matched={kv.matched_skills}")
    print(
        f"[live] flops  full_prefill={fp.flops:.3g}  kv_packet={kv.flops:.3g}  "
        f"(ratio {fp.flops / kv.flops:.0f}x)"
    )

    # RECORD yields a usable TrainingSample with the skill spans
    sample = agent.record(task, max_new_tokens=8)
    assert sample.num_generated > 0
    assert set(sample.chunk_spans) == set(task.skills)
    print(f"[live] RECORD sample: {sample}")
    print("[live] D3 checks passed.")


if __name__ == "__main__":
    _run_light()
    if "--no-model" not in sys.argv:
        _run_live()
    print("\nAll Phase 2 agentic checks passed.")
