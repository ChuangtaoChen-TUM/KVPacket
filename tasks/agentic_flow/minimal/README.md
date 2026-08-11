# `tasks/agentic_flow/minimal` — a tiny in-house agentic framework

A dependency-free agentic flow to validate the **agentic path end-to-end on
something small and fully controlled**, before wiring popular/complex frameworks.

This is the Agent-Skills setting: a **skill** is reusable static text (a `SKILL.md`
with `name` / `description` / body). Loading a skill into context ("L2 activation")
is a re-prefill of that static block at a dynamic position — **exactly** KV Packet's
SERVE-mode substitution. So each skill body is registered as a chunk and served from a
packet, regardless of what precedes it.

## Pieces

- `skills/*.md` — 5 toy skills (caesar-shift, word-reverse, roman-numeral, initials,
  pig-latin), each a rule + worked example.
- `skills.py` — `SKILL.md` frontmatter loader.
- `tasks.py` — 15 controlled tasks; each activates a **known** skill subset so the
  independent variable (#activated skills, 1..5) is exact. Answers are computed
  programmatically (correct by construction).
- `agent.py` — `MinimalAgent`: assembles `system + [skill bodies] + task` as blocks,
  drives a `PacketSession`, generates. The caller never says which block is which
  skill — the **library matches** the registered skill bodies.
- `run.py` — 3-way compare + a RECORD pass, with verification.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 uv run python tasks/agentic_flow/minimal/run.py
```

Three modes (same engine, different config):

| mode | config | role |
|---|---|---|
| `full_prefill` | no chunks — all skills prefilled every task | oracle |
| `naive_concat` | skills as chunks, no wrapper | naive bound |
| `kv_packet`    | skills as chunks + `PacketWrapper` | the method |

Plus a **RECORD** pass producing one `TrainingSample` per task (skill spans +
generated-token mask + teacher logits) for the future trainer.

## What it verifies

- The loop runs in all three modes on `Qwen2.5-1.5B`.
- **KV Packet auto-matches exactly the activated skills** on every task (the caller
  passes no `chunk_id`).
- **FLOPs**: `full_prefill` spends the full skill-prefill cost (grows with #skills,
  ~1e11–1.5e12), while `kv_packet` spends only re-rotation (~4e6–2e7) — a **~5×10⁴×**
  reduction whose gap *widens* with more activated skills.
- RECORD yields a `list[TrainingSample]` usable by the trainer.

## Honest notes

- **Task quality**: the wrapper here is **untrained** (there is no adapter for
  `Qwen2.5-1.5B`), so `kv_packet ≈ naive_concat` on success and both trail the
  `full_prefill` oracle. The *structural* path is what this validates; quality
  requires Phase-5 training (or a trained adapter), after which `kv_packet` should
  approach `full_prefill`. A RECORD → train loop over this suite is the intended next
  step.
- **Wallclock TTFT**: on a 1.5B model with ~100-token skills, `kv_packet`'s wallclock
  TTFT can *exceed* `full_prefill`'s, because re-rotation is currently a per-layer,
  per-packet Python loop of small GPU ops, while `full_prefill` is one fused matmul
  that is nearly free at this size. The **FLOPs** advantage is fundamental and
  hardware-independent; the wallclock win appears at scale (larger models / longer
  reusable content, where prefill dominates) and/or once re-rotation is batched across
  layers. Report FLOPs as the primary efficiency metric.
