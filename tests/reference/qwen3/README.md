# Reference parity harness

These manual reference-validation scripts compare replay, training differentials,
checkpoint reload, and short convergence against the exact public
`Qwen/Qwen3-4B-Instruct-2507` revision used by the old experiment.

Set `KVP_STORAGE` to a writable large-artifact directory first. Model weights, frozen
teacher logits, samples, and checkpoints are stored below
`$KVP_STORAGE/kvpacket/reference_parity/qwen3`; only JSON reports are stored
in `outputs/`.

```bash
export KVP_STORAGE="/path/to/large/storage"
CUDA_VISIBLE_DEVICES=0 uv run python tests/reference/qwen3/replay.py
CUDA_VISIBLE_DEVICES=0 uv run python tests/reference/qwen3/training_differential.py
CUDA_VISIBLE_DEVICES=0 uv run python tests/reference/qwen3/short_convergence.py
```

- `replay.py`: Gates 0–1, including the frozen old/new dataset digest,
  100-sample full-recompute and packet generation, and cache/logit diagnostics.
- `training_differential.py`: Gate 2, using one saved teacher trajectory and
  initial wrapper tensors for the legacy summed-KL differential.
- `short_convergence.py`: Gate 3, using 16 saved trajectories, four optimizer
  steps, checkpoint reload, and held-out free generation.

The scripts require the pinned Qwen3 snapshot to be present locally and use
`local_files_only=True`; they never silently download model weights.
