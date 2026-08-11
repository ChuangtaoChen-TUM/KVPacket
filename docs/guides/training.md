# Training objectives, batching, and initialization

`PacketTrainer` separates three independent choices:

1. the statistical unit used by the objective;
2. the samples or tokens included in one optimizer step; and
3. the number of samples evaluated simultaneously by the backend.

## Teacher objective

`PacketTrainer` supports two self-distillation targets:

- `objective="teacher_kl"` (default) minimizes KL divergence against the frozen
  full-attention teacher distribution. Each `TrainingSample` must store one
  full-vocabulary logit vector per generated token.
- `objective="teacher_ce"` minimizes cross-entropy against the token IDs generated
  by that same teacher. It does not use task gold answers or add capabilities beyond
  the teacher trajectory. The sample retains `token_ids`, `generated_mask`, and
  `chunk_spans`, while `logits` may be `None`.

For the config-driven multi-document example, set `"objective": "teacher_ce"`
before RECORD. The recorder then calls generation with logit capture disabled, so
the persisted training samples are compact. Do not reuse one RECORD directory across
KL and CE configs: the runner checks the sample representation and rejects a mismatch.

The objective and `loss_reduction` are independent. CE still supports token-,
sample-, and legacy-sum reductions. KL remains the default because the paper ablation
showed a meaningful CE regression on MuSiQue even though other tasks were comparable.

## Loss reduction

`loss_reduction="token_mean"` is the recommended default for teacher-logit
distillation. It sums KL over every generated position in the optimizer batch and
divides by the number of generated tokens belonging to samples that use at least
one optimized wrapper.

`loss_reduction="sample_mean"` first averages each sample over its generated
positions and then averages the active samples. This gives every sample equal
weight regardless of generation length.

`loss_reduction="legacy_sum"` reproduces the old unnormalized backward scaling and
is retained for diagnostics. Metrics remain token-normalized so runs are readable.

Samples containing only frozen wrappers may be evaluated in a mixed batch, but
they do not dilute the training denominator.

## Optimizer batching

Sample-level training uses fixed sample-count batches:

```python
trainer = PacketTrainer(
    backend,
    wrapper_registry,
    loss_reduction="sample_mean",
)
metrics = trainer.fit(
    samples,
    batch_size=32,
    forward_batch_size=4,
)
```

The final partial batch is processed unless `drop_last=True`.

Token-level training can instead use an approximate generated-token budget:

```python
trainer = PacketTrainer(
    backend,
    wrapper_registry,
    loss_reduction="token_mean",
)
metrics = trainer.fit(
    samples,
    target_tokens_per_step=2048,
    batch_size=64,          # maximum samples in one optimizer step
    forward_batch_size=4,
)
```

Samples are added until their generated-token count reaches the target or the
sample cap is reached. A single long sample may exceed the target. Token-budget
batching is rejected for `sample_mean` and `legacy_sum`.

## Forward microbatching

`forward_batch_size=1` is the memory-efficient default. Larger values activate the
optional batched 4-D path:

- wrapped embeddings are right-padded;
- independent packet masks are placed into `[B, 1, L, L]`;
- real rows cannot attend padding;
- padded rows receive only a self edge and are never selected for loss;
- generated-logit positions remain sample-specific; and
- normalization is performed over the complete optimizer batch, not separately
  per forward microbatch.

Changing `forward_batch_size` changes performance and floating-point execution
order, but not the mathematical objective.

## Wrapper initialization

Wrapper initialization is explicit and backend-aware:

```python
from kvpacket import create_wrapper

wrapper = create_wrapper(
    backend,
    header_len=8,
    trailer_len=8,
    dtype=torch.bfloat16,
    initialization={
        "strategy": "fixed",
        "mean": 0.0,
        "std": 0.02,
    },
)
```

or:

```python
wrapper = create_wrapper(
    backend,
    header_len=8,
    trailer_len=8,
    dtype=torch.bfloat16,
    initialization={"strategy": "model-embedding"},
)
```

For `model-embedding`, the selected `TrainingBackend` owns access to its model and
returns the input-embedding mean and standard deviation. `PacketWrapper` remains
backend-independent and receives ordinary numeric initialization parameters.

Users who want deterministic or semantic soft-token initialization can copy any
compatible embeddings into the randomly initialized tensors. Do this before creating
the `PacketTrainer` so its optimizer starts from the edited values, and preserve the
existing tensor objects:

```python
header_ids = torch.tensor([[101, 102, 103, 104]], device=backend.device)
trailer_ids = torch.tensor([[201, 202]], device=backend.device)

with torch.no_grad():
    wrapper.header.copy_(backend.embed(header_ids).to(wrapper.header))
    wrapper.trailer.copy_(backend.embed(trailer_ids).to(wrapper.trailer))
```

The token counts must equal `header_len` and `trailer_len`, and the embedding width
must equal `wrapper.dim`. Do not assign new tensors to `wrapper.header` or
`wrapper.trailer` after a trainer has been created: replacing them would leave the
optimizer referencing the old parameter objects. If initialization is changed after
training has begun, create a fresh trainer so AdamW moment state does not carry over.
