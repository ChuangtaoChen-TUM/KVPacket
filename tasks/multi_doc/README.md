# `tasks/multi_doc` — RAG engine verification & paper reproduction

Drives a `PacketSession` over a retrieval dataset in three modes and reports
token-F1 / exact-match / TTFT / FLOPs. This is the **well-understood setting** that
confirms the engine is correct before agentic work.

The three methods are the *same engine* configured differently:

| method | session config | role |
|---|---|---|
| `full_recompute` | no chunks → whole context forwarded fresh | **oracle** (upper bound) |
| `no_recompute`   | docs as chunks, **no** wrapper → raw caches, contiguous re-rotation | **naive** (lower bound) |
| `kv_packet`      | docs as chunks, trained `PacketWrapper` → packets | **the method** |

## Minimal adapter-training example

`train.py` demonstrates the complete public workflow without relying on internal
experiment scripts:

1. RECORD full-attention teacher trajectories for a training split.
2. Train one document wrapper with `PacketTrainer`.
3. Save and reload the trainer checkpoint.
4. Evaluate Full Recompute, No Recompute, and KV Packet on held-out samples.

The supplied config uses eight synthetic training examples, four held-out examples,
and `objective="teacher_ce"`, so recorded samples contain teacher-generated token IDs
without full-vocabulary logits. It is a plumbing example rather than a quality claim.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tasks/multi_doc/train.py \
  --config tasks/multi_doc/training_configs/synthetic_qwen.json
```

The stages can also run separately with `--phase record`, `--phase train`, and
`--phase eval`. Artifacts are written below `outputs/multi_doc_training_example/`;
delete that directory to restart from a fresh seeded wrapper. Change the config to
`objective="teacher_kl"` when full-distribution distillation is required.

The former task-level recipes have also been converted to the same standalone schema:

- `biography_qwen2_5_1_5b.json`
- `biography_llama3_1_8b.json`
- `hotpot_qa_llama3_2_3b.json`
- `musique_llama3_2_3b.json`
- `gsm8k_dependent_llama3_2_3b.json`
- `multinews_llama3_2_3b.json`

These larger configs store RECORD samples under
`$KVP_STORAGE/kvpacket/multi_doc_training/`; set `KVP_STORAGE` before running
them. They preserve the former model, data, wrapper, objective, batch, token-budget,
and optimizer settings. The public example runner uses deterministic per-epoch
shuffling and a constant AdamW learning rate; it intentionally does not reproduce the
old LR-sweep or linear-decay experiment harness.

## Paper reproduction (stretch)

The exact Qwen3-4B biography reproduction config, converted adapter fixture, and
manual parity harness are kept together under `tests/reference/`; see
`tests/reference/qwen3/README.md`. They need the pinned
`Qwen/Qwen3-4B-Instruct-2507` snapshot in `HF_HOME`. With a trained adapter,
`kv_packet` should approach `full_recompute` F1 while spending only re-rotation FLOPs
— the paper's headline. An untrained wrapper, as used on `Qwen2.5-1.5B`, makes
`kv_packet ≈ no_recompute`: the plumbing runs but adds no quality until training.

## Notes

- **FLOPs**: `full_recompute` reports the analytic document-prefill cost;
  `no_recompute` / `kv_packet` report only re-rotation FLOPs (orders of magnitude less).
- **TTFT** is measured inside the engine (assemble + prefill), synchronized on CUDA.
- `get_ret_eval_generator(..., tokenizer=tokenizer)` applies the tokenizer's native
  chat format once while producing samples. Passing `tokenizer=None` yields plain
  samples. The formatter splits around a document placeholder, keeping documents
  unchanged as independently cacheable packets; evaluators need no template setting.

## Included datasets

The task registry contains every generator from the old project plus the local
synthetic fixture:

| Name | Source | Split behavior | Document behavior |
|---|---|---|---|
| `biography` | `alex-karev/biographies` | deterministic train/test partition | configured 2/3/5/7/etc. documents |
| `hotpot_qa` | `hotpotqa/hotpot_qa` | native train/validation/test | one document per context paragraph |
| `musique` | `dgslibisey/MuSiQue` | native train/validation/test | one document per paragraph |
| `gsm8k_dependent` | `openai/gsm8k` | native train/test | one masked reasoning trace split into ordered chunks |
| `multinews` | `Awesome075/multi_news_parquet` | native train/validation/test | one packet per source article |
| `niah` | `simonjegou/ruler` | fixed-seed half train/test partition | all sentence-packed context chunks |
| `synthetic` | repository-local | deterministic fixture | configured document count |

HotpotQA, MuSiQue, MultiNews, and NIAH use dataset-defined or context-derived document counts, so
their `num_documents` must be `0`. NIAH sentence packing uses NLTK when its tokenizer
data is already installed and otherwise uses a local regex fallback; it never downloads
NLTK data implicitly.

The old `mixture` name was not a separate dataset. It was a training recipe that
iterated biography, NIAH, MuSiQue, and HotpotQA entries through this same registry.
