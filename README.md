<p align="center">
  <h3 align="center"><strong>KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs </strong></h3>

<p align="center">
    Chuangtao Chen<sup>1</sup>,
    Grace Li Zhang<sup>2</sup>,
    XunZhao Yin<sup>3</sup>,
    Cheng Zhuo<sup>3</sup>,
    Bing Li<sup>4</sup>,
    Ulf Schlichtmann<sup>1</sup><br>
    <sup>1</sup>Technical University of Munich,
    <sup>2</sup>Technical University of Darmstadt<br>
    <sup>3</sup>Zhejiang Univerity,
    <sup>4</sup>Technische Universität Ilmenau
</p>



<div align="center">

<a href='https://arxiv.org/abs/2604.13226'><img src='https://img.shields.io/badge/arXiv-2604.13226-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/ChuangtaoChen-TUM/KVPacket/blob/master/LICENSE'><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</div>

<p align="center">
    <img src="./res/cache_pipeline.png" alt width="700" >
    <em><br>(a) Recomputation-based approaches require inference-time algorithms to select and recompute important tokens to repair contextual staleness; (b) the proposed KV Packet approach wraps documents with global adapters.</em>
</p>

<p align="center">
    <img src="./res/main_results.png" alt width="600" >
    <em><br>Evaluation results (F1 score, FLOPs, Time-to-First-Token) of Llama-3.1-8B / Qwen-3-4B on datasets: Needle-in-a-Haystack, Biography, HotpotQA, and MusiQue.</em>
</p>

# KV Packet

**KV Packet** is a framework for reusing precomputed KV caches across documents in multi-document RAG settings, without recomputation.

## Core Idea

Each document's KV cache is wrapped with a small set of trainable soft-token vectors — a **header** prepended before the document and a **trailer** appended after it. At inference time, independently cached documents are directly concatenated. No recomputation is needed.

**Why it works:** Naive KV cache concatenation fails because of boundary artifacts — disrupted attention sinks and abrupt token distribution shifts at block boundaries. The learned adapter vectors act as smooth delimiters that absorb these artifacts, restoring output quality close to full-attention inference.

**Training:** Adapters are trained via self-supervised KL distillation. The model's own full-attention output serves as the teacher signal, and only the small adapter tensors receive gradients. No labeled data or base model modification is needed.

**Key results:**
- ~4–6 orders of magnitude fewer FLOPs than recomputation-based baselines (CacheBlend, EPIC, A3)
- Lower TTFT than all recomputation methods
- Competitive F1 on retrieval (NIAH, Biography) and reasoning (HotpotQA, MuSiQue) benchmarks
- Naturally compatible with KV compression techniques
- Storage overhead of 0.4%–6% for realistic document lengths (≥512 tokens)

**Models tested:** Llama-3.1-8B-Instruct and Qwen2.5/3.

---

## Project Structure

```
kv_packet_clean/
├── run_train_filler.py         # Phase 1: train header/trailer adapters
├── run_eval.py                 # Phase 2: evaluate on benchmarks
├── run_build_packet.py         # Ablation A.1: initialize wrapper from handcrafted tokens
│
├── kv_packet/                  # Core library
│   ├── packet_wrapper/         # PacketWrapper: header/trailer parameter tensors
│   ├── cache/                  # KV cache storage, quantization, compression, re-rotation
│   ├── cache_comb/             # Cache combination methods (KV Packet + all baselines)
│   ├── dataset/                # Dataset loaders (Biography, HotpotQA, NIAH, MuSiQue)
│   ├── model/                  # Supported model type definitions
│   └── utils/                  # Training loop, generation cache, metrics, config loader
│
├── packet_wrapper_config/      # Training configs organised by model and dataset
├── eval_config/                # Evaluation configs organised by model and dataset
├── ablation_study/             # Ablation experiments (loss type, explicit tokens)
└── plot_scripts/               # Result visualisation
```

---

## Workflow

### Train

Trains the header and trailer adapters on one or more retrieval datasets. Only the wrapper parameters receive gradients; the base model is frozen throughout.

```bash
python run_train_filler.py <config.json> [<config2.json> ...]
# or pass a directory to pick up all configs inside it
python run_train_filler.py packet_wrapper_config/llama_3_1_8b/mixture/
```

Checkpoints can be saved using the config. Training can be resumed from a checkpoint by setting `resume: true` in the config.

### Evaluate

Runs one or more evaluation configs and writes results as JSON files alongside each config.

```bash
python run_eval.py <config.json or directory> [--overwrite] [--debug]
```

`--overwrite` re-runs configs that already have a result file. `--debug` disables the progress bar.

Results are written to `eval_results/<config_name>_result.json` next to each config file.

### Setup

Before running training or evaluation, update the `model.model_path` field in all config files to point to your local model directory or a HuggingFace model identifier. The default paths in the example configs may not exist on your system.


---

## Training Configuration Reference

Training configs are JSON files. A `_default.json` in the same directory sets shared defaults; individual configs extend (not overwrite) those defaults.

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_epoch` | int | yes | Number of complete passes over the training data. |
| `batch_size` | int | yes | Number of samples per gradient update (accumulation boundary). Must divide the total number of samples evenly. |
| `gen_batch_size` | int | yes | Batch size used when generating teacher targets with the frozen model. Set lower if GPU memory is tight. |
| `forward_batch_size` | int | yes | Micro-batch size for the forward pass with gradients. Must divide `batch_size` evenly. Set to `1` to enable the single-sample training path with gradient checkpointing. |
| `header_len` | int | yes | Number of trainable soft tokens prepended to each document. |
| `trailer_len` | int | yes | Number of trainable soft tokens appended to each document. |
| `use_logits` | bool | yes | If `true`, use KL-divergence against the teacher's full logits. If `false`, use cross-entropy against the greedy-decoded token IDs. KL distillation is stronger and is used in all main experiments. |
| `seed` | int | no | Global random seed for sample shuffling (default: `42`). |
| `cache_device` | str | no | Device for storing the generation cache (e.g. `"cpu"`, `"cuda:0"`). Defaults to `"cuda:0"`. Using `"cpu"` saves GPU memory at the cost of transfer overhead. |
| `cache_path` | str \| null | no | Path to persist the generation cache to disk. If set and the file exists, it is loaded at startup and updated after new generations are added. Useful to avoid regenerating the same targets across runs. |
| `ckpt_epoch` | int | yes | Save an intermediate checkpoint every this many epochs. Set to `0` to disable intermediate checkpoints. The final model is always saved. |
| `save_path` | str | yes | Directory where checkpoints and the final wrapper are saved. Created automatically if it does not exist. |
| `file_name` | str | yes | Filename for the final saved wrapper (e.g. `"8_8.pt"`). Intermediate checkpoints are saved as `<file_name>.epoch<N>`. |
| `dtype` | str \| null | no | dtype for the wrapper parameters. If `null`, inherits from `model.dtype`. One of `"float32"`, `"float16"`, `"bfloat16"`. |
| `use_cache` | bool | no | If `true`, reuse the loaded model and tokenizer across multiple configs in the same run. Defaults to `false`. |
| `resume` | bool | no | If `true`, resume training from an existing checkpoint. Defaults to `false`. |
| `resume_epoch` | int \| null | no | Specific epoch checkpoint to resume from. If `null`, the latest available checkpoint in `save_path` is used. |

### `model` block

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | str | yes | Path (local) or HuggingFace identifier for the base model. |
| `dtype` | str | no | Model loading dtype. One of `"float32"`, `"float16"`, `"bfloat16"`. Defaults to `"bfloat16"`. |
| `device` | str | no | Device for the model. Use `"auto"` for multi-GPU distribution, or a specific device like `"cuda:0"`. Defaults to `"cuda:0"`. |
| `generation_kwargs` | dict | no | Forwarded to `GenerationConfig` for teacher target generation. Common keys: `max_new_tokens`, `stop_strings`, `do_sample`, `use_cache`. |

### `opt_config` block

Passed directly to `torch.optim.AdamW`. Only the wrapper parameters (header and trailer) are optimised.

| Field | Type | Description |
|-------|------|-------------|
| `lr` | float | Learning rate (e.g. `5e-4`). |
| `weight_decay` | float | L2 regularisation strength (typically `0.0`). |

### `scheduler_config` block

Controls the `torch.optim.lr_scheduler.LinearLR` schedule applied after each gradient step.

| Field | Type | Description |
|-------|------|-------------|
| `start_factor` | float | Multiplier applied to `lr` at the first step (e.g. `1.0` = full lr). |
| `end_factor` | float | Multiplier applied to `lr` at the last step (e.g. `0.0` = linear decay to zero). |
| `total_iters` | int | Steps over which the schedule runs. Automatically set to `(num_samples / batch_size) * total_epoch` if omitted or `0`. |

### `data_configs` block

A list of dataset configurations. All entries are concatenated into a single training pool.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_name` | str | yes | Dataset to use. One of `"biography"`, `"hotpot_qa"`, `"niah"`, `"musique"`. |
| `num_samples` | int | yes | Number of samples to draw from the dataset. |
| `num_data_strs` | int | yes | Number of documents per sample. For NIAH and multi-hop datasets that pack documents internally, set to `0` to use the dataset's own document count. |
| `num_shots` | int | yes | Number of few-shot examples prepended to the prompt. Typically `0` during training. |
| `subset` | str | yes | Dataset-specific split variant (e.g. `"10k"` for Biography, `"fullwiki"` for HotpotQA, `"8192"` for NIAH context length, `"default"` for MuSiQue). |
| `split` | str | no | HuggingFace dataset split. One of `"train"`, `"validation"`, `"test"`. Defaults to `"train"`. |
| `seed` | int | no | Seed for sampling from the dataset. Defaults to `42`. Use different seeds for data augmentation across multiple entries. |
| `data_kwargs` | dict | no | Dataset-specific keyword arguments (see below). |
| `template` | str | no | Chat template to apply. One of `"llama_chat"`, `"qwen_3_chat"`, `"default"` (pass-through). |
| `template_kwargs` | dict | no | Additional keyword arguments for the template (e.g. `{"question_pos": "end"}` to place the question after the documents). |

#### `data_kwargs` options by dataset

**`biography`**
| Key | Description |
|-----|-------------|
| `question_type` | `"QA"` for question-answer format. |
| `cache_dataset` | If `true`, caches the loaded dataset in memory. |

**`niah`**
| Key | Description |
|-----|-------------|
| `chunk_size` | Token length of each document chunk (e.g. `4096`). |

**`hotpot_qa`** and **`musique`**
| Key | Description |
|-----|-------------|
| `add_inst` | If `true`, adds an instruction prefix to the prompt. |
| `add_cot` | If `true`, adds chain-of-thought formatting. |

### Example training config

```json
{
    "total_epoch": 30,
    "batch_size": 64,
    "gen_batch_size": 4,
    "forward_batch_size": 8,
    "header_len": 8,
    "trailer_len": 8,
    "use_logits": true,
    "seed": 42,
    "cache_device": "cpu",
    "ckpt_epoch": 5,
    "save_path": "./packet_wrapper/llama_3_1_8b/mixture/",
    "file_name": "8_8.pt",
    "model": {
        "model_path": "/path/to/llama-3.1-8b-instruct",
        "dtype": "bfloat16",
        "device": "auto",
        "generation_kwargs": {
            "max_new_tokens": 512,
            "stop_strings": ["<|eot_id|>"],
            "do_sample": false,
            "use_cache": true
        }
    },
    "opt_config": { "lr": 5e-4, "weight_decay": 0.0 },
    "scheduler_config": { "start_factor": 1.0, "end_factor": 0.0 },
    "data_configs": [
        {
            "dataset_name": "biography",
            "num_samples": 16,
            "num_data_strs": 5,
            "num_shots": 0,
            "subset": "10k",
            "split": "train",
            "seed": 0,
            "data_kwargs": { "question_type": "QA", "cache_dataset": true },
            "template": "llama_chat",
            "template_kwargs": { "question_pos": "end" }
        }
    ]
}
```

---

## Evaluation Configuration Reference

Evaluation configs also use `_default.json` inheritance. The `model` and `dataset` blocks follow the same structure as in training configs. Per-method override files only need to specify the fields that differ from the default.

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | dict | yes | See model block below. |
| `dataset` | dict | yes | See dataset block below. |
| `cache_comb` | dict | yes | Cache combination method and its arguments (see below). |
| `packet_wrapper` | str \| null | no | Path to a trained `.pt` wrapper file. Required only when `cache_comb.method` is `"kv_packet"`. |
| `compress` | dict \| null | no | Optional KV cache compression. See compress block below. |
| `quantization` | dict \| null | no | Optional KV cache quantization. See quantization block below. |
| `seed` | int | yes | Random seed for evaluation sample selection. |

### `model` block

Same fields as in training: `model_path`, `dtype`, `device`, `generation_kwargs`. During evaluation `max_new_tokens` is typically much smaller (e.g. `32`) since only the final answer needs to be generated.

### `dataset` block

Same fields as a single `data_configs` entry in training: `dataset_name`, `num_samples`, `num_data_strs`, `num_shots`, `subset`, `split`, `seed`, `data_kwargs`, `template`, `template_kwargs`.

### `cache_comb` block

Selects how independently cached document KV caches are combined before generation.

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | One of the method names listed below. |
| `kwargs` | dict | Method-specific parameters (see per-method tables). |

#### Available methods

| Method | Description |
|--------|-------------|
| `"kv_packet"` | **Proposed method.** Concatenates wrapped document caches and re-rotates positional embeddings to reflect the new token positions. Requires `packet_wrapper` to be set. |
| `"full_recompute"` | Oracle baseline. Concatenates all documents into a single sequence and runs a full forward pass. Highest quality, highest cost. |
| `"no_cache"` | Recomputes each document's KV cache independently at inference time without reuse. |
| `"no_recompute"` | Naive concatenation of precomputed caches without any position correction. Cheapest but lowest quality. |
| `"cache_blend"` | Weighted average of token KV vectors across cached documents. |
| `"rand_recompute"` | Randomly selects a fraction of tokens to recompute in the combined cache. |
| `"epic"` | EPIC: recomputes a fixed number of boundary tokens to repair attention sinks. |
| `"a3"` | A³: anchor-based attention repair at cache boundaries. |
| `"sam_kv"` | SAM-KV: uses stable attention layers to merge caches selectively. |
| `"single_cache"` | Concatenates all documents into one cache before compression. Only meaningful when `compress` is set. |

#### `kwargs` per method

**`kv_packet`** — no required kwargs.

**`no_recompute`** — no kwargs.

**`full_recompute`** — no kwargs.

**`no_cache`** — no kwargs.

**`cache_blend`**

| Key | Type | Description |
|-----|------|-------------|
| `recompute_ratio` | float | Fraction of tokens to blend (0–1). Higher = more recomputation. |

**`rand_recompute`**

| Key | Type | Description |
|-----|------|-------------|
| `recompute_ratio` | float | Fraction of tokens to recompute at random (0–1). |

**`epic`**

| Key | Type | Description |
|-----|------|-------------|
| `recompute_tokens` | int | Number of boundary tokens to recompute per document boundary. |

**`a3`**

| Key | Type | Description |
|-----|------|-------------|
| `recompute_ratio` | float | Fraction of tokens to recompute using the anchor mechanism (0–1). |

**`sam_kv`**

| Key | Type | Description |
|-----|------|-------------|
| `stable_layers` | list[int] | Layer indices considered stable for cache merging (e.g. the last few layers). |
| `num_initial_tokens` | int | Number of leading (sink) tokens to always keep. |
| `num_local_tokens` | int | Number of local context tokens to retain around each boundary. |
| `block_size` | int | Block size for grouped cache merging. |
| `fuse_theta` | float | Fusion threshold (0–1). Higher = more aggressive merging. |

### `compress` block

Applies a KV cache compression method from the [kvpress](https://github.com/IsaacRe/kvpress) library to each document's cache before combination.

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | Name of the kvpress compressor class (e.g. `"SnapKVPress"`, `"StreamingLLMPress"`, `"TOVAPress"`, etc.). |
| `compression_ratio` | float | Fraction of KV entries to **retain** (0–1). E.g. `0.5` keeps half. |
| `keep_filler_tokens` | bool | If `true`, the header and trailer positions from the PacketWrapper are excluded from compression and always kept. Requires a packet wrapper to be set. |
| `kwargs` | dict | Additional keyword arguments forwarded to the compressor constructor. |

### `quantization` block (not used in our paper)

Quantizes KV cache tensors after compression (if any) and before combination.

| Field | Type | Description |
|-------|------|-------------|
| `num_bits` | int | Quantization bit-width (e.g. `4` or `8`). |
| `axis` | int | Axis along which to compute quantization scales. `0` = per-head, `1` = per-token. Defaults to `0`. |
| `group_size` | int | Group size for grouped quantization. Defaults to `64`. |

### Example evaluation configs

**KV Packet (proposed method)**
```json
{
    "cache_comb": {
        "method": "kv_packet",
        "kwargs": {}
    },
    "packet_wrapper": "./packet_wrapper/llama_3_1_8b/biography/8_8.pt"
}
```

**CacheBlend at 50% recompute ratio**
```json
{
    "cache_comb": {
        "method": "cache_blend",
        "kwargs": { "recompute_ratio": 0.5 }
    }
}
```

**EPIC with 50 boundary tokens**
```json
{
    "cache_comb": {
        "method": "epic",
        "kwargs": { "recompute_tokens": 50 }
    }
}
```

**SAM-KV**
```json
{
    "cache_comb": {
        "method": "sam_kv",
        "kwargs": {
            "stable_layers": [28, 29, 30, 31],
            "num_initial_tokens": 64,
            "num_local_tokens": 128,
            "block_size": 64,
            "fuse_theta": 0.9
        }
    }
}
```

---

## Config Inheritance

Both training and evaluation configs support a `_default.json` file in the same directory. When a config is loaded, its values extend (but do not overwrite) the defaults. This means a per-method override file only needs to list the fields that differ.

Fields present only in the override are added. Fields present in both default and override are kept from the **default** unless `broadcast_dict` is called in overwrite mode. Cycle detection prevents circular inheritance.

---

## Evaluation Metrics

Each evaluation run reports the following per-config:

| Metric | Description |
|--------|-------------|
| `precision` | Token-level precision of the generated answer vs. the ground-truth answer. |
| `recall` | Token-level recall. |
| `f1` | Harmonic mean of precision and recall. Primary benchmark metric. |
| `ttft` | Average time-to-first-token in seconds across samples. |
| `flops` | Average FLOPs for KV re-rotation (relevant to KV Packet; zero for non-rotation methods). |
| `num_orig_tokens` | Total original document tokens processed (before wrapping). |
| `num_wrapped_tokens` | Total tokens after adding header and trailer padding. |

---

## Supported Models

| Model | Template key |
|-------|-------------|
| Llama-3.1-8B-Instruct | `"llama_chat"` |
| Qwen2.5 / Qwen3-4B | `"qwen_3_chat"` |

Adding a new model requires implementing a model-specific KV re-rotation in `kv_packet/cache_comb/recompute_kv/` and registering it in `kv_packet/model/`.


## Citation

To cite our work:
```
@misc{
    chen2026kvpacket,
    title={KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs}, 
    author={Chuangtao Chen and Grace Li Zhang and Xunzhao Yin and Cheng Zhuo and Bing Li and Ulf Schlichtmann},
    year={2026},
    eprint={2604.13226},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2604.13226},
}
```