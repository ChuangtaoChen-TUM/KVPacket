# Training Configuration Reference

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
