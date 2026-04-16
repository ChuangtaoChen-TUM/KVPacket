# Evaluation Configuration Reference

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
