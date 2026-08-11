# Environment setup

## Standard installation

KV Packet requires Python 3.12 or newer. Runtime dependencies are declared in
`pyproject.toml` and locked in `uv.lock`. Install the PyTorch build appropriate for
your compute platform; the published package does not force a CUDA-specific wheel.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install repository task, development, analysis, or public feature dependencies as needed:

```bash
uv sync --group tasks
uv sync --group dev
uv sync --group analysis
uv sync --extra compress
```

No repository-specific environment script is required.

## Optional storage and cache redirection

Model weights, datasets, recorded teacher samples, and training checkpoints can be
large. On quota-limited systems, choose a storage directory and export cache locations
before running `uv sync` or downloading models:

```bash
export KVP_STORAGE="/path/to/large/storage"
export UV_PROJECT_ENVIRONMENT="$KVP_STORAGE/venvs/kvpacket"
export UV_CACHE_DIR="$KVP_STORAGE/cache/uv"
export PIP_CACHE_DIR="$KVP_STORAGE/cache/pip"
export HF_HOME="$KVP_STORAGE/cache/huggingface"
export TORCH_HOME="$KVP_STORAGE/cache/torch"
export XDG_CACHE_HOME="$KVP_STORAGE/cache/xdg"
export NLTK_DATA="$KVP_STORAGE/cache/nltk_data"

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$NLTK_DATA" \
  "$(dirname "$UV_PROJECT_ENVIRONMENT")"
```

These values are intentionally not committed because suitable paths differ by
machine. Put them in your shell profile or another untracked local setup file if you
want a reusable setup. Never store access tokens in a tracked file.

`KVP_STORAGE` is also used by the supplied training and reference-parity configs to
keep large run artifacts outside the repository. Set it to any writable directory
before running those experiments. A config that references `${KVP_STORAGE}` fails
early with a configuration error when the variable is unset; it will never create a
literal `${KVP_STORAGE}` directory. Normal library use, unit tests, and configs with
ordinary local paths (including the synthetic training example) do not require it.

## Common commands

```bash
# Core test suite
uv run pytest -q

# Static checks
uv run ty check kvpacket tasks tests
uv run ruff check kvpacket tasks tests

# Select a GPU for a task
CUDA_VISIBLE_DEVICES=0 uv run python path/to/task.py
```

Hugging Face gated models require the usual `HF_TOKEN` authentication. Supply tokens
through your shell or a secret manager rather than a repository file.
