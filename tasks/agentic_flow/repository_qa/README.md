# Minimal Codex agent example

This example keeps one complete, inspectable path from Codex-style skill use to KV
Packet serving:

1. record compact Full Recompute teacher samples;
2. train one repository-navigation wrapper;
3. materialize the skill as a reusable packet;
4. serve it through the Responses-compatible endpoint used by Codex.

The initial model context contains only the skill name, description, and path. The full
`SKILL.md` enters the transcript after conditional activation, following Codex
progressive disclosure. Repository files remain dynamic tool results, and MCP tool
schemas remain ordinary structured request metadata; neither becomes a packet.

## Files

| Path | Purpose |
|---|---|
| `.agents/skills/repository-navigation/SKILL.md` | reusable packet source and Codex skill |
| `agent.py` | small programmatic record/serve loop |
| `tools.py` | bounded read-only repository tools |
| `mcp_server.py` | the same tools exposed to Codex over stdio MCP |
| `pipeline.py` | record, train, and materialize stages |
| `training_configs/qwen3_4b.json` | eight local prompts and Qwen3-4B training settings |
| `checkpoints/qwen3_4b_repository_navigation.pt` | packaged trained wrapper |
| `codex_config.toml.example` | isolated Codex provider and MCP configuration |

The packaged wrapper was trained with token-ID cross entropy on 110 post-activation
samples. It is a `PacketTrainer` checkpoint rather than a materialized KV store, so the
45 MB model cache does not need to be checked into the repository.

## Environment

Run commands from the project root. Keep generated samples and stores outside the Git
checkout:

```bash
export KVP_STORAGE=/path/to/writable/storage
export HF_HOME=/path/to/huggingface/cache  # optional
```

The default prompts inspect this KV Packet checkout (`repository_root` is `.`). Change
that field and the prompt list to record samples for another repository. The eight
checked-in prompts are a mechanics smoke test, not enough data for a production wrapper;
implicit skill activation is model-dependent. The packaged checkpoint retains the
validated 110-sample training run.

## 1. Record samples

```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.agentic_flow.repository_qa.pipeline \
  --config tasks/agentic_flow/repository_qa/training_configs/qwen3_4b.json \
  --phase record
```

Only model turns containing an activated skill are saved under
`$KVP_STORAGE/examples/codex_agent/samples`. Pre-activation routing turns are retained
in `recorded_runs.jsonl` but are not wrapper-training samples.

## 2. Train a replacement wrapper

```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.agentic_flow.repository_qa.pipeline \
  --config tasks/agentic_flow/repository_qa/training_configs/qwen3_4b.json \
  --phase train
```

This writes `$KVP_STORAGE/examples/codex_agent/trainer.pt`. The example deliberately
keeps only the final checkpoint, not an epoch-by-epoch sweep.

## 3. Materialize a serving store

```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.agentic_flow.repository_qa.pipeline \
  --config tasks/agentic_flow/repository_qa/training_configs/qwen3_4b.json \
  --phase materialize
```

Materialization uses the newly trained checkpoint when present; otherwise it uses the
packaged wrapper. It writes
`$KVP_STORAGE/examples/codex_agent/kv_packet_store.pt`.

## 4. Serve and connect Codex

Install the optional serving and MCP dependencies, then start the endpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.run_responses_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --packet-store "$KVP_STORAGE/examples/codex_agent/kv_packet_store.pt" \
  --model-alias qwen3-4b-kvpacket \
  --tool-parser qwen
```

Use a separate experiment home so this local provider cannot modify a normal Codex
profile:

```bash
project_root="$PWD"
experiment_root="$KVP_STORAGE/examples/codex_agent/codex"
codex_home="$experiment_root/codex_home"
workspace="$experiment_root/workspace"

mkdir -p "$codex_home" "$workspace/.agents"
cp tasks/agentic_flow/repository_qa/codex_config.toml.example "$codex_home/config.toml"
cp -R tasks/agentic_flow/repository_qa/.agents/. "$workspace/.agents/"

export KVP_AGENT_REPOSITORY="$project_root"
export PYTHONPATH="$project_root"
CODEX_HOME="$codex_home" codex --cd "$workspace"
```

Codex discovers the skill from the isolated workspace and executes `list_files`,
`search_text`, and `read_file` through the path-confined MCP server. The Responses
server remains stateless with respect to the agent transcript while keeping the packet
store resident across requests.

## Safety and scope

The example never executes repository code. Tools are read-only, path-confined,
output-bounded, and reject path traversal and symlink escapes. This is a minimal
integration example, not an agent benchmark or an evaluation suite.
