# Serving Architecture

KV Packet's serving stack is layered so an existing inference system can adopt only
packet matching and cache substitution, while a smaller application can use the
included codec, state manager, and provider adapters.

```text
agent framework
    │ provider request
    ▼
wire adapter (optional: Responses / Anthropic Messages)
    │ InferenceRequest
    ▼
ModelCodec (optional: structured messages ↔ model tokens/tool calls)
    │ token IDs
    ▼
session manager (optional: cache lifetime, locking, LRU/TTL)
    │ PacketCacheState
    ▼
PacketExecutor (core: match, substitute, rerotate, prefill, decode step)
    │
    ▼
ComputeBackend
```

## Core integration

`PacketExecutor` is reentrant. It retains the backend, packet store, and matching
catalog, but never a conversation cache. Existing serving systems can therefore keep
their current session manager:

```python
from kvpacket import PacketCacheState, PacketExecutor

executor = PacketExecutor(backend=backend, store=store)

state = existing_session.cache_state or PacketCacheState()
prefill = executor.prefill(state, rendered_input_ids)
existing_session.cache_state = prefill.cache_state
```

The state records both the cache and its next **logical** model position. They must be
kept together: a compressed packet may occupy fewer physical KV slots than its
original logical span.

## Structured completion

`PacketCompletionEngine` composes an executor with a model-specific `ModelCodec`. It
renders a structured request, performs packet-aware prefill, selects all generated
tokens with the framework sampling implementation, and returns the replacement cache
state. TTFT ends when the first token is selected; committing that token for a later
turn is not included.

```python
from kvpacket.serve import HFChatCodec, PacketCompletionEngine

codec = HFChatCodec(tokenizer, tool_call_parser=model_family_parser)
engine = PacketCompletionEngine(
    executor,
    codec,
    eos_token_ids=[tokenizer.eos_token_id],
)
result = engine.complete(request, state)
state = result.cache_state
```

Tool syntax belongs in the codec because it varies by model family. Before rendering,
the codec validates whether the request's features can be represented by the selected
model family. For example, `HFChatCodec` rejects tool definitions and tool transcripts
unless a model-family `ToolCallParser` is configured. The Responses endpoint reports
that mismatch as a client error before packet matching or model prefill begins.

This boundary separates three different kinds of compatibility:

- the provider adapter checks the agent framework's wire format;
- the model codec checks chat-template and tool-call syntax;
- the compute backend checks token, attention-mask, position, and KV-cache mechanics.

Supplying a parser declares that the selected weights and tokenizer implement its
syntax. Static validation cannot prove that arbitrary weights learned reliable tool
use, so the server configuration must pair the model with the correct family codec.
Packet matching, cache assembly, and the compute backend remain structured-message
agnostic.

## Optional state manager

`PacketStateManager` is a convenience, not a required owner above the executor. It
provides bounded per-session state, per-session serialization, optional backend-wide
serialization, idle expiry, and least-recently-used eviction. `complete_managed()`
holds one session lock across the entire prefill and decode transaction. The manager
also retains the external token trajectory. A self-contained later request reuses the
cache only when its rendered tokens exactly extend that trajectory; otherwise it
safely starts from an empty cache and rematches the full request.

```python
from kvpacket.serve import PacketStateManager

manager = PacketStateManager(executor, max_sessions=128, idle_timeout_s=1800)
result = engine.complete_managed(manager, conversation_id, request)
```

## Provider adapters

`OpenAIResponsesAdapter` and `AnthropicMessagesAdapter` are pure dictionary
translators. They do not perform HTTP, manage conversation history, render prompts, or
run a model. A server may parse a provider request, call the completion engine, and
format either a completed response or provider-native streaming events.

```python
from kvpacket.serve import OpenAIResponsesAdapter

adapter = OpenAIResponsesAdapter()
request = adapter.parse_request(json_payload)
result = engine.complete(request, state)
response_body = adapter.format_response(
    result.response,
    response_id=response_id,
    model=request.model,
    usage=result.usage,
)
```

The adapters support text messages, function tools, tool calls/results, standard
sampling fields, and stop sequences. The Responses adapter additionally preserves
free-form custom tools and namespaced tools. Opaque hosted-model reasoning items are
omitted when a self-contained Codex transcript is rendered for the local model.
Unsupported multimodal input and `previous_response_id` fail explicitly. See
[`examples/serve_requests.py`](../../examples/serve_requests.py) for both state-ownership
styles.

## Responses HTTP/SSE endpoint

The optional `kvpacket.serve.http` module composes the Responses adapter and
`PacketCompletionEngine` into a small Starlette application. It is an application
layer, not a dependency of the packet runtime or direct serving APIs.

Install and launch it with:

```bash
uv sync --extra serve
export KVP_API_KEY="replace-with-a-local-secret"  # optional
uv run python -m examples.run_responses_server \
  --model Qwen/Qwen3-8B \
  --packet-store checkpoints/agent-packets.pt \
  --model-alias qwen3-8b-kvpacket \
  --device cuda:0 \
  --bearer-token-env KVP_API_KEY
```

Omit `--packet-store` to exercise the same endpoint as ordinary full-attention local
generation. The executable exposes:

- `POST /v1/responses` for non-streaming JSON or live SSE responses;
- `GET /v1/models` for SDK discovery and Codex fallback discovery;
- `GET /healthz` for process liveness without model work.

The server emits `response.created` before prefill, then text deltas after each token
becomes safe to expose. It ends with output-item completion and
`response.completed`/`response.incomplete`. Function calls use the standard argument
events. Responses `custom` tools use custom-input events and a final
`custom_tool_call` item, which is required for clients such as Codex to dispatch
free-form `apply_patch`. The event shape follows the
[OpenAI Responses streaming reference](https://developers.openai.com/api/reference/resources/responses/streaming-events).

### Tool compatibility

Hugging Face chat templates normally accept JSON-schema function tools. The codec
therefore renders a free-form custom tool as a temporary function with one required
string field named `input`; after model parsing, the completion engine restores the
original custom-tool identity and raw input for the Responses wire format. Namespace
members are similarly rendered as `namespace__name` and restored to separate
`namespace` and `name` fields before emission. These conversions are local-model
syntax adapters and do not change packet matching.

Codecs are organized by responsibility under `kvpacket.serve.codecs`: `base` defines
the protocols, `hf` provides the default chat-template implementation, and family
modules such as `qwen` contain only family-specific output syntax.

Provider-hosted declarations (`web_search`, `web_search_preview`, and `tool_search`)
are accepted but omitted from the local model's tool catalog because this endpoint
does not implement provider-side tool execution. Ordinary client-executed function
and custom tools remain available. Images, audio, implicit response-history lookup,
and hosted reasoning-state replay are not implemented.

### Codex configuration

Point a custom Codex Responses provider at the endpoint. For ordinary interactive
use, place the provider in a Codex profile or trusted project configuration:

```toml
model = "qwen3-8b-kvpacket"
model_provider = "kvpacket"

[model_providers.kvpacket]
name = "Local KV Packet"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
env_key = "KVP_API_KEY"  # omit when the local server has auth disabled
```

This routing follows Codex's
[custom model-provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced#custom-model-providers).
The agent framework still owns its loop, tool execution, and transcript. KV Packet
owns only local inference and reusable packet substitution. Model capability matters:
wire compatibility does not make a base model proficient at coding or tool use.

Do not reuse a personal `~/.codex` state directory for local-model experiments. The
minimal example in `tasks/agentic_flow/repository_qa/` shows how to create a separate
`CODEX_HOME` and workspace, install only its repository-navigation skill, and connect
the confined repository MCP server. This isolates provider configuration, logs,
sessions, skills, and project files from the operator's normal Codex environment.

### State, TTFT, and operational limits

Each HTTP request begins with an empty caller-owned `PacketCacheState`. The packet
store and matching catalog stay resident, but response history does not. This fits
Responses clients such as Codex that resend a self-contained visible transcript and
avoids conflicting with their own session manager.

SSE is generated from the same canonical token iterator as non-streaming completion.
TTFT ends at first-token selection, and the first text delta is yielded before the
forward that prepares token two. Client cancellation closes the iterator after the
current backend operation and releases its concurrency slot.

The transport enforces bounded request bytes, rendered input tokens, output tokens,
model identity, and simultaneous model executions. It does not log request bodies or
exception details. Keep the default loopback binding for local use; for remote access,
set bearer auth and put TLS, rate limiting, and deployment-level timeouts in a trusted
reverse proxy.
