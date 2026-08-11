"""Run a local OpenAI Responses-compatible KV Packet endpoint.

Install the optional transport with ``uv sync --extra serve``. The process loads one
Hugging Face model and an optional packet-store checkpoint, then serves stateless
Responses requests at ``/v1/responses``. Agent frameworks remain responsible for
sending their self-contained conversation transcript on every turn.
"""

from __future__ import annotations

import argparse
import os
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpacket import PacketStore, HFBackend, PacketExecutor
from kvpacket.serve import (
    ChatTokenizer,
    HFChatCodec,
    PacketCompletionEngine,
)
from kvpacket.serve.codecs import QwenToolCallParser
from kvpacket.serve.http import ResponsesServerConfig, create_responses_app

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _arguments() -> argparse.Namespace:
    """Parse model, packet-store, transport, and safety options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument(
        "--model-alias",
        help="External Responses model name; defaults to --model",
    )
    parser.add_argument("--packet-store", help="Optional materialized packet-store checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Single model device")
    parser.add_argument(
        "--dtype",
        choices=["auto", *_DTYPES],
        default="auto",
        help="Model weight dtype",
    )
    parser.add_argument(
        "--attention-implementation",
        help="Optional Transformers attention implementation, such as eager or sdpa",
    )
    parser.add_argument(
        "--mixed-prefill-policy",
        choices=["sequential", "fused"],
        default="fused",
        help="How to process online spans interleaved with packet matches",
    )
    parser.add_argument(
        "--tool-parser",
        choices=["auto", "qwen", "none"],
        default="auto",
        help="Model-family generated tool-call parser",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-request-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-input-tokens", type=int, default=131_072)
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument(
        "--bearer-token-env",
        default="KVP_API_KEY",
        help="Environment variable containing the optional bearer token; empty disables auth",
    )
    return parser.parse_args()


def _tool_parser(name: str, model: str):
    """Select the explicit or safely inferred model-family tool parser."""
    selected = "qwen" if name == "auto" and "qwen" in model.lower() else name
    return QwenToolCallParser() if selected == "qwen" else None


def main() -> None:
    """Construct the packet stack once and run the local ASGI server."""
    args = _arguments()
    load_options: dict[str, object] = {
        "dtype": "auto" if args.dtype == "auto" else _DTYPES[args.dtype],
        "device_map": args.device,
    }
    if args.attention_implementation:
        load_options["attn_implementation"] = args.attention_implementation
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if loaded_tokenizer is None:
        raise RuntimeError("AutoTokenizer returned no tokenizer")
    tokenizer = cast(ChatTokenizer, loaded_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_options).eval()
    backend = HFBackend(model)
    store = (
        PacketStore.load(args.packet_store, backend=backend)
        if args.packet_store
        else PacketStore(backend=backend)
    )
    executor = PacketExecutor(
        backend=backend,
        store=store,
        mixed_prefill_policy=args.mixed_prefill_policy,
    )
    codec = HFChatCodec(
        tokenizer,
        tool_call_parser=_tool_parser(args.tool_parser, args.model),
    )
    eos_token_id = getattr(loaded_tokenizer, "eos_token_id", None)
    engine = PacketCompletionEngine(executor, codec, eos_token_ids=eos_token_id)
    bearer_token = os.environ.get(args.bearer_token_env) or None
    app = create_responses_app(
        engine,
        config=ResponsesServerConfig(
            model_alias=args.model_alias or args.model,
            max_concurrency=args.max_concurrency,
            max_request_bytes=args.max_request_bytes,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            bearer_token=bearer_token,
        ),
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
