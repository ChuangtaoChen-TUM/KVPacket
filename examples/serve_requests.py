"""Minimal transport integration with caller-owned or library-managed cache state.

This example intentionally omits HTTP-framework code. A FastAPI, Starlette, Flask, or
existing inference server handler can call the same small functions after constructing
its model-specific ``PacketCompletionEngine`` once at process startup.
"""

from __future__ import annotations

from collections.abc import Mapping

from kvpacket.runtime import PacketCacheState
from kvpacket.serve import PacketCompletionEngine, PacketStateManager
from kvpacket.serve.adapters import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
)


def openai_response(
    payload: Mapping[str, object],
    *,
    engine: PacketCompletionEngine,
    cache_state: PacketCacheState,
    response_id: str,
) -> tuple[dict[str, object], PacketCacheState]:
    """Handle a self-contained Responses request with caller-owned cache lifetime."""
    adapter = OpenAIResponsesAdapter()
    request = adapter.parse_request(payload)
    result = engine.complete(request, cache_state)
    body = adapter.format_response(
        result.response,
        response_id=response_id,
        model=request.model,
        usage=result.usage,
    )
    return body, result.cache_state


def anthropic_message(
    payload: Mapping[str, object],
    *,
    engine: PacketCompletionEngine,
    manager: PacketStateManager,
    session_id: str,
    response_id: str,
) -> dict[str, object]:
    """Handle a Messages request with the optional bounded state manager."""
    adapter = AnthropicMessagesAdapter()
    request = adapter.parse_request(payload)
    result = engine.complete_managed(manager, session_id, request)
    return adapter.format_response(
        result.response,
        response_id=response_id,
        model=request.model,
        usage=result.usage,
    )
