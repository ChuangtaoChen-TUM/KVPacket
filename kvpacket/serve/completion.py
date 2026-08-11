"""Compose a model codec with reentrant packet prefill and token generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from time import perf_counter

import torch

from ..runtime import PacketCacheState, PacketExecutor, select_next_token
from ..runtime.matching import Match
from .codecs import ModelCodec
from .manager import ManagedPacketState, PacketStateManager
from .types import InferenceRequest, ModelResponse, TokenUsage

__all__ = [
    "CompletionEvent",
    "CompletionFinished",
    "CompletionPrefilled",
    "CompletionResult",
    "CompletionToken",
    "PacketCompletionEngine",
]


@dataclass(frozen=True)
class CompletionResult:
    """Completed structured response, caller-owned cache, and packet diagnostics.

    Attributes:
        response: Model-family-decoded assistant output.
        cache_state: Cache after committing the prompt and generated continuation.
        usage: Logical prompt and generated token counts.
        prompt_len: Unmatched trailing prompt length in the packet-aware prefill.
        cover: Reusable source matches in rendered-input coordinates.
        rerotation_flops: Work spent moving cached packet positions.
        ttft: Seconds through selection of the first generated token.
        history_reused: Whether a managed completion extended an exact cached prefix.
    """

    response: ModelResponse
    cache_state: PacketCacheState
    usage: TokenUsage
    prompt_len: int
    cover: tuple[Match, ...]
    rerotation_flops: int
    ttft: float
    history_reused: bool = False


@dataclass(frozen=True)
class CompletionPrefilled:
    """Prompt-prefill diagnostics emitted before continuation decoding."""

    input_tokens: int
    prompt_len: int
    cover: tuple[Match, ...]
    rerotation_flops: int


@dataclass(frozen=True)
class CompletionToken:
    """One selected token and any newly safe decoded text."""

    token_id: int | None
    token_index: int
    text_delta: str
    ttft: float | None = None


@dataclass(frozen=True)
class CompletionFinished:
    """Final structured output and replacement cache state."""

    result: CompletionResult


CompletionEvent = CompletionPrefilled | CompletionToken | CompletionFinished


class PacketCompletionEngine:
    """Reentrant structured completion over a shared packet executor.

    This class owns model-independent orchestration but no conversation state. Existing
    serving systems pass their current ``PacketCacheState`` to ``complete`` and retain
    the returned state. Small deployments may instead use ``complete_managed`` with
    the optional ``PacketStateManager``.

    Args:
        executor: Shared packet-aware token executor.
        codec: Model-family prompt renderer and output parser.
        eos_token_ids: Tokens that end generation after being committed.
    """

    def __init__(
        self,
        executor: PacketExecutor,
        codec: ModelCodec,
        *,
        eos_token_ids: int | Sequence[int] | None = None,
    ) -> None:
        self.executor = executor
        self.codec = codec
        if eos_token_ids is None:
            eos = ()
        elif isinstance(eos_token_ids, bool):
            raise TypeError("eos_token_ids must contain integers, not booleans")
        elif isinstance(eos_token_ids, int):
            eos = (eos_token_ids,)
        else:
            eos = eos_token_ids
        self.eos_token_ids = frozenset(int(token_id) for token_id in eos)

    def complete(
        self,
        request: InferenceRequest,
        cache_state: PacketCacheState = PacketCacheState(),
    ) -> CompletionResult:
        """Render, packet-prefill, decode, and return replacement caller state."""
        return self.complete_rendered(
            request,
            self.render_input(request),
            cache_state,
        )

    def render_input(self, request: InferenceRequest) -> list[int]:
        """Render a structured request for validation or direct token-level reuse."""
        return self.codec.render(request.input)

    def complete_rendered(
        self,
        request: InferenceRequest,
        input_ids: Sequence[int],
        cache_state: PacketCacheState = PacketCacheState(),
    ) -> CompletionResult:
        """Complete a request whose codec output was already validated by the caller."""
        result: CompletionResult | None = None
        for event in self.stream_rendered(request, input_ids, cache_state):
            if isinstance(event, CompletionFinished):
                result = event.result
        if result is None:
            raise RuntimeError("completion stream ended without a final result")
        return result

    def stream(
        self,
        request: InferenceRequest,
        cache_state: PacketCacheState = PacketCacheState(),
    ) -> Iterator[CompletionEvent]:
        """Generate incrementally without retaining the caller's conversation state."""
        yield from self.stream_rendered(request, self.render_input(request), cache_state)

    def stream_rendered(
        self,
        request: InferenceRequest,
        input_ids: Sequence[int],
        cache_state: PacketCacheState = PacketCacheState(),
    ) -> Iterator[CompletionEvent]:
        """Stream a request whose codec output was already validated by the caller."""
        tokens = [int(token_id) for token_id in input_ids]
        yield from self._stream_tokens(
            request,
            tokens,
            cache_state,
            input_token_count=len(tokens),
        )

    def _complete_tokens(
        self,
        request: InferenceRequest,
        input_ids: list[int],
        cache_state: PacketCacheState,
        *,
        input_token_count: int,
        history_reused: bool = False,
    ) -> CompletionResult:
        """Collect the canonical event iterator for an already rendered token suffix."""
        result: CompletionResult | None = None
        for event in self._stream_tokens(
            request,
            input_ids,
            cache_state,
            input_token_count=input_token_count,
            history_reused=history_reused,
        ):
            if isinstance(event, CompletionFinished):
                result = event.result
        if result is None:
            raise RuntimeError("completion stream ended without a final result")
        return result

    def _stream_tokens(
        self,
        request: InferenceRequest,
        input_ids: list[int],
        cache_state: PacketCacheState,
        *,
        input_token_count: int,
        history_reused: bool = False,
    ) -> Iterator[CompletionEvent]:
        """Generate canonical prefill, token, and completion events."""
        with torch.no_grad():
            self._sync()
            started_at = perf_counter()
            prefill = self.executor.prefill(cache_state, input_ids)
            yield CompletionPrefilled(
                input_tokens=input_token_count,
                prompt_len=prefill.prompt_len,
                cover=tuple(prefill.cover),
                rerotation_flops=prefill.rerotation_flops,
            )
            state = prefill.cache_state
            next_token_logits = prefill.first_token_logits
            generated: list[int] = []
            emitted_text = ""
            ttft: float | None = None
            stopped = False

            for token_index in range(request.max_output_tokens):
                token_id = select_next_token(next_token_logits, request.sampling)
                generated.append(token_id)
                token_ttft: float | None = None
                if ttft is None:
                    self._sync()
                    ttft = perf_counter() - started_at
                    token_ttft = ttft

                stopped = token_id in self.eos_token_ids or self._has_stop_sequence(
                    generated, request.stop_sequences
                )
                visible_text = self.codec.stream_text(generated)
                streamable_text = self._streamable_text(
                    visible_text,
                    request.stop_sequences,
                    stopped=stopped,
                )
                text_delta = self._text_delta(emitted_text, streamable_text)
                emitted_text += text_delta
                # Yield before forwarding this token: the caller can observe the first
                # token without paying for the forward that prepares token two.
                yield CompletionToken(token_id, token_index, text_delta, token_ttft)

                forwarded = self.executor.forward_tokens(state, [token_id])
                state = forwarded.cache_state
                next_token_logits = forwarded.logits[:, -1, :]
                if stopped:
                    break

            assert ttft is not None
            response = self.codec.parse(
                generated,
                reached_limit=len(generated) == request.max_output_tokens and not stopped,
            )
            response = self._restore_tool_kinds(response, request)
            response = self._trim_stop_sequence(response, request.stop_sequences)
            final_delta = self._text_delta(emitted_text, response.content)
            if final_delta:
                emitted_text += final_delta
                yield CompletionToken(None, len(generated), final_delta)
            result = CompletionResult(
                response=response,
                cache_state=state,
                usage=TokenUsage(input_token_count, len(generated)),
                prompt_len=prefill.prompt_len,
                cover=tuple(prefill.cover),
                rerotation_flops=prefill.rerotation_flops,
                ttft=ttft,
                history_reused=history_reused,
            )
            yield CompletionFinished(result)

    def complete_managed(
        self,
        manager: PacketStateManager,
        session_id: str,
        request: InferenceRequest,
    ) -> CompletionResult:
        """Complete one request atomically against manager-owned conversation state."""
        if manager.executor is not self.executor:
            raise ValueError("completion engine and state manager must share one executor")

        full_input_ids = self.render_input(request)

        def operation(
            state: ManagedPacketState,
        ) -> tuple[CompletionResult, ManagedPacketState]:
            history = state.token_ids
            can_reuse = (
                bool(history)
                and len(history) < len(full_input_ids)
                and tuple(full_input_ids[: len(history)]) == history
            )
            if can_reuse:
                cache_state = state.cache_state
                new_input_ids = full_input_ids[len(history) :]
            else:
                cache_state = PacketCacheState()
                new_input_ids = full_input_ids
            result = self._complete_tokens(
                request,
                new_input_ids,
                cache_state,
                input_token_count=len(full_input_ids),
                history_reused=can_reuse,
            )
            next_state = ManagedPacketState(
                result.cache_state,
                tuple(full_input_ids) + tuple(result.response.token_ids),
            )
            return result, next_state

        return manager._transact(session_id, operation)

    def _has_stop_sequence(self, token_ids: list[int], stop_sequences: tuple[str, ...]) -> bool:
        """Detect decoded stop strings without coupling the executor to a tokenizer."""
        if not stop_sequences:
            return False
        content = self.codec.decode(token_ids)
        return any(content.endswith(stop) for stop in stop_sequences)

    @staticmethod
    def _streamable_text(
        text: str,
        stop_sequences: tuple[str, ...],
        *,
        stopped: bool,
    ) -> str:
        """Withhold possible stop suffixes so streamed text matches the final response."""
        if not stop_sequences:
            return text
        if stopped:
            matches = [stop for stop in stop_sequences if text.endswith(stop)]
            if matches:
                stop = max(matches, key=len)
                return text[: -len(stop)]
            return text
        reserve = 0
        for stop in stop_sequences:
            max_partial = min(len(text), len(stop) - 1)
            for length in range(max_partial, reserve, -1):
                if stop.startswith(text[-length:]):
                    reserve = length
                    break
        return text if reserve == 0 else text[:-reserve]

    @staticmethod
    def _text_delta(previous: str, current: str) -> str:
        """Extract an append-only text delta or reject an unstable codec prefix."""
        if not current.startswith(previous):
            raise RuntimeError("model codec exposed a non-monotonic streaming text prefix")
        return current[len(previous) :]

    @staticmethod
    def _trim_stop_sequence(
        response: ModelResponse,
        stop_sequences: tuple[str, ...],
    ) -> ModelResponse:
        """Hide a matched stop suffix while retaining committed token provenance."""
        matches = [stop for stop in stop_sequences if response.content.endswith(stop)]
        if not matches:
            return response
        stop = max(matches, key=len)
        return replace(response, content=response.content[: -len(stop)], stop_reason="stop")

    @staticmethod
    def _restore_tool_kinds(
        response: ModelResponse,
        request: InferenceRequest,
    ) -> ModelResponse:
        """Restore free-form tool identity after function-shaped model rendering."""
        definitions = {tool.model_name: tool for tool in request.input.tools}
        calls = []
        for call in response.tool_calls:
            definition = definitions.get(call.model_name)
            if definition is None:
                calls.append(call)
                continue
            call = replace(
                call,
                name=definition.name,
                namespace=definition.namespace,
            )
            if definition.kind == "function":
                calls.append(call)
                continue
            raw_input = call.raw_input
            if raw_input is None:
                candidate = call.arguments.get("input")
                if not isinstance(candidate, str):
                    raise ValueError(
                        f"custom tool {call.name!r} must return a string `input` argument"
                    )
                raw_input = candidate
            calls.append(replace(call, kind="custom", raw_input=raw_input))
        if tuple(calls) == response.tool_calls:
            return response
        return replace(response, tool_calls=tuple(calls))

    def _sync(self) -> None:
        """Resolve asynchronous GPU work at TTFT boundaries."""
        if self.executor.backend.device.type == "cuda":
            torch.cuda.synchronize(self.executor.backend.device)
