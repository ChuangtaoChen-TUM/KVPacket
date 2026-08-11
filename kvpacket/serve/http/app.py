"""Expose packet-aware generation through an OpenAI Responses-compatible ASGI app."""

from __future__ import annotations

import json
import logging
from asyncio import Semaphore
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from secrets import compare_digest
from time import time
from typing import cast
from uuid import uuid4

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ...runtime import PacketCacheState
from ..adapters import (
    OpenAIResponsesAdapter,
    OpenAIResponsesStreamEncoder,
    WireFormatError,
)
from ..completion import (
    CompletionEvent,
    CompletionFinished,
    CompletionToken,
    PacketCompletionEngine,
)
from ..types import InferenceRequest

__all__ = ["ResponsesServerConfig", "create_responses_app"]

_LOGGER = logging.getLogger(__name__)


class _End:
    """Private sentinel returned after the blocking iterator is exhausted."""


_END = _End()


@dataclass(frozen=True)
class ResponsesServerConfig:
    """Safety and compatibility limits for one local Responses endpoint.

    Attributes:
        model_alias: Only externally accepted model name.
        max_concurrency: Maximum simultaneous model executions.
        max_request_bytes: Maximum JSON request body size.
        max_input_tokens: Maximum rendered prompt length before packet substitution.
        max_output_tokens: Server-side cap on requested generation length.
        bearer_token: Optional exact bearer token required on every API request.
    """

    model_alias: str
    max_concurrency: int = 1
    max_request_bytes: int = 4 * 1024 * 1024
    max_input_tokens: int = 131_072
    max_output_tokens: int = 16_384
    bearer_token: str | None = None

    def __post_init__(self) -> None:
        """Validate externally enforced limits before the server starts."""
        if not self.model_alias:
            raise ValueError("model_alias must be non-empty")
        for name in (
            "max_concurrency",
            "max_request_bytes",
            "max_input_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bearer_token == "":
            raise ValueError("bearer_token must be non-empty or None")


class _ResponsesService:
    """Own transport limits and bound access to one completion engine."""

    def __init__(
        self,
        engine: PacketCompletionEngine,
        config: ResponsesServerConfig,
    ) -> None:
        self.engine = engine
        self.config = config
        self.adapter = OpenAIResponsesAdapter()
        self._slots = Semaphore(config.max_concurrency)
        self._model_created_at = int(time())

    async def create_response(self, request: Request) -> Response:
        """Validate a Responses request and run it synchronously or over SSE."""
        auth_error = self._authentication_error(request)
        if auth_error is not None:
            return auth_error
        body_or_error = await self._read_body(request)
        if isinstance(body_or_error, Response):
            return body_or_error
        try:
            inference = self.adapter.parse_request(body_or_error)
            self._validate_request(inference)
            input_ids = self.engine.render_input(inference)
            if len(input_ids) > self.config.max_input_tokens:
                return self._error(
                    400,
                    "context_length_exceeded",
                    f"rendered input has {len(input_ids)} tokens; limit is "
                    f"{self.config.max_input_tokens}",
                    param="input",
                )
        except (TypeError, ValueError, WireFormatError) as error:
            return self._error(400, "invalid_request_error", str(error))

        response_id = f"resp_{uuid4().hex}"
        if inference.stream:
            return StreamingResponse(
                self._stream(request, inference, input_ids, response_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            async with self._slots:
                result = await run_in_threadpool(
                    self.engine.complete_rendered,
                    inference,
                    input_ids,
                    PacketCacheState(),
                )
        except Exception:
            _LOGGER.exception("Responses completion failed")
            return self._error(
                500,
                "server_error",
                "The model failed to generate a response.",
            )
        payload = self.adapter.format_response(
            result.response,
            response_id=response_id,
            model=inference.model,
            usage=result.usage,
        )
        return JSONResponse(payload)

    async def health(self, request: Request) -> Response:
        """Liveness route that performs no model work."""
        del request
        return JSONResponse({"status": "ok", "model": self.config.model_alias})

    async def models(self, request: Request) -> Response:
        """Expose the configured local model using the common list schema."""
        auth_error = self._authentication_error(request)
        if auth_error is not None:
            return auth_error
        return JSONResponse(
            {
                "object": "list",
                # Codex currently reads a provider-specific ``models`` wrapper,
                # while OpenAI SDKs read the standard ``data`` list. An empty Codex
                # catalog makes it use configured fallback metadata without changing
                # the externally configured model alias.
                "models": [],
                "data": [
                    {
                        "id": self.config.model_alias,
                        "object": "model",
                        "created": self._model_created_at,
                        "owned_by": "kvpacket",
                    }
                ],
            }
        )

    async def _stream(
        self,
        request: Request,
        inference: InferenceRequest,
        input_ids: list[int],
        response_id: str,
    ) -> AsyncIterator[bytes]:
        """Bridge the blocking token iterator to an asynchronous SSE body."""
        encoder = OpenAIResponsesStreamEncoder(
            self.adapter,
            response_id=response_id,
            model=inference.model,
        )
        yield self._sse(encoder.created_event())
        events: Iterator[CompletionEvent] | None = None
        try:
            async with self._slots:
                events = self.engine.stream_rendered(
                    inference,
                    input_ids,
                    PacketCacheState(),
                )
                while not await request.is_disconnected():
                    event = await run_in_threadpool(self._next_event, events)
                    if isinstance(event, _End):
                        break
                    if isinstance(event, CompletionToken):
                        if event.text_delta:
                            for payload in encoder.text_delta_events(event.text_delta):
                                yield self._sse(payload)
                    elif isinstance(event, CompletionFinished):
                        for payload in encoder.finished_events(
                            event.result.response,
                            event.result.usage,
                        ):
                            yield self._sse(payload)
        except GeneratorExit:
            raise
        except Exception:
            _LOGGER.exception("streaming Responses completion failed")
            yield self._sse(
                encoder.failed_event(
                    code="server_error",
                    message="The model failed to generate a response.",
                )
            )
        finally:
            if events is not None:
                await run_in_threadpool(self._close_iterator, events)

    async def _read_body(self, request: Request) -> Mapping[str, object] | Response:
        """Read bounded JSON without retaining or logging request content."""
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                return self._error(400, "invalid_request_error", "invalid Content-Length")
            if declared > self.config.max_request_bytes:
                return self._error(413, "request_too_large", "request body exceeds server limit")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > self.config.max_request_bytes:
                return self._error(413, "request_too_large", "request body exceeds server limit")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error(400, "invalid_request_error", "request body must be valid JSON")
        if not isinstance(payload, Mapping):
            return self._error(400, "invalid_request_error", "request body must be a JSON object")
        return cast(Mapping[str, object], payload)

    def _validate_request(self, request: InferenceRequest) -> None:
        """Apply model identity and generation limits not owned by the adapter."""
        if request.model != self.config.model_alias:
            raise WireFormatError(f"unknown model {request.model!r}")
        if request.max_output_tokens > self.config.max_output_tokens:
            raise WireFormatError(
                f"max_output_tokens exceeds server limit {self.config.max_output_tokens}"
            )

    def _authentication_error(self, request: Request) -> Response | None:
        """Require an exact bearer token when configured."""
        expected = self.config.bearer_token
        if expected is None:
            return None
        prefix = "Bearer "
        authorization = request.headers.get("authorization", "")
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        if not compare_digest(supplied, expected):
            return self._error(401, "invalid_api_key", "Invalid bearer token.")
        return None

    @staticmethod
    def _next_event(events: Iterator[CompletionEvent]) -> CompletionEvent | _End:
        """Advance a blocking model iterator without leaking ``StopIteration`` to async code."""
        return next(events, _END)

    @staticmethod
    def _close_iterator(events: Iterator[CompletionEvent]) -> None:
        """Release the model slot after completion, failure, or client disconnect."""
        close = getattr(events, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _sse(payload: Mapping[str, object]) -> bytes:
        """Serialize one Responses event as an SSE data record."""
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()

    @staticmethod
    def _error(
        status_code: int,
        code: str,
        message: str,
        *,
        param: str | None = None,
    ) -> JSONResponse:
        """Format a pre-stream OpenAI-style error response."""
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if status_code < 500 else "server_error",
                    "param": param,
                    "code": code,
                }
            },
            status_code=status_code,
        )


def create_responses_app(
    engine: PacketCompletionEngine,
    *,
    config: ResponsesServerConfig,
) -> Starlette:
    """Build a local Responses-compatible ASGI application.

    The app is stateless at the request boundary: every request receives a new
    ``PacketCacheState`` while the shared executor still substitutes registered packet
    content. Codex remains responsible for conversation history and tool execution.
    """
    service = _ResponsesService(engine, config)
    return Starlette(
        routes=[
            Route("/healthz", service.health, methods=["GET"]),
            Route("/v1/models", service.models, methods=["GET"]),
            Route("/v1/responses", service.create_response, methods=["POST"]),
        ]
    )
