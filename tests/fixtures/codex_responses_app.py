"""Deterministic ASGI app used for a manual Codex CLI wire-compatibility smoke test."""

from __future__ import annotations

from collections.abc import Sequence

from kvpacket.runtime import PacketExecutor
from kvpacket.serve import ModelRequest, ModelResponse, PacketCompletionEngine
from kvpacket.serve.http import ResponsesServerConfig, create_responses_app
from tests.test_phase2 import _FakeBackend, _packet_store


class _AnyTextCodec:
    """Ignore prompt content and produce one deterministic assistant token."""

    def validate(self, request: ModelRequest) -> None:
        """Accept any normalized request used by the manual wire smoke test."""
        del request

    def render(self, request: ModelRequest) -> list[int]:
        assert request.messages
        return [7]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join("x" for _ in token_ids)

    def stream_text(self, token_ids: Sequence[int]) -> str:
        return self.decode(token_ids)

    def parse(self, token_ids: Sequence[int], *, reached_limit: bool = False) -> ModelResponse:
        del reached_limit
        ids = [int(token_id) for token_id in token_ids]
        return ModelResponse(content=self.decode(ids), token_ids=tuple(ids))


_backend = _FakeBackend()
_executor = PacketExecutor(backend=_backend, store=_packet_store(_backend, {}))
_engine = PacketCompletionEngine(_executor, _AnyTextCodec(), eos_token_ids=1)

app = create_responses_app(
    _engine,
    config=ResponsesServerConfig(model_alias="kvpacket-smoke", max_output_tokens=2_048),
)
