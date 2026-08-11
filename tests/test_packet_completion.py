"""End-to-end structured completion over the reentrant packet runtime."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from kvpacket.runtime import PacketCacheState, PacketExecutor, SamplingConfig
from kvpacket.serve import (
    ChatMessage,
    InferenceRequest,
    ModelRequest,
    ModelResponse,
    PacketCompletionEngine,
    PacketStateManager,
    CompletionToken,
    ToolCall,
    ToolDefinition,
)
from tests.test_phase2 import _FakeBackend, _packet_store


class _Codec:
    """Deterministic codec exposing a packet-bearing rendered prompt."""

    def validate(self, request: ModelRequest) -> None:
        """Accept every normalized request used by the test fixture."""
        del request

    def render(self, request: ModelRequest) -> list[int]:
        content = request.messages[-1].content
        if content == "question":
            return [7, 10, 11, 8]
        if content == "follow-up":
            return [7, 10, 11, 8, 1, 9]
        raise AssertionError(f"unexpected test message: {content}")

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join("x" if token_id == 1 else "!" for token_id in token_ids)

    def stream_text(self, token_ids: Sequence[int]) -> str:
        return self.decode(token_ids)

    def parse(self, token_ids: Sequence[int], *, reached_limit: bool = False) -> ModelResponse:
        return ModelResponse(
            content=self.decode(token_ids),
            stop_reason="length" if reached_limit else "stop",
            token_ids=tuple(int(token_id) for token_id in token_ids),
        )


def _request(
    *,
    content: str = "question",
    max_output_tokens: int = 2,
    stop_sequences: tuple[str, ...] = (),
):
    return InferenceRequest(
        model="fake",
        input=ModelRequest((ChatMessage("user", content),)),
        max_output_tokens=max_output_tokens,
        sampling=SamplingConfig(),
        stop_sequences=stop_sequences,
    )


def _engine() -> PacketCompletionEngine:
    backend = _FakeBackend()
    store = _packet_store(backend, {"doc": [10, 11]})
    executor = PacketExecutor(backend=backend, store=store)
    return PacketCompletionEngine(executor, _Codec())


def test_completion_returns_replacement_state_and_packet_diagnostics() -> None:
    engine = _engine()

    result = engine.complete(_request(), PacketCacheState())

    assert result.response.content == "xx"
    assert result.response.stop_reason == "length"
    assert result.response.token_ids == (1, 1)
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 2
    assert result.prompt_len == 1
    assert [match.chunk_id for match in result.cover] == ["doc"]
    assert result.cache_state.next_position == 6
    assert result.ttft >= 0


def test_completion_hides_stop_suffix_but_commits_its_tokens() -> None:
    engine = _engine()

    result = engine.complete(_request(max_output_tokens=5, stop_sequences=("xx",)))

    assert result.response.content == ""
    assert result.response.stop_reason == "stop"
    assert result.response.token_ids == (1, 1)
    assert result.cache_state.next_position == 6


def test_completion_streams_text_unrelated_to_a_possible_stop_prefix() -> None:
    events = list(_engine().stream(_request(max_output_tokens=2, stop_sequences=("zz",))))

    deltas = [event.text_delta for event in events if isinstance(event, CompletionToken)]
    assert deltas == ["x", "x"]


def test_completion_restores_custom_tool_kind_after_model_parsing() -> None:
    class CustomCodec(_Codec):
        def stream_text(self, token_ids: Sequence[int]) -> str:
            del token_ids
            return ""

        def parse(
            self,
            token_ids: Sequence[int],
            *,
            reached_limit: bool = False,
        ) -> ModelResponse:
            del token_ids, reached_limit
            return ModelResponse(
                tool_calls=(ToolCall("call-1", "apply_patch", {"input": "patch"}),),
                stop_reason="tool_use",
            )

    backend = _FakeBackend()
    executor = PacketExecutor(backend=backend, store=_packet_store(backend, {}))
    engine = PacketCompletionEngine(executor, CustomCodec())
    request = InferenceRequest(
        model="fake",
        input=ModelRequest(
            (ChatMessage("user", "question"),),
            (
                ToolDefinition(
                    "apply_patch",
                    "Apply a patch.",
                    {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                    },
                    kind="custom",
                ),
            ),
        ),
        max_output_tokens=1,
        sampling=SamplingConfig(),
    )

    result = engine.complete(request)

    call = result.response.tool_calls[0]
    assert call.kind == "custom"
    assert call.raw_input == "patch"


def test_managed_completion_retains_state_atomically() -> None:
    engine = _engine()
    manager = PacketStateManager(engine.executor)

    first = engine.complete_managed(manager, "conversation", _request(max_output_tokens=1))
    second = engine.complete_managed(
        manager,
        "conversation",
        _request(content="follow-up", max_output_tokens=1),
    )

    assert first.cache_state.next_position == 5
    assert first.history_reused is False
    assert second.cache_state.next_position == 7
    assert second.history_reused is True
    snapshot = manager.snapshot("conversation")
    assert snapshot.cache_state.next_position == 7
    assert snapshot.token_ids == (7, 10, 11, 8, 1, 9, 1)


def test_managed_completion_recomputes_when_full_history_is_not_an_exact_extension() -> None:
    engine = _engine()
    manager = PacketStateManager(engine.executor)
    engine.complete_managed(manager, "conversation", _request(max_output_tokens=1))

    result = engine.complete_managed(manager, "conversation", _request(max_output_tokens=1))

    assert result.history_reused is False
    assert result.cache_state.next_position == 5


def test_managed_completion_requires_shared_executor() -> None:
    engine = _engine()
    other = _engine()
    manager = PacketStateManager(other.executor)

    try:
        engine.complete_managed(manager, "conversation", _request())
    except ValueError as error:
        assert "share one executor" in str(error)
    else:
        raise AssertionError("mismatched executor should be rejected")


def test_completion_runs_on_a_real_hugging_face_backend() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket.backend import HFBackend

    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    backend = HFBackend(LlamaForCausalLM(config).float().eval())
    executor = PacketExecutor(backend=backend, store=_packet_store(backend, {}))
    engine = PacketCompletionEngine(executor, _Codec())

    result = engine.complete(_request(max_output_tokens=2))

    assert len(result.response.token_ids) == 2
    assert result.cache_state.next_position == 6
    assert result.cache_state.cache is not None
    assert result.cache_state.cache.seq_len == 6
