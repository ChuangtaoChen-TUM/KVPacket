"""Exercise the optional OpenAI Responses HTTP and SSE transport."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import cast

from starlette.testclient import TestClient

from kvpacket.runtime import PacketExecutor
from kvpacket.serve import ModelRequest, ModelResponse, PacketCompletionEngine, ToolCall
from kvpacket.serve.http import ResponsesServerConfig, create_responses_app
from tests.test_packet_completion import _Codec
from tests.test_phase2 import _FakeBackend, _packet_store


def _engine(codec: _Codec | None = None) -> PacketCompletionEngine:
    backend = _FakeBackend()
    store = _packet_store(backend, {"doc": [10, 11]})
    executor = PacketExecutor(backend=backend, store=store)
    return PacketCompletionEngine(executor, codec or _Codec())


def _client(
    *,
    config: ResponsesServerConfig | None = None,
    codec: _Codec | None = None,
) -> TestClient:
    app = create_responses_app(
        _engine(codec),
        config=config or ResponsesServerConfig(model_alias="fake"),
    )
    return TestClient(app)


def _events(response) -> list[dict[str, object]]:
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_non_streaming_response_and_model_routes() -> None:
    with _client() as client:
        assert client.get("/healthz").json() == {"status": "ok", "model": "fake"}
        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "fake"
        assert models["models"] == []

        response = client.post(
            "/v1/responses",
            json={
                "model": "fake",
                "input": "question",
                "max_output_tokens": 2,
                "temperature": 0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "incomplete"
    assert body["output"][0]["content"][0]["text"] == "xx"
    assert body["usage"]["input_tokens"] == 4
    assert body["usage"]["output_tokens"] == 2


def test_streaming_response_emits_live_ordered_text_events() -> None:
    with _client() as client:
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "fake",
                "input": "question",
                "max_output_tokens": 2,
                "temperature": 0,
                "stream": True,
            },
        ) as response:
            events = _events(response)

    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    assert types.count("response.output_text.delta") == 2
    assert types[-1] == "response.incomplete"
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert [
        event["delta"] for event in events if event["type"] == "response.output_text.delta"
    ] == ["x", "x"]


def test_streaming_response_emits_codex_custom_tool_item() -> None:
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
                tool_calls=(ToolCall("call_patch", "apply_patch", {"input": "patch"}),),
                stop_reason="tool_use",
            )

        def render(self, request: ModelRequest) -> list[int]:
            assert request.tools[0].kind == "custom"
            return super().render(request)

    with _client(codec=CustomCodec()) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "fake",
                "input": "question",
                "max_output_tokens": 1,
                "temperature": 0,
                "stream": True,
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply a patch.",
                        "format": {
                            "type": "grammar",
                            "syntax": "lark",
                            "definition": "start: /.+/",
                        },
                    }
                ],
            },
        ) as response:
            events = _events(response)

    done = [event for event in events if event["type"] == "response.output_item.done"]
    created_response = cast(dict[str, object], events[0]["response"])
    assert len(done) == 1
    assert done[0]["item"] == {
        "id": f"ctc_{created_response['id']}_0",
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": "call_patch",
        "name": "apply_patch",
        "input": "patch",
    }
    assert events[-1]["type"] == "response.completed"


def test_transport_enforces_auth_model_and_token_limits() -> None:
    config = ResponsesServerConfig(
        model_alias="fake",
        bearer_token="secret",
        max_input_tokens=3,
        max_output_tokens=2,
    )
    with _client(config=config) as client:
        unauthorized = client.post("/v1/responses", json={"model": "fake", "input": "question"})
        headers = {"Authorization": "Bearer secret"}
        wrong_model = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "other", "input": "question"},
        )
        long_input = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "fake", "input": "question", "max_output_tokens": 2},
        )
        long_output = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "fake", "input": "question", "max_output_tokens": 3},
        )

    assert unauthorized.status_code == 401
    assert wrong_model.status_code == 400
    assert long_input.status_code == 400
    assert long_input.json()["error"]["code"] == "context_length_exceeded"
    assert long_output.status_code == 400
