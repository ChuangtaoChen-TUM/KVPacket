"""Translate OpenAI Responses API payloads to the provider-neutral serving API."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from time import time
import json

from ...runtime import SamplingConfig
from ..types import (
    ChatMessage,
    InferenceRequest,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from ._parsing import (
    boolean,
    integer,
    json_arguments,
    mapping,
    number,
    sequence,
    string,
    tool_definition,
)
from .base import WireFormatError

__all__ = ["OpenAIResponsesAdapter", "OpenAIResponsesStreamEncoder"]


class OpenAIResponsesAdapter:
    """Pure dictionary adapter for the OpenAI Responses request and event schemas.

    The adapter intentionally does not perform HTTP, retain response history, render
    model prompts, or execute the model. A web server may therefore place it around a
    caller-owned ``PacketExecutor`` or the optional ``PacketStateManager``.
    """

    def parse_request(self, payload: Mapping[str, object]) -> InferenceRequest:
        """Normalize text messages, function calls, tools, and generation options."""
        if payload.get("previous_response_id") is not None:
            raise WireFormatError(
                "previous_response_id requires a response-history store; provide self-contained input"
            )
        model = string(payload.get("model"), where="model")
        messages: list[ChatMessage] = []
        instructions = payload.get("instructions")
        if instructions is not None:
            messages.append(ChatMessage("system", self._text_content(instructions, "instructions")))
        messages.extend(self._messages(payload.get("input")))

        tools_value = payload.get("tools", [])
        tools: list[ToolDefinition] = []
        for index, item in enumerate(sequence(tools_value, where="tools")):
            tools.extend(self._tool_definitions(item, where=f"tools[{index}]"))
        max_tokens = integer(payload.get("max_output_tokens", 1024), where="max_output_tokens")
        stream = boolean(payload.get("stream", False), where="stream")
        sampling = self._sampling(payload)
        return InferenceRequest(
            model=model,
            input=ModelRequest(tuple(messages), tuple(tools)),
            max_output_tokens=max_tokens,
            sampling=sampling,
            stream=stream,
        )

    def format_response(
        self,
        response: ModelResponse,
        *,
        response_id: str,
        model: str,
        usage: TokenUsage = TokenUsage(),
    ) -> dict[str, object]:
        """Build a completed Responses object from normalized assistant output."""
        return self._response_object(
            response_id=response_id,
            model=model,
            status="incomplete" if response.stop_reason == "length" else "completed",
            output=self._output_items(response, response_id),
            usage=usage,
            created_at=int(time()),
        )

    def _messages(self, value: object) -> list[ChatMessage]:
        """Parse Responses input strings and supported input-item objects."""
        if isinstance(value, str):
            return [ChatMessage("user", value)]
        items = sequence(value, where="input")
        messages: list[ChatMessage] = []
        for index, raw in enumerate(items):
            where = f"input[{index}]"
            item = mapping(raw, where=where)
            item_type = item.get("type", "message")
            if item_type == "message":
                role = string(item.get("role"), where=f"{where}.role")
                normalized_role = "system" if role == "developer" else role
                if normalized_role not in ("system", "user", "assistant"):
                    raise WireFormatError(f"{where}.role is unsupported: {role!r}")
                messages.append(
                    ChatMessage(
                        normalized_role, self._text_content(item.get("content"), f"{where}.content")
                    )  # type: ignore[arg-type]
                )
            elif item_type == "function_call":
                call_id = string(item.get("call_id", item.get("id")), where=f"{where}.call_id")
                call = ToolCall(
                    call_id,
                    string(item.get("name"), where=f"{where}.name"),
                    json_arguments(item.get("arguments", {}), where=f"{where}.arguments"),
                    namespace=self._optional_name(
                        item.get("namespace"), where=f"{where}.namespace"
                    ),
                )
                messages.append(ChatMessage("assistant", tool_calls=(call,)))
            elif item_type == "custom_tool_call":
                call_id = string(item.get("call_id", item.get("id")), where=f"{where}.call_id")
                raw_input = string(item.get("input", ""), where=f"{where}.input")
                call = ToolCall(
                    call_id,
                    string(item.get("name"), where=f"{where}.name"),
                    {"input": raw_input},
                    kind="custom",
                    raw_input=raw_input,
                    namespace=self._optional_name(
                        item.get("namespace"), where=f"{where}.namespace"
                    ),
                )
                messages.append(ChatMessage("assistant", tool_calls=(call,)))
            elif item_type == "function_call_output":
                messages.append(
                    ChatMessage(
                        "tool",
                        self._text_content(item.get("output", ""), f"{where}.output"),
                        tool_call_id=string(item.get("call_id"), where=f"{where}.call_id"),
                    )
                )
            elif item_type == "custom_tool_call_output":
                messages.append(
                    ChatMessage(
                        "tool",
                        self._text_content(item.get("output", ""), f"{where}.output"),
                        tool_call_id=string(item.get("call_id"), where=f"{where}.call_id"),
                        name=self._optional_name(item.get("name"), where=f"{where}.name"),
                    )
                )
            elif item_type == "reasoning":
                # Hosted-model reasoning state has no equivalent in an open model's
                # chat template. Preserve the surrounding visible messages and tool
                # transcript while intentionally omitting this opaque item.
                continue
            else:
                raise WireFormatError(f"{where}.type is unsupported: {item_type!r}")
        return messages

    @staticmethod
    def _tool_definitions(value: object, *, where: str) -> tuple[ToolDefinition, ...]:
        """Flatten Responses namespaces into uniquely rendered local tools."""
        item = mapping(value, where=where)
        item_type = item.get("type", "function")
        if item_type in {"web_search", "web_search_preview", "tool_search"}:
            # These tools require provider-side execution rather than a client tool
            # round trip. A local model server cannot execute them, so omit them from
            # the rendered tool catalog while retaining the rest of the request.
            return ()
        if item_type == "namespace":
            namespace = string(item.get("name"), where=f"{where}.name")
            nested = sequence(item.get("tools", []), where=f"{where}.tools")
            definitions: list[ToolDefinition] = []
            for index, raw_tool in enumerate(nested):
                nested_where = f"{where}.tools[{index}]"
                parsed = OpenAIResponsesAdapter._tool_definitions(
                    raw_tool,
                    where=nested_where,
                )
                if any(tool.namespace is not None for tool in parsed):
                    raise WireFormatError(f"{nested_where} cannot contain a nested namespace")
                definitions.extend(replace(tool, namespace=namespace) for tool in parsed)
            return tuple(definitions)
        if item_type == "function":
            return (tool_definition(item, where=where, nested_function=False),)
        if item_type != "custom":
            raise WireFormatError(f"{where}.type is unsupported: {item_type!r}")
        name = string(item.get("name"), where=f"{where}.name")
        description_value = item.get("description", "")
        description = string(description_value, where=f"{where}.description")
        raw_format = item.get("format")
        custom_format = (
            None
            if raw_format is None
            else {
                str(key): value
                for key, value in mapping(raw_format, where=f"{where}.format").items()
            }
        )
        return (
            ToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
                kind="custom",
                custom_format=custom_format,
            ),
        )

    @staticmethod
    def _optional_name(value: object, *, where: str) -> str | None:
        """Parse an optional tool name retained on custom output items."""
        if value is None:
            return None
        return string(value, where=where)

    @staticmethod
    def _text_content(value: object, where: str) -> str:
        """Join supported text blocks and reject multimodal input explicitly."""
        if isinstance(value, str):
            return value
        blocks = sequence(value, where=where)
        texts: list[str] = []
        for index, raw in enumerate(blocks):
            block = mapping(raw, where=f"{where}[{index}]")
            block_type = block.get("type")
            if block_type not in ("input_text", "output_text", "text"):
                raise WireFormatError(
                    f"{where}[{index}].type is unsupported: {block_type!r}; only text is supported"
                )
            texts.append(string(block.get("text"), where=f"{where}[{index}].text"))
        return "".join(texts)

    @staticmethod
    def _sampling(payload: Mapping[str, object]) -> SamplingConfig:
        """Map Responses sampling fields to framework token selection."""
        temperature = number(payload.get("temperature", 1.0), where="temperature")
        top_p = number(payload.get("top_p", 1.0), where="top_p")
        if temperature < 0:
            raise WireFormatError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise WireFormatError("top_p must be in (0, 1]")
        return SamplingConfig(
            do_sample=temperature > 0,
            temperature=temperature or 1.0,
            top_p=top_p,
        )

    @staticmethod
    def _output_items(response: ModelResponse, response_id: str) -> list[dict[str, object]]:
        """Build completed Responses output items in conversation order."""
        items: list[dict[str, object]] = []
        if response.content or not response.tool_calls:
            items.append(
                {
                    "id": f"msg_{response_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response.content,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )
        for index, call in enumerate(response.tool_calls):
            if call.kind == "custom":
                item: dict[str, object] = {
                    "id": f"ctc_{response_id}_{index}",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call.call_id,
                    "name": call.name,
                    "input": call.raw_input or "",
                }
                if call.namespace is not None:
                    item["namespace"] = call.namespace
                items.append(item)
            else:
                item: dict[str, object] = {
                    "id": f"fc_{response_id}_{index}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(dict(call.arguments), separators=(",", ":")),
                }
                if call.namespace is not None:
                    item["namespace"] = call.namespace
                items.append(item)
        return items

    @staticmethod
    def _usage(usage: TokenUsage) -> dict[str, object]:
        """Format normalized counts using the Responses usage schema."""
        return {
            "input_tokens": usage.input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.total_tokens,
        }

    @classmethod
    def _response_object(
        cls,
        *,
        response_id: str,
        model: str,
        status: str,
        output: list[dict[str, object]],
        usage: TokenUsage | None,
        created_at: int,
        error: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the stable subset of a Responses object consumed by Codex clients."""
        incomplete = {"reason": "max_tokens"} if status == "incomplete" else None
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "completed_at": int(time()) if status == "completed" else None,
            "status": status,
            "error": error,
            "incomplete_details": incomplete,
            "instructions": None,
            "max_output_tokens": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": False,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": None if usage is None else cls._usage(usage),
            "metadata": {},
        }


class OpenAIResponsesStreamEncoder:
    """Statefully encode live normalized deltas as ordered Responses SSE events."""

    def __init__(
        self,
        adapter: OpenAIResponsesAdapter,
        *,
        response_id: str,
        model: str,
        created_at: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.response_id = response_id
        self.model = model
        self.created_at = int(time()) if created_at is None else created_at
        self.sequence_number = 0
        self.message_started = False
        self.text = ""

    def created_event(self) -> dict[str, object]:
        """Initial in-progress response event emitted before model prefill."""
        return self._event(
            "response.created",
            response=self.adapter._response_object(
                response_id=self.response_id,
                model=self.model,
                status="in_progress",
                output=[],
                usage=None,
                created_at=self.created_at,
            ),
        )

    def text_delta_events(self, delta: str) -> Iterator[dict[str, object]]:
        """Open the assistant text item lazily and append one visible delta."""
        if not delta:
            return
        if not self.message_started:
            yield from self._start_message()
        self.text += delta
        yield self._event(
            "response.output_text.delta",
            item_id=self._message_id,
            output_index=0,
            content_index=0,
            delta=delta,
            logprobs=[],
        )

    def finished_events(
        self,
        response: ModelResponse,
        usage: TokenUsage,
    ) -> Iterator[dict[str, object]]:
        """Finalize visible text, function calls, output items, and the response."""
        has_message = bool(response.content) or not response.tool_calls
        if has_message:
            if not response.content.startswith(self.text):
                raise RuntimeError("streamed text is not a prefix of the final response")
            if not self.message_started:
                yield from self._start_message()
            remaining = response.content[len(self.text) :]
            if remaining:
                yield from self.text_delta_events(remaining)
            part = {
                "type": "output_text",
                "text": response.content,
                "annotations": [],
                "logprobs": [],
            }
            yield self._event(
                "response.output_text.done",
                item_id=self._message_id,
                output_index=0,
                content_index=0,
                text=response.content,
                logprobs=[],
            )
            yield self._event(
                "response.content_part.done",
                item_id=self._message_id,
                output_index=0,
                content_index=0,
                part=part,
            )
            message = self.adapter._output_items(response, self.response_id)[0]
            yield self._event(
                "response.output_item.done",
                output_index=0,
                item=message,
            )

        output_index = 1 if has_message else 0
        for call_index, call in enumerate(response.tool_calls):
            if call.kind == "custom":
                yield from self._custom_tool_events(call, call_index, output_index)
                output_index += 1
                continue
            item_id = f"fc_{self.response_id}_{call_index}"
            arguments = json.dumps(call.arguments, separators=(",", ":"))
            pending = {
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": "",
            }
            if call.namespace is not None:
                pending["namespace"] = call.namespace
            yield self._event(
                "response.output_item.added",
                output_index=output_index,
                item=pending,
            )
            if arguments:
                yield self._event(
                    "response.function_call_arguments.delta",
                    item_id=item_id,
                    output_index=output_index,
                    delta=arguments,
                )
            yield self._event(
                "response.function_call_arguments.done",
                item_id=item_id,
                output_index=output_index,
                arguments=arguments,
                name=call.name,
            )
            completed = dict(pending)
            completed["status"] = "completed"
            completed["arguments"] = arguments
            yield self._event(
                "response.output_item.done",
                output_index=output_index,
                item=completed,
            )
            output_index += 1

        completed = self.adapter.format_response(
            response,
            response_id=self.response_id,
            model=self.model,
            usage=usage,
        )
        terminal = (
            "response.incomplete" if response.stop_reason == "length" else "response.completed"
        )
        yield self._event(terminal, response=completed)

    def _custom_tool_events(
        self,
        call: ToolCall,
        call_index: int,
        output_index: int,
    ) -> Iterator[dict[str, object]]:
        """Emit a free-form tool call using the Responses custom-tool events."""
        item_id = f"ctc_{self.response_id}_{call_index}"
        raw_input = call.raw_input or ""
        pending = {
            "id": item_id,
            "type": "custom_tool_call",
            "status": "in_progress",
            "call_id": call.call_id,
            "name": call.name,
            "input": "",
        }
        if call.namespace is not None:
            pending["namespace"] = call.namespace
        yield self._event(
            "response.output_item.added",
            output_index=output_index,
            item=pending,
        )
        if raw_input:
            yield self._event(
                "response.custom_tool_call_input.delta",
                item_id=item_id,
                call_id=call.call_id,
                output_index=output_index,
                delta=raw_input,
            )
        yield self._event(
            "response.custom_tool_call_input.done",
            item_id=item_id,
            call_id=call.call_id,
            output_index=output_index,
            input=raw_input,
        )
        completed = dict(pending)
        completed["status"] = "completed"
        completed["input"] = raw_input
        yield self._event(
            "response.output_item.done",
            output_index=output_index,
            item=completed,
        )

    def failed_event(self, *, code: str, message: str) -> dict[str, object]:
        """Terminal failed response for an exception after streaming has begun."""
        response = self.adapter._response_object(
            response_id=self.response_id,
            model=self.model,
            status="failed",
            output=[],
            usage=None,
            created_at=self.created_at,
            error={"code": code, "message": message},
        )
        return self._event("response.failed", response=response)

    def _start_message(self) -> Iterator[dict[str, object]]:
        """Open output item zero and its text content part exactly once."""
        if self.message_started:
            return
        self.message_started = True
        yield self._event(
            "response.output_item.added",
            output_index=0,
            item={
                "id": self._message_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        yield self._event(
            "response.content_part.added",
            item_id=self._message_id,
            output_index=0,
            content_index=0,
            part={
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        )

    def _event(self, event_type: str, **fields: object) -> dict[str, object]:
        """Attach a strictly increasing sequence number."""
        event = {"type": event_type, **fields, "sequence_number": self.sequence_number}
        self.sequence_number += 1
        return event

    @property
    def _message_id(self) -> str:
        """Stable assistant output-item ID for this response."""
        return f"msg_{self.response_id}"
