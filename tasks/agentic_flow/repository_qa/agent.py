"""Minimal read-only Codex-style loop with recordable packetized instructions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from kvpacket.backend import ComputeBackend
from kvpacket.serve import (
    ChatMessage,
    ChatTokenizer,
    HFChatCodec,
    ModelRequest,
    ModelResponse,
)
from kvpacket.serve.codecs import QwenToolCallParser
from kvpacket.session import PacketStore, Mode, PacketSession, SessionTokenizer, TrainingSample

from .skills import (
    AgentSkill,
    AgentTokenizer,
    loaded_skill_message,
    load_skills,
    skill_token_ids,
    system_prompt,
)
from .tools import RepositoryTools, agent_tools, skill_loader_tool

__all__ = ["AgentMethod", "AgentRun", "AgentTurn", "RepositoryAgent"]


class AgentMethod(str, Enum):
    """How reusable instruction blocks are handled during model inference."""

    FULL_RECOMPUTE = "full_recompute"
    KV_PACKET = "kv_packet"


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One model continuation and any tool calls executed after it."""

    turn_index: int
    response: ModelResponse
    tool_outputs: tuple[tuple[str, str], ...]
    input_tokens: int
    output_tokens: int
    ttft: float
    rerotation_flops: int
    sample: TrainingSample | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Completed or failed repository-agent trajectory."""

    run_id: str
    method: AgentMethod
    prediction: str
    turns: tuple[AgentTurn, ...]
    valid_tool_calls: int
    total_tool_calls: int
    skill_load_calls: int
    successful_skill_loads: int
    error: str | None = None

    @property
    def ttft(self) -> float:
        """Total first-token latency across agent turns."""
        return sum(turn.ttft for turn in self.turns)

    @property
    def rerotation_flops(self) -> int:
        """Total packet re-rotation work across agent turns."""
        return sum(turn.rerotation_flops for turn in self.turns)

    @property
    def samples(self) -> tuple[TrainingSample, ...]:
        """Compact token-teacher samples recorded for generated turns."""
        return tuple(turn.sample for turn in self.turns if turn.sample is not None)


class RepositoryAgent:
    """Execute self-contained tool turns over one repository snapshot.

    Full Recompute can export compact token-ID teacher samples from turns containing
    the conditionally loaded skill. KV Packet uses a materialized store containing that
    same instruction block with its trained wrapper tensors.

    Args:
        backend: Model execution backend shared by every stateless agent turn.
        tokenizer: Model tokenizer with a Hugging Face-compatible chat template.
        method: Full Recompute recording or trained-packet inference.
        store: Required preprocessed store for KV Packet inference.
        skills: Optional replacement reusable skill set.
    """

    def __init__(
        self,
        *,
        backend: ComputeBackend,
        tokenizer: object,
        method: AgentMethod | str,
        store: PacketStore | None = None,
        skills: tuple[AgentSkill, ...] | None = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = cast(SessionTokenizer, tokenizer)
        self.codec = HFChatCodec(
            cast(ChatTokenizer, tokenizer), tool_call_parser=QwenToolCallParser()
        )
        self.method = AgentMethod(method)
        self.skills = load_skills() if skills is None else skills
        if not self.skills:
            raise ValueError("the Codex example requires at least one skill")
        self.skill_tokens = skill_token_ids(cast(AgentTokenizer, tokenizer), self.skills)
        if self.method is AgentMethod.FULL_RECOMPUTE and store is not None:
            raise ValueError("full_recompute does not accept a packet store")
        if self.method is AgentMethod.KV_PACKET and store is None:
            raise ValueError("kv_packet requires a preprocessed packet store")
        self.store = store

    def run(
        self,
        prompt: str,
        repository_root: str,
        *,
        run_id: str = "prompt",
        max_turns: int = 5,
        max_new_tokens: int = 192,
        capture_samples: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> AgentRun:
        """Run bounded tool turns for one repository question."""
        if not prompt.strip() or not run_id:
            raise ValueError("prompt and run_id must be non-empty")
        if max_turns <= 0 or max_new_tokens <= 0:
            raise ValueError("max_turns and max_new_tokens must be positive")
        if capture_samples and self.method is not AgentMethod.FULL_RECOMPUTE:
            raise ValueError("only full_recompute can record teacher samples")

        messages: list[ChatMessage] = [
            ChatMessage("system", system_prompt(self.skills)),
            ChatMessage("user", prompt),
        ]
        tools = RepositoryTools(repository_root)
        turns: list[AgentTurn] = []
        valid_tool_calls = 0
        total_tool_calls = 0
        skill_load_calls = 0
        successful_skill_loads = 0
        loaded_skills: set[str] = set()
        skills_by_name = {skill.name: skill for skill in self.skills}
        final_text = ""
        error: str | None = None

        for turn_index in range(max_turns):
            request_tools = agent_tools()
            if skills_by_name.keys() - loaded_skills:
                request_tools += (
                    skill_loader_tool(tuple(sorted(skills_by_name.keys() - loaded_skills))),
                )
            request = ModelRequest(tuple(messages), request_tools)
            rendered = self.codec.render(request)
            session = self._session()
            session.push(rendered)
            matches = {match.chunk_id for match in session.preview_matches()}
            expected = {
                skills_by_name[name].skill_id for name in loaded_skills if name in skills_by_name
            }
            if matches != expected:
                missing = sorted(expected - matches)
                extra = sorted(matches - expected)
                raise RuntimeError(
                    f"rendered prompt packet mismatch: missing={missing}, extra={extra}"
                )
            generated = session.generate(
                max_new_tokens=max_new_tokens,
                capture_logits=False,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            reached_limit = len(generated.token_ids) >= max_new_tokens
            try:
                response = self.codec.parse(
                    generated.token_ids,
                    reached_limit=reached_limit,
                )
            except ValueError as parse_error:
                response = ModelResponse(
                    content=generated.text,
                    stop_reason="length" if reached_limit else "stop",
                    token_ids=tuple(generated.token_ids),
                )
                error = f"tool-call parse failed: {parse_error}"

            sample = session.export_sample() if capture_samples else None
            if sample is not None and not sample.chunk_spans:
                sample = None
            messages.append(
                ChatMessage(
                    "assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            outputs: list[tuple[str, str]] = []
            for call in response.tool_calls:
                total_tool_calls += 1
                if call.name == "load_skill":
                    skill_load_calls += 1
                    name = call.arguments.get("name")
                    if set(call.arguments) != {"name"} or not isinstance(name, str):
                        output = "ERROR: load_skill requires exactly one string name"
                    elif name in loaded_skills:
                        output = f"ERROR: skill {name!r} is already loaded"
                    elif name not in skills_by_name:
                        output = f"ERROR: unknown skill {name!r}"
                    else:
                        output = loaded_skill_message(skills_by_name[name])
                        loaded_skills.add(name)
                        successful_skill_loads += 1
                else:
                    output = tools.execute(call)
                if not output.startswith("ERROR:"):
                    valid_tool_calls += 1
                outputs.append((call.name, output))
                messages.append(
                    ChatMessage(
                        "tool",
                        output,
                        tool_call_id=call.call_id,
                        name=call.name,
                    )
                )
            turns.append(
                AgentTurn(
                    turn_index=turn_index,
                    response=response,
                    tool_outputs=tuple(outputs),
                    input_tokens=len(rendered),
                    output_tokens=len(generated.token_ids),
                    ttft=generated.ttft,
                    rerotation_flops=generated.flops,
                    sample=sample,
                )
            )
            if not response.tool_calls:
                final_text = response.content
                break
            if error is not None:
                break
        else:
            error = f"agent exceeded {max_turns} model turns"

        return AgentRun(
            run_id=run_id,
            method=self.method,
            prediction=final_text.strip(),
            turns=tuple(turns),
            valid_tool_calls=valid_tool_calls,
            total_tool_calls=total_tool_calls,
            skill_load_calls=skill_load_calls,
            successful_skill_loads=successful_skill_loads,
            error=error,
        )

    def _session(self) -> PacketSession:
        """Create one stateless model turn with identical registered sources."""
        if self.method is AgentMethod.FULL_RECOMPUTE:
            session = PacketSession(
                backend=self.backend,
                mode=Mode.RECORD,
                tokenizer=self.tokenizer,
                match="token",
            )
            for skill_id, token_ids in self.skill_tokens.items():
                session.register_chunk(skill_id, token_ids=token_ids)
            return session
        assert self.store is not None
        session = PacketSession(
            backend=self.backend,
            mode=Mode.SERVE,
            tokenizer=self.tokenizer,
            store=self.store,
            match="token",
            mixed_prefill_policy="fused",
        )
        session.register_store_chunks(list(self.skill_tokens))
        return session
