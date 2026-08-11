"""Tests for the real-repository agent example without model downloads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import pytest

from kvpacket.serve import ToolCall
from tasks.agentic_flow.repository_qa.mcp_server import create_server
from tasks.agentic_flow.repository_qa.pipeline import _prompt_specs, load_config
from tasks.agentic_flow.repository_qa.skills import (
    loaded_skill_message,
    load_skills,
    skill_token_ids,
    system_prompt,
)
from tasks.agentic_flow.repository_qa.tools import (
    RepositoryTools,
    agent_tools,
    skill_loader_tool,
)


class _CharacterTokenizer:
    """Small deterministic tokenizer sufficient for document-boundary tests."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, object]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _repository(root: Path) -> Path:
    """Create a repository snapshot with exact AST-grounded facts."""
    repository = root / "repository"
    package = repository / "src" / "demo"
    package.mkdir(parents=True)
    (package / "core.py").write_text(
        """LIMIT = 7

class Worker(BaseWorker):
    def run(self, retries=3):
        if retries < 0:
            raise ValueError("negative")
        return helper(retries)

def helper(value=None):
    return value
"""
    )
    return repository


def test_repository_tools_are_bounded_and_cannot_escape(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    tools = RepositoryTools(repository)
    assert tools.list_files(".") == "src/"
    assert "src/demo/core.py:1: LIMIT = 7" in tools.search_text(".", "LIMIT")
    assert tools.read_file("src/demo/core.py", 1, 2).startswith("src/demo/core.py:1:")

    escaped = tools.execute(ToolCall("call-1", "read_file", {"path": "../outside.txt"}))
    assert escaped.startswith("ERROR:")
    malformed = tools.execute(ToolCall("call-2", "list_files", {"path": ".", "extra": 1}))
    assert malformed.startswith("ERROR:")


def test_skill_uses_progressive_disclosure_and_only_its_body_is_packetized() -> None:
    skills = load_skills()
    prompt = system_prompt(skills)
    tokens = skill_token_ids(_CharacterTokenizer(), skills)
    assert len(skills) == 1
    assert set(tokens) == {skill.skill_id for skill in skills}
    for skill in skills:
        assert skill.content.startswith("---\n")
        assert skill.block.startswith("<kvpacket-skill")
        assert skill.block.endswith("</kvpacket-skill")
        assert skill.name in prompt
        assert skill.description in prompt
        assert skill.path in prompt
        assert skill.block not in prompt
        loaded = loaded_skill_message(skill)
        assert loaded.count(f"{skill.block}>") == 1

    tools = agent_tools()
    assert {tool.name for tool in tools} == {"list_files", "search_text", "read_file"}
    assert all(tool.input_schema["type"] == "object" for tool in tools)
    assert all(tool.description not in prompt for tool in tools)
    loader = skill_loader_tool(tuple(skill.name for skill in skills))
    assert loader.name == "load_skill"
    properties = cast(Mapping[str, object], loader.input_schema["properties"])
    name_schema = cast(Mapping[str, object], properties["name"])
    assert name_schema["enum"] == [skill.name for skill in skills]


def test_no_skill_prompt_omits_activation_without_removing_tool_guidance() -> None:
    prompt = system_prompt(())

    assert "Available skills" not in prompt
    assert "<kvpacket-skill" not in prompt
    assert "available tools" in prompt


@pytest.mark.skipif(find_spec("mcp") is None, reason="optional agent dependency is absent")
def test_mcp_server_registers_over_the_confined_repository(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    server = create_server(repository)

    async def inspect_server() -> None:
        tools = await server.list_tools()
        assert {tool.name for tool in tools} == {"list_files", "search_text", "read_file"}
        for tool in tools:
            definition = next(item for item in agent_tools() if item.name == tool.name)
            assert tool.description == definition.description
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
        result = await server.call_tool("list_files", {"path": "."})
        assert result.is_error is False
        assert result.content[0].text == "src/"

    asyncio.run(inspect_server())


@pytest.mark.skipif(find_spec("mcp") is None, reason="optional agent dependency is absent")
def test_mcp_stdio_round_trip(tmp_path: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    repository = _repository(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "tasks.agentic_flow.repository_qa.mcp_server",
            "--repository-root",
            str(repository),
        ],
        env=environment,
    )

    async def round_trip() -> None:
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "list_files",
                "search_text",
                "read_file",
            }
            result = await session.call_tool("list_files", {"path": "../outside"})
            assert result.is_error is False
            assert getattr(result.content[0], "text", "").startswith("ERROR:")

    asyncio.run(round_trip())


def test_config_requires_external_storage_for_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "repository_root": ".",
                "artifact_dir": "${KVP_STORAGE}/artifacts",
                "pretrained_wrapper": "checkpoint.pt",
                "prompts": [{"id": "one", "question": "Where is Worker?"}],
            }
        )
    )
    monkeypatch.delenv("KVP_STORAGE", raising=False)
    with pytest.raises(ValueError, match="KVP_STORAGE"):
        load_config(path)

    monkeypatch.setenv("KVP_STORAGE", str(tmp_path / "storage"))
    config = load_config(path)
    assert config["artifact_dir"] == str(tmp_path / "storage" / "artifacts")


def test_prompt_ids_cannot_escape_the_sample_directory() -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        _prompt_specs({"prompts": [{"id": "../escape", "question": "Where?"}]})


def test_packaged_wrapper_is_the_recorded_qwen_checkpoint() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "tasks"
        / "agentic_flow"
        / "repository_qa"
        / "checkpoints"
        / "qwen3_4b_repository_navigation.pt"
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == "060006f3c0a6f636c07054918e57c13c4f0b2a3e536ed318efc086bb8c7d3a3f"
