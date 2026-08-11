"""Expose the repository QA tools through a local read-only MCP server.

The optional MCP dependency is imported only when the server is created. Install it
with ``uv sync --extra agent``. The server writes protocol messages to standard I/O;
tool implementations themselves never write to standard output.
"""

from __future__ import annotations

import argparse
import os
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any

from kvpacket.serve import ToolCall

from .tools import RepositoryTools, agent_tools

__all__ = ["create_server", "main"]


def _description(name: str) -> str:
    """Look up one canonical tool description shared with model requests."""
    for definition in agent_tools():
        if definition.name == name:
            return definition.description
    raise KeyError(name)


def _mcp_classes() -> tuple[type[Any], type[Any]]:
    """Load the ergonomic server and protocol annotations from the optional SDK."""
    try:
        server_module = import_module("mcp.server")
        server_type = server_module.MCPServer
    except (ImportError, AttributeError):
        try:
            server_type = import_module("mcp.server.fastmcp").FastMCP
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "the repository MCP server requires the 'agent' extra; run `uv sync --extra agent`"
            ) from error
    try:
        annotations = import_module("mcp.types").ToolAnnotations
    except (ImportError, AttributeError) as error:
        raise RuntimeError("the installed MCP SDK does not expose ToolAnnotations") from error
    return server_type, annotations


def create_server(repository_root: str | Path) -> Any:
    """Create a stdio MCP server confined to one repository snapshot.

    The MCP schemas stay ordinary structured tool metadata. They are deliberately
    not registered with the packet store; only the separately installed skill body
    is eligible for KV Packet matching.
    """
    mcp_server_type, annotations_type = _mcp_classes()
    executor = RepositoryTools(repository_root)
    server = mcp_server_type(
        "kvpacket-repository-tools",
        instructions=(
            "Inspect only the configured repository snapshot. All paths are relative "
            "to that repository, and every operation is read-only."
        ),
    )
    annotations = annotations_type(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    def list_files(path: str) -> str:
        """List immediate entries in one repository-relative directory."""
        return executor.execute(ToolCall("mcp-list-files", "list_files", {"path": path}))

    def search_text(path: str, query: str) -> str:
        """Search a literal substring in repository text files."""
        return executor.execute(
            ToolCall("mcp-search-text", "search_text", {"path": path, "query": query})
        )

    def read_file(path: str, start_line: int, end_line: int) -> str:
        """Read an inclusive range of numbered lines from one text file."""
        return executor.execute(
            ToolCall(
                "mcp-read-file",
                "read_file",
                {"path": path, "start_line": start_line, "end_line": end_line},
            )
        )

    field = import_module("pydantic").Field
    list_files.__annotations__ = {
        "path": Annotated[
            str,
            field(
                min_length=1,
                description="Repository-relative path; use '.' for the repository root.",
            ),
        ],
        "return": str,
    }
    search_text.__annotations__ = {
        "path": Annotated[
            str,
            field(
                min_length=1,
                description="Repository-relative path; use '.' for the repository root.",
            ),
        ],
        "query": Annotated[
            str,
            field(min_length=1, description="Non-empty case-sensitive literal substring."),
        ],
        "return": str,
    }
    read_file.__annotations__ = {
        "path": Annotated[
            str,
            field(
                min_length=1,
                description="Repository-relative path; use '.' for the repository root.",
            ),
        ],
        "start_line": Annotated[
            int,
            field(ge=1, description="One-based first line to read."),
        ],
        "end_line": Annotated[
            int,
            field(ge=1, description="One-based final line to read, inclusive."),
        ],
        "return": str,
    }

    for name, handler in (
        ("list_files", list_files),
        ("search_text", search_text),
        ("read_file", read_file),
    ):
        server.tool(
            name=name,
            description=_description(name),
            annotations=annotations,
        )(handler)
    return server


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing the MCP SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=os.environ.get("KVP_AGENT_REPOSITORY"),
        help="repository snapshot root; defaults to KVP_AGENT_REPOSITORY",
    )
    return parser


def main() -> None:
    """Run the repository MCP server over standard I/O."""
    parser = _parser()
    arguments = parser.parse_args()
    if not arguments.repository_root:
        parser.error("--repository-root or KVP_AGENT_REPOSITORY is required")
    server = create_server(arguments.repository_root)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
