"""Bounded read-only repository tools used by the minimal agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from kvpacket.serve import ToolCall, ToolDefinition

__all__ = ["RepositoryTools", "ToolExecutionError", "agent_tools", "skill_loader_tool"]

_TEXT_SUFFIXES = {".cfg", ".ini", ".md", ".py", ".rst", ".toml", ".txt", ".yaml", ".yml"}
_MAX_OUTPUT_CHARS = 16_000

_PATH_SCHEMA: dict[str, object] = {
    "type": "string",
    "minLength": 1,
    "description": "Repository-relative path; use '.' for the repository root.",
}

_TOOL_DEFINITIONS = (
    ToolDefinition(
        "list_files",
        "List the immediate entries in one repository-relative directory.",
        {
            "type": "object",
            "properties": {"path": _PATH_SCHEMA},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "search_text",
        "Search for a case-sensitive literal substring in repository text files.",
        {
            "type": "object",
            "properties": {
                "path": _PATH_SCHEMA,
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Non-empty case-sensitive literal substring.",
                },
            },
            "required": ["path", "query"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "read_file",
        "Read an inclusive range of numbered lines from a repository text file.",
        {
            "type": "object",
            "properties": {
                "path": _PATH_SCHEMA,
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "One-based first line to read.",
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "One-based final line to read, inclusive.",
                },
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    ),
)


def agent_tools() -> tuple[ToolDefinition, ...]:
    """Canonical structured tool schemas rendered on every model request."""
    return _TOOL_DEFINITIONS


def skill_loader_tool(skill_names: tuple[str, ...]) -> ToolDefinition:
    """Structured operation used to load one listed skill after task matching."""
    if not skill_names or len(set(skill_names)) != len(skill_names):
        raise ValueError("skill_names must contain unique available skills")
    return ToolDefinition(
        "load_skill",
        "Load the full instructions for one available skill when the task matches it.",
        {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": list(skill_names),
                    "description": "Exact skill name from the available-skills catalog.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    )


class ToolExecutionError(ValueError):
    """Invalid or unsafe read-only repository tool request."""


def _string_argument(arguments: Mapping[str, object], name: str) -> str:
    """Extract one required non-empty string argument."""
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f"{name} must be a non-empty string")
    return value


def _integer_argument(arguments: Mapping[str, object], name: str) -> int:
    """Extract one required positive integer argument."""
    value = arguments.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ToolExecutionError(f"{name} must be a positive integer")
    return value


class RepositoryTools:
    """Read-only filesystem view constrained to one prepared repository snapshot."""

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"repository root is not a directory: {resolved}")
        self.root = resolved

    def execute(self, call: ToolCall) -> str:
        """Dispatch one normalized tool call and convert validation failures to output."""
        try:
            if call.name == "list_files":
                self._exact_arguments(call.arguments, {"path"})
                return self.list_files(_string_argument(call.arguments, "path"))
            if call.name == "search_text":
                self._exact_arguments(call.arguments, {"path", "query"})
                return self.search_text(
                    _string_argument(call.arguments, "path"),
                    _string_argument(call.arguments, "query"),
                )
            if call.name == "read_file":
                self._exact_arguments(call.arguments, {"path", "start_line", "end_line"})
                return self.read_file(
                    _string_argument(call.arguments, "path"),
                    _integer_argument(call.arguments, "start_line"),
                    _integer_argument(call.arguments, "end_line"),
                )
            raise ToolExecutionError(f"unknown tool {call.name!r}")
        except (OSError, UnicodeError, ToolExecutionError) as error:
            return f"ERROR: {error}"

    def list_files(self, relative_path: str) -> str:
        """List immediate children of a repository-relative directory."""
        directory = self._resolve(relative_path)
        if not directory.is_dir():
            raise ToolExecutionError(f"not a directory: {relative_path}")
        entries = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.name == ".kvpacket-snapshot.json":
                continue
            suffix = "/" if path.is_dir() else ""
            entries.append(f"{path.relative_to(self.root).as_posix()}{suffix}")
        return self._bounded("\n".join(entries) or "(empty directory)")

    def search_text(self, relative_path: str, query: str) -> str:
        """Search a literal substring in one file or recursively below a directory."""
        if "\n" in query or "\r" in query:
            raise ToolExecutionError("query must fit on one line")
        target = self._resolve(relative_path)
        paths = [target] if target.is_file() else sorted(target.rglob("*"))
        matches: list[str] = []
        for path in paths:
            if not path.is_file() or not self._is_text_file(path):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            relative = path.relative_to(self.root).as_posix()
            for line_number, line in enumerate(lines, 1):
                if query in line:
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) == 50:
                        return self._bounded("\n".join(matches) + "\n[results truncated]")
        return self._bounded("\n".join(matches) or "(no matches)")

    def read_file(self, relative_path: str, start_line: int, end_line: int) -> str:
        """Read an inclusive, numbered line range from a repository text file."""
        if end_line < start_line:
            raise ToolExecutionError("end_line must be greater than or equal to start_line")
        if end_line - start_line + 1 > 200:
            raise ToolExecutionError("read_file accepts at most 200 lines")
        path = self._resolve(relative_path)
        if not path.is_file() or not self._is_text_file(path):
            raise ToolExecutionError(f"not a supported text file: {relative_path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if start_line > len(lines):
            raise ToolExecutionError(f"start_line {start_line} exceeds file length {len(lines)}")
        stop = min(end_line, len(lines))
        relative = path.relative_to(self.root).as_posix()
        output = "\n".join(
            f"{relative}:{line_number}: {lines[line_number - 1]}"
            for line_number in range(start_line, stop + 1)
        )
        return self._bounded(output)

    def _resolve(self, relative_path: str) -> Path:
        """Resolve an existing path and reject traversal or symlink escapes."""
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ToolExecutionError("path must remain inside the repository")
        try:
            resolved = (self.root / candidate).resolve(strict=True)
        except FileNotFoundError as error:
            raise ToolExecutionError(f"path does not exist: {relative_path}") from error
        if not resolved.is_relative_to(self.root):
            raise ToolExecutionError("path resolves outside the repository")
        return resolved

    @staticmethod
    def _exact_arguments(arguments: Mapping[str, object], expected: set[str]) -> None:
        """Reject omitted or invented tool arguments."""
        if set(arguments) != expected:
            raise ToolExecutionError(
                f"arguments must be exactly {sorted(expected)}; got {sorted(arguments)}"
            )

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        """Supported source/config/document file selected during preparation."""
        name = path.name.upper()
        return path.suffix.lower() in _TEXT_SUFFIXES or name.startswith(("LICENSE", "COPYING"))

    @staticmethod
    def _bounded(output: str) -> str:
        """Cap tool output before it is returned to the model context."""
        if len(output) <= _MAX_OUTPUT_CHARS:
            return output
        return output[:_MAX_OUTPUT_CHARS] + "\n[output truncated]"
