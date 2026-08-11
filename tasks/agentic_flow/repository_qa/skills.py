"""Load the reusable Codex skill that becomes the example's KV packet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from kvpacket.packet import RawText

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "AgentSkill",
    "AgentTokenizer",
    "loaded_skill_message",
    "load_skills",
    "skill_token_ids",
    "system_prompt",
]

BASE_SYSTEM_PROMPT = (
    "You are a read-only repository question-answering agent. Use the available tools "
    "when needed, and obey the exact answer format requested by the user."
)

_SKILL_FILES = {
    "skill.repository-navigation": Path("repository-navigation") / "SKILL.md",
}


class AgentTokenizer(Protocol):
    """Tokenizer operation needed to materialize exact skill tokens."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Mapping[str, object]:
        """Tokenize one skill block without model-level special tokens."""
        ...


@dataclass(frozen=True, slots=True)
class AgentSkill:
    """Installed Codex skill metadata, instructions, and packetizable interior.

    Codex initially sees only ``name``, ``description``, and ``path``. ``content`` is
    the complete ``SKILL.md`` text loaded after activation. ``block`` is the tagged
    reusable portion, excluding only the final ``>``. Keeping that boundary character
    online prevents common BPE tokenizers from merging it with following prompt text
    while leaving the model-visible skill unchanged.
    """

    skill_id: str
    name: str
    description: str
    path: str
    content: str
    block: str

    @property
    def source(self) -> RawText:
        """Original tagged skill text used for packet provenance."""
        return RawText(self.block, source_id=self.skill_id)


def _packet_block(skill_id: str, content: str) -> str:
    """Extract the uniquely tagged packet interior from one skill file."""
    opening = f'<kvpacket-skill id="{skill_id}">'
    closing = "</kvpacket-skill>"
    if content.count(opening) != 1 or content.count(closing) != 1:
        raise ValueError(f"skill {skill_id!r} must contain one packet marker pair")
    start = content.index(opening)
    stop = content.index(closing, start) + len(closing) - 1
    return content[start:stop]


def _frontmatter(content: str, *, path: Path) -> tuple[str, str]:
    """Read the required name and description from a small skill frontmatter block."""
    lines = content.splitlines()
    if len(lines) < 4 or lines[0] != "---":
        raise ValueError(f"skill {path} must start with YAML frontmatter")
    try:
        stop = lines.index("---", 1)
    except ValueError as error:
        raise ValueError(f"skill {path} has unterminated YAML frontmatter") from error
    fields: dict[str, str] = {}
    for line in lines[1:stop]:
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"skill {path} has unsupported frontmatter line {line!r}")
        fields[key.strip()] = value.strip()
    name = fields.get("name")
    description = fields.get("description")
    if not name or not description:
        raise ValueError(f"skill {path} requires name and description")
    return name, description


def load_skills() -> tuple[AgentSkill, ...]:
    """Load checked-in project skills in stable prompt order."""
    directory = Path(__file__).with_name(".agents") / "skills"
    skills = []
    for skill_id, filename in _SKILL_FILES.items():
        path = directory / filename
        content = path.read_text(encoding="utf-8").strip()
        name, description = _frontmatter(content, path=path)
        relative_path = (Path(".agents") / "skills" / filename).as_posix()
        skills.append(
            AgentSkill(
                skill_id,
                name,
                description,
                relative_path,
                content,
                _packet_block(skill_id, content),
            )
        )
    return tuple(skills)


def system_prompt(skills: tuple[AgentSkill, ...] | None = None) -> str:
    """Assemble Codex-style initial metadata without eagerly loading skill bodies."""
    selected = load_skills() if skills is None else skills
    if not selected:
        return BASE_SYSTEM_PROMPT
    catalog = "\n".join(
        f"- {skill.name}: {skill.description} (file: {skill.path})" for skill in selected
    )
    return (
        BASE_SYSTEM_PROMPT
        + "\n\nAvailable skills (metadata only):\n"
        + catalog
        + "\nLoad a skill when the task matches its description before following its "
        "instructions."
    )


def loaded_skill_message(skill: AgentSkill) -> str:
    """Render the full skill result inserted after conditional activation."""
    return (
        f"<skill>\n<name>{skill.name}</name>\n<path>{skill.path}</path>\n{skill.content}\n</skill>"
    )


def skill_token_ids(
    tokenizer: AgentTokenizer,
    skills: tuple[AgentSkill, ...] | None = None,
) -> dict[str, list[int]]:
    """Tokenize exact reusable blocks for matching and packet preprocessing."""
    selected = load_skills() if skills is None else skills
    encoded: dict[str, list[int]] = {}
    for skill in selected:
        raw = tokenizer(skill.block, add_special_tokens=False).get("input_ids")
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw)
        ):
            raise TypeError("agent tokenizer must return one non-empty input_ids list")
        encoded[skill.skill_id] = cast(list[int], raw)
    return encoded
