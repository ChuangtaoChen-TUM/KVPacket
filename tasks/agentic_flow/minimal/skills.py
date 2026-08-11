"""Load Agent-Skills-style ``SKILL.md`` files (YAML-ish frontmatter + body).

A skill is reusable static text: a ``name``, a one-line ``description``, and a body
(a short how-to). In the KV Packet flow the **body** is what gets registered as a
chunk and served from a packet — the L2 "activation" of Agent Skills is exactly the
SERVE-mode substitution of that static block.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["Skill", "load_skills", "SKILLS_DIR"]

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str

    @property
    def block(self) -> str:
        """The reusable text block injected into the prompt (registered as a chunk).

        The trailing blank line is part of the block so that adjacent activated skills
        sit back-to-back as packets with **no** interior unmatched run between them —
        the engine then spends one re-rotation per skill and no extra forward.
        """
        return self.body + "\n\n"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    meta: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        if ":" in lines[i]:
            k, v = lines[i].split(":", 1)
            meta[k.strip()] = v.strip()
        i += 1
    body = "\n".join(lines[i + 1:]).strip()
    return meta, body


def load_skills(directory: str = SKILLS_DIR) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(directory, fname)) as f:
            meta, body = _parse_frontmatter(f.read())
        name = meta.get("name", os.path.splitext(fname)[0])
        skills[name] = Skill(name=name, description=meta.get("description", ""), body=body)
    return skills
