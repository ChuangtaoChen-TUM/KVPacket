"""A small, fully-controlled agentic task set.

Each task activates a **known** subset of the skill library (ground-truth activation
— no learned retrieval yet), so the independent variable, the *number of activated
skills*, is exact. Answers are computed programmatically so they are correct by
construction; a task's answer depends on its **primary** skill, while the remaining
activated skills are realistic distractors that still occupy context (and, in KV
Packet mode, still get served from packets).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AgentTask", "build_tasks"]

_VOWELS = "aeiou"


# -- reference skill implementations (compute gold answers) -------------------
def _caesar_decode(word: str, shift: int = 3) -> str:
    out = []
    for c in word:
        if c.isalpha():
            out.append(chr((ord(c) - ord("a") - shift) % 26 + ord("a")))
        else:
            out.append(c)
    return "".join(out)


def _caesar_encode(word: str, shift: int = 3) -> str:
    return _caesar_decode(word, -shift)


def _reverse(word: str) -> str:
    return word[::-1]


def _roman(n: int) -> str:
    table = [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
             (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in table:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _initials(name: str) -> str:
    return "".join(w[0].upper() + "." for w in name.split())


def _pig_latin(word: str) -> str:
    i = 0
    while i < len(word) and word[i] not in _VOWELS:
        i += 1
    if i == 0 or i == len(word):
        return word + "ay"
    return word[i:] + word[:i] + "ay"


# -- per-skill instances + prompt/answer builders ----------------------------
def _caesar_task(word):
    enc = _caesar_encode(word)
    return f"Decode this Caesar-shifted word (shift 3): '{enc}'.", _caesar_decode(enc)


def _reverse_task(word):
    return f"Reverse the letters of the word '{word}'.", _reverse(word)


def _roman_task(n):
    return f"Write the number {n} as a Roman numeral.", _roman(n)


def _initials_task(name):
    return f"Give the dotted initials of the name '{name}'.", _initials(name)


def _pig_task(word):
    return f"Convert the word '{word}' to Pig Latin.", _pig_latin(word)


# primary skill -> (builder, [instances])
_PRIMARIES = {
    "caesar-shift": (_caesar_task, ["world", "table", "puzzle"]),
    "word-reverse": (_reverse_task, ["drawer", "planets", "silent"]),
    "roman-numeral": (_roman_task, [24, 49, 88]),
    "initials": (_initials_task, ["grace hopper", "alan turing", "john von neumann"]),
    "pig-latin": (_pig_task, ["happy", "trust", "brave"]),
}
_SKILL_ORDER = list(_PRIMARIES.keys())


@dataclass
class AgentTask:
    task_id: str
    prompt: str
    answer: str
    skills: list[str]          # activated set (primary first), the controlled variable
    primary: str
    meta: dict = field(default_factory=dict)

    @property
    def num_skills(self) -> int:
        return len(self.skills)


def build_tasks() -> list[AgentTask]:
    """~15 tasks spanning 1..5 activated skills (3 per activation-count)."""
    tasks: list[AgentTask] = []
    for n in range(1, len(_SKILL_ORDER) + 1):          # #activated skills = 1..5
        for k in range(3):                             # 3 tasks per count
            primary = _SKILL_ORDER[(n + k) % len(_SKILL_ORDER)]
            builder, instances = _PRIMARIES[primary]
            prompt, answer = builder(instances[k])
            # activated set: primary + (n-1) distractors, keeping library order
            others = [s for s in _SKILL_ORDER if s != primary]
            activated = [primary] + others[: n - 1]
            activated_in_order = [s for s in _SKILL_ORDER if s in activated]
            tasks.append(
                AgentTask(
                    task_id=f"n{n}_{k}_{primary}",
                    prompt=prompt,
                    answer=answer,
                    skills=activated_in_order,
                    primary=primary,
                    meta={"instance": instances[k]},
                )
            )
    return tasks
