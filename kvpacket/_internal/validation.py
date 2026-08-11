"""Small internal validation helpers shared by persistent formats."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast


def string_mapping(
    value: object,
    *,
    where: str,
    error: Callable[[str], Exception] = ValueError,
) -> dict[str, object]:
    """Require a concrete mapping whose keys are all strings."""
    if not isinstance(value, dict):
        raise error(f"{where} must be a mapping")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise error(f"{where} keys must be strings")
    return cast(dict[str, object], raw)
