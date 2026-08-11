"""Share strict JSON-like payload parsing across wire adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import cast

from ..types import ToolDefinition
from .base import WireFormatError


def mapping(value: object, *, where: str) -> Mapping[str, object]:
    """Require a JSON object."""
    if not isinstance(value, Mapping):
        raise WireFormatError(f"{where} must be an object")
    return cast(Mapping[str, object], value)


def sequence(value: object, *, where: str) -> Sequence[object]:
    """Require a JSON array while rejecting strings."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WireFormatError(f"{where} must be an array")
    return value


def string(value: object, *, where: str) -> str:
    """Require a string."""
    if not isinstance(value, str):
        raise WireFormatError(f"{where} must be a string")
    return value


def optional_string(value: object, *, where: str) -> str | None:
    """Accept a string or null."""
    if value is None:
        return None
    return string(value, where=where)


def integer(value: object, *, where: str) -> int:
    """Require an integer distinct from a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireFormatError(f"{where} must be an integer")
    return value


def number(value: object, *, where: str) -> float:
    """Require a finite integer or floating-point number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireFormatError(f"{where} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), -float("inf")):
        raise WireFormatError(f"{where} must be finite")
    return result


def boolean(value: object, *, where: str) -> bool:
    """Require a boolean."""
    if not isinstance(value, bool):
        raise WireFormatError(f"{where} must be a boolean")
    return value


def json_arguments(value: object, *, where: str) -> dict[str, object]:
    """Normalize function arguments supplied as an object or JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise WireFormatError(f"{where} is not valid JSON") from error
    result = mapping(value, where=where)
    return {str(key): item for key, item in result.items()}


def tool_definition(
    value: object,
    *,
    where: str,
    nested_function: bool,
) -> ToolDefinition:
    """Parse a function tool in Responses or chat-template conventions."""
    item = mapping(value, where=where)
    if item.get("type", "function") != "function":
        raise WireFormatError(f"{where}.type must be 'function'")
    body = mapping(item.get("function"), where=f"{where}.function") if nested_function else item
    name = string(body.get("name"), where=f"{where}.name")
    description = optional_string(body.get("description", ""), where=f"{where}.description") or ""
    schema_key = "parameters" if "parameters" in body else "input_schema"
    schema = mapping(body.get(schema_key, {}), where=f"{where}.{schema_key}")
    return ToolDefinition(name, description, {str(key): item for key, item in schema.items()})
