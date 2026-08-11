"""Deterministic fingerprints for wrapper registries and cache identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch
from torch import Tensor

__all__ = [
    "FingerprintValue",
    "canonical_json_bytes",
    "fingerprint_json",
    "fingerprint_bytes",
    "fingerprint_tensor",
    "is_fingerprint",
]

FingerprintValue: TypeAlias = (
    None | bool | int | float | str | list["FingerprintValue"] | dict[str, "FingerprintValue"]
)

_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _validate_json_value(value: object, path: str = "$") -> FingerprintValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        out: dict[str, FingerprintValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains non-string mapping key {key!r}")
            out[key] = _validate_json_value(item, f"{path}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_validate_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{path} contains non-JSON value of type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return strict canonical JSON bytes, rejecting lossy/non-finite values."""
    checked = _validate_json_value(value)
    return json.dumps(
        checked,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_json(value: object, *, domain: str) -> str:
    """Fingerprint a JSON-compatible value with an explicit domain separator."""
    digest = hashlib.sha256(canonical_json_bytes({"domain": domain, "value": value})).hexdigest()
    return f"sha256:{digest}"


def fingerprint_bytes(value: bytes, *, domain: str) -> str:
    """Fingerprint opaque bytes with an explicit domain separator."""
    metadata = canonical_json_bytes({"domain": domain, "length": len(value)})
    digest = hashlib.sha256(metadata + b"\0" + value).hexdigest()
    return f"sha256:{digest}"


def fingerprint_tensor(tensor: Tensor) -> str:
    """Fingerprint tensor dtype, shape, and contiguous row-major bytes."""
    if tensor.layout != torch.strided:
        raise ValueError(f"only strided tensors can be fingerprinted, got {tensor.layout}")
    cpu = tensor.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {
            "domain": "kvpacket.tensor/1",
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "shape": list(cpu.shape),
        }
    )
    # Viewing as uint8 works for dtypes (notably bfloat16) unsupported by NumPy.
    raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256(metadata + b"\0" + raw).hexdigest()
    return f"sha256:{digest}"


def is_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _FINGERPRINT_RE.fullmatch(value) is not None
