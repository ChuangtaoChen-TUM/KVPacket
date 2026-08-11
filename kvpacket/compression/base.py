"""Define backend-extension contracts and persistent cache-compression metadata."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from ..backend import ComputeBackend
from ..fingerprint import FingerprintValue, canonical_json_bytes, fingerprint_json

__all__ = [
    "NO_COMPRESSION_FINGERPRINT",
    "CacheCompressor",
    "CompressionSpec",
]


@dataclass(frozen=True)
class CompressionSpec:
    """Serializable identity of one offline cache-compression configuration.

    Attributes:
        provider: Compression implementation family, such as ``"kvpress"``.
        method: Provider-specific compression method.
        implementation_version: Installed provider version used to build the cache.
        config: JSON-compatible method parameters that affect compressed tensors.
    """

    provider: str
    method: str
    implementation_version: str
    config: dict[str, FingerprintValue]

    def __post_init__(self) -> None:
        """Validate identifiers and detach configuration from caller-owned mappings."""
        for name in ("provider", "method", "implementation_version"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        normalized = json.loads(canonical_json_bytes(self.config).decode("utf-8"))
        if not isinstance(normalized, dict):
            raise ValueError("compression config must be a mapping")
        object.__setattr__(self, "config", normalized)

    @property
    def fingerprint(self) -> str:
        """Canonical identity of the provider, version, method, and parameters."""
        return fingerprint_json(self.to_dict(), domain="kvpacket.cache_compression/1")

    def to_dict(self) -> dict[str, object]:
        """Serialize the specification into detached primitive values."""
        config = json.loads(canonical_json_bytes(self.config).decode("utf-8"))
        return {
            "provider": self.provider,
            "method": self.method,
            "implementation_version": self.implementation_version,
            "config": config,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CompressionSpec":
        """Parse and validate persisted compression metadata."""
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("compression specification must be a string-keyed mapping")
        data = cast(dict[str, object], value)
        expected = {"provider", "method", "implementation_version", "config"}
        if set(data) != expected:
            raise ValueError(
                "invalid compression specification keys: "
                f"missing={sorted(expected - set(data))}, "
                f"unknown={sorted(set(data) - expected)}"
            )
        provider = data["provider"]
        method = data["method"]
        implementation_version = data["implementation_version"]
        config = data["config"]
        if not all(isinstance(item, str) for item in (provider, method, implementation_version)):
            raise ValueError("compression provider, method, and version must be strings")
        if not isinstance(config, dict):
            raise ValueError("compression config must be a mapping")
        return cls(
            provider=cast(str, provider),
            method=cast(str, method),
            implementation_version=cast(str, implementation_version),
            config=cast(dict[str, FingerprintValue], config),
        )


NO_COMPRESSION_FINGERPRINT = fingerprint_json(
    {"provider": "none"}, domain="kvpacket.cache_compression/1"
)


@runtime_checkable
class CacheCompressor(Protocol):
    """Optional extension that activates compression around one offline prefill."""

    @property
    def spec(self) -> CompressionSpec:
        """Persistent identity of the active compression configuration."""
        ...

    def activate(self, backend: ComputeBackend) -> AbstractContextManager[None]:
        """Activate compression for forwards executed inside the returned context."""
        ...
