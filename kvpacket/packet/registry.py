"""Manage the named adapters used to construct and validate reusable packets.

A KV Packet application may use different ``PacketWrapper`` instances for documents,
skills, tools, or other reusable context types. ``WrapperRegistry`` gives each
adapter a stable ID and binds the group to the exact backend model that can use it.
Source assignments select these IDs during offline packet preprocessing,
and trainer checkpoints retain the same registry structure.

The aggregate registry fingerprint validates registry checkpoints only. Individual
wrapper fingerprints govern packet invalidation; unrelated registry entries do not.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

from .._internal.persistence import atomic_torch_save
from .._internal.validation import string_mapping
from ..backend.base import BackendCompatibility
from ..errors import BackendCompatibilityError, WrapperFormatError
from ..fingerprint import canonical_json_bytes, fingerprint_json, fingerprint_tensor, is_fingerprint
from .wrapper import PacketWrapper

__all__ = [
    "WRAPPER_REGISTRY_FORMAT",
    "WrapperEntry",
    "WrapperRegistry",
    "fingerprint_wrapper",
]

WRAPPER_REGISTRY_FORMAT = "kvpacket.wrapper_registry/1"
_WRAPPER_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _validate_wrapper_id(wrapper_id: str) -> None:
    """Validate a wrapper ID used in configs, checkpoints, and packet identities.

    Args:
        wrapper_id: Candidate lowercase identifier.

    Raises:
        WrapperFormatError: If the ID is empty, too long, or contains unsupported
            characters.
    """
    if _WRAPPER_ID_RE.fullmatch(wrapper_id) is None:
        raise WrapperFormatError(
            f"invalid wrapper_id {wrapper_id!r}; expected ^[a-z][a-z0-9_.-]{{0,127}}$"
        )


def _checked_metadata(value: Mapping[str, object] | None, *, field_name: str) -> dict[str, object]:
    """Copy metadata after verifying that it has a stable JSON representation.

    Args:
        value: Optional metadata mapping.
        field_name: Human-readable field name included in validation errors.

    Returns:
        A deep copy that callers may mutate without changing their input mapping.

    Raises:
        WrapperFormatError: If the metadata contains a non-JSON-compatible value.
    """
    copied = deepcopy(dict(value or {}))
    try:
        canonical_json_bytes(copied)
    except ValueError as error:
        raise WrapperFormatError(f"{field_name} must be JSON-compatible: {error}") from error
    return copied


def fingerprint_wrapper(wrapper: PacketWrapper) -> str:
    """Fingerprint the functional header and trailer tensor state of a wrapper.

    Training configuration and provenance are intentionally excluded so editing
    non-functional metadata does not invalidate previously built packets.

    Args:
        wrapper: Adapter whose current parameters define the identity.

    Returns:
        SHA-256 fingerprint covering both parameter tensors.
    """
    return fingerprint_json(
        {
            "header": fingerprint_tensor(wrapper.header),
            "trailer": fingerprint_tensor(wrapper.trailer),
        },
        domain="kvpacket.wrapper/1",
    )


@dataclass
class WrapperEntry:
    """One wrapper together with non-functional training and provenance metadata.

    Attributes:
        wrapper: Adapter used to construct reusable packet caches.
        training_config: JSON-compatible record of how the adapter was trained.
        meta: JSON-compatible metrics, labels, or provenance.
    """

    wrapper: PacketWrapper
    training_config: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach entry metadata from caller mappings and validate its representation."""
        self.training_config = _checked_metadata(self.training_config, field_name="training_config")
        self.meta = _checked_metadata(self.meta, field_name="wrapper meta")

    @property
    def fingerprint(self) -> str:
        """Current parameter fingerprint, updated implicitly as training changes tensors."""
        return fingerprint_wrapper(self.wrapper)

    @property
    def header_len(self) -> int:
        """Number of learned header tokens."""
        return self.wrapper.header_len

    @property
    def trailer_len(self) -> int:
        """Number of learned trailer tokens."""
        return self.wrapper.trailer_len

    def state_dict(self) -> dict[str, object]:
        """Build the entry portion of a registry checkpoint.

        Tensor values are detached and placed on CPU. Metadata is deep-copied so later
        edits to the entry do not mutate the returned metadata mappings.

        Returns:
            Serialized tensors, geometry, fingerprint, and non-functional metadata.
        """
        return {
            "header": self.wrapper.header.detach().cpu(),
            "trailer": self.wrapper.trailer.detach().cpu(),
            "header_len": self.header_len,
            "trailer_len": self.trailer_len,
            "fingerprint": self.fingerprint,
            "training_config": deepcopy(self.training_config),
            "meta": deepcopy(self.meta),
        }


class WrapperRegistry:
    """Named wrappers that share one backend model compatibility domain.

    A registry is the adapter registry used by source preprocessing, heterogeneous
    training, and packet identity validation. It must remain non-empty, and every
    wrapper must match the backend hidden size recorded by ``compatibility``.
    """

    def __init__(
        self,
        *,
        compatibility: BackendCompatibility,
        wrappers: Mapping[str, WrapperEntry],
        meta: Mapping[str, object] | None = None,
    ) -> None:
        """Create and validate a wrapper registry.

        Args:
            compatibility: Exact backend model identity for the wrappers.
            wrappers: Initial mapping from stable wrapper IDs to entries.
            meta: JSON-compatible registry-level provenance.

        Raises:
            WrapperFormatError: If the registry is empty, an ID or entry is invalid,
                metadata is not JSON-compatible, or wrapper geometry is incompatible.
        """
        self.compatibility = compatibility
        self._wrappers = dict(wrappers)
        self.meta = _checked_metadata(meta, field_name="registry meta")
        self._validate()

    def _validate(self) -> None:
        """Validate registry membership, geometry, and tensor data types."""
        if not self._wrappers:
            raise WrapperFormatError("a wrapper registry must contain at least one wrapper")
        for wrapper_id, entry in self._wrappers.items():
            _validate_wrapper_id(wrapper_id)
            if not isinstance(entry, WrapperEntry):
                raise WrapperFormatError(f"wrapper {wrapper_id!r} is not a WrapperEntry")
            wrapper = entry.wrapper
            if wrapper.dim != self.compatibility.hidden_size:
                raise WrapperFormatError(
                    f"wrapper {wrapper_id!r} hidden size {wrapper.dim} does not match "
                    f"registry hidden size {self.compatibility.hidden_size}"
                )
            if (
                not wrapper.header.dtype.is_floating_point
                or not wrapper.trailer.dtype.is_floating_point
            ):
                raise WrapperFormatError(f"wrapper {wrapper_id!r} tensors must be floating point")

    @property
    def fingerprint(self) -> str:
        """Functional identity of the compatibility domain and all named wrappers.

        Wrapper IDs are sorted before hashing, so insertion order has no effect.
        Registry and entry metadata are deliberately excluded.
        """
        return fingerprint_json(
            {
                "compatibility": self.compatibility.fingerprint,
                "wrappers": {
                    wrapper_id: self._wrappers[wrapper_id].fingerprint
                    for wrapper_id in sorted(self._wrappers)
                },
            },
            domain=WRAPPER_REGISTRY_FORMAT,
        )

    def __getitem__(self, wrapper_id: str) -> WrapperEntry:
        """Retrieve an entry by wrapper ID."""
        return self._wrappers[wrapper_id]

    def __contains__(self, wrapper_id: str) -> bool:
        """Whether a wrapper ID is registered."""
        return wrapper_id in self._wrappers

    def __len__(self) -> int:
        """Number of registered wrappers."""
        return len(self._wrappers)

    def ids(self) -> list[str]:
        """Return wrapper IDs in insertion order."""
        return list(self._wrappers)

    def add(
        self,
        wrapper_id: str,
        wrapper: PacketWrapper,
        *,
        training_config: Mapping[str, object] | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> WrapperEntry:
        """Register a new wrapper in the compatibility domain.

        Args:
            wrapper_id: Stable ID used by source assignments and trainer state.
            wrapper: Adapter to register.
            training_config: JSON-compatible training settings for provenance.
            meta: JSON-compatible wrapper-level provenance.

        Returns:
            The newly registered entry.

        Raises:
            WrapperFormatError: If the ID already exists, the ID is invalid, metadata
                is not JSON-compatible, or the hidden size is incompatible.
        """
        _validate_wrapper_id(wrapper_id)
        if wrapper_id in self._wrappers:
            raise WrapperFormatError(f"wrapper_id {wrapper_id!r} already exists")
        entry = WrapperEntry(
            wrapper=wrapper,
            training_config=dict(training_config or {}),
            meta=dict(meta or {}),
        )
        if wrapper.dim != self.compatibility.hidden_size:
            raise WrapperFormatError(
                f"wrapper {wrapper_id!r} hidden size {wrapper.dim} does not match "
                f"registry hidden size {self.compatibility.hidden_size}"
            )
        self._wrappers[wrapper_id] = entry
        return entry

    def remove(self, wrapper_id: str) -> WrapperEntry:
        """Remove a wrapper while preserving the non-empty registry invariant.

        Args:
            wrapper_id: ID of the entry to remove.

        Returns:
            The removed entry.

        Raises:
            WrapperFormatError: If the ID is unknown or it is the registry's final
                wrapper.
        """
        if wrapper_id not in self._wrappers:
            raise WrapperFormatError(f"unknown wrapper_id {wrapper_id!r}")
        if len(self._wrappers) == 1:
            raise WrapperFormatError("a wrapper registry cannot be empty")
        return self._wrappers.pop(wrapper_id)

    def validate_compatibility(self, runtime: BackendCompatibility) -> None:
        """Require an exact functional and diagnostic match with a runtime backend.

        The exact model ID is checked in addition to its config fingerprint. The
        current model-config fingerprint describes architecture rather than model
        weights, so accepting a differently named checkpoint could otherwise reuse a
        weight-specific adapter with the wrong model.

        Args:
            runtime: Compatibility descriptor produced by the selected backend.

        Raises:
            BackendCompatibilityError: If any backend, model, or hidden-size identity
                field differs.
        """
        expected = self.compatibility
        comparisons = {
            "backend": (expected.backend, runtime.backend),
            "model_id": (expected.model_id, runtime.model_id),
            "model_config_fingerprint": (
                expected.model_config_fingerprint,
                runtime.model_config_fingerprint,
            ),
            "model_artifact_fingerprint": (
                expected.model_artifact_fingerprint,
                runtime.model_artifact_fingerprint,
            ),
            "hidden_size": (expected.hidden_size, runtime.hidden_size),
        }
        mismatches = [
            f"{name}: registry={left!r}, runtime={right!r}"
            for name, (left, right) in comparisons.items()
            if left != right
        ]
        if mismatches:
            raise BackendCompatibilityError(
                "wrapper registry is incompatible with the selected backend: "
                + "; ".join(mismatches)
            )

    def state_dict(self) -> dict[str, object]:
        """Build a complete self-validating registry checkpoint.

        Returns:
            Current-format registry state with CPU entry tensors, compatibility,
            fingerprints, and copied metadata.
        """
        return {
            "format": WRAPPER_REGISTRY_FORMAT,
            "compatibility": self.compatibility.to_dict(),
            "wrappers": {
                wrapper_id: entry.state_dict() for wrapper_id, entry in self._wrappers.items()
            },
            "fingerprint": self.fingerprint,
            "meta": deepcopy(self.meta),
        }

    @classmethod
    def from_state_dict(
        cls,
        checkpoint: object,
        *,
        device: torch.device | str = "cpu",
        requires_grad: bool = True,
    ) -> "WrapperRegistry":
        """Validate and reconstruct a registry checkpoint.

        Structure, tensor geometry, recorded lengths, wrapper fingerprints, and the
        aggregate registry fingerprint are verified before the registry is
        returned. Stored tensors are initially read on CPU by ``load`` and materialized
        on ``device`` here.

        Args:
            checkpoint: Untrusted serialized registry value to validate.
            device: Destination device for reconstructed wrapper tensors.
            requires_grad: Whether reconstructed wrapper tensors participate in
                autograd.

        Returns:
            Validated wrapper registry.

        Raises:
            WrapperFormatError: If any checkpoint field, wrapper, metadata value,
                recorded fingerprint, or geometry constraint is invalid.
        """
        data = string_mapping(
            checkpoint,
            where="wrapper registry checkpoint",
            error=WrapperFormatError,
        )
        expected_keys = {"format", "compatibility", "wrappers", "fingerprint", "meta"}
        unknown = set(data) - expected_keys
        missing = expected_keys - set(data)
        if unknown or missing:
            raise WrapperFormatError(
                f"invalid registry keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        format_id = data["format"]
        if format_id != WRAPPER_REGISTRY_FORMAT:
            raise WrapperFormatError(
                f"unrecognized registry format {data['format']!r}; "
                f"expected {WRAPPER_REGISTRY_FORMAT!r}"
            )
        try:
            compatibility = BackendCompatibility.from_dict(data["compatibility"])
        except (TypeError, ValueError) as error:
            raise WrapperFormatError(f"invalid registry compatibility: {error}") from error

        raw_wrappers = data["wrappers"]
        if not isinstance(raw_wrappers, dict):
            raise WrapperFormatError("wrappers must be a mapping")
        wrapper_items = cast(dict[object, object], raw_wrappers)

        wrappers: dict[str, WrapperEntry] = {}
        wrapper_keys = {
            "header",
            "trailer",
            "header_len",
            "trailer_len",
            "fingerprint",
            "training_config",
            "meta",
        }
        for wrapper_id, raw_value in wrapper_items.items():
            if not isinstance(wrapper_id, str):
                raise WrapperFormatError("wrapper IDs must be strings")
            _validate_wrapper_id(wrapper_id)
            raw = string_mapping(
                raw_value,
                where=f"wrapper {wrapper_id!r}",
                error=WrapperFormatError,
            )
            unknown_wrapper = set(raw) - wrapper_keys
            missing_wrapper = wrapper_keys - set(raw)
            if unknown_wrapper or missing_wrapper:
                raise WrapperFormatError(
                    f"invalid keys for wrapper {wrapper_id!r}: "
                    f"missing={sorted(missing_wrapper)}, unknown={sorted(unknown_wrapper)}"
                )
            header, trailer = raw["header"], raw["trailer"]
            if not isinstance(header, Tensor) or not isinstance(trailer, Tensor):
                raise WrapperFormatError(f"wrapper {wrapper_id!r} header/trailer must be tensors")
            if header.dim() != 3 or trailer.dim() != 3:
                raise WrapperFormatError(
                    f"wrapper {wrapper_id!r} header/trailer must be 3-D"
                )
            if header.size(0) != 1 or trailer.size(0) != 1:
                raise WrapperFormatError(
                    f"wrapper {wrapper_id!r} header/trailer must have batch size one"
                )
            if header.size(2) != trailer.size(2):
                raise WrapperFormatError(f"wrapper {wrapper_id!r} hidden sizes do not match")
            if header.dtype != trailer.dtype:
                raise WrapperFormatError(f"wrapper {wrapper_id!r} data types do not match")
            wrapper = PacketWrapper(
                header_len=header.size(1),
                trailer_len=trailer.size(1),
                dim=header.size(2),
                dtype=header.dtype,
                device=device,
                requires_grad=requires_grad,
            )
            wrapper.load_state_dict(
                {
                    "header": header.detach().clone().to(device),
                    "trailer": trailer.detach().clone().to(device),
                }
            )
            if raw["header_len"] != wrapper.header_len or raw["trailer_len"] != wrapper.trailer_len:
                raise WrapperFormatError(
                    f"wrapper {wrapper_id!r} recorded lengths do not match tensors"
                )
            training_config = string_mapping(
                raw["training_config"],
                where=f"wrapper {wrapper_id!r} training_config",
                error=WrapperFormatError,
            )
            wrapper_meta = string_mapping(
                raw["meta"],
                where=f"wrapper {wrapper_id!r} meta",
                error=WrapperFormatError,
            )
            entry = WrapperEntry(
                wrapper=wrapper,
                training_config=training_config,
                meta=wrapper_meta,
            )
            stored_fingerprint = raw["fingerprint"]
            if not is_fingerprint(stored_fingerprint) or stored_fingerprint != entry.fingerprint:
                raise WrapperFormatError(f"wrapper {wrapper_id!r} fingerprint mismatch")
            wrappers[wrapper_id] = entry

        registry_meta = string_mapping(
            data["meta"], where="registry meta", error=WrapperFormatError
        )
        registry = cls(
            compatibility=compatibility,
            wrappers=wrappers,
            meta=registry_meta,
        )
        stored_registry_fingerprint = data["fingerprint"]
        expected_fingerprint = registry.fingerprint
        if (
            not is_fingerprint(stored_registry_fingerprint)
            or stored_registry_fingerprint != expected_fingerprint
        ):
            raise WrapperFormatError("wrapper registry fingerprint mismatch")
        return registry

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save the registry with tensors offloaded to CPU.

        Args:
            path: Destination checkpoint path. Parent directories are created as
                needed by the atomic save helper.
        """
        atomic_torch_save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        device: torch.device | str = "cpu",
        requires_grad: bool = True,
        runtime: BackendCompatibility | None = None,
    ) -> "WrapperRegistry":
        """Load and optionally runtime-validate a registry checkpoint.

        Args:
            path: Registry checkpoint path.
            device: Destination device for wrapper tensors.
            requires_grad: Whether loaded wrapper tensors participate in autograd.
            runtime: Optional backend descriptor that must exactly match the registry.

        Returns:
            Validated wrapper registry.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            WrapperFormatError: If checkpoint validation fails.
            BackendCompatibilityError: If ``runtime`` is incompatible.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        registry = cls.from_state_dict(checkpoint, device=device, requires_grad=requires_grad)
        if runtime is not None:
            registry.validate_compatibility(runtime)
        return registry

    def __repr__(self) -> str:
        """Return a concise summary of registry size and backend identity."""
        return (
            f"WrapperRegistry(n={len(self)}, backend={self.compatibility.backend!r}, "
            f"model={self.compatibility.model_id!r})"
        )
