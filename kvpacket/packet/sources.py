"""Declare reusable content and assign its adapter before packet construction.

KV Packet's offline preprocessing needs two pieces of information for every reusable
context block: what content the framework should encode and which ``PacketWrapper``
should surround that content.

This module represents the content as an immutable ``Tokenizable`` and loads mappings
of ``{Tokenizable: wrapper_id}`` from YAML/JSON configuration and JSONL manifests.

``PacketPreprocessor.preprocess`` consumes that mapping. A framework-side source encoder
resolves each source before the selected wrapper and compute backend materialize its KV
cache. This module only declares and validates preprocessing inputs; it does not encode
content or construct caches itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import torch
import yaml

from .._internal.validation import string_mapping
from ..errors import SourceConfigError, SourceConflictError
from ..fingerprint import fingerprint_json
from .registry import WrapperRegistry

__all__ = [
    "SourceKind",
    "Tokenizable",
    "RawText",
    "FileSource",
    "WrapperAssignmentConfig",
    "load_wrapper_assignments",
    "load_wrapper_registry_and_assignments",
]

SourceKind = str
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _default_source_id(kind: SourceKind, value: str) -> str:
    """Derive a stable source ID from a source's kind and exact supplied value."""
    digest = fingerprint_json(
        {"kind": kind, "value": value}, domain="kvpacket.tokenizable_identity/1"
    ).removeprefix("sha256:")
    if kind == "raw-text":
        prefix = "text"
    elif kind == "file":
        prefix = "file"
    else:
        raise SourceConfigError(f"unsupported source kind {kind!r}")
    return f"{prefix}-{digest[:24]}"


@dataclass(frozen=True, slots=True, eq=False)
class Tokenizable:
    """Immutable logical source that can be assigned to a named wrapper.

    Identity includes the source kind, logical ID, value, and media type. Including the
    kind prevents equal strings representing literal text and file locations from
    colliding in assignment mappings.

    Attributes:
        kind: How preprocessing should interpret ``value``.
        source_id: Stable logical identifier used for conflict detection and storage.
        value: Literal text, local path, or URI. Config loaders normalize file values.
        media_type: Optional content type supplied to the preprocessing backend.
    """

    kind: SourceKind
    source_id: str
    value: str
    media_type: str | None = None

    def __eq__(self, other: object) -> bool:
        """Compare source identity across ``Tokenizable`` subclasses."""
        if not isinstance(other, Tokenizable):
            return NotImplemented
        return (
            self.kind,
            self.source_id,
            self.value,
            self.media_type,
        ) == (
            other.kind,
            other.source_id,
            other.value,
            other.media_type,
        )

    def __hash__(self) -> int:
        """Hash the complete immutable source identity."""
        return hash((self.kind, self.source_id, self.value, self.media_type))

    def __post_init__(self) -> None:
        """Validate fields shared by all source kinds.

        Raises:
            TypeError: If ``value`` is not text.
            ValueError: If the source kind, ID, or media type is invalid.
        """
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("source kind must be a non-empty string")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.value, str):
            raise TypeError("value must be a string")
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type
        ):
            raise ValueError("media_type must be a non-empty string or None")


class RawText(Tokenizable):
    """Literal text that is never interpreted as a filesystem path or URI."""

    def __init__(
        self,
        value: str,
        source_id: str | None = None,
        media_type: str = "text/plain",
    ) -> None:
        """Create a literal-text source.

        Args:
            value: Exact text passed to the preprocessing backend.
            source_id: Logical identifier. By default, derive one from ``value``.
            media_type: Content type associated with the text.
        """
        super().__init__(
            kind="raw-text",
            source_id=source_id or _default_source_id("raw-text", value),
            value=value,
            media_type=media_type,
        )


class FileSource(Tokenizable):
    """Local file or URI whose content is resolved during preprocessing."""

    def __init__(
        self,
        value: str,
        source_id: str | None = None,
        media_type: str | None = None,
    ) -> None:
        """Create a file-backed source.

        Args:
            value: Local path or URI consumed by the preprocessing backend.
            source_id: Logical identifier. By default, derive one from ``value``.
            media_type: Optional content type override.
        """
        super().__init__(
            kind="file",
            source_id=source_id or _default_source_id("file", value),
            value=value,
            media_type=media_type,
        )


@dataclass
class WrapperAssignmentConfig:
    """Resolved wrapper registry path and expanded source assignments.

    Attributes:
        registry_path: Absolute local path to the wrapper registry checkpoint.
        assignments: Mutable mapping from validated sources to wrapper IDs.
    """

    registry_path: Path
    assignments: dict[Tokenizable, str]


def _reject_duplicate_yaml_keys(
    loader: yaml.SafeLoader,
    node: object,
    *,
    deep: bool,
) -> None:
    """Reject duplicate or unhashable keys in one YAML mapping node."""
    if not isinstance(node, yaml.MappingNode):
        return

    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as error:
            raise SourceConfigError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise SourceConfigError(f"duplicate YAML mapping key {key!r}")
        seen.add(key)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of overwriting them."""

    def construct_mapping(self, node, deep: bool = False):
        """Construct a mapping after checking that each YAML key is unique."""
        _reject_duplicate_yaml_keys(self, node, deep=deep)
        return super().construct_mapping(node, deep=deep)


def _exact_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    where: str,
) -> None:
    """Require a mapping to contain exactly the supported keys.

    Args:
        value: Mapping being validated.
        required: Keys that must be present.
        optional: Additional keys that may be present.
        where: Human-readable location included in validation errors.

    Raises:
        SourceConfigError: If required keys are missing or unknown keys are present.
    """
    optional_keys = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional_keys
    if missing or unknown:
        raise SourceConfigError(
            f"{where} has invalid keys: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _non_empty_string(value: object, *, where: str) -> str:
    """Validate and return a non-empty configuration string."""
    if not isinstance(value, str) or not value:
        raise SourceConfigError(f"{where} must be a non-empty string")
    return value


def _is_uri(value: str) -> bool:
    """Whether a value begins with an RFC-style URI scheme."""
    return _URI_RE.match(value) is not None


def _normalize_uri(value: str) -> str:
    """Normalize the case-insensitive scheme and authority of a URI."""
    parts = urlsplit(value)
    # Paths, queries, and fragments may be case-sensitive and must remain untouched.
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, parts.fragment)
    )


def _local_path(value: str, *, base: Path, where: str, must_exist: bool = True) -> Path:
    """Resolve a local file path relative to its declaring file.

    Args:
        value: Absolute path or path relative to ``base``.
        base: Directory containing the config or manifest that declared the path.
        where: Human-readable location included in validation errors.
        must_exist: Whether the resolved path must identify an existing regular file.

    Returns:
        The normalized absolute path.

    Raises:
        SourceConfigError: If ``value`` is a URI or the required file does not exist.
    """
    if _is_uri(value):
        raise SourceConfigError(f"{where} must be a local path, not a URI")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if must_exist and not path.is_file():
        raise SourceConfigError(f"{where} does not exist or is not a file: {path}")
    return path


def _file_value(value: str, *, base: Path, where: str) -> str:
    """Normalize a file declaration as either a URI or an absolute local path."""
    if _is_uri(value):
        return _normalize_uri(value)
    return str(_local_path(value, base=base, where=where))


def _media_type(value: object, *, where: str, default: str | None) -> str | None:
    """Validate an optional media type, applying a source-kind default."""
    if value is None:
        return default
    return _non_empty_string(value, where=where)


def _text_record(value: object, *, where: str, strict: bool) -> RawText:
    """Parse one literal-text declaration.

    Non-strict inline declarations may use a bare string and receive a deterministic
    source ID. Strict manifest records must explicitly provide ``id`` and ``text``.

    Args:
        value: Bare text or a structured source record.
        where: Human-readable location included in validation errors.
        strict: Whether the shorthand bare-string form is forbidden.

    Returns:
        A validated literal-text source.

    Raises:
        SourceConfigError: If the record shape or any field is invalid.
    """
    if isinstance(value, str) and not strict:
        return RawText(value)
    record = string_mapping(value, where=where, error=SourceConfigError)
    _exact_keys(
        record,
        required={"id", "text"},
        optional={"media_type"},
        where=where,
    )
    source_id = _non_empty_string(record["id"], where=f"{where}.id")
    text = record["text"]
    if not isinstance(text, str):
        raise SourceConfigError(f"{where}.text must be a string")
    media_type = _media_type(
        record.get("media_type"), where=f"{where}.media_type", default="text/plain"
    )
    assert media_type is not None
    return RawText(text, source_id=source_id, media_type=media_type)


def _file_record(value: object, *, base: Path, where: str, strict: bool) -> FileSource:
    """Parse one file or URI declaration.

    Non-strict inline declarations may use a bare path or URI and receive a
    deterministic source ID. Strict manifest records must explicitly provide ``id``
    and ``path``.

    Args:
        value: Bare location or a structured source record.
        base: Directory against which local relative paths are resolved.
        where: Human-readable location included in validation errors.
        strict: Whether the shorthand bare-string form is forbidden.

    Returns:
        A validated file-backed source.

    Raises:
        SourceConfigError: If the record or referenced local file is invalid.
    """
    if isinstance(value, str) and not strict:
        normalized = _file_value(value, base=base, where=where)
        return FileSource(normalized)
    record = string_mapping(value, where=where, error=SourceConfigError)
    _exact_keys(
        record,
        required={"id", "path"},
        optional={"media_type"},
        where=where,
    )
    source_id = _non_empty_string(record["id"], where=f"{where}.id")
    raw_path = _non_empty_string(record["path"], where=f"{where}.path")
    media_type = _media_type(record.get("media_type"), where=f"{where}.media_type", default=None)
    return FileSource(
        _file_value(raw_path, base=base, where=f"{where}.path"),
        source_id=source_id,
        media_type=media_type,
    )


def _manifest_records(path: Path, *, kind: SourceKind) -> list[Tokenizable]:
    """Load strict source records from a non-empty JSONL manifest.

    Local paths inside file manifests are interpreted relative to the manifest rather
    than the top-level assignment config.

    Args:
        path: Local JSONL manifest path.
        kind: Source kind expected for every record.

    Returns:
        Parsed sources in manifest order, excluding blank lines.

    Raises:
        SourceConfigError: If the manifest cannot be read, contains malformed records,
            or contains no sources.
    """
    records: list[Tokenizable] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SourceConfigError(f"cannot read manifest {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise SourceConfigError(f"invalid JSON at {where}: {error.msg}") from error
        if kind == "raw-text":
            records.append(_text_record(raw, where=where, strict=True))
        elif kind == "file":
            records.append(_file_record(raw, base=path.parent, where=where, strict=True))
        else:
            raise SourceConfigError(f"unsupported source kind {kind!r} at {where}")
    if not records:
        raise SourceConfigError(f"manifest contains no source records: {path}")
    return records


def _read_config(path: Path) -> dict[str, object]:
    """Read and validate the top level of a YAML or JSON assignment config.

    Args:
        path: Local configuration file.

    Returns:
        A mapping containing ``registry_path`` and ``assignments``.

    Raises:
        SourceConfigError: If the file is unreadable, malformed, or has invalid keys.
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except OSError as error:
        raise SourceConfigError(f"cannot read assignment config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise SourceConfigError(f"invalid YAML in assignment config {path}: {error}") from error
    config = string_mapping(raw, where=f"assignment config {path}", error=SourceConfigError)
    _exact_keys(
        config, required={"registry_path", "assignments"}, where=f"assignment config {path}"
    )
    return config


def _assignment_groups(config: dict[str, object]) -> dict[str, object]:
    """Validate and return the configured wrapper assignment groups."""
    groups = string_mapping(config["assignments"], where="assignments", error=SourceConfigError)
    if not groups:
        raise SourceConfigError("assignments must contain at least one wrapper group")
    return groups


def _wrapper_declarations(
    wrapper_id: str,
    value: object,
    registry: WrapperRegistry,
) -> list[object]:
    """Validate one wrapper ID and its non-empty declaration list."""
    if wrapper_id not in registry:
        raise SourceConfigError(f"unknown wrapper ID {wrapper_id!r} in assignments")
    if not isinstance(value, list) or not value:
        raise SourceConfigError(f"assignments.{wrapper_id} must be a non-empty list")
    return cast(list[object], value)


def _declaration_items(value: object, *, where: str) -> tuple[str, list[object]]:
    """Validate one source declaration and return its type and document items."""
    declaration = string_mapping(value, where=where, error=SourceConfigError)
    _exact_keys(declaration, required={"type", "doc"}, where=where)
    source_type = _non_empty_string(declaration["type"], where=f"{where}.type")
    docs = declaration["doc"]
    if not isinstance(docs, list) or not docs:
        raise SourceConfigError(f"{where}.doc must be a non-empty list")
    return source_type, cast(list[object], docs)


def _inline_sources(
    items: list[object],
    *,
    kind: SourceKind,
    base: Path,
    where: str,
) -> Iterator[tuple[Tokenizable, str]]:
    """Parse inline source items while retaining their config locations."""
    for index, item in enumerate(items):
        item_where = f"{where}.doc[{index}]"
        if kind == "raw-text":
            source = _text_record(item, where=item_where, strict=False)
        elif kind == "file":
            source = _file_record(item, base=base, where=item_where, strict=False)
        else:
            raise SourceConfigError(f"unsupported source kind {kind!r} at {item_where}")
        yield source, item_where


def _manifest_sources(
    items: list[object],
    *,
    kind: SourceKind,
    base: Path,
    where: str,
) -> Iterator[tuple[Tokenizable, str]]:
    """Expand manifest declarations while retaining each manifest location."""
    for index, item in enumerate(items):
        item_where = f"{where}.doc[{index}]"
        manifest_value = _non_empty_string(item, where=item_where)
        manifest_path = _local_path(manifest_value, base=base, where=item_where)
        location = str(manifest_path)
        yield from ((source, location) for source in _manifest_records(manifest_path, kind=kind))


def _declaration_sources(
    value: object,
    *,
    base: Path,
    where: str,
) -> Iterator[tuple[Tokenizable, str]]:
    """Expand one inline or manifest-backed source declaration."""
    source_type, items = _declaration_items(value, where=where)
    if source_type == "raw-text":
        yield from _inline_sources(items, kind="raw-text", base=base, where=where)
        return
    if source_type == "file":
        yield from _inline_sources(items, kind="file", base=base, where=where)
        return
    if source_type in ("text-manifest", "file-manifest"):
        manifest_kind: SourceKind = "raw-text" if source_type == "text-manifest" else "file"
        yield from _manifest_sources(items, kind=manifest_kind, base=base, where=where)
        return
    raise SourceConfigError(
        f"unknown source type {source_type!r} at {where}; expected raw-text, "
        "file, text-manifest, or file-manifest"
    )


def _add_sources(
    sources: Iterator[tuple[Tokenizable, str]],
    wrapper_id: str,
    *,
    assignments: dict[Tokenizable, str],
    seen_ids: dict[str, Tokenizable],
) -> None:
    """Add expanded sources after enforcing global source uniqueness."""
    for source, where in sources:
        previous = seen_ids.get(source.source_id)
        if previous is not None:
            raise SourceConflictError(f"duplicate source_id {source.source_id!r} at {where}")
        if source in assignments:
            raise SourceConflictError(
                f"source {source.source_id!r} is assigned more than once at {where}"
            )
        seen_ids[source.source_id] = source
        assignments[source] = wrapper_id


def _expand_wrapper_group(
    wrapper_id: str,
    value: object,
    *,
    config_base: Path,
    registry: WrapperRegistry,
    assignments: dict[Tokenizable, str],
    seen_ids: dict[str, Tokenizable],
) -> None:
    """Expand every declaration assigned to one wrapper."""
    declarations = _wrapper_declarations(wrapper_id, value, registry)
    for index, declaration in enumerate(declarations):
        where = f"assignments.{wrapper_id}[{index}]"
        _add_sources(
            _declaration_sources(declaration, base=config_base, where=where),
            wrapper_id,
            assignments=assignments,
            seen_ids=seen_ids,
        )


def _expand_config(
    config_path: Path,
    config: dict[str, object],
    registry: WrapperRegistry,
) -> WrapperAssignmentConfig:
    """Expand validated assignment declarations against a wrapper registry.

    All wrapper IDs, source records, manifests, local paths, and uniqueness constraints
    are checked before returning the assignment mapping.

    Args:
        config_path: Path used to resolve relative declarations.
        config: Parsed top-level assignment configuration.
        registry: Registry whose wrapper IDs may receive sources.

    Returns:
        The resolved registry path and mutable source assignment mapping.

    Raises:
        SourceConfigError: If a declaration is malformed or references an unknown
            wrapper or missing file.
        SourceConflictError: If a logical source ID or full source is assigned more
            than once.
    """
    wrapper_value = _non_empty_string(config["registry_path"], where="registry_path")
    registry_path = _local_path(wrapper_value, base=config_path.parent, where="registry_path")
    assignments: dict[Tokenizable, str] = {}
    # Full source equality includes the value and media type, so track logical IDs
    # separately to reject two different declarations claiming the same storage ID.
    seen_ids: dict[str, Tokenizable] = {}

    for wrapper_id, value in _assignment_groups(config).items():
        _expand_wrapper_group(
            wrapper_id,
            value,
            config_base=config_path.parent,
            registry=registry,
            assignments=assignments,
            seen_ids=seen_ids,
        )

    return WrapperAssignmentConfig(registry_path=registry_path, assignments=assignments)


def load_wrapper_assignments(
    config_path: str | Path,
    registry: WrapperRegistry,
) -> dict[Tokenizable, str]:
    """Expand source assignments against an already-loaded wrapper registry.

    Relative paths are resolved from the config file, except paths inside manifests,
    which are resolved from their manifest. The returned ordinary dictionary may be
    edited before packet preprocessing.

    Args:
        config_path: YAML or JSON assignment configuration path.
        registry: Loaded registry used to validate wrapper IDs.

    Returns:
        Mutable mapping from validated sources to wrapper IDs.

    Raises:
        SourceConfigError: If the config, a manifest, a source, or a local path is
            invalid.
        SourceConflictError: If source declarations conflict.
    """
    path = Path(config_path).resolve()
    return _expand_config(path, _read_config(path), registry).assignments


def load_wrapper_registry_and_assignments(
    config_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> tuple[WrapperRegistry, dict[Tokenizable, str]]:
    """Load a wrapper registry and its expanded source assignments.

    Args:
        config_path: YAML or JSON assignment configuration path.
        device: Device on which to materialize registry tensors.
        requires_grad: Whether loaded wrapper tensors participate in autograd.

    Returns:
        The validated registry and mutable source-to-wrapper mapping.

    Raises:
        SourceConfigError: If the config, registry path, manifest, source, or local
            path is invalid.
        SourceConflictError: If source declarations conflict.
    """
    path = Path(config_path).resolve()
    config = _read_config(path)
    wrapper_value = _non_empty_string(config["registry_path"], where="registry_path")
    registry_path = _local_path(wrapper_value, base=path.parent, where="registry_path")
    registry = WrapperRegistry.load(registry_path, device=device, requires_grad=requires_grad)
    loaded = _expand_config(path, config, registry)
    return registry, loaded.assignments
