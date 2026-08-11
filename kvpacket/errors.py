"""Public kvpacket validation errors."""

__all__ = [
    "WrapperFormatError",
    "BackendCompatibilityError",
    "SourceConfigError",
    "SourceConflictError",
    "UnsupportedModalityError",
    "PacketIdentityError",
]


class WrapperFormatError(ValueError):
    """A wrapper or wrapper-registry checkpoint is malformed."""


class BackendCompatibilityError(ValueError):
    """A wrapper registry is incompatible with the selected backend/model."""


class SourceConfigError(ValueError):
    """A source assignment config or manifest is malformed."""


class SourceConflictError(ValueError):
    """Expanded sources contain a duplicate or conflicting assignment."""


class UnsupportedModalityError(ValueError):
    """The selected backend cannot encode a source's media type."""


class PacketIdentityError(ValueError):
    """A persisted packet is stale or incompatible with the current runtime."""
