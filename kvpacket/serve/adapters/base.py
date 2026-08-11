"""Define the transport-only contract shared by external API adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ..types import InferenceRequest, ModelResponse, TokenUsage

__all__ = ["ProviderAdapter", "WireFormatError"]


class WireFormatError(ValueError):
    """An external request cannot be represented by the normalized model API."""


@runtime_checkable
class ProviderAdapter(Protocol):
    """Translate one provider's dictionaries without performing model work."""

    def parse_request(self, payload: Mapping[str, object]) -> InferenceRequest:
        """Normalize one provider request."""
        ...

    def format_response(
        self,
        response: ModelResponse,
        *,
        response_id: str,
        model: str,
        usage: TokenUsage = TokenUsage(),
    ) -> dict[str, object]:
        """Format one completed normalized response."""
        ...
