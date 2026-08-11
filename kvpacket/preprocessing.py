"""Construct reusable KV packets from framework-owned source representations.

``PacketPreprocessor`` coordinates source encoding, wrapper selection, packet identity,
model prefill, and insertion into a ``PacketStore``. These responsibilities live
above compute backends so a new execution platform only needs to implement tensor and
cache operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import torch

from .backend import PacketBuildBackend
from .compression import CacheCompressor, CompressionSpec, NO_COMPRESSION_FINGERPRINT
from .errors import PacketIdentityError, SourceConfigError, SourceConflictError
from .fingerprint import fingerprint_bytes, fingerprint_tensor
from .packet.encoding import EncodedBody, SourceEncoder, encode_pretokenized
from .packet.registry import WrapperRegistry
from .packet.sources import Tokenizable
from .packet.wrapper import PacketWrapper
from .session.store import PacketStore, PacketIdentity, StoredPacket

__all__ = ["PacketPreprocessor"]

TokenizableT = TypeVar("TokenizableT", bound=Tokenizable)


class PacketPreprocessor:
    """Materialize reusable packets with wrappers selected from one registry.

    Args:
        backend: Model execution backend used for embedding and cache prefill.
        wrapper_registry: Named wrappers compatible with ``backend``.
        store: Destination packet store. A new empty store is created when omitted.
        source_encoder: Optional framework-side encoder for typed sources.
        cache_compressor: Optional offline cache-compression extension.

    Raises:
        TypeError: If an argument does not implement its required protocol.
        ValueError: If the store targets another backend.
        BackendCompatibilityError: If the registry targets another backend.
    """

    def __init__(
        self,
        *,
        backend: PacketBuildBackend,
        wrapper_registry: WrapperRegistry,
        store: PacketStore | None = None,
        source_encoder: SourceEncoder | None = None,
        cache_compressor: CacheCompressor | None = None,
    ) -> None:
        if not isinstance(backend, PacketBuildBackend):
            raise TypeError("backend must implement the PacketBuildBackend protocol")
        if not isinstance(wrapper_registry, WrapperRegistry):
            raise TypeError("wrapper_registry must be a WrapperRegistry")
        if source_encoder is not None and not isinstance(source_encoder, SourceEncoder):
            raise TypeError("source_encoder must implement the SourceEncoder protocol")
        if cache_compressor is not None and not isinstance(cache_compressor, CacheCompressor):
            raise TypeError("cache_compressor must implement the CacheCompressor protocol")
        destination = store if store is not None else PacketStore(backend=backend)
        if destination.backend.compatibility.fingerprint != backend.compatibility.fingerprint:
            raise ValueError("store and preprocessor must target the same backend")
        wrapper_registry.validate_compatibility(backend.compatibility)

        self.backend = backend
        self.store = destination
        self.source_encoder = source_encoder
        self.wrapper_registry = wrapper_registry
        self.cache_compressor = cache_compressor

    @property
    def compression_spec(self) -> CompressionSpec | None:
        """Active offline compression metadata, or ``None`` for full-size caches."""
        compressor = self.cache_compressor
        return None if compressor is None else compressor.spec

    def _identity(
        self,
        *,
        source_id: str,
        source_kind: str,
        encoded: EncodedBody,
        wrapper_id: str,
        wrapper_fingerprint: str,
    ) -> PacketIdentity:
        """Combine source, model, and wrapper identities for one packet."""
        return PacketIdentity(
            source_id=source_id,
            source_kind=source_kind,
            source_content_fingerprint=fingerprint_bytes(
                encoded.content, domain="kvpacket.source_content/1"
            ),
            backend_compatibility_fingerprint=self.backend.compatibility.fingerprint,
            wrapper_id=wrapper_id,
            wrapper_fingerprint=wrapper_fingerprint,
            encoded_input_fingerprint=fingerprint_tensor(encoded.encoded_input),
            cache_compression_fingerprint=(
                NO_COMPRESSION_FINGERPRINT
                if self.compression_spec is None
                else self.compression_spec.fingerprint
            ),
        )

    @torch.no_grad()
    def _materialize(
        self,
        encoded: EncodedBody,
        wrapper: PacketWrapper,
        identity: PacketIdentity,
    ) -> StoredPacket:
        """Insert wrapper embeddings and prefill one encoded source cache."""
        if encoded.input_ids is not None:
            embeddings = self.backend.embed(encoded.input_ids.to(self.backend.device))
        else:
            assert encoded.inputs_embeds is not None
            embeddings = encoded.inputs_embeds.to(self.backend.device)
        embeddings = wrapper(embeddings)
        compressor = self.cache_compressor
        if compressor is None:
            cache = self.backend.prefill_caches(inputs_embeds=embeddings)[0]
        else:
            with compressor.activate(self.backend):
                cache = self.backend.prefill_caches(inputs_embeds=embeddings)[0]
        wrapped_len = int(embeddings.shape[1])
        if cache.seq_len <= 0:
            raise ValueError(
                "cache compression removed every wrapped position; reduce compression_ratio"
            )
        if cache.seq_len > wrapped_len:
            raise ValueError("cache compression increased the physical sequence length")
        if cache.doc_len != wrapped_len:
            raise ValueError(
                "backend prefill did not preserve the full logical wrapped-document span"
            )
        return StoredPacket(
            chunk_id=identity.source_id,
            token_ids=None if encoded.token_ids is None else list(encoded.token_ids),
            match_text=encoded.match_text,
            packet=cache,
            body_len=encoded.body_len,
            wrapped_len=wrapped_len,
            identity=identity,
            compression=self.compression_spec,
        )

    def _reuse_or_materialize(
        self,
        *,
        encoded: EncodedBody,
        wrapper: PacketWrapper,
        identity: PacketIdentity,
        rebuild: bool,
    ) -> StoredPacket:
        """Reuse an identical packet or replace a stale packet when permitted."""
        existing = self.store.get(identity.source_id) if identity.source_id in self.store else None
        if existing is not None and existing.identity.cache_key == identity.cache_key:
            return existing
        if existing is not None and not rebuild:
            raise PacketIdentityError(
                f"packet {identity.source_id!r} is stale; pass rebuild=True to replace it"
            )
        packet = self._materialize(encoded, wrapper, identity)
        return self.store.put(packet, replace=existing is not None)

    def preprocess_pretokenized(
        self,
        source: Tokenizable,
        token_ids: list[int],
        *,
        wrapper_id: str,
        rebuild: bool = False,
    ) -> StoredPacket:
        """Build or reuse one packet from an original source and supplied token IDs.

        Args:
            source: Original raw-text or local-file source.
            token_ids: Non-empty tokenization of ``source``.
            wrapper_id: Named wrapper selected from the configured registry.
            rebuild: Whether an existing stale packet may be replaced.

        Returns:
            Existing identity-equivalent packet or a newly materialized packet.

        Raises:
            SourceConfigError: If ``wrapper_id`` is absent from the registry.
            PacketIdentityError: If a stale packet exists and rebuilding is disabled.
        """
        if not isinstance(source, Tokenizable):
            raise TypeError("source must be a Tokenizable")
        registry = self.wrapper_registry
        if registry is None:
            raise RuntimeError("preprocessor wrapper registry has been released")
        if wrapper_id not in registry:
            raise SourceConfigError(
                f"unknown wrapper ID {wrapper_id!r} for source {source.source_id!r}"
            )
        encoded = encode_pretokenized(source, token_ids)
        entry = registry[wrapper_id]
        identity = self._identity(
            source_id=source.source_id,
            source_kind=source.kind,
            encoded=encoded,
            wrapper_id=wrapper_id,
            wrapper_fingerprint=entry.fingerprint,
        )
        return self._reuse_or_materialize(
            encoded=encoded,
            wrapper=entry.wrapper,
            identity=identity,
            rebuild=rebuild,
        )

    @torch.no_grad()
    def preprocess(
        self,
        assignments: Mapping[TokenizableT, str],
        *,
        rebuild: bool = False,
    ) -> list[StoredPacket]:
        """Encode typed sources and materialize their selected named wrappers.

        All assignments are encoded and validated before any stale packet is replaced.
        Identical cache keys are reused; ``rebuild=True`` permits stale entries to be
        regenerated.

        Args:
            assignments: Sources mapped to wrapper IDs in the configured registry.
            rebuild: Whether stale packets may be replaced.

        Returns:
            Stored packets in assignment iteration order.

        Raises:
            RuntimeError: If the registry or source encoder is missing.
            SourceConfigError: If assignments are empty or reference unknown wrappers.
            SourceConflictError: If source IDs or matchable tokens conflict.
            PacketIdentityError: If a stale packet exists and rebuilding is disabled.
        """
        registry = self.wrapper_registry
        if registry is None:
            raise RuntimeError("typed preprocessing requires wrapper_registry=")
        encoder = self.source_encoder
        if encoder is None:
            raise RuntimeError("typed preprocessing requires source_encoder=")
        if not assignments:
            raise SourceConfigError("assignments must not be empty")
        registry.validate_compatibility(self.backend.compatibility)

        prepared: list[tuple[Tokenizable, EncodedBody, PacketIdentity, PacketWrapper]] = []
        seen_ids: set[str] = set()
        matchable: dict[tuple[int, ...], tuple[str, str]] = {}
        for source, wrapper_id in assignments.items():
            if not isinstance(source, Tokenizable):
                raise TypeError("assignment keys must be Tokenizable values")
            if source.source_id in seen_ids:
                raise SourceConflictError(f"duplicate source_id {source.source_id!r}")
            seen_ids.add(source.source_id)
            if wrapper_id not in registry:
                raise SourceConfigError(
                    f"unknown wrapper ID {wrapper_id!r} for source {source.source_id!r}"
                )
            entry = registry[wrapper_id]
            encoded = encoder.encode(source)
            if encoded.token_ids is not None:
                previous = matchable.get(encoded.token_ids)
                if previous is not None and previous[0] != wrapper_id:
                    raise SourceConflictError(
                        f"sources {previous[1]!r} and {source.source_id!r} encode to the "
                        "same matchable tokens but select different wrappers"
                    )
                matchable[encoded.token_ids] = (wrapper_id, source.source_id)

            identity = self._identity(
                source_id=source.source_id,
                source_kind=source.kind,
                encoded=encoded,
                wrapper_id=wrapper_id,
                wrapper_fingerprint=entry.fingerprint,
            )
            existing = self.store.get(source.source_id) if source.source_id in self.store else None
            if (
                existing is not None
                and existing.identity.cache_key != identity.cache_key
                and not rebuild
            ):
                raise PacketIdentityError(
                    f"packet {source.source_id!r} is stale; pass rebuild=True to replace it"
                )
            prepared.append((source, encoded, identity, entry.wrapper))

        results = [
            self._reuse_or_materialize(
                encoded=encoded,
                wrapper=wrapper,
                identity=identity,
                rebuild=rebuild,
            )
            for _source, encoded, identity, wrapper in prepared
        ]
        return results
