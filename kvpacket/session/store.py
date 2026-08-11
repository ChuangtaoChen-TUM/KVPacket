"""Identify, persist, and retrieve materialized reusable packet caches.

``PacketStore`` is the offline-to-online artifact boundary. Packet construction is
owned by ``PacketPreprocessor``; this module stores completed packets and rejects stale
or model-incompatible artifacts before serving.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import cast

import torch

from .._internal.persistence import atomic_torch_save
from .._internal.validation import string_mapping
from ..backend import ComputeBackend
from ..compression import CompressionSpec, NO_COMPRESSION_FINGERPRINT
from ..errors import PacketIdentityError
from ..fingerprint import fingerprint_json, is_fingerprint
from ..kvcache.cache import KeyValue, KVCache

__all__ = [
    "PACKET_STORE_FORMAT",
    "PacketIdentity",
    "StoredPacket",
    "PacketStore",
]

PACKET_STORE_FORMAT = "kvpacket.packet_store/1"


@dataclass(frozen=True)
class PacketIdentity:
    """Content, model, and wrapper identity governing packet reuse.


    Attributes:
        source_id: Logical source and store key.
        source_kind: Source representation used during preprocessing.
        source_content_fingerprint: Fingerprint of the source content.
        backend_compatibility_fingerprint: Fingerprint of model execution identity.
        wrapper_id: Selected wrapper name.
        wrapper_fingerprint: Fingerprint of the selected wrapper parameters.
        encoded_input_fingerprint: Fingerprint of the backend-encoded body.
        cache_compression_fingerprint: Fingerprint of offline compression or its
            explicit no-compression identity.
    """

    source_id: str
    source_kind: str
    source_content_fingerprint: str
    backend_compatibility_fingerprint: str
    wrapper_id: str
    wrapper_fingerprint: str
    encoded_input_fingerprint: str
    cache_compression_fingerprint: str = NO_COMPRESSION_FINGERPRINT

    def __post_init__(self) -> None:
        """Validate required identifiers and fingerprint syntax."""
        for name in ("source_id", "source_kind", "wrapper_id"):
            if not getattr(self, name):
                raise PacketIdentityError(f"{name} must be non-empty")
        for name in (
            "source_content_fingerprint",
            "backend_compatibility_fingerprint",
            "wrapper_fingerprint",
            "encoded_input_fingerprint",
            "cache_compression_fingerprint",
        ):
            if not is_fingerprint(getattr(self, name)):
                raise PacketIdentityError(f"{name} must be a sha256 fingerprint")

    @property
    def cache_key(self) -> str:
        """Functional identity of the source, backend, wrapper, and compression."""
        return fingerprint_json(
            {
                "source_content": self.source_content_fingerprint,
                "backend_compatibility": self.backend_compatibility_fingerprint,
                "wrapper_id": self.wrapper_id,
                "wrapper": self.wrapper_fingerprint,
                "encoded_input": self.encoded_input_fingerprint,
                "cache_compression": self.cache_compression_fingerprint,
            },
            domain="kvpacket.packet_identity/1",
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the identity into primitive values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "PacketIdentity":
        """Parse an identity from untrusted persisted data."""
        data = string_mapping(value, where="packet identity", error=PacketIdentityError)
        fields = {
            "source_id",
            "source_kind",
            "source_content_fingerprint",
            "backend_compatibility_fingerprint",
            "wrapper_id",
            "wrapper_fingerprint",
            "encoded_input_fingerprint",
            "cache_compression_fingerprint",
        }
        missing = fields - set(data)
        unknown = set(data) - fields
        if missing or unknown:
            raise PacketIdentityError(
                f"invalid packet identity keys: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if not all(isinstance(data[name], str) for name in fields):
            raise PacketIdentityError("all packet identity fields must be strings")
        return cls(**{name: cast(str, data[name]) for name in fields})


@dataclass
class StoredPacket:
    """Materialized packet cache and its matching and identity metadata.

    Attributes:
        chunk_id: Store key, equal to ``identity.source_id``.
        token_ids: Optional source-body tokens used by token matching.
        match_text: Optional source text used by string matching.
        packet: Cache of the wrapped source.
        body_len: Source length before wrapper insertion.
        wrapped_len: Original logical sequence span after wrapper insertion.
        identity: Fingerprints controlling reuse and compatibility.
        compression: Compression configuration, or ``None`` for a full-size cache.
    """

    chunk_id: str
    token_ids: list[int] | None
    match_text: str | None
    packet: KVCache
    body_len: int
    wrapped_len: int
    identity: PacketIdentity
    compression: CompressionSpec | None = None

    def __post_init__(self) -> None:
        """Validate packet lengths, matching metadata, and source identity."""
        if self.chunk_id != self.identity.source_id:
            raise PacketIdentityError("stored chunk_id does not match packet source_id")
        if self.body_len <= 0 or self.wrapped_len < self.body_len:
            raise PacketIdentityError("invalid stored packet lengths")
        if self.token_ids is not None and len(self.token_ids) != self.body_len:
            raise PacketIdentityError("stored token_ids length does not match body_len")
        if self.match_text is not None and not isinstance(self.match_text, str):
            raise PacketIdentityError("stored match_text must be a string or None")
        if self.packet.doc_len != self.wrapped_len:
            raise PacketIdentityError("stored cache positions do not match wrapped_len")
        if self.packet.seq_len <= 0 or self.packet.seq_len > self.wrapped_len:
            raise PacketIdentityError("invalid stored cache length")
        batch_size = self.packet.batch_size
        if batch_size != 1:
            raise PacketIdentityError("stored packets must contain exactly one source")
        if any(
            layer.key.shape[0] != batch_size
            or layer.value.shape[0] != batch_size
            or layer.logical_positions.shape != (batch_size, self.wrapped_len)
            or layer.key.shape[2] != self.packet.seq_len
            or layer.value.shape[2] != self.packet.seq_len
            for layer in self.packet.layers
        ):
            raise PacketIdentityError("stored cache layers disagree on physical or logical length")
        for layer in self.packet.layers:
            expected = torch.arange(
                self.wrapped_len,
                dtype=layer.logical_positions.dtype,
                device=layer.logical_positions.device,
            ).unsqueeze(0)
            if not torch.equal(layer.logical_positions, expected):
                raise PacketIdentityError("stored packet logical positions must start at zero")
        compression_fingerprint = (
            NO_COMPRESSION_FINGERPRINT if self.compression is None else self.compression.fingerprint
        )
        if self.identity.cache_compression_fingerprint != compression_fingerprint:
            raise PacketIdentityError("stored compression metadata does not match packet identity")

    @property
    def stored_len(self) -> int:
        """Number of physical key/value positions retained in every cache layer."""
        return self.packet.seq_len


class PacketStore:
    """In-memory store for packets materialized by a preprocessor.

    Args:
        backend: Runtime backend used for model-compatibility checks and load placement.

    Raises:
        TypeError: If ``backend`` does not implement ``ComputeBackend``.
    """

    def __init__(self, *, backend: ComputeBackend) -> None:
        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend must implement the ComputeBackend protocol")
        self.backend = backend
        self._store: dict[str, StoredPacket] = {}

    def put(self, packet: StoredPacket, *, replace: bool = False) -> StoredPacket:
        """Insert a completed packet after checking its model identity.

        Args:
            packet: Packet materialized by a framework preprocessor.
            replace: Whether an existing packet with the same source ID may be replaced.

        Returns:
            The inserted packet.

        Raises:
            PacketIdentityError: If the packet targets another backend or conflicts with
                an existing source ID.
        """
        self._validate_runtime_identity(packet.identity)
        existing = self._store.get(packet.chunk_id)
        if existing is not None and not replace:
            raise PacketIdentityError(f"store already contains packet {packet.chunk_id!r}")
        self._store[packet.chunk_id] = packet
        return packet

    def get(self, chunk_id: str) -> StoredPacket:
        """Fetch a stored packet by source ID."""
        return self._store[chunk_id]

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._store

    def __len__(self) -> int:
        return len(self._store)

    def ids(self) -> list[str]:
        """Stored source IDs in insertion order."""
        return list(self._store)

    def _validate_runtime_identity(self, identity: PacketIdentity) -> None:
        """Reject a persisted identity incompatible with current runtime state."""
        runtime_fingerprint = self.backend.compatibility.fingerprint
        if identity.backend_compatibility_fingerprint != runtime_fingerprint:
            raise PacketIdentityError(
                f"packet {identity.source_id!r} was built for a different backend/model"
            )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save every packet and CPU-resident cache tensor."""
        blob: dict[str, object] = {
            "format": PACKET_STORE_FORMAT,
            "packets": {
                chunk_id: {
                    "token_ids": packet.token_ids,
                    "match_text": packet.match_text,
                    "body_len": packet.body_len,
                    "wrapped_len": packet.wrapped_len,
                    "stored_len": packet.stored_len,
                    "compression": (
                        None if packet.compression is None else packet.compression.to_dict()
                    ),
                    "identity": packet.identity.to_dict(),
                    "cache_key": packet.identity.cache_key,
                    "layers": [
                        (layer.key.cpu(), layer.value.cpu(), layer.logical_positions.cpu())
                        for layer in packet.packet.layers
                    ],
                }
                for chunk_id, packet in self._store.items()
            },
        }
        atomic_torch_save(blob, path)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        backend: ComputeBackend,
        device: torch.device | str | None = None,
    ) -> "PacketStore":
        """Load a validated packet-store checkpoint.

        Args:
            path: Versioned packet-store checkpoint.
            backend: Runtime backend used for compatibility checks.
            device: Cache destination, defaulting to the backend device.

        Returns:
            A new store containing all persisted packets.

        Raises:
            PacketIdentityError: If the checkpoint schema, tensors, identities, runtime
                compatibility, or source IDs are invalid.
        """
        store = cls(backend=backend)
        dev = torch.device(device) if device is not None else backend.device
        raw_blob = torch.load(path, map_location="cpu", weights_only=True)
        blob = string_mapping(raw_blob, where="packet store checkpoint", error=PacketIdentityError)
        if blob.get("format") != PACKET_STORE_FORMAT:
            raise PacketIdentityError(
                f"unrecognized or legacy packet store format {blob.get('format')!r}; "
                "rebuild packets with the current preprocessor"
            )
        if set(blob) != {"format", "packets"}:
            raise PacketIdentityError("packet store checkpoint has invalid top-level keys")
        raw_packets = blob["packets"]
        if not isinstance(raw_packets, dict):
            raise PacketIdentityError("packet store packets must be a mapping")

        loaded: dict[str, StoredPacket] = {}
        for raw_chunk_id, raw_packet in cast(dict[object, object], raw_packets).items():
            if not isinstance(raw_chunk_id, str):
                raise PacketIdentityError("packet IDs must be strings")
            packet_data = string_mapping(
                raw_packet,
                where=f"packet {raw_chunk_id!r}",
                error=PacketIdentityError,
            )
            required = {
                "token_ids",
                "match_text",
                "body_len",
                "wrapped_len",
                "stored_len",
                "compression",
                "identity",
                "cache_key",
                "layers",
            }
            if set(packet_data) != required:
                raise PacketIdentityError(f"packet {raw_chunk_id!r} has invalid keys")
            identity = PacketIdentity.from_dict(packet_data["identity"])
            if packet_data["cache_key"] != identity.cache_key:
                raise PacketIdentityError(f"packet {raw_chunk_id!r} cache identity mismatch")
            if identity.source_id != raw_chunk_id:
                raise PacketIdentityError(
                    f"packet mapping key {raw_chunk_id!r} does not match source identity"
                )
            store._validate_runtime_identity(identity)
            raw_layers = packet_data["layers"]
            if not isinstance(raw_layers, list) or not raw_layers:
                raise PacketIdentityError(f"packet {raw_chunk_id!r} layers must be non-empty")
            layers: list[KeyValue] = []
            for raw_layer in cast(list[object], raw_layers):
                if not isinstance(raw_layer, (tuple, list)) or len(raw_layer) != 3:
                    raise PacketIdentityError(f"packet {raw_chunk_id!r} has malformed layer")
                key, value, positions = raw_layer
                if not all(isinstance(item, torch.Tensor) for item in raw_layer):
                    raise PacketIdentityError(
                        f"packet {raw_chunk_id!r} layer values must be tensors"
                    )
                layers.append(
                    KeyValue(
                        cast(torch.Tensor, key).to(dev),
                        cast(torch.Tensor, value).to(dev),
                        cast(torch.Tensor, positions).to(dev),
                    )
                )
            raw_token_ids = packet_data["token_ids"]
            if raw_token_ids is not None and not (
                isinstance(raw_token_ids, list)
                and all(
                    isinstance(token, int) and not isinstance(token, bool)
                    for token in raw_token_ids
                )
            ):
                raise PacketIdentityError(f"packet {raw_chunk_id!r} token_ids are malformed")
            match_text = packet_data["match_text"]
            if match_text is not None and not isinstance(match_text, str):
                raise PacketIdentityError(f"packet {raw_chunk_id!r} match_text is malformed")
            body_len = packet_data["body_len"]
            wrapped_len = packet_data["wrapped_len"]
            stored_len = packet_data["stored_len"]
            if not all(
                isinstance(length, int) and not isinstance(length, bool)
                for length in (body_len, wrapped_len, stored_len)
            ):
                raise PacketIdentityError(f"packet {raw_chunk_id!r} lengths must be integers")
            body_len = cast(int, body_len)
            wrapped_len = cast(int, wrapped_len)
            stored_len = cast(int, stored_len)
            packet_cache = KVCache(layers)
            if packet_cache.seq_len != stored_len:
                raise PacketIdentityError(
                    f"packet {raw_chunk_id!r} stored_len does not match cache tensors"
                )
            raw_compression = packet_data["compression"]
            try:
                compression = (
                    None if raw_compression is None else CompressionSpec.from_dict(raw_compression)
                )
            except ValueError as exc:
                raise PacketIdentityError(
                    f"packet {raw_chunk_id!r} compression metadata is malformed"
                ) from exc
            loaded[raw_chunk_id] = StoredPacket(
                chunk_id=raw_chunk_id,
                token_ids=cast(list[int] | None, raw_token_ids),
                match_text=match_text,
                packet=packet_cache,
                body_len=body_len,
                wrapped_len=wrapped_len,
                identity=identity,
                compression=compression,
            )

        store._store.update(loaded)
        return store

    def __repr__(self) -> str:
        return f"PacketStore(n={len(self._store)})"
