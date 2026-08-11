"""Public model-execution protocols, compatibility metadata, and implementations."""

from .base import (
    BackendCompatibility,
    BackendGenerationOutput,
    CacheStore,
    ComputeBackend,
    GenerationBackend,
    PacketBuildBackend,
    TrainingBackend,
    ForwardOutput,
    Packet,
    RerotationError,
)
from .impl.hf import HFBackend

__all__ = [
    "ComputeBackend",
    "TrainingBackend",
    "BackendCompatibility",
    "BackendGenerationOutput",
    "GenerationBackend",
    "PacketBuildBackend",
    "CacheStore",
    "ForwardOutput",
    "Packet",
    "RerotationError",
    "HFBackend",
]
