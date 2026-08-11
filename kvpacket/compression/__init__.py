"""Optional cache-compression interfaces and KVPress integration."""

from .base import CacheCompressor, CompressionSpec, NO_COMPRESSION_FINGERPRINT
from .kvpress import KVPressCompressor, SUPPORTED_KVPRESS_METHODS

__all__ = [
    "NO_COMPRESSION_FINGERPRINT",
    "CacheCompressor",
    "CompressionSpec",
    "KVPressCompressor",
    "SUPPORTED_KVPRESS_METHODS",
]
