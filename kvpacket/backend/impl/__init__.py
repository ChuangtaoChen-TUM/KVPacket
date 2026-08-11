"""Concrete implementations of KV Packet's model-execution protocols.

Each module adapts a model framework while keeping its native cache representation
behind the framework-neutral backend boundary.
"""

from .hf import HFBackend

__all__ = ["HFBackend"]
