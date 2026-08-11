"""KV Packet's concise end-to-end public API.

Extension protocols, persistence records, and low-level result types remain available
from their focused subpackages. The package root contains the common path from model
setup through packet construction, training, and serving.
"""

from .backend import HFBackend
from .compression import KVPressCompressor
from .packet import FileSource, PacketWrapper, RawText, TextSourceEncoder, WrapperRegistry
from .preprocessing import PacketPreprocessor
from .runtime import MixedPrefillPolicy, PacketCacheState, PacketExecutor
from .session import GenerationResult, Mode, PacketSession, PacketStore, TrainingSample
from .training import PacketTrainer, create_wrapper

__all__ = [
    "HFBackend",
    "PacketWrapper",
    "WrapperRegistry",
    "RawText",
    "FileSource",
    "TextSourceEncoder",
    "PacketPreprocessor",
    "PacketStore",
    "PacketExecutor",
    "PacketCacheState",
    "MixedPrefillPolicy",
    "PacketSession",
    "Mode",
    "GenerationResult",
    "TrainingSample",
    "PacketTrainer",
    "create_wrapper",
    "KVPressCompressor",
]

__version__ = "0.1.0"
