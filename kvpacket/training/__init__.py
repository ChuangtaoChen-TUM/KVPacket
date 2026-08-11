"""Train packet wrappers from full-attention teacher trajectories.

The package reconstructs the packet-isolated student attention topology from
``TrainingSample`` records, optimizes selected wrappers with KL or token-level
cross-entropy, and persists the wrapper registry together with optimizer state.
"""

from .initialization import InitializationStrategy, create_wrapper
from .mask import batch_attention_masks, packet_attention_mask
from .trainer import TRAINER_CHECKPOINT_FORMAT, PacketTrainer, TrainerMetrics

__all__ = [
    "TRAINER_CHECKPOINT_FORMAT",
    "InitializationStrategy",
    "PacketTrainer",
    "TrainerMetrics",
    "batch_attention_masks",
    "create_wrapper",
    "packet_attention_mask",
]
