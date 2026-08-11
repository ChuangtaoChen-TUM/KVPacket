"""The KV Packet adapter: trainable header/trailer soft tokens.

A ``PacketWrapper`` holds two small parameter tensors — a ``header`` prepended and
a ``trailer`` appended to each document's *embeddings* before its KV cache is
built. These are the only parameters that receive gradients during training.

Wrappers are regular ``torch.nn.Module`` instances. Complete persistence, including
backend identity and training metadata, is owned by ``WrapperRegistry`` and
``PacketTrainer`` rather than by a second wrapper-only checkpoint format.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

__all__ = ["PacketWrapper"]


class PacketWrapper(nn.Module):
    """Trainable soft-token adapter placed around a sequence of embeddings.

    The header and trailer are independent leaf tensors with shape ``(1, N, dim)``.
    They are expanded across the input batch when wrapping a sequence; the base model
    embeddings are not modified in place.
    """

    def __init__(
        self,
        header_len: int,
        trailer_len: int,
        dim: int,
        mean: float = 0.0,
        std: float = 0.02,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        requires_grad: bool = True,
    ) -> None:
        """Initialize header and trailer parameters from a normal distribution.

        Args:
            header_len: Number of learned tokens prepended to each sequence.
            trailer_len: Number of learned tokens appended to each sequence.
            dim: Embedding width expected by ``forward``.
            mean: Mean used to initialize both tensors.
            std: Standard deviation used to initialize both tensors.
            dtype: Storage data type for the adapter tensors.
            device: Initial storage device for the adapter tensors.
            requires_grad: Whether the adapter tensors participate in autograd.
        """
        super().__init__()
        self.dim = dim
        self.mean = mean
        self.std = std

        self.header = nn.Parameter(
            torch.empty((1, header_len, dim), dtype=dtype, device=device),
            requires_grad=requires_grad,
        )
        self.trailer = nn.Parameter(
            torch.empty((1, trailer_len, dim), dtype=dtype, device=device),
            requires_grad=requires_grad,
        )
        # Initialization changes parameter values but must not become part of a future
        # training graph.
        with torch.no_grad():
            torch.nn.init.normal_(self.header, mean=mean, std=std)
            torch.nn.init.normal_(self.trailer, mean=mean, std=std)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Place the learned header and trailer around an embedding sequence.

        A single sequence may omit the batch dimension. The returned tensor preserves
        the input rank and uses the input device and data type.

        Args:
            hidden_states: Embeddings shaped ``(S, dim)`` or ``(B, S, dim)``.

        Returns:
            Wrapped embeddings whose sequence length is increased by ``filler_len``.

        Raises:
            ValueError: If the input rank is unsupported or its embedding width does
                not equal ``dim``.
        """
        if hidden_states.dim() not in (2, 3):
            raise ValueError("hidden_states must be 2-D (S, dim) or 3-D (B, S, dim)")
        no_batch = hidden_states.dim() == 2
        if no_batch:
            hidden_states = hidden_states.unsqueeze(0)

        batch, _, dim = hidden_states.shape
        if dim != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {dim}")

        header = self.header.expand(batch, -1, -1).to(hidden_states.device, hidden_states.dtype)
        trailer = self.trailer.expand(batch, -1, -1).to(hidden_states.device, hidden_states.dtype)
        out = torch.cat([header, hidden_states, trailer], dim=1)
        return out.squeeze(0) if no_batch else out

    @property
    def header_len(self) -> int:
        """Number of learned header tokens."""
        return self.header.size(1)

    @property
    def trailer_len(self) -> int:
        """Number of learned trailer tokens."""
        return self.trailer.size(1)

    @property
    def filler_len(self) -> int:
        """Total number of learned tokens added by the wrapper."""
        return self.header_len + self.trailer_len

    @property
    def device(self) -> torch.device:
        """Storage device of the wrapper parameters."""
        return self.header.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage data type of the wrapper parameters."""
        return self.header.dtype

    def __repr__(self) -> str:
        """Return a concise representation of the adapter geometry and storage."""
        return (
            f"PacketWrapper(header_len={self.header_len}, trailer_len={self.trailer_len}, "
            f"dim={self.dim}, device={self.device}, dtype={self.dtype})"
        )
