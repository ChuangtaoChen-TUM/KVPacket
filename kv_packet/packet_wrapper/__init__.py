""" Packet Wrapper for handling header and trailer tokens """
import torch
from typing import TypedDict

__all__ = [
    "WrapperStateDict",
    "PacketWrapper",
    "load_wrapper",
]

class WrapperStateDict(TypedDict):
    header: torch.Tensor
    trailer: torch.Tensor
    train_config: dict


class PacketWrapper:
    def __init__(
        self,
        header_len: int,
        trailer_len: int,
        dim: int,
        mean: float = 0.0,
        std: float = 0.02,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        requires_grad: bool = True,
    ):
        """
        Initialize a PacketWrapper with header and trailer of given lengths and dimension.

        Args:
            header_len (int): Length of the header to add.
            trailer_len (int): Length of the trailer to add.
            dim (int): Dimension of the hidden states.
            mean (float): Mean for normal initialization of header and trailer.
            std (float): Standard deviation for normal initialization of header and trailer.
            dtype (torch.dtype): Data type of the header and trailer.
            device (torch.device): Device to store the header and trailer.
            requires_grad (bool): Whether the header and trailer require gradients.
        """
        self.dim = dim
        self.dtype = dtype
        self.device = device

        self.header = torch.zeros(
            (1, header_len, dim),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
        self.trailer = torch.zeros(
            (1, trailer_len, dim),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
        self.mean = mean
        self.std = std
        torch.nn.init.normal_(self.header, mean=self.mean, std=self.std)
        torch.nn.init.normal_(self.trailer, mean=self.mean, std=self.std)


    def wrap(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Wrap the hidden states with header and trailer
        
        Args:
            hidden_states (torch.Tensor): The hidden states to wrap. Shape can be (seq_len, dim) or (batch_size, seq_len, dim)
        Returns:
            torch.Tensor: The wrapped hidden states with header and trailer added.
        
        Note: the returned tensor will have shape (seq_len + header_len + trailer_len, dim) when input is (seq_len, dim) and 
            shape (batch_size, seq_len + header_len + trailer_len, dim) when input is (batch_size, seq_len, dim)
        """
        assert hidden_states.dim() in (2, 3)
        no_batch_dim = hidden_states.dim() == 2

        if no_batch_dim:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size, _, dim = hidden_states.size()
        assert dim == self.dim, f"Expected hidden_states last dimension to be {self.dim}, but got {dim}"
        hidden_states = torch.cat(
            [
                self.header.expand(batch_size, -1, -1).to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ),
                hidden_states,
                self.trailer.expand(batch_size, -1, -1).to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            ],
            dim=1,
        )
        if no_batch_dim:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states


    def to(self, device: torch.device|None = None, dtype: torch.dtype|None = None):
        """ Move the PacketWrapper to a different device and/or dtype """
        if device is None and dtype is None:
            return self
        if device is not None:
            self.device = device
            self.header = self.header.to(device)
            self.trailer = self.trailer.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.header = self.header.to(dtype)
            self.trailer = self.trailer.to(dtype)
        return self


    def state_dict(self) -> WrapperStateDict:
        """ Get the state dict of the PacketWrapper """
        return {
            "header": self.header,
            "trailer": self.trailer,
            "train_config": {}
        }


    def load_state_dict(self, state_dict: WrapperStateDict):
        """ Load the state dict into the PacketWrapper """
        self.header = state_dict["header"].to(self.device).to(self.dtype)
        self.trailer = state_dict["trailer"].to(self.device).to(self.dtype)
    

    @classmethod
    def from_state_dict(cls, state_dict: WrapperStateDict, device: torch.device|None = None):
        """ Create a PacketWrapper from a state dict """
        header = state_dict["header"]
        trailer = state_dict["trailer"]

        if (dim := header.size(2)) != trailer.size(2):
            raise ValueError(f"Header and trailer must have the same dimension, but got {header.size(2)} and {trailer.size(2)}")
        
        if (dtype := header.dtype) != trailer.dtype:
            raise ValueError(f"Header and trailer must have the same dtype, but got {header.dtype} and {trailer.dtype}")

        if device is not None:
            header = header.to(device)
            trailer = trailer.to(device)
        elif (device := header.device) != trailer.device:
            raise ValueError(f"Header and trailer must be on the same device, but got {header.device} and {trailer.device}")

        if header.size(0) != 1 or trailer.size(0) != 1:
            raise ValueError(f"Header and trailer must have batch size of 1, but got {header.size(0)} and {trailer.size(0)}")

        if header.dim() != 3 or trailer.dim() != 3:
            raise ValueError(f"Header and trailer must be 3-dimensional, but got {header.dim()} and {trailer.dim()}")

        wrapper = cls(
            header_len=header.size(1),
            trailer_len=trailer.size(1),
            dim=dim,
            device=device,
            dtype=dtype,
        )
        wrapper.load_state_dict(state_dict)

        if device is not None:
            wrapper.to(device)
        return wrapper

    @property
    def header_len(self) -> int:
        return self.header.size(1)

    @property
    def trailer_len(self) -> int:
        return self.trailer.size(1)

    def __repr__(self) -> str:
        return (
            f"PacketWrapper(header_len={self.header_len}, "
            f"trailer_len={self.trailer_len}, dim={self.dim}, "
            f"device={self.device}, dtype={self.dtype})"
        )


def load_wrapper(
    wrapper_path: str,
    device: torch.device|str = "cpu",
) -> PacketWrapper:
    """
    Load a PacketWrapper from path.
    """
    device = torch.device(device) if isinstance(device, str) else device
    wrapper = PacketWrapper.from_state_dict(
        torch.load(wrapper_path),
        device=device,
    )
    return wrapper
