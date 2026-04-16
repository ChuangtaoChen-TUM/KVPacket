""" Abstract base class and utilities """
from torch import Tensor
from typing import Literal, overload, TypedDict
from optimum.quanto import WeightQBitsTensor, WeightQBytesTensor
from .quantization import quantize_tensor

QuantizedType = WeightQBytesTensor | WeightQBitsTensor

""" Dimensions for key and value tensors """
KVDim = {
    "batch": 0,
    "head": 1,
    "seq": 2,
    "dim": 3
}

""" Dimensions for position IDs tensor """
PosIDDim = {
    "batch": 0,
    "seq": 1
}


class KeyValueDict(TypedDict):
    key: Tensor
    value: Tensor
    position_ids: Tensor


class KeyValue:
    """
    Key-Value pair for attention mechanism at a layer.
    Dim: (batch_size, num_heads, seq_len, head_dim)
    Position IDs: (batch_size, orig_seq_len)

    If quantized, key and value tensors will be of type QuantizedType
    If a KV is compressed, we have seq_len < orig_seq_len
    """
    def __init__(
        self,
        key: Tensor,
        value: Tensor,
        position_ids: Tensor
    ):
        self.key: Tensor = key
        self.value: Tensor = value
        self.position_ids: Tensor = position_ids

        if isinstance(key, QuantizedType) != isinstance(value, QuantizedType):
            raise ValueError("Both key and value must be either quantized or not quantized.")
    
        self._is_quantized = isinstance(key, QuantizedType)


    @overload
    def __getitem__(self, index: Literal["key", "value", "position_ids"]) -> 'Tensor': ...
    @overload
    def __getitem__(self, index: slice|tuple) -> 'KeyValue': ...

    def __getitem__(self, index: Literal["key", "value", "position_ids"]|slice|tuple) -> 'Tensor|KeyValue':
        if isinstance(index, str):
            if index == "key":
                return self.key
            elif index == "value":
                return self.value
            elif index == "position_ids":
                return self.position_ids
            else:
                raise KeyError(f"KeyValue has no attribute '{index}'.")
        elif isinstance(index, slice) or isinstance(index, tuple):
            return KeyValue(
                key=self.key[index],
                value=self.value[index],
                position_ids=self.position_ids,
            )
        else:
            raise TypeError("Index must be a string or a slice.")


    def __iter__(self):
        yield self.key
        yield self.value


    def __post_init__(self):
        assert self.key.dim() == 4, "Key tensor must be 4-dimensional"
        assert self.value.dim() == 4, "Value tensor must be 4-dimensional"
        assert self.key.size(KVDim["batch"]) == self.value.size(KVDim["batch"]), "Batch size of key and value must match"
        assert self.key.size(KVDim["head"]) == self.value.size(KVDim["head"]), "Number of heads of key and value must match"
        assert self.key.size(KVDim["seq"]) == self.value.size(KVDim["seq"]), "Sequence length of key and value must match"

        assert self.position_ids.dim() == 2, "Position IDs tensor must be 2-dimensional"
        # assert self.position_ids.size(0) == self.key.size(KVDim["batch"]), "Batch size of position IDs must match key and value"


    def __repr__(self) -> str:
        shape_str = ", ".join(
            [
                f"{dim_name}={self.key.size(dim_idx)}"
                for dim_name, dim_idx in KVDim.items()
            ]
        )
        return f"KeyValue({shape_str})"


    def quantize(
        self,
        num_bits: int,
        q_group_size: int|None = 64,
        axis: int = 0,
    ) -> 'KeyValue':
        if self._is_quantized:
            raise RuntimeError("KeyValue is already quantized.")

        quantized_key = quantize_tensor(
            self.key,
            num_bits=num_bits,
            axis=axis,
            q_group_size=q_group_size
        )
        self.key = quantized_key

        quantized_value = quantize_tensor(
            self.value,
            num_bits=num_bits,
            axis=axis,
            q_group_size=q_group_size
        )
        self.value = quantized_value
        self._is_quantized = True
        return self
