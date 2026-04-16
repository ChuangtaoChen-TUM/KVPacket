import torch
from typing import TypeAlias
from optimum.quanto import quantize_weight, MaxOptimizer, AbsmaxOptimizer
from optimum.quanto import qint4, qint8, qint2, qtype
from optimum.quanto import WeightQBitsTensor, WeightQBytesTensor


__all__ = [
    "quantize_tensor",
    "dequantize_tensor",
    "QuantizedTensor",
]


QTYPES: dict[int, qtype] = {
    2: qint2,
    4: qint4,
    8: qint8,
}

QuantizedTensor: TypeAlias = WeightQBytesTensor | WeightQBitsTensor


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    axis: int = 0,
    q_group_size: int|None = 64,
) -> QuantizedTensor:
    if num_bits not in QTYPES:
        raise ValueError(f"num_bits must be one of {list(QTYPES.keys())}, got {num_bits}")
    qtype = QTYPES[num_bits]
    if num_bits == 8:
        zero_point = None
        q_group_size = None
        optimizer = AbsmaxOptimizer()
        scale = optimizer(
            tensor,
            qtype,
            axis=axis
        )
    else:
        optimizer = MaxOptimizer()
        scale, zero_point = optimizer(
            tensor,
            qtype,
            axis=axis,
            group_size=q_group_size
        )
    qtensor = quantize_weight(
        tensor,
        qtype,
        axis,
        scale,
        shift=zero_point,
        group_size=q_group_size,
        optimized=True
    )
    if num_bits == 8:
        assert isinstance(qtensor, WeightQBytesTensor)
    else:
        assert isinstance(qtensor, WeightQBitsTensor)
    return qtensor


def dequantize_tensor(
    qtensor: WeightQBytesTensor|WeightQBitsTensor
) -> torch.Tensor:
    tensor = qtensor.dequantize()
    assert isinstance(tensor, torch.Tensor)
    return tensor
