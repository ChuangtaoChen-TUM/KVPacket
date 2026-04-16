""" State dict utilities for KVCache for saving/loading and quantization """
import torch
from typing import TypedDict
from .abc import KeyValueDict
from .quantization import QuantizedTensor, dequantize_tensor, quantize_tensor

__all__ = [
    "KVCacheStateDict",
    "quantize_kv_cache_sd",
    "dequantize_kv_cache_sd",
    "kv_cache_sd_to",
]

class KVCacheStateDict(TypedDict):
    layers: dict[int, KeyValueDict|None]
    compact_kv: torch.Tensor|None
    position_ids: torch.Tensor|None


def quantize_kv_cache_sd(
    state_dict: KVCacheStateDict,
    num_bits: int,
    axis: int = 0,
    q_group_size: int|None = 64,
) -> KVCacheStateDict:
    """
    Quantize the KeyValue pairs in a KVCacheStateDict

    Args:
        - state_dict: KVCacheStateDict to quantize
        - num_bits: Number of bits for quantization (2, 4, or 8)
        - axis: Axis along which to quantize
        - q_group_size: Group size for quantization
    Returns:
        - KVCacheStateDict: New KVCacheStateDict with quantized KeyValue pairs
    """
    new_layers: dict[int, KeyValueDict|None] = {}

    for layer, kv_dict in state_dict["layers"].items():
        if kv_dict is None:
            new_layers[layer] = None
            continue
        quantized_key = quantize_tensor(
            kv_dict["key"],
            num_bits=num_bits,
            axis=axis,
            q_group_size=q_group_size
        )
        quantized_value = quantize_tensor(
            kv_dict["value"],
            num_bits=num_bits,
            axis=axis,
            q_group_size=q_group_size
        )
        new_kv_dict: KeyValueDict = {
            "key": quantized_key,
            "value": quantized_value,
            "position_ids": kv_dict["position_ids"]
        }
        new_layers[layer] = new_kv_dict

    if state_dict["compact_kv"] is not None:
        quantized_compact_kv = quantize_tensor(
            state_dict["compact_kv"],
            num_bits=num_bits,
            axis=axis,
            q_group_size=q_group_size
        )
    else:
        quantized_compact_kv = None
    
    return KVCacheStateDict(
        layers=new_layers,
        compact_kv=quantized_compact_kv,
        position_ids=state_dict["position_ids"],
    )


def dequantize_if(
    tensor: torch.Tensor|QuantizedTensor,
) -> torch.Tensor:
    """
    Dequantize a tensor if it is quantized

    Args:
        - tensor: Tensor to dequantize
    Returns:
        - torch.Tensor: Dequantized tensor
    """
    if isinstance(tensor, QuantizedTensor):
        return dequantize_tensor(tensor)
    return tensor


def dequantize_kv_cache_sd(
    state_dict: KVCacheStateDict,
) -> KVCacheStateDict:
    """
    Dequantize the KeyValue pairs in a KVCacheStateDict

    Args:
        - state_dict: KVCacheStateDict to dequantize
    Returns:
        - KVCacheStateDict: New KVCacheStateDict with dequantized KeyValue pairs
    """
    new_layers: dict[int, KeyValueDict|None] = {}

    for layer, kv_dict in state_dict["layers"].items():
        if kv_dict is None:
            new_layers[layer] = None
            continue
        dequantized_key = dequantize_if(kv_dict["key"])
        dequantized_value = dequantize_if(kv_dict["value"])
        new_kv_dict: KeyValueDict = {
            "key": dequantized_key,
            "value": dequantized_value,
            "position_ids": kv_dict["position_ids"]
        }
        new_layers[layer] = new_kv_dict

    if state_dict["compact_kv"] is not None:
        dequantized_compact_kv = dequantize_if(state_dict["compact_kv"])
    else:
        dequantized_compact_kv = None
    
    return KVCacheStateDict(
        layers=new_layers,
        compact_kv=dequantized_compact_kv,
        position_ids=state_dict["position_ids"],
    )


def kv_cache_sd_to(
    state_dict: KVCacheStateDict,
    device: torch.device|None = None,
    dtype: torch.dtype|None = None,
    non_blocking: bool=False
) -> KVCacheStateDict:
    """
    Move all tensors in a KVCacheStateDict to the specified device

    Args:
        - state_dict: KVCacheStateDict to move
        - device: Target device
        - non_blocking: Whether to use non-blocking transfers
    Returns:
        - KVCacheStateDict: New KVCacheStateDict on the target device
    """
    for layer, kv_dict in state_dict["layers"].items():
        if kv_dict is None:
            continue
        kv_dict["key"] = kv_dict["key"].to(device, dtype=dtype, non_blocking=non_blocking)
        kv_dict["value"] = kv_dict["value"].to(device, dtype=dtype, non_blocking=non_blocking)
        kv_dict["position_ids"] = kv_dict["position_ids"].to(device, non_blocking=non_blocking)
        state_dict["layers"][layer] = kv_dict

    if state_dict["compact_kv"] is not None:
        state_dict["compact_kv"] = state_dict["compact_kv"].to(device, dtype=dtype, non_blocking=non_blocking)

    if state_dict["position_ids"] is not None:
        state_dict["position_ids"] = state_dict["position_ids"].to(device, non_blocking=non_blocking)

    return state_dict
