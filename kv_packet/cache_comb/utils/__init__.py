import torch
from . import flops

__all__ = [
    "flops",
]

def get_cumsum_pos(cache_lens: list[int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Get the cumulative sum positions for a list of document lengths.

    Args:
        cache_lens (list[int]): A list of integers representing the lengths of documents.
        device (torch.device, optional): The device on which to create the tensor. Defaults to CPU.
    Returns:
        torch.Tensor: A 1D tensor containing the cumulative sum positions. Shape: (sum(cache_lens),)

    Example::
        >>> cache_lens = [3, 5, 2]
        >>> get_cumsum_pos(cache_lens)
        tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1])
    """
    cache_lens_t = torch.tensor(cache_lens, dtype=torch.long, device=device)
    offset = torch.cumsum(cache_lens_t, dim=0) - cache_lens_t
    repeated_offset = torch.repeat_interleave(offset, cache_lens_t)

    result = torch.arange(sum(cache_lens), device=device) - repeated_offset
    return result


def get_shifted_cumsum_pos(
    cache_lens: list[int],
    doc_lens: list[int], 
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Get the shifted cumulative sum positions for a list of document lengths and cache lengths.
    doc_lens[i] >= cache_lens[i] for all i.
    Args:
        cache_lens (list[int]): A list of integers representing the lengths of caches.
        doc_lens (list[int]): A list of integers representing the lengths of documents.
        device (torch.device, optional): The device on which to create the tensor. Defaults to CPU.
    
    Returns:
        torch.Tensor: A 1D tensor containing the shifted cumulative sum positions. Shape: (sum(cache_lens),)
    
    Example:
        >>> cache_lens = [3, 5, 2]
        >>> doc_lens = [4, 6, 3]
        >>> get_shifted_cumsum_pos(cache_lens, doc_lens)
        tensor([0, 1, 2, 4, 5, 6, 7, 8, 10, 11])
    """
    cache_lens_t = torch.tensor(cache_lens, dtype=torch.long, device=device)
    doc_lens_t = torch.tensor(doc_lens, dtype=torch.long, device=device)
    
    offset = torch.zeros_like(cache_lens_t)
    offset[1:] = doc_lens_t[:-1] - cache_lens_t[:-1]
    
    assert torch.all(offset >= 0), "All document lengths must be greater than or equal to cache lengths."
    
    cum_offset = torch.cumsum(offset, dim=0)
    repeated_offset = torch.repeat_interleave(cum_offset, cache_lens_t)
    
    result = torch.arange(sum(cache_lens), device=device) + repeated_offset
    return result