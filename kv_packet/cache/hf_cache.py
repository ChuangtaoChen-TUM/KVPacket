""" HuggingFace cache utilities for concatenation and selection """
import torch
from transformers.cache_utils import Cache, DynamicCache

def concat_hf_caches(caches: list[DynamicCache]) -> DynamicCache:
    """
    Concatenate multiple HuggingFace caches along the sequence length dimension.
    
    This function is useful for speculative decoding or strategies where multiple 
    independent generation branches need to be merged into a single history.

    Args:
        caches (list[DynamicCache]): List of HuggingFace DynamicCache objects to concatenate. 
                              Must contain compatible layers (same head_dim, num_heads).

    Returns:
        DynamicCache: A new DynamicCache object containing the concatenated history.
    """
    if not caches:
        raise ValueError("Cannot concatenate an empty list of caches.")

    # We use DynamicCache as the output container as it can accommodate 
    # the arbitrary length of the concatenated results.
    out_cache = DynamicCache()
    
    # We assume all caches have the same number of layers as the first one
    num_layers = len(caches[0])

    for layer_idx in range(num_layers):
        layer_keys = []
        layer_values = []

        for i, cache in enumerate(caches):
            if layer_idx >= len(cache):
                raise ValueError(
                    f"Cache at index {i} has fewer layers ({len(cache)}) "
                    f"than the first cache ({num_layers})."
                )

            # Retrieve the raw key/value tensors for this layer
            # k, v shape: [batch_size, num_heads, max_len/seq_len, head_dim]
            k, v = cache.layers[layer_idx].keys, cache.layers[layer_idx].values

            if k is None or v is None:
                raise ValueError(f"Cache at index {i} has uninitialized keys/values at layer {layer_idx}.")

            valid_length = cache.get_seq_length(layer_idx)
            
            # Slice the tensors to their valid length along the sequence dimension (dim -2)
            k = k[..., :valid_length, :]
            v = v[..., :valid_length, :]

            layer_keys.append(k)
            layer_values.append(v)

        # Concatenate along the sequence length dimension
        concat_k = torch.cat(layer_keys, dim=-2)
        concat_v = torch.cat(layer_values, dim=-2)

        # Populate the output cache. 
        # Calling update on a fresh DynamicCache layer will initialize it 
        # and set these tensors as the history.
        out_cache.update(concat_k, concat_v, layer_idx)

    return out_cache


def select_hf_cache(
    cache: Cache,
    batch_indices: torch.Tensor|None = None,
    seq_indices: torch.Tensor|None = None,
) -> DynamicCache:
    """
    Select specific token positions at seq dim from a HuggingFace Cache object to create a new cache.

    This is useful for KV Cache eviction/pruning strategies where you want to 
    retain only specific tokens (e.g., heavy hitters or local windows) and 
    discard the rest to save memory.

    Args:
        cache (Cache): The original HuggingFace Cache object.
        indices (torch.Tensor): A 1D tensor of token indices to select from the cache.
                                Must be within the bounds of the existing sequence length.

    Returns:
        DynamicCache: A new DynamicCache object containing only the selected tokens.
    """
    # Initialize the output container
    if batch_indices is None and seq_indices is None:
        raise ValueError("At least one of batch_indices or seq_indices must be provided.")

    new_cache = DynamicCache()
    num_layers = len(cache)

    for layer_idx in range(num_layers):
        # Retrieve key/value states from the source cache
        # Shape: [batch_size, num_heads, seq_len, head_dim]
        key_states, value_states = cache.layers[layer_idx].keys, cache.layers[layer_idx].values
        
        if key_states is None or value_states is None:
            raise ValueError(f"Cache has uninitialized keys/values at layer {layer_idx}.")

        if batch_indices is not None:
            # Select the batches along the batch dimension (dimension 0)
            # We use index_select as it is generally efficient for non-contiguous selection
            key_states = key_states.index_select(dim=0, index=batch_indices)
            value_states = value_states.index_select(dim=0, index=batch_indices)
        
        if seq_indices is not None:
            # Select the tokens along the sequence length dimension (dimension -2)
            # We use index_select as it is generally efficient for non-contiguous selection
            key_states = key_states.index_select(dim=-2, index=seq_indices)
            value_states = value_states.index_select(dim=-2, index=seq_indices)

        # Update the new cache with the pruned states
        new_cache.update(key_states, value_states, layer_idx)

    return new_cache
