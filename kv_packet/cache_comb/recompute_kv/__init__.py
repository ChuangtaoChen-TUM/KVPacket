""" Utils for recomputing KV cache """
import torch
from transformers import (
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from .utils import prepare_pos_embed_and_mask, RecomputeResult
from ...model import SupportedModel
from ...cache import KVCache

from .llama import llama_recompute_kv, llama_recompute_query
from .qwen3 import qwen3_recompute_kv, qwen3_recompute_query

__all__ = [
    "recompute_kv",
    "prepare_pos_embed_and_mask",
]

def recompute_kv(
    model: SupportedModel,
    kv_cache: KVCache,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    token_idx: list[int],
    layer_idx: int,
    update_cache: bool=False,
    token_position_ids: torch.Tensor|None=None,
    pos_embed: tuple[torch.Tensor, torch.Tensor]|torch.Tensor|None=None,
    recompute_mask: torch.Tensor|None=None,
    update_indices: list[int]|None=None,
    fuse_theta: float|None=None,
    return_query_states: bool=False,
    return_attention_weights: bool=False
) -> RecomputeResult:
    """
    Recompute key and value states for a specific layer and token indices.
    
    Args:
        model (SupportedModel): The model or its base model.
        kv_cache (KVCache): The KVCache instance containing cached key-value pairs. The keys and values should 
            have shape [1, num_heads, seq_len, head_dim].
        hidden_states (torch.Tensor): The hidden states tensor of shape [1, len(token_idx), hidden_size].
            The positions of hidden states should be within the kv_cache.
        pos_ids (torch.Tensor): The position IDs tensor of shape [1, seq_len] corresponding to kv_cache.
        token_idx (list of int): List of token indices to recompute. The length should match hidden_states' seq_len.
            All token indices must be within the range of kv_cache sequence length.
        layer_idx (int): The layer index to recompute.
        update_cache (bool): Whether to update the KVCache with the recomputed states. Default is True.
        token_position_ids (torch.Tensor|None): Shape [1, len(token_idx)]
            positional_ids to store in the KV cache. If not given, pos_ids[:,token_idx] is used.
        pos_embed: pos_embedding to avoid repeat computation.
        recompute_mask: recompute mask for attention. Used to avoid repeat computation.
        update_indices: which indices of KV cache to update, must be a subset of token_idx. If update_cache is True
            and update_indices is not given, all indices in token_idx will be updated.
        fuse_theta: the value to mix old and new KVs. If not given, new KVs will overwrite old ones (If update cache).
        return_query_states (bool): Whether to return the recomputed query states. Default is False.
        return_attention_weights (bool): Whether to return the attention weights used during recomputation. Default
    Returns:
        - result_dict (dict): A dictionary containing additional information such as query states and attention weights.
    """
    if isinstance(model, LlamaForCausalLM):
        recompute_func = llama_recompute_kv
    elif isinstance(model, Qwen3ForCausalLM):
        recompute_func = qwen3_recompute_kv
    else:
        raise ValueError(f"Unsupported model type for recompute_kv: {type(model)}")

    with torch.no_grad():
        result_dict = recompute_func(
            model=model, # type: ignore
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            pos_ids=pos_ids,
            token_idx=token_idx,
            layer_idx=layer_idx,
            update_cache=update_cache,
            token_position_ids=token_position_ids,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
            update_indices=update_indices,
            fuse_theta=fuse_theta,
            return_query_states=return_query_states,
            return_attention_weights=return_attention_weights
        )
    return result_dict



def recompute_query_states(
    model: SupportedModel,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    token_idx: list[int],
    layer_idx: int,
    pos_embed: tuple[torch.Tensor, torch.Tensor]|torch.Tensor|None=None,
) -> torch.Tensor:
    """
    Recompute only the query states for a specific layer and token indices.
    
    Args:
        model (SupportedModel): The model or its base model.
        hidden_states (torch.Tensor): The hidden states tensor of shape [1, len(token_idx), hidden_size].
        pos_ids (torch.Tensor): The position IDs tensor of shape [1, seq_len].
        token_idx (list of int): List of token indices to recompute. The length should match hidden_states' seq_len.
        layer_idx (int): The layer index to recompute.
        pos_embed (tuple or torch.Tensor or None): Precomputed positional embeddings or None.

    Returns:
        torch.Tensor: The recomputed query states of shape [1, num_heads, len(token_idx), head_dim].
    """
    with torch.no_grad():
        if isinstance(model, LlamaForCausalLM):
            recompute_query_func = llama_recompute_query
        elif isinstance(model, Qwen3ForCausalLM):
            recompute_query_func = qwen3_recompute_query
        else:
            raise ValueError(f"Unsupported model type for recompute_query_states: {type(model)}")
        query_states = recompute_query_func(
            model=model, # type: ignore
            hidden_states=hidden_states,
            pos_ids=pos_ids,
            token_idx=token_idx,
            layer_idx=layer_idx,
            pos_embed=pos_embed,
        )
    return query_states
