import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    eager_attention_forward as qwen3_eager_attention_forward,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)
from ...cache import KVCache, KeyValue
from .utils import create_recompute_mask, RecomputeResult

__all__ = [
    "qwen3_recompute_kv",
    "qwen3_recompute_query",
]

def qwen3_recompute_kv(
    model: Qwen3ForCausalLM|Qwen3Model,
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
    if isinstance(model, Qwen3ForCausalLM):
        model = model.model  # Extract the base model if a full model is provided

    assert isinstance(model.layers, torch.nn.ModuleList), "Model does not have layers attribute"
    assert 0 <= layer_idx < len(model.layers), "Invalid layer index"

    assert hidden_states.dim() == 3, "Hidden states must be a 3D tensor"
    assert hidden_states.size(0) == 1, "Batch size must be 1 for recomputation"
    assert hidden_states.size(1) == len(token_idx), "Hidden states sequence length must match token_idx length"
    assert isinstance(pos_embed, tuple|None), "pos_embed must be a tuple of (cos, sin) or None"
    decoder_layer = model.layers[layer_idx]
    assert isinstance(decoder_layer, Qwen3DecoderLayer), "Layer is not a Qwen3DecoderLayer"

    if pos_embed is None:
        pos_embed = model.rotary_emb(
            hidden_states,
            pos_ids,
        )
        assert isinstance(pos_embed, tuple)
        cos, sin = pos_embed

        cos = cos[:, token_idx].to(hidden_states.device)
        sin = sin[:, token_idx].to(hidden_states.device)
    else:
        cos, sin = pos_embed

    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)

    attn: Qwen3Attention = decoder_layer.self_attn

    ## Attention
    input_shape = hidden_states.size()[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    # [1, num_heads, len(token_idx), head_dim]
    # Qwen3 Difference: Uses q_norm and k_norm after projection and view, before transpose
    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states_from_hs = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states_from_hs = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states, key_states_from_hs = qwen3_apply_rotary_pos_emb(
        query_states, key_states_from_hs, cos, sin
    )

    if token_position_ids is None:
        token_position_ids = pos_ids[:, token_idx]
    kv_from_hs = KeyValue(key=key_states_from_hs, value=value_states_from_hs, position_ids=token_position_ids)
    key_states = kv_cache[layer_idx]["key"] # [1, num_heads, seq_len, head_dim]
    value_states = kv_cache[layer_idx]["value"]

    if not update_cache:
        key_states = key_states.clone()
        value_states = value_states.clone()
    
    if update_indices is None:
        if fuse_theta is None:
            key_states[:, :, token_idx, :] = key_states_from_hs
            value_states[:, :, token_idx, :] = value_states_from_hs
        else:
            key_states[:, :, token_idx, :] = (1 - fuse_theta) * key_states[:, :, token_idx, :] + fuse_theta * key_states_from_hs
            value_states[:, :, token_idx, :] = (1 - fuse_theta) * value_states[:, :, token_idx, :] + fuse_theta * value_states_from_hs
    else:
        index_map = {val: i for i, val in enumerate(token_idx)}
        local_index = [index_map[item] for item in update_indices]
        if fuse_theta is None:
            key_states[:, :, update_indices, :] = key_states_from_hs[:, :, local_index, :]
            value_states[:, :, update_indices, :] = value_states_from_hs[:, :, local_index, :]
        else:
            key_states[:, :, update_indices, :] = (1 - fuse_theta) * key_states[:, :, update_indices, :] + fuse_theta * key_states_from_hs[:, :, local_index, :]
            value_states[:, :, update_indices, :] = (1 - fuse_theta) * value_states[:, :, update_indices, :] + fuse_theta * value_states_from_hs[:, :, local_index, :]
    

    if recompute_mask is None:
        recompute_mask = create_recompute_mask(
            query_len=query_states.size(2),
            key_len=key_states.size(2),
            token_idx=token_idx,
            to_4d=True,
            device=hidden_states.device,
        )

    attention_interface = qwen3_eager_attention_forward
    if attn.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

    hidden_states, attn_weights = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask=recompute_mask,
        dropout=0.0,
        scaling=attn.scaling,
    )

    hidden_states = hidden_states.reshape(*input_shape, -1).contiguous()
    hidden_states = attn.o_proj(hidden_states)

    hidden_states = residual + hidden_states # [1, len(token_idx), hidden_size]

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    result_dict = RecomputeResult(
        recomputed_hidden_states=hidden_states,
        kv_from_hs=kv_from_hs,
        query_states=query_states if return_query_states else None,
        attention_weights=attn_weights if return_attention_weights else None,
    )
    return result_dict


def qwen3_recompute_query(
    model: Qwen3ForCausalLM|Qwen3Model,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    token_idx: list[int],
    layer_idx: int,
    pos_embed: tuple[torch.Tensor, torch.Tensor]|torch.Tensor|None=None
) -> torch.Tensor:
    if isinstance(model, Qwen3ForCausalLM):
        model = model.model  # Extract the base model if a full model is provided

    assert isinstance(model.layers, torch.nn.ModuleList), "Model does not have layers attribute"
    assert 0 <= layer_idx < len(model.layers), "Invalid layer index"

    assert hidden_states.dim() == 3, "Hidden states must be a 3D tensor"
    assert hidden_states.size(0) == 1, "Batch size must be 1 for recomputation"
    assert hidden_states.size(1) == len(token_idx), "Hidden states sequence length must match token_idx length"
    assert isinstance(pos_embed, tuple|None), "pos_embed must be a tuple of (cos, sin) or None"
    decoder_layer = model.layers[layer_idx]
    assert isinstance(decoder_layer, Qwen3DecoderLayer), "Layer is not a Qwen3DecoderLayer"

    if pos_embed is None:
        pos_embed = model.rotary_emb(
            hidden_states,
            pos_ids,
        )
        assert isinstance(pos_embed, tuple)
        cos, sin = pos_embed

        cos = cos[:, token_idx].to(hidden_states.device)
        sin = sin[:, token_idx].to(hidden_states.device)
    else:
        cos, sin = pos_embed

    hidden_states = decoder_layer.input_layernorm(hidden_states)

    attn: Qwen3Attention = decoder_layer.self_attn

    ## Attention
    input_shape = hidden_states.size()[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    # [1, num_heads, len(token_idx), head_dim]
    # Qwen3 Difference: Apply q_norm
    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    
    # We also need keys for RoPE rotation context, though we only return queries
    key_states_from_hs = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    query_states, _ = qwen3_apply_rotary_pos_emb(
        query_states, key_states_from_hs, cos, sin
    )

    return query_states