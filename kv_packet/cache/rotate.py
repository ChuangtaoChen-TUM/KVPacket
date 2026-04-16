import torch
from typing import TypeAlias
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.cache_utils import Cache as HFCache
from . import KVCache, KeyValue

SupportedRotaryEmbedding: TypeAlias = \
    LlamaRotaryEmbedding | \
    MistralRotaryEmbedding | \
    DeepseekV3RotaryEmbedding | \
    DeepseekV2RotaryEmbedding | \
    Qwen3MoeRotaryEmbedding | \
    Qwen3RotaryEmbedding

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The input tensor (query or key) shape [...,head_dim].
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `torch.Tensor`: The tensor after applying rotary position embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def apply_rotary_emb_ds_v2(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1).to(x_.device)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)
    return x_out


def rerotate_embeddings(
    x_p_orig: torch.Tensor,
    rotary_emb: SupportedRotaryEmbedding,
    p_orig: torch.Tensor,
    p_new: torch.Tensor,
    nope_dim: int|None = None,
    unsqueeze_dim: int = 1,
):
    """
    Transforms query/key tensors from being rotated by p_orig to being rotated by p_new.

    Args:
        x_p_orig (`torch.Tensor`): The query or key tensor already rotated by p_orig.
        rotary_emb (`LlamaRotaryEmbedding`): The rotary embedding module.
        p_orig (`torch.Tensor`): The original position IDs used.
        p_new (`torch.Tensor`): The new position IDs to rotate to.
        nope_dim (`int`, *optional*): The dimension to not apply rotation on. If None, apply rotation on all dimensions.
        unsqueeze_dim (`int`, *optional*, defaults to 1): The dimension to unsqueeze cos/sin.

    Returns:
        `tuple(torch.Tensor)`: The re-rotated query and key tensors.
    """
    # 1. Calculate the difference in positions
    p_delta = p_new - p_orig

    if hasattr(rotary_emb, "attention_scaling") and abs(rotary_emb.attention_scaling - 1.0) > 1e-6:
        raise NotImplementedError(
            f"KV Cache re-rotation is not currently supported for models with "
            f"attention_scaling != 1.0 (found {rotary_emb.attention_scaling}). "
            "Re-rotating would compound the scaling factor and corrupt the embeddings."
        )

    if nope_dim is not None:
        x_nope = x_p_orig[..., :nope_dim]
        x_rope = x_p_orig[..., nope_dim:]

        if isinstance(rotary_emb, DeepseekV2RotaryEmbedding):
            freqs_cis = rotary_emb.forward(x_rope, position_ids=p_delta)
            x_rope_new = apply_rotary_emb_ds_v2(
                x_rope, freqs_cis=freqs_cis
            )
        else:
            cos_delta, sin_delta = rotary_emb.forward(x_rope, position_ids=p_delta)
            x_rope_new = apply_rotary_pos_emb(
                x_rope, cos_delta, sin_delta, unsqueeze_dim=unsqueeze_dim
            )
        x_p_new = torch.cat([x_nope, x_rope_new], dim=-1)
    else:
        # 2. Get the rotary embeddings for this difference
        # We pass x_p_orig just to get the correct device and dtype
        if isinstance(rotary_emb, DeepseekV2RotaryEmbedding):
            freqs_cis = rotary_emb.forward(x_p_orig, position_ids=p_delta)
            x_p_new = apply_rotary_emb_ds_v2(
                x_p_orig, freqs_cis=freqs_cis
            )
        else:
            cos_delta, sin_delta = rotary_emb.forward(x_p_orig, position_ids=p_delta)
            # 3. Apply this delta rotation to the already-rotated x
            x_p_new = apply_rotary_pos_emb(
                x_p_orig, cos_delta, sin_delta, unsqueeze_dim=unsqueeze_dim
            )

    return x_p_new


def rerotate_kv[T: KVCache|HFCache](
    kv: T,
    rotary_emb: SupportedRotaryEmbedding,
    shift: int,
    nope_dim: int|None,
) -> T:
    """
    Shifts the position IDs in the KV cache by a specified amount and re-rotates the key tensors accordingly.

    Args:
    - kv (KVCache|HFCache): The KV cache containing key tensors to be re-rotated.
    - rotary_emb (LlamaRotaryEmbedding): The rotary embedding module.
    - shift (int): The amount to shift the position IDs by.

    Warnings:
    - For HFCache, the original position IDs are assumed to start from 0.
    """
    if isinstance(kv, KVCache):
        for layer in kv.layers:
            key_value = kv[layer]
            key_states = key_value["key"]  # [batch_size, num_heads, seq_len, head_dim]
            p_orig = key_value["position_ids"]
            p_new = p_orig + shift

            key_states = rerotate_embeddings(
                x_p_orig=key_states,
                rotary_emb=rotary_emb,
                p_orig=p_orig,
                p_new=p_new,
                nope_dim=nope_dim,
            )
            kv[layer] = KeyValue(
                key=key_states,
                value=key_value["value"],
                position_ids=p_new,
            )
    else:
        for kv_layer in kv.layers:
            key_states = kv_layer.keys
            value_states = kv_layer.values
            if key_states is None and value_states is None:
                continue
            assert key_states is not None
            assert value_states is not None

            batch_size, _, seq_len, _ = key_states.size()
            p_orig = torch.arange(
                seq_len,
                dtype=torch.long,
                device=key_states.device,
            ).unsqueeze(0).expand(batch_size, -1)
            p_new = p_orig + shift
            key_states = rerotate_embeddings(
                x_p_orig=key_states,
                rotary_emb=rotary_emb,
                p_orig=p_orig,
                p_new=p_new,
                nope_dim=nope_dim,
            )
            kv_layer.keys = key_states
    return kv


def rerotate_kv_p[T: KVCache|HFCache](
    kv: T,
    rotary_emb: SupportedRotaryEmbedding,
    old_pos: torch.Tensor,
    new_pos: torch.Tensor,
    nope_dim: int|None,
) -> T:
    """
    Re-rotates the key tensors in the KV cache from old_pos to new_pos.
    
    Args:
    - kv (KVCache|HFCache): The KV cache containing key tensors to be re-rotated.
    - rotary_emb (LlamaRotaryEmbedding): The rotary embedding module.
    - old_pos (torch.Tensor [batch_size, seq_len]): The original position IDs used for rotation.
    - new_pos (torch.Tensor [batch_size, seq_len]): The new position IDs to rotate to.
    """
    if isinstance(kv, KVCache):
        for layer in kv.layers:
            key_value = kv[layer]
            key_states = key_value["key"]  # [batch_size, num_heads, seq_len, head_dim]

            assert old_pos.shape == (key_states.size(0), key_states.size(2))
            assert new_pos.shape == (key_states.size(0), key_states.size(2))

            key_states = rerotate_embeddings(
                x_p_orig=key_states,
                rotary_emb=rotary_emb,
                p_orig=old_pos,
                p_new=new_pos,
                nope_dim=nope_dim,
            )
            kv[layer] = KeyValue(
                key=key_states,
                value=key_value["value"],
                position_ids=new_pos,
            )
    else:
        for layer in kv.layers:
            key_states = layer.keys
            value_states = layer.values
            if key_states is None and value_states is None:
                continue
            assert key_states is not None
            assert value_states is not None
            
            assert old_pos.shape == (key_states.size(0), key_states.size(2))
            assert new_pos.shape == (key_states.size(0), key_states.size(2))

            key_states = rerotate_embeddings(
                x_p_orig=key_states,
                rotary_emb=rotary_emb,
                p_orig=old_pos,
                p_new=new_pos,
                nope_dim=nope_dim,
            )
            layer.keys = key_states
    return kv



def rerotate_kv_flops(
    kv: KVCache|HFCache,
    nope_dim: int|None,
) -> int:
    """
    Estimates the number of floating point operations (FLOPs) required to rerotate the key tensors in the KV cache.

    Args:
    - kv (KVCache|HFCache): The KV cache containing key tensors to be re-rotated.
    Returns:
    - int: The estimated number of FLOPs required for the rerotation.
    """
    num_flops = 0

    def cal_layer_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int, nope_dim: int|None) -> int:
        rope_dim = head_dim if nope_dim is None else head_dim - nope_dim
        return batch_size * (4 + 3 * num_heads) * seq_len * rope_dim

    if isinstance(kv, KVCache):
        for layer in kv.layers:
            key_value = kv[layer]
            key_states = key_value["key"]  # [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = key_states.size()
            num_flops += cal_layer_flops(batch_size, num_heads, seq_len, head_dim, nope_dim)
    else:
        for kv_layer in kv.layers:
            key_states = kv_layer.keys
            value_states = kv_layer.values
            if key_states is None and value_states is None:
                continue
            assert key_states is not None

            batch_size, num_heads, seq_len, head_dim = key_states.size()

            num_flops += cal_layer_flops(batch_size, num_heads, seq_len, head_dim, nope_dim)
    return num_flops


