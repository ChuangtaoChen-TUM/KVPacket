import torch
from typing import TypedDict
from ...cache.abc import KeyValue
from ...model import SupportedModel

class RecomputeResult(TypedDict):
    recomputed_hidden_states: torch.Tensor
    kv_from_hs: KeyValue
    query_states: torch.Tensor | None
    attention_weights: torch.Tensor | None


def create_recompute_mask(
    query_len: int, 
    key_len: int,
    token_idx: list[int] | torch.Tensor,
    device: torch.device,
    to_4d: bool = False, 
    sliding_window: int | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Creates a boolean mask for attention recomputation.
    True = Attend (Keep), False = Mask out.
    """
    # 1. Ensure token_idx is a tensor (setup cost)
    if isinstance(token_idx, list):
        token_idx = torch.tensor(token_idx, device=device, dtype=torch.long)
    
    assert token_idx.dim() == 1 and token_idx.size(0) == query_len, \
        "token_idx must be a 1D tensor of length query_len"

    # 2. Create column indices [0, 1, 2, ..., key_len-1] (The Key positions)
    # Shape: [1, key_len]
    col_indices = torch.arange(key_len, device=device).unsqueeze(0)
    
    # 3. Create row thresholds (The Query positions)
    # Shape: [query_len, 1]
    row_limits = token_idx.unsqueeze(1)
    
    # 4. Broadcast comparison: 
    # Condition A (Causal): Key index (col) <= Query index (row)
    # Shape: [query_len, key_len]
    causal_mask = col_indices <= row_limits

    # 5. Apply Sliding Window Constraint (if provided)
    if sliding_window is not None:
        # Condition B (Window): Key index (col) > Query index (row) - window_size
        # Example: Query at 10000, Window 4096. 
        # Allowed keys: 10000 - 4096 < k <= 10000.
        # k > 5904
        lower_bound = row_limits - sliding_window
        window_mask = col_indices > lower_bound
        
        # Combine with Logical AND
        attn_mask = causal_mask & window_mask
    else:
        attn_mask = causal_mask

    # 6. Reshape for 4D
    if to_4d:
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
    return attn_mask


def prepare_pos_embed_and_mask(
    model: SupportedModel,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    recompute_indices: list[int],
) -> tuple[tuple[torch.Tensor, torch.Tensor] | torch.Tensor, torch.Tensor]:
    pos_embed = model.model.rotary_emb(
        hidden_states,
        pos_ids,
    )
    assert isinstance(pos_embed, tuple|torch.Tensor)
    if isinstance(pos_embed, tuple):
        cos, sin = pos_embed
        cos = cos[:, recompute_indices].to(hidden_states.device)
        sin = sin[:, recompute_indices].to(hidden_states.device)
        pos_embed = (cos, sin)
    else:
        pos_embed = pos_embed[:, recompute_indices].to(hidden_states.device)

    recompute_mask = create_recompute_mask(
        query_len=len(recompute_indices),
        key_len=pos_ids.size(1),
        token_idx=recompute_indices,
        to_4d=True,
        device=model.device,
        sliding_window=getattr(model.config, "sliding_window", None),
    )
    return pos_embed, recompute_mask
