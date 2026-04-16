import torch
from typing import Callable
from transformers import GenerationConfig
from time import time
from ..abc import ResultDict, TokenizerType
from ..utils import get_cumsum_pos, get_shifted_cumsum_pos
from ...model import SupportedModel
from ...cache.rotate import rerotate_kv_flops, rerotate_kv_p
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache, concate_kv_caches, get_kv_caches

__all__ = [
    "kv_packet_eval",
]


def kv_packet_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None,
    preamble: str,
    documents: list[str],
    task_prompt: str,
    document_kvs: list[KVCache],
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    kwargs: dict|None = None
) -> ResultDict:
    """
    Proposed packet-KV evaluation method.
    """
    if kwargs is not None and kwargs != {}:
        print(f"Warning: kv_packet_eval got unexpected kwargs: {kwargs}")

    if preamble:
        context_tokens = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_tokens["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1
        context_cache = get_kv_caches(
            model,
            context_ids,
        )[0]
        document_kvs.insert(0, context_cache)

    cache_lens: list[int] = [
        kv[0].key.shape[2] for kv in document_kvs
    ]
    doc_lens: list[int] = [
        kv[0].position_ids.shape[1] for kv in document_kvs
    ]
    total_cache_len = sum(cache_lens)
    total_doc_len = sum(doc_lens)
    full_kv = concate_kv_caches(document_kvs).to_hf_cache(config=model.config)

    ## We assume all cache starts from position 0
    old_pos = get_cumsum_pos(cache_lens, model.device).unsqueeze(0)
    new_pos = get_shifted_cumsum_pos(cache_lens, doc_lens, model.device).unsqueeze(0)

    num_flops = 0
    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)

    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, total_cache_len), device=model.device, dtype=torch.long) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    # Start to count TTFT
    torch.cuda.synchronize()
    start_time = time()

    # Shift data KV caches
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)
    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)

    with torch.no_grad():
        model.forward(
            input_ids=q_ids, # type: ignore
            past_key_values=full_kv,
            position_ids=torch.arange(
                total_doc_len,
                total_doc_len + q_ids.size(1),
                device=model.device,
            ).unsqueeze(0), # type: ignore
        )

    torch.cuda.synchronize()
    end_time = time()
    ttft = end_time - start_time

    # Start generation
    full_kv.crop(total_cache_len)
    pos_ids = torch.arange(
        total_doc_len,
        total_doc_len + q_ids.size(1),
        device=model.device,
    ).unsqueeze(0)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=full_kv,
            position_ids=pos_ids,
        )

    assert isinstance(generation, torch.Tensor)
    pred_answer = get_answers(generation, input_ids, tokenizer)[0]


    if answer_postprocess_func is not None:
        pred_answer, answer = answer_postprocess_func(pred_answer, answer)

    pred_tokens = pred_answer.split()

    tp, fp, fn = f1_states(gold_tokens=answer.split(), pred_tokens=pred_tokens)

    return_dict: ResultDict = {
        "ttft": ttft,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flops": num_flops,
    }
    return return_dict



def kv_packet_eval_attn(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None,
    preamble: str,
    documents: list[str],
    task_prompt: str,
    document_kvs: list[KVCache],
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    kwargs: dict|None = None
) -> tuple[ResultDict, torch.Tensor]:
    """
    The same as `kv_packet_eval` but also returns the attention weights for the query tokens attending to the KV cache tokens.
    """
    if kwargs is not None and kwargs != {}:
        print(f"Warning: kv_packet_eval got unexpected kwargs: {kwargs}")

    if preamble:
        context_tokens = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_tokens["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1
        context_cache = get_kv_caches(
            model,
            context_ids,
        )[0]
        document_kvs.insert(0, context_cache)

    cache_lens: list[int] = [
        kv[0].key.shape[2] for kv in document_kvs
    ]
    doc_lens: list[int] = [
        kv[0].position_ids.shape[1] for kv in document_kvs
    ]

    total_cache_len = sum(cache_lens)
    full_kv = concate_kv_caches(document_kvs)

    ## We assume all cache starts from position 0
    old_pos = get_cumsum_pos(cache_lens, model.device).unsqueeze(0)
    new_pos = get_shifted_cumsum_pos(cache_lens, doc_lens, model.device).unsqueeze(0)

    num_flops = 0
    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)
    query_len: int = q_ids.size(1)

    # Shift data KV caches
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)
    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)

    num_head: int = document_kvs[0][0].key.size(1)
    key_head_dim: int = document_kvs[0][0].key.size(3)
    value_head_dim: int = document_kvs[0][0].value.size(3)

    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(document_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=document_kvs[0][0].key.dtype,
    )
    full_kv = concate_kv_caches([full_kv, dummy_query_cache])
    recompute_indices = list(
        range(total_cache_len, total_cache_len + query_len)
    )
    hidden_states: torch.Tensor = model.model.embed_tokens(q_ids)
    pos_ids = torch.arange(
        0, total_cache_len + query_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    recomputed_hs = hidden_states
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model,
        hidden_states=recomputed_hs,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )

    query_states_dict: dict[int, torch.Tensor] = {}
    assert isinstance(model.config.num_hidden_layers, int)

    for layer in range(model.config.num_hidden_layers):
        results = recompute_kv(
            model=model,
            kv_cache=full_kv,
            hidden_states=recomputed_hs,
            pos_ids=pos_ids,
            token_idx=recompute_indices,
            layer_idx=layer,
            update_cache=True,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
            return_query_states=True,
        )
        recomputed_hs = results["recomputed_hidden_states"]
        query_states = results["query_states"]
        assert isinstance(query_states, torch.Tensor)
        query_states_dict[layer] = query_states

    # Start generation
    full_kv_hf = full_kv.to_hf_cache(config=model.config)
    full_kv_hf.crop(total_cache_len)
    pos_ids = torch.arange(
        total_cache_len,
        total_cache_len + q_ids.size(1),
        device=model.device,
    ).unsqueeze(0)

    return_dict: ResultDict = {
        "ttft": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "flops": num_flops,
    }

    attn_weights_list: list[torch.Tensor] = []

    for layer in range(model.config.num_hidden_layers):
        key = full_kv[layer].key.squeeze(0) # [num_heads, seq_len, head_dim]
        key = key[:, :total_cache_len, :] # Only data KVs
        query = query_states_dict[layer].squeeze(0) # [num_heads, query_len, head_dim]

        n_key_head = key.size(0)
        n_query_head = query.size(0)
        if n_key_head != n_query_head:
            n_group = n_query_head // n_key_head
            query = query.reshape(
                n_group, n_key_head, query_len, -1
            ).mean(dim=0) # [num_heads, query_len, head_dim]
        attn_weights = torch.matmul(
            query, key.transpose(-1, -2)
        ) / (key.size(-1) ** 0.5) # [num_heads, query_len, seq_len]
        reduced_attn_weights = attn_weights.mean(dim=[0,1]) # [seq_len]
        attn_weights_list.append(reduced_attn_weights)
    
    attn_weights_tensor = torch.stack(attn_weights_list, dim=0) # [num_layers, seq_len]
    return return_dict, attn_weights_tensor
