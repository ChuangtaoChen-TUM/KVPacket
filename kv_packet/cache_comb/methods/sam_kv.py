import torch
from time import time
from typing import Callable
from transformers import GenerationConfig
from torch.nn import functional as F
import itertools
from ..abc import ResultDict, TokenizerType
# from ...instrument.hf_fix import GenerateFix
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ..utils.flops import AutoFlopsCalculator
from ...model import SupportedModel
from ...cache.rotate import rerotate_kv, rerotate_kv_flops
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache, concate_kv_caches, get_kv_caches


def sam_kv_eval(
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
    assert model.config.num_hidden_layers is not None
    if kwargs is None:
        raise ValueError("sam_kv_eval requires 'recompute_ratio', 'stable_layers', 'num_initial_tokens', and 'num_local_tokens' kwargs")
    else:
        kwargs = kwargs.copy()
        stable_layers: list[int] = kwargs.pop("stable_layers")
        num_initial_tokens: int = kwargs.pop("num_initial_tokens")
        num_local_tokens: int = kwargs.pop("num_local_tokens")
        block_size: int = kwargs.pop("block_size")
        fuse_theta: float = kwargs.pop("fuse_theta")

    if kwargs != {}:
        print(f"Warning: sam_kv_eval got unexpected kwargs: {kwargs}")

    for kv_cache in document_kvs:
        if any(kv.position_ids.size(1) != kv.key.size(2) for kv in kv_cache):
            raise ValueError("SamKV does not support compressed KV caches")

    doc_tokens: list[torch.Tensor] = []
    for document in documents:
        tokens = tokenizer(
            [document], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        input_ids = tokens["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        doc_tokens.append(input_ids)

    for doc_token, kv_cache in zip(doc_tokens, document_kvs):
        if doc_token.size(1) != kv_cache[0].key.size(2):
            raise ValueError("The length of document tokens does not match the length of KV cache.")

    if preamble:
        context_token = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_token["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        saved_kv = get_kv_caches(model, input_ids=context_ids)[0]
        document_kvs.insert(0, saved_kv)
        doc_tokens.insert(0, context_ids)
    else:
        context_ids = None

    sample_len: list[int] = [
        kv[0].position_ids.size(1) for kv in document_kvs
    ]
    total_data_len = sum(sample_len)

    # get the query states
    stable_layer_set = set(stable_layers)
    doc_query_states: list[dict[int, torch.Tensor]] = []
    for i in range(len(document_kvs)):
        hidden_states = model.model.embed_tokens(doc_tokens[i][:, -num_local_tokens:])
        assert isinstance(hidden_states, torch.Tensor)

        if num_local_tokens >= sample_len[i]:
            token_idx = list(range(sample_len[i]))
        else:
            token_idx = list(range(sample_len[i] - num_local_tokens, sample_len[i]))
    
        pos_embed, recompute_mask = prepare_pos_embed_and_mask(
            model=model,
            hidden_states=hidden_states,
            pos_ids=document_kvs[i][0].position_ids,
            recompute_indices=token_idx,
        )
        doc_query_state_dict: dict[int, torch.Tensor] = {}

        for layer_index in range(model.config.num_hidden_layers):
            recompute_result = recompute_kv(
                model=model,
                kv_cache=document_kvs[i],
                hidden_states=hidden_states,
                pos_ids=document_kvs[i][0].position_ids,
                token_idx=token_idx,
                layer_idx=layer_index,
                update_cache=False,
                pos_embed=pos_embed,
                recompute_mask=recompute_mask,
                return_query_states=True,
            )
            hidden_states = recompute_result["recomputed_hidden_states"]
            query_states = recompute_result["query_states"]
            assert isinstance(query_states, torch.Tensor)
            if layer_index in stable_layer_set:
                doc_query_state_dict[layer_index] = query_states.mean(dim=2)  # [1, num_heads, head_dim]
        doc_query_states.append(doc_query_state_dict)
            

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)
    query_len = q_ids.size(1)
    doc_tokens.append(q_ids)

    num_head, _ = document_kvs[0][0].key.size(1), document_kvs[0][0].key.size(3)
    flops_calculator = AutoFlopsCalculator(model)
    num_flops = 0

    # Start to count TTFT
    torch.cuda.synchronize()
    start_time = time()
    
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)

    start_pos = 0
    for i in range(len(document_kvs)):
        document_kvs[i] = rerotate_kv(
            document_kvs[i], model.model.rotary_emb, 
            shift=start_pos, nope_dim=nope_dim
        )
        num_flops += rerotate_kv_flops(document_kvs[i], nope_dim=nope_dim)
        start_pos += sample_len[i]

    sparse_kvs, inv_sparse_kvs = get_sparse_kvs(
        kv_caches=document_kvs,
        num_initial_tokens=num_initial_tokens,
        num_local_tokens=num_local_tokens,
    )

    full_sparse_kv = concate_kv_caches(sparse_kvs)
    sparse_background_len = full_sparse_kv[0].key.size(2)
    key_head_dim = full_sparse_kv[0].key.size(3)
    value_head_dim = full_sparse_kv[0].value.size(3)

    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(sparse_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=sparse_kvs[0][0].key.dtype,
    )

    full_sparse_kv = concate_kv_caches([full_sparse_kv, dummy_query_cache])

    query_indices_in_sparse = list(
        range(sparse_background_len,  sparse_background_len + query_len)
    )
    pos_ids_sparse = torch.arange(
        0, sparse_background_len + query_len, device=model.device, dtype=torch.long
    )
    pos_ids_sparse[-query_len:] = torch.arange(
        total_data_len, total_data_len + query_len,
        device=model.device, dtype=torch.long
    )

    hidden_states = model.model.embed_tokens(q_ids)
    assert isinstance(hidden_states, torch.Tensor)

    pos_ids_sparse = pos_ids_sparse.unsqueeze(0)
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids_sparse,
        recompute_indices=query_indices_in_sparse,
    )

    generic_query_states: dict[int, torch.Tensor] = {}

    for layer_index in range(model.config.num_hidden_layers):
        recompute_result = recompute_kv(
            model=model,
            kv_cache=full_sparse_kv,
            hidden_states=hidden_states,
            pos_ids=pos_ids_sparse,
            token_idx=query_indices_in_sparse,
            layer_idx=layer_index,
            update_cache=True,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
            return_query_states=True,
        )
        hidden_states = recompute_result["recomputed_hidden_states"]
        query_states = recompute_result["query_states"]

        assert isinstance(query_states, torch.Tensor)
        if layer_index in stable_layer_set:
            generic_query_states[layer_index] = query_states.mean(dim=2)  # [1, num_heads, head_dim]

        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1,
            seq_len=query_len,
            cache_len=sparse_background_len + query_len,
        )
    
    query_doc_sim: list[dict[int, torch.Tensor]] = []
    for i in range(len(doc_query_states)):
        weighted_q_doc: dict[int, torch.Tensor] = {}
        for layer_index in stable_layer_set:
            dqs_doc_layer = doc_query_states[i][layer_index]  # [1, num_heads, head_dim]
            dqs_generic_layer = generic_query_states[layer_index]  # [1, num_heads, head_dim]
            cos_sim = torch.abs(
                torch.nn.functional.cosine_similarity(
                    dqs_doc_layer,
                    dqs_generic_layer,
                    dim=2,
                )
            ) # [1, num_heads]
            weighted_q_doc[layer_index] = dqs_doc_layer * cos_sim.unsqueeze(2)  # [1, num_heads, head_dim]
        query_doc_sim.append(weighted_q_doc)

    query_doc: list[dict[int, torch.Tensor]] = [{} for _ in range(len(doc_query_states))] # [num_documents, num_stable_layers]
    for j in stable_layer_set:
        sum_of_query_doc_sim = torch.sum(
            torch.stack([query_doc_sim[i][j] for i in range(len(doc_query_states))], dim=0),
            dim=0
        ) # [1, num_heads, head_dim]
        dqs_generic_layer = generic_query_states[j]  # [1, num_heads, head_dim]
        # query_doc_dict: dict[int, torch.Tensor] = {}
        for i in range(len(doc_query_states)):
            q_doc_i = (sum_of_query_doc_sim - query_doc_sim[i][j]) / (len(doc_query_states) - 1) + dqs_generic_layer
            query_doc[i][j] = q_doc_i

    p_doc_layer: list[dict[int, float]] = []
    doc_inner_list: list[dict[int, torch.Tensor|None]] = []
    for i in range(len(document_kvs)):
        p_dict: dict[int, float] = {}
        doc_inner_dict: dict[int, torch.Tensor|None] = {}
        for j in stable_layer_set:
            inv_kv_i = inv_sparse_kvs[i]
            if inv_kv_i is None or inv_kv_i[j].key.numel() == 0:
                p_dict[j] = 0.0
                doc_inner_dict[j] = None
            else:
                k_i_j_anc = sparse_kvs[i][j].key.mean(dim=2) # [1, num_heads, head_dim]
                k_i_j_doc = inv_kv_i[j].key # [1, num_heads, seq_len, head_dim]
                k_i_j_doc = interleaved_mean(k_i_j_doc, dim=2, block_size=block_size) # [1, num_heads, num_block, head_dim]
                qd_ij = query_doc[i][j] # [1, num_heads, head_dim]
                if qd_ij.size(1) != k_i_j_anc.size(1):
                    n_group = qd_ij.size(1) // k_i_j_anc.size(1)
                    qd_ij = qd_ij.view(1, n_group, k_i_j_anc.size(1), -1).mean(dim=1)
                anc_inner = torch.sum(k_i_j_anc * qd_ij) # []
                doc_inner = torch.sum(k_i_j_doc * qd_ij.unsqueeze(2), dim=(0, 1, 3)) # [num_block]
                max_inner = torch.max(doc_inner)
                min_inner = torch.min(doc_inner)

                if min_inner.item() < anc_inner.item() <= max_inner.item():
                    p_doc_i_j = (max_inner.item() - anc_inner.item()) / (max_inner.item() - min_inner.item() + 1e-8)
                else:
                    p_doc_i_j = 0.0
                p_dict[j] = p_doc_i_j
                doc_inner_dict[j] = doc_inner
        doc_inner_list.append(doc_inner_dict)
        p_doc_layer.append(p_dict)

    recompute_indices: list[dict[int, list[int]]] = []
    for i in range(len(document_kvs)):
        p_list = list(p_doc_layer[i].values())
        recompute_ratio = sum(p_list) / len(p_list)
        indices_dict: dict[int, list[int]] = {}
        for j in stable_layer_set:
            doc_inner = doc_inner_list[i][j]
            cache_len = sample_len[i]
            if doc_inner is None:
                indices_dict[j] = list(range(cache_len))
            else:
                num_recompute_blocks = int(doc_inner.shape[0] * recompute_ratio)
                top_k_indices = torch.topk(doc_inner, k=num_recompute_blocks).indices.tolist()
                indices_dict[j] = get_recompute_indices(
                    top_k_indices,
                    block_size,
                    num_initial_tokens,
                    num_local_tokens,
                    cache_len
                )
        recompute_indices.append(indices_dict)

    abs_indices_doc_max: list[list[int]] = [] # max indices to recompute absolute in doc
    or_indices_new: list[dict[int, list[int]]] = [] # or indices relative to new cache in doc

    for item in recompute_indices:
        _, or_indices_dict_new, max_indices = get_or_indices(item)
        abs_indices_doc_max.append(max_indices)
        or_indices_new.append(or_indices_dict_new)
    
    abs_recompute_indices = to_continue_indices(abs_indices_doc_max, sample_len)
    abs_recompute_indices.extend(list(range(total_data_len, total_data_len + query_len)))

    compressed_doc_lens = [len(item) for item in abs_indices_doc_max]
    total_sparse_len = sum(compressed_doc_lens)

    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(sparse_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=sparse_kvs[0][0].key.dtype,
    )

    sparse_kv = concate_kv_caches(document_kvs + [dummy_query_cache]).select_seq(
        torch.tensor(abs_recompute_indices, dtype=torch.long, device=model.device)
    ) # [total_sparse_len + query_len]
    hidden_states = model.model.embed_tokens(
        torch.cat(doc_tokens+[q_ids], dim=1)[:, abs_recompute_indices]
    )
    assert isinstance(hidden_states, torch.Tensor)
    pos_ids = torch.arange(
        0, total_data_len + query_len,
        dtype=torch.long, device=model.device
    ).unsqueeze(0)

    pos_ids = pos_ids[:, abs_recompute_indices]
    recompute_token_indices = list(range(total_sparse_len + query_len))
    query_indices = list(range(total_sparse_len, total_sparse_len+query_len))

    for layer in range(model.config.num_hidden_layers):
        or_indices_all_doc = [item.get(layer, []) for item in or_indices_new]
        new_recompute_token_indices = to_continue_indices(or_indices_all_doc, compressed_doc_lens) + query_indices

        if len(new_recompute_token_indices) == 0:
            break

        recompute_token_indices_map = {val: i for i, val in enumerate(recompute_token_indices)}
        new_relative_indices = [recompute_token_indices_map[item] for item in new_recompute_token_indices]
        hidden_states = hidden_states[:, new_relative_indices, :]
        recompute_token_indices = new_recompute_token_indices

        hidden_states = recompute_kv(
            model=model,
            kv_cache=sparse_kv,
            hidden_states=hidden_states,
            pos_ids=pos_ids,
            token_idx=recompute_token_indices,
            layer_idx=layer,
            update_cache=True,
            fuse_theta=fuse_theta
        )["recomputed_hidden_states"]
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1, seq_len=len(recompute_token_indices),
            cache_len=total_sparse_len+query_len-len(recompute_token_indices)
        )

    torch.cuda.synchronize()
    end_time = time()

    ttft = end_time - start_time

    sparse_kv_hf = sparse_kv.to_hf_cache(config=model.config)
    sparse_kv_hf.crop(total_sparse_len)
    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_input_ids = torch.ones((1, total_sparse_len), dtype=torch.long, device=model.device) * dummy_id
    input_ids = torch.cat([dummy_input_ids, q_ids], dim=1)
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=sparse_kv_hf,
            position_ids=pos_ids
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



def get_sparse_kvs(
    kv_caches: list[KVCache],
    num_initial_tokens: int,
    num_local_tokens: int,
) -> tuple[list[KVCache], list[KVCache|None]]:
    sparse_kvs: list[KVCache] = []
    inv_sparse_kvs: list[KVCache|None] = []
    for kv_cache in kv_caches:
        cache_len = kv_cache[0].key.size(2)
        if cache_len <= num_initial_tokens + num_local_tokens:
            sparse_kvs.append(kv_cache.select_seq(
                torch.arange(0, cache_len, device=kv_cache[0].key.device, dtype=torch.long)
            ))
            inv_sparse_kvs.append(None)
        else:
            device = kv_cache[0].key.device
            initial_indices = torch.arange(
                0, num_initial_tokens, device=device, dtype=torch.long
            )
            local_indices = torch.arange(
                cache_len - num_local_tokens,
                cache_len,
                device=device,
                dtype=torch.long,
            )
            selected_indices = torch.cat([initial_indices, local_indices], dim=0)
            sparse_kvs.append(
                kv_cache.select_seq(selected_indices)
            )
            inv_sparse_kvs.append(
                kv_cache.select_seq(
                    torch.arange(
                        num_initial_tokens,
                        cache_len - num_local_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                )
            )
    return sparse_kvs, inv_sparse_kvs


def interleaved_mean(tensor: torch.Tensor, dim: int, block_size: int):
    """
    Computes the mean of a tensor along a specific dimension in interleaved blocks.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension to perform the mean on.
        block_size (int): The size of the stride/window to average.
        
    Returns:
        torch.Tensor: The tensor with the reduced dimension.
    """
    ndims = tensor.dim()
    dim = dim % ndims

    dims_order = list(range(ndims))
    dims_order.append(dims_order.pop(dim))
    
    x = tensor.permute(*dims_order)
    original_permuted_shape = x.shape
    x = x.reshape(-1, 1, original_permuted_shape[-1])
    x = F.avg_pool1d(x, kernel_size=block_size, stride=block_size, ceil_mode=True)
    new_shape = list(original_permuted_shape)
    new_shape[-1] = x.shape[-1]
    x = x.view(*new_shape)
    inverse_dims_order = [0] * ndims
    for i, p in enumerate(dims_order):
        inverse_dims_order[p] = i
        
    return x.permute(*inverse_dims_order)


def get_recompute_indices(
    block_indices: list[int],
    block_size: int,
    num_initial_tokens: int,
    num_local_tokens: int,
    doc_len: int
) -> list[int]:
    block_indices.sort()
    indices: list[int] = list(range(num_initial_tokens))

    for block_index in block_indices:
        start_index = num_initial_tokens + block_index * block_size
        end_index = min(num_initial_tokens + (block_index + 1 ) * block_size, doc_len - num_local_tokens)
        indices.extend(range(start_index, end_index))

    indices.extend(range(doc_len - num_local_tokens, doc_len))
    return indices


def get_or_indices(indices_dict: dict[int, list[int]]):
    max_indices = list(set(itertools.chain.from_iterable(list(indices_dict.values()))))
    max_indices.sort()

    index_map = {val: i for i, val in enumerate(max_indices)}
    layer_indices = list(indices_dict.keys())
    layer_indices.sort(reverse=True)

    or_indices_dict_new: dict[int, list[int]] = {} # local index
    indices_dict_new: dict[int, list[int]] = {} # local index
    indices_set: set[int] = set()
    for layer_index in layer_indices:
        indices_set = indices_set | set(indices_dict[layer_index])
        indices_list = list(indices_set)
        indices_list.sort()
        or_indices_dict_new[layer_index] = [index_map[item] for item in indices_list]
        indices_dict_new[layer_index] = [index_map[item] for item in indices_dict[layer_index]]

    for layer_index in range(layer_indices[0], -1, -1):
        if layer_index not in or_indices_dict_new:
            or_indices_dict_new[layer_index] = or_indices_dict_new[layer_index+1]

    return indices_dict_new, or_indices_dict_new, max_indices


def to_continue_indices(
    indices_list: list[list[int]],
    doc_len_list: list[int]
):
    total_indices: list[int] = []
    current_index = 0
    for indices, doc_len in zip(indices_list, doc_len_list):
        total_indices.extend(
            [current_index + item for item in indices]
        )
        current_index += doc_len
    return total_indices
