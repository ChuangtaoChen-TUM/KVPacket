import torch
from time import time
from typing import Callable
from transformers import GenerationConfig
from ...model import SupportedModel
from ..abc import ResultDict, TokenizerType
from ..utils import get_cumsum_pos
from ..utils.flops import AutoFlopsCalculator
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ...cache.rotate import rerotate_kv_p, rerotate_kv_flops
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache, concate_kv_caches, get_kv_caches


def a3_eval(
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
    if kwargs is None:
        recompute_ratio = None
    else:
        kwargs = kwargs.copy()
        recompute_ratio = kwargs.pop("recompute_ratio", None)

    if not isinstance(recompute_ratio, float) or not (0.0 <= recompute_ratio <= 1.0):
        raise ValueError("a3_eval requires a float 'recompute_ratio' kwarg between 0.0 and 1.0")
    
    if kwargs is not None and kwargs != {}:
        print(f"Warning: a3_eval got unexpected kwargs: {kwargs}")
    for kv_cache in document_kvs:
        if any(kv.position_ids.size(1) != kv.key.size(2) for kv in kv_cache):
            raise ValueError("a3_eval does not support compressed KV caches")

    # Prepare context KV cache
    if preamble:
        context_token = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_token["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        saved_kv = get_kv_caches(model, input_ids=context_ids)[0]
    else:
        context_ids = None
        saved_kv = None

    # Check document tokens length
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

    if saved_kv is not None:
        document_kvs.insert(0, saved_kv)

    if context_ids is not None:
        doc_tokens.insert(0, context_ids)

    sample_len: list[int] = [
        kv[0].position_ids.size(1) for kv in document_kvs
    ]
    total_data_len = sum(sample_len)

    # Prepare shift positions
    full_kv = concate_kv_caches(document_kvs)
    old_pos = get_cumsum_pos(sample_len, model.device).unsqueeze(0)
    new_pos = torch.arange(total_data_len, device=model.device).unsqueeze(0)

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)

    query_len = q_ids.size(1)
    doc_tokens.append(q_ids)

    num_head, key_head_dim, value_head_dim = document_kvs[0][0].key.size(1), document_kvs[0][0].key.size(3), document_kvs[0][0].value.size(3)
    dummy_query_cache = KVCache.create_dummy(
        batch_size=1,
        num_layers=len(document_kvs[0].layers),
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=document_kvs[0][0].key.dtype,
    )

    num_flops = 0
    flops_calculator = AutoFlopsCalculator(model)

    # Start to count TTFT
    torch.cuda.synchronize()
    start_time = time()

    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)

    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)
    full_kv = concate_kv_caches([full_kv, dummy_query_cache])

    ## Hidden states for the first layer
    first_doc_len = sample_len[0]
    seq_len = total_data_len - first_doc_len

    hidden_states = model.model.embed_tokens(
        torch.cat(doc_tokens, dim=1)
    )[:, first_doc_len:, :]  # [1, seq_len + query_len, hidden_size]
    pos_ids = torch.arange(0, total_data_len + query_len, dtype=torch.long, device=model.device).unsqueeze(0)  # [1, total_data_len]

    ## Only recompute tokens after the first doc
    seq_indices = list(range(first_doc_len, total_data_len))
    query_indices = list(range(total_data_len, total_data_len + query_len))
    recompute_indices = seq_indices + query_indices
    num_flops += 2 * flops_calculator.decoder_layer_flops(
        batch_size=1, seq_len=seq_len, cache_len=first_doc_len
    )
    token_position_ids = pos_ids[:, recompute_indices]
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )
    ## zero-th layer recompute, the KV cache are the same
    recomputed_result = recompute_kv(
        model=model,
        kv_cache=full_kv,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        token_idx=recompute_indices,
        layer_idx=0,
        update_cache=True,
        token_position_ids=token_position_ids,
        pos_embed=pos_embed,
        recompute_mask=recompute_mask,
        return_query_states=True,
    )
    recomputed_hs = recomputed_result["recomputed_hidden_states"]
    query_states = recomputed_result["query_states"] # [1, num_heads, seq_len + query_len, head_dim]
    key_states = recomputed_result["kv_from_hs"]["key"] # [1, num_key_heads, seq_len + query_len, head_dim]

    assert isinstance(query_states, torch.Tensor)

    q_view = query_states[:, :, -query_len:, :]  # [1, num_heads, query_len, head_dim]
    k_target = key_states[:, :, :seq_len, :] # [1, num_key_heads, seq_len, head_dim]

    num_heads = q_view.size(1)
    num_key_heads = k_target.size(1)

    if num_heads != num_key_heads:
        num_rep = num_heads // num_key_heads
        # Expand: [1, n_kv, 1, ...] -> [1, n_kv, n_rep, ...] -> [1, n_heads, ...]
        k_target = k_target[:, :, None, :, :].expand(1, num_key_heads, num_rep, seq_len, k_target.size(-1))
        k_target = k_target.reshape(1, num_heads, seq_len, k_target.size(-1))

    # Calculate token importance scores
    with torch.no_grad():
        attn_logits = torch.matmul(q_view, k_target.transpose(-1, -2))
        head_dim = q_view.size(-1)
        attn_logits = attn_logits / (head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
        diff_along_seq = attn_weights.sum(dim=(0,1,2))  # [seq_len]

    num_recompute = int(recompute_ratio * seq_len)

    # topk_indices_t: range from 0 to seq_len-1
    _, topk_indices_t = torch.topk(diff_along_seq, k=num_recompute, largest=True, sorted=False)
    topk_indices_t = torch.sort(topk_indices_t).values  # sort indices

    # also include the query positions
    hs_indices = topk_indices_t.cpu().tolist() + list(range(seq_len, seq_len + query_len))
    recomputed_hs = recomputed_hs[:, hs_indices, :]

    # absolute token indices in the full_kv, range from 0 to total_data_len-1
    topk_indices = (topk_indices_t + first_doc_len).cpu().tolist()

    max_index = max(topk_indices) + 1
    topk_indices += list(range(total_data_len, total_data_len + query_len))

    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=recomputed_hs,
        pos_ids=pos_ids,
        recompute_indices=topk_indices,
    )

    assert model.config.num_hidden_layers is not None
    for layer in range(1, model.config.num_hidden_layers):
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1, seq_len=num_recompute,
            cache_len=max_index-num_recompute
        ) # This has equivalent FLOPs as the rectangular attention with (num_recompute, context_len+total_data_len)
        recomputed_hs = recompute_kv(
            model=model,
            kv_cache=full_kv,
            hidden_states=recomputed_hs,
            pos_ids=pos_ids,
            token_idx=topk_indices,
            layer_idx=layer,
            update_cache=True,
            token_position_ids=token_position_ids,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
        )["recomputed_hidden_states"]

    torch.cuda.synchronize()
    end_time = time()
    ttft = end_time - start_time

    # Start generation
    full_kv_hf = full_kv.to_hf_cache(config=model.config)
    full_kv_hf.crop(total_data_len)


    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, total_data_len), device=model.device, dtype=torch.long) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=full_kv_hf,
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
