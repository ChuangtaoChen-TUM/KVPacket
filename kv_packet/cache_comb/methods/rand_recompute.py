import torch
import random
from time import time
from typing import Callable
from transformers import GenerationConfig
from ..abc import ResultDict, TokenizerType
from ..utils import get_cumsum_pos
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ..utils.flops import AutoFlopsCalculator
from ...model import SupportedModel
from ...cache.rotate import rerotate_kv_p, rerotate_kv_flops
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache, concate_kv_caches, get_kv_caches


def rand_recompute_eval(
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
        recompute_ratio = None
        seed = None
    else:
        kwargs = kwargs.copy()
        recompute_ratio = kwargs.pop("recompute_ratio", None)
        seed = kwargs.pop("seed", None)
    if not isinstance(recompute_ratio, float) or recompute_ratio < 0:
        raise ValueError("rand_eval requires a non-negative 'recompute_ratio' kwarg")

    if kwargs is not None and kwargs != {}:
        print(f"Warning: epic_eval got unexpected kwargs: {kwargs}")

    for kv_cache in document_kvs:
        if any(kv.position_ids.size(1) != kv.key.size(2) for kv in kv_cache):
            raise ValueError("Random does not support compressed KV caches")

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
        num_layers=len(document_kvs[0].layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=document_kvs[0][0].key.dtype,
    )
    flops_calculator = AutoFlopsCalculator(model)
    num_flops = 0

    full_kv = concate_kv_caches(document_kvs)

    # Start to count TTFT
    torch.cuda.synchronize()
    start_time = time()

    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)
    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)
    full_kv = concate_kv_caches([full_kv, dummy_query_cache])

    random.seed(seed)
    recompute_len = int((total_data_len - sample_len[0]) * recompute_ratio)
    recompute_indices = random.sample(
        range(sample_len[0], total_data_len), recompute_len
    )
    recompute_indices.sort()

    recompute_len = len(recompute_indices)
    if recompute_len > 0:
        max_index = recompute_indices[-1] + 1
    else:
        max_index = 0

    recompute_indices += list(range(total_data_len, total_data_len + query_len))
    hidden_states = model.model.embed_tokens(
        torch.cat(doc_tokens, dim=1)[:, recompute_indices]
    )
    pos_ids = torch.arange(
        0, total_data_len + query_len,
        dtype=torch.long, device=model.device
    ).unsqueeze(0)
    token_position_ids = pos_ids[:, recompute_indices]

    recomputed_hs = hidden_states
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )
    for layer in range(model.config.num_hidden_layers):
        recomputed_hs  = recompute_kv(
            model=model,
            kv_cache=full_kv,
            hidden_states=recomputed_hs,
            pos_ids=pos_ids,
            token_idx=recompute_indices,
            layer_idx=layer,
            update_cache=True,
            token_position_ids=token_position_ids,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
        )["recomputed_hidden_states"]
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1, seq_len=recompute_len,
            cache_len=max_index-recompute_len
        )

    torch.cuda.synchronize()
    end_time = time()

    ttft = end_time - start_time


    full_kv_hf = full_kv.to_hf_cache(config=model.config)
    full_kv_hf.crop(total_data_len)

    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_input_ids = torch.ones((1, total_data_len), dtype=torch.long, device=model.device) * dummy_id
    input_ids = torch.cat([dummy_input_ids, q_ids], dim=1)
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
