import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Callable
from transformers import DynamicCache, GenerationConfig
from warnings import warn
from time import time
from ..abc import ResultDict, TokenizerType
from ..utils.flops import AutoFlopsCalculator
from ...model import SupportedModel
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache, get_kv_caches
from ...cache.hf_cache import concat_hf_caches, select_hf_cache


def sink_eval(
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
    if kwargs is not None and kwargs != {}:
        print(f"Warning: sink_eval got unexpected kwargs: {kwargs}")

    num_flops = 0
    if preamble:
        context_tokens = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_tokens["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1

        context_len = context_ids.size(1)
        context_cache = get_kv_caches(
            model,
            input_ids=context_ids,
        )[0].to_hf_cache(config=model.config)
    else:
        context_cache = None
        context_len = 0
        warn("sink_eval requires a preamble to create the sink cache, but none was provided.")

    doc_ids: list[torch.Tensor] = []
    doc_lens: list[int] = []
    for document in documents:
        doc_tokens = tokenizer(
            [document], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        input_ids = doc_tokens["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        input_ids.squeeze_(0)
        doc_ids.append(input_ids)
        doc_lens.append(input_ids.size(0))

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    assert isinstance(pad_id, int)

    torch.cuda.synchronize()
    start_time = time()

    batched_doc_ids = pad_sequence(
        doc_ids,
        batch_first=True,
        padding_value=pad_id
    )  # [batch_size, max_seq_len]
    batch_size, max_seq_len = batched_doc_ids.shape
    global_offset = [context_len]
    current_offset = context_len

    for length in doc_lens[:-1]:
        current_offset += length
        global_offset.append(current_offset)

    batch_pos_ids = torch.arange(
        max_seq_len,
        dtype=torch.long,
        device=model.device
    ).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, max_seq_len]

    offset_t = torch.tensor(
        global_offset,
        dtype=torch.long,
        device=model.device
    ).unsqueeze(1)  # [batch_size, 1]

    batch_pos_ids = batch_pos_ids + offset_t  # [batch_size, max_seq_len]
    attn_mask = (batched_doc_ids != pad_id).long()  # [batch_size, max_seq_len]

    if context_cache is not None:
        context_cache.batch_repeat_interleave(batch_size)
        ctx_mask = torch.ones(
            (batch_size, context_len),
            dtype=torch.long,
            device=model.device
        )
        attn_mask = torch.cat([ctx_mask, attn_mask], dim=1)

    flops_counter = AutoFlopsCalculator(model)
    num_flops += flops_counter.total_flops(
        batch_size=batch_size,
        seq_len=max_seq_len,
        cache_len=context_len
    )

    with torch.no_grad():
        result = model.forward(
            input_ids=batched_doc_ids, # type: ignore
            past_key_values=context_cache,
            position_ids=batch_pos_ids, # type: ignore
            attention_mask=attn_mask, # type: ignore
            use_cache=True,
        )
    
    assert result.past_key_values is not None
    new_kvs = result.past_key_values

    final_caches: list[DynamicCache] = []
    context_cache = select_hf_cache(
        cache=new_kvs,
        batch_indices=torch.tensor([0], dtype=torch.long, device=model.device),
        seq_indices=torch.arange(0, context_len, dtype=torch.long, device=model.device)
    )
    final_caches.append(context_cache)

    for i, doc_len in enumerate(doc_lens):
        doc_slice = select_hf_cache(
            cache=new_kvs,
            batch_indices=torch.tensor([i], dtype=torch.long, device=model.device),
            seq_indices=torch.arange(context_len, context_len + doc_len, dtype=torch.long, device=model.device)
        )
        final_caches.append(doc_slice)
    
    full_kv = concat_hf_caches(final_caches)
    assert full_kv.get_seq_length(0) == context_len + sum(doc_lens)
    with torch.no_grad():
        model.forward(
            input_ids=q_ids, # type: ignore
            past_key_values=full_kv,
        )
    
    torch.cuda.synchronize()
    end_time = time()

    ttft = end_time - start_time

    cache_len = context_len + sum(doc_lens)
    full_kv.crop(cache_len)


    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, cache_len), dtype=torch.long, device=model.device) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=full_kv,
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
