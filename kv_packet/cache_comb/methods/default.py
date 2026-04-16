import torch
from time import time
from typing import Callable
from transformers import GenerationConfig
from ...model import SupportedModel
from ..abc import ResultDict, TokenizerType
from ...cache import KVCache, get_kv_caches
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ..utils.flops import AutoFlopsCalculator

def full_context_eval(
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
    Full Context Evaluation: concatenate all documents and prompt, and feed into the model without using cache
    This is only for reference of performance
    """
    if kwargs is not None and kwargs != {}:
        print(f"Warning: full_context_eval got unexpected kwargs: {kwargs}")

    if preamble:
        documents = [preamble] + documents

    context = " ".join(documents) + " " + task_prompt
    q_tokens = tokenizer(
        [context], return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)

    # measure TTFT
    torch.cuda.synchronize()
    start_time = time()
    model.forward(
        input_ids=q_ids, # type: ignore
    )
    torch.cuda.synchronize()
    end_time = time()
    ttft = end_time - start_time

    # generation
    with torch.no_grad():
        generation = model.generate(
            input_ids=q_ids,
            generation_config=generation_config,
            attention_mask=q_tokens["attention_mask"],
            tokenizer=tokenizer,
        )

    assert isinstance(generation, torch.Tensor)
    pred_answer = get_answers(generation, q_ids, tokenizer)[0]
    
    if answer_postprocess_func is not None:
        pred_answer, answer = answer_postprocess_func(pred_answer, answer)

    pred_tokens = pred_answer.split()
    tp, fp, fn = f1_states(gold_tokens=answer.split(), pred_tokens=pred_tokens)
    return_dict: ResultDict = {
        "ttft": ttft,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flops": -1,
    }
    return return_dict


def no_cache_eval(
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
    No Cache Evaluation: feed each document and prompt into the model without using cache
    """
    if kwargs is not None and kwargs != {}:
        print(f"Warning: no_cache_eval got unexpected kwargs: {kwargs}")

    if preamble:
        context_token = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_token["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        context_len = context_ids.size(1)
        saved_kv = get_kv_caches(model, input_ids=context_ids)[0].to_hf_cache(config=model.config)
    else:
        context_ids = None
        context_len = 0
        saved_kv = None

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)

    dummy_ids = torch.zeros((1, context_len), device=model.device, dtype=torch.long)
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    # measure TTFT
    torch.cuda.synchronize()
    start_time = time()

    with torch.no_grad():
        model.forward(
            input_ids=q_ids, # type: ignore
            past_key_values=saved_kv,
        )
    torch.cuda.synchronize()
    end_time = time()

    ttft = end_time - start_time

    if saved_kv is not None:
        saved_kv.crop(context_len)
    # generation
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=saved_kv,
        )

    assert isinstance(generation, torch.Tensor)
    pred_answer = get_answers(generation, q_ids, tokenizer)[0]

    if answer_postprocess_func is not None:
        pred_answer, answer = answer_postprocess_func(pred_answer, answer)

    pred_tokens = pred_answer.split()

    tp, fp, fn = f1_states(gold_tokens=answer.split(), pred_tokens=pred_tokens)

    return_dict: ResultDict = {
        "ttft": ttft,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flops": 0,
    }
    return return_dict


def full_recompute(
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
    Full Recompute Evaluation: feed each document and prompt into the model sequentially,
    and recompute the all KV caches for each document (except the preamble at the beginning)
    """
    if kwargs is not None and kwargs != {}:
        print(f"Warning: whole_cache_eval got unexpected kwargs: {kwargs}")

    if preamble:
        context_token = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_token["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1

        context_len = context_ids.size(1)
        saved_kv = get_kv_caches(
            model,
            input_ids=context_ids,
        )[0].to_hf_cache(config=model.config)
    else:
        context_ids = None
        context_len = 0
        saved_kv = None

    data_id_list: list[torch.Tensor] = []

    for doc in documents:
        doc_token = tokenizer(
            [doc], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        doc_ids = doc_token["input_ids"]
        assert isinstance(doc_ids, torch.Tensor)
        assert doc_ids.size(0) == 1
        data_id_list.append(doc_ids)

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)
    assert q_ids.size(0) == 1
    task_prompt_len = q_ids.size(1)
    data_id_list.append(q_ids)

    q_ids = torch.cat(data_id_list, dim=1)


    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, context_len), device=model.device, dtype=torch.long) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    # measure TTFT
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        model.forward(
            input_ids=q_ids, # type: ignore
            past_key_values=saved_kv,
        )

    torch.cuda.synchronize()
    end_time = time()
    ttft = end_time - start_time

    if saved_kv is not None:
        saved_kv.crop(context_len)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            past_key_values=saved_kv,
        )

    assert isinstance(generation, torch.Tensor)
    pred_answer = get_answers(generation, input_ids, tokenizer)[0]

    if answer_postprocess_func is not None:
        pred_answer, answer = answer_postprocess_func(pred_answer, answer)

    pred_tokens = pred_answer.split()
    tp, fp, fn = f1_states(gold_tokens=answer.split(), pred_tokens=pred_tokens)

    # FLOPS calculation
    document_len = q_ids.size(1) - task_prompt_len # exclude task prompt
    flops_calculator = AutoFlopsCalculator(model)
    flops = flops_calculator.total_flops(
        batch_size=1,
        seq_len=document_len,
        cache_len=context_len,
    )
    return_dict: ResultDict = {
        "ttft": ttft,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flops": flops,
    }
    return return_dict


def single_cache(
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
        print(f"Warning: whole_cache_eval got unexpected kwargs: {kwargs}")

    if len(document_kvs) != 1:
        raise ValueError(f"single_cache method expects exactly one document_kv, but got {len(document_kvs)}")

    if len(documents) != 1:
        raise ValueError

    data_id_list: list[torch.Tensor] = []

    if preamble:
        context_token = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_token["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1
        data_id_list.append(context_ids)

    doc_token = tokenizer(
        documents, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    doc_ids = doc_token["input_ids"]
    assert isinstance(doc_ids, torch.Tensor)
    assert doc_ids.size(0) == 1
    data_id_list.append(doc_ids)

    total_doc_len = sum([data_ids.size(1) for data_ids in data_id_list])
    orig_cache_len = document_kvs[0][0].position_ids.size(1)
    assert total_doc_len == orig_cache_len, f"Total token length {total_doc_len} does not match cache length {orig_cache_len}"
    cache_len = document_kvs[0][0].key.shape[2]

    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)
    assert q_ids.size(0) == 1

    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, cache_len), device=model.device, dtype=torch.long) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    ttft = -1

    kv_cache_hf = document_kvs[0].to_hf_cache(config=model.config)
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
            past_key_values=kv_cache_hf,
            position_ids=pos_ids,
        )

    assert isinstance(generation, torch.Tensor)
    pred_answer = get_answers(generation, input_ids, tokenizer)[0]

    if answer_postprocess_func is not None:
        pred_answer, answer = answer_postprocess_func(pred_answer, answer)

    pred_tokens = pred_answer.split()
    tp, fp, fn = f1_states(gold_tokens=answer.split(), pred_tokens=pred_tokens)

    flops = -1
    return_dict: ResultDict = {
        "ttft": ttft,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flops": flops,
    }
    return return_dict
