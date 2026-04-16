import torch
from time import time
from typing import Callable
from transformers import GenerationConfig
from ..utils import get_cumsum_pos
from ..abc import ResultDict, TokenizerType
from ...model import SupportedModel
from ...cache.rotate import rerotate_kv_p, rerotate_kv_flops
from ...utils.generate import get_answers
from ...utils.metric import f1_states
from ...cache import KVCache,  concate_kv_caches, get_kv_caches


def no_recompute_eval(
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
        print(f"Warning: no_recompute_eval got unexpected kwargs: {kwargs}")

    if preamble:
        context_tokens = tokenizer(
            [preamble], return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        context_ids = context_tokens["input_ids"]
        assert isinstance(context_ids, torch.Tensor)
        assert context_ids.size(0) == 1

        context_cache = get_kv_caches(
            model,
            input_ids=context_ids,
        )[0]
        document_kvs.insert(0, context_cache)

    num_flops = 0

    sample_len: list[int] = [
        kv[0].position_ids.size(1) for kv in document_kvs
    ]
    total_data_len = sum(sample_len)

    full_kv = concate_kv_caches(document_kvs).to_hf_cache(config=model.config)

    ## We assume all cache starts from position 0
    old_pos = get_cumsum_pos(sample_len, model.device).unsqueeze(0)
    new_pos = torch.arange(total_data_len, device=model.device).unsqueeze(0)

    num_flops = 0
    q_tokens = tokenizer(
        [task_prompt], return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    q_ids = q_tokens["input_ids"]
    assert isinstance(q_ids, torch.Tensor)


    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.ones((1, total_data_len), dtype=torch.long, device=model.device) * dummy_id
    input_ids = torch.cat([dummy_ids, q_ids], dim=1)

    # Start to count TTFT
    torch.cuda.synchronize()
    start_time = time()

    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)
    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)

    with torch.no_grad():
        model.forward(
            input_ids=q_ids, # type: ignore
            past_key_values=full_kv,
        )
    
    torch.cuda.synchronize()
    end_time = time()
    ttft = end_time - start_time

    full_kv.crop(total_data_len)
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
