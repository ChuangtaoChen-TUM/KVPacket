import json
import os
import torch
import argparse
import glob
from typing import TypedDict, Iterator, Callable
from alive_progress import alive_it
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GenerationConfig
from kv_packet.cache.compress import ScorerPress, PRESS_CLASSES
from kv_packet.packet_wrapper import load_wrapper, PacketWrapper
from kv_packet.cache import KVCache, get_kv_caches, quantize_kv_cache_sd
from kv_packet.cache_comb import EvalCombFunc, get_cache_comb_func
from kv_packet.dataset import get_ret_eval_generator, ANSWER_POSTPROCESS_DICT
from kv_packet.dataset.abc import RetEvalEntry
from kv_packet.utils.metric import calculate_metrics
from kv_packet.utils.config import gather_config_files, load_config_file
from kv_packet.model import SupportedModel

class ModelConfig(TypedDict):
    model_path: str
    dtype: str
    device: str
    generation_kwargs: dict



class DatasetConfig(TypedDict):
    dataset_name: str
    num_samples: int
    num_data_strs: int
    num_shots: int
    subset: str
    split: str
    seed: int
    data_kwargs: dict
    template: str
    template_kwargs: dict


class CompressConfig(TypedDict):
    method: str
    compression_ratio: float
    keep_filler_tokens: bool
    kwargs: dict


class QuantizationConfig(TypedDict):
    num_bits: int
    axis: int
    group_size: int


class CacheCombConfig(TypedDict):
    method: str
    kwargs: dict


class EvalConfig(TypedDict):
    model: ModelConfig
    dataset: DatasetConfig
    cache_comb: CacheCombConfig
    packet_wrapper: str|None
    compress: CompressConfig|None
    quantization: QuantizationConfig|None
    seed: int


class EvalResult(TypedDict):
    precision: float
    recall: float
    f1: float
    ttft: float
    flops: float
    num_orig_tokens: int
    num_wrapped_tokens: int


def load_eval_config(loaded_json: dict) -> EvalConfig:
    model = ModelConfig(
        model_path=loaded_json["model"]["model_path"],
        dtype=loaded_json["model"].get("dtype", "float32"),
        device=loaded_json["model"].get("device", "cuda:0"),
        generation_kwargs=loaded_json["model"].get("generation_kwargs", {}),
    )

    dataset = DatasetConfig(
        dataset_name=loaded_json["dataset"]["dataset_name"],
        num_samples=loaded_json["dataset"]["num_samples"],
        num_data_strs=loaded_json["dataset"]["num_data_strs"],
        num_shots=loaded_json["dataset"]["num_shots"],
        subset=loaded_json["dataset"]["subset"],
        split=loaded_json["dataset"]["split"],
        seed=loaded_json["dataset"]["seed"],
        data_kwargs=loaded_json["dataset"].get("data_kwargs", {}),
        template=loaded_json["dataset"].get("template", "default"),
        template_kwargs=loaded_json["dataset"].get("template_kwargs", {}),
    )
    cache_comb = CacheCombConfig(
        method=loaded_json["cache_comb"]["method"],
        kwargs=loaded_json["cache_comb"].get("kwargs", {}),
    )
    quant_config_dict = loaded_json.get("quantization", None)
    if quant_config_dict is not None:
        quantization_config = QuantizationConfig(
            num_bits=quant_config_dict["num_bits"],
            axis=quant_config_dict.get("axis", 0),
            group_size=quant_config_dict.get("group_size", 64)
        )
    else:
        quantization_config = None
    
    compress_config_dict = loaded_json.get("compress", None)
    if compress_config_dict is not None:
        compress_config = CompressConfig(
            method=compress_config_dict["method"],
            compression_ratio=compress_config_dict["compression_ratio"],
            keep_filler_tokens=compress_config_dict.get("keep_filler_tokens", False),
            kwargs=compress_config_dict.get("kwargs", {}),
        )
    else:
        compress_config = None

    return EvalConfig(
        model=model,
        dataset=dataset,
        cache_comb=cache_comb,
        packet_wrapper=loaded_json.get("packet_wrapper", None),
        compress=compress_config,
        quantization=quantization_config,
        seed=loaded_json["seed"],
    )


def run_eval(
    model: SupportedModel,
    tokenizer: PreTrainedTokenizer|PreTrainedTokenizerFast,
    eval_generator: Iterator[RetEvalEntry],
    cache_comb_func: EvalCombFunc,
    cache_comb_kwargs: dict,
    packet_wrapper: PacketWrapper|None = None,
    compressor: ScorerPress|None = None,
    keep_filler_tokens: bool = False,
    quantization_config: QuantizationConfig|None = None,
    generation_config: GenerationConfig|None = None,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    eval_config: EvalConfig|None = None,
    debug: bool = False,
) -> EvalResult:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ttft = 0.0
    total_flops = 0.0
    num_orig_tokens: int = 0
    num_wrapped_tokens: int = 0

    num_eval = 0

    for eval_entry in alive_it(list(eval_generator)) if not debug else eval_generator:
        query = eval_entry["query"]
        gt_answer = eval_entry["answer"]
        preamble = eval_entry["preamble"]
        documents = eval_entry["documents"]

        task_prompt = eval_entry["task_prompt"]

        if len(documents) == 0:
            raise ValueError("No documents retrieved for the query.")

        if eval_config is not None and eval_config["cache_comb"]["method"] == "single_cache":
            documents = ["".join(documents)]
            preamble_token = tokenizer(
                [preamble], return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            preamble_ids = preamble_token["input_ids"]
            assert isinstance(preamble_ids, torch.Tensor)
            preamble_len = preamble_ids.size(1)
            doc_tokens = tokenizer(
                documents,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            doc_ids = doc_tokens["input_ids"]
            assert isinstance(doc_ids, torch.Tensor)
            doc_len = doc_ids.size(1)

            if compressor is not None:
                assert eval_config["compress"] is not None
                compressor.compression_ratio = eval_config["compress"]["compression_ratio"] * (doc_len / (preamble_len + doc_len))
                indices_to_keep: list[int]|None = list(range(preamble_len))
            inputs_ids = torch.cat([preamble_ids, doc_ids], dim=1)
            attn_mask = torch.ones_like(inputs_ids, dtype=torch.long)
            input_embeds = model.model.embed_tokens(inputs_ids)
        else:
            inputs = tokenizer(
                documents,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                padding_side='right',
            ).to(model.device)
            attn_mask = inputs["attention_mask"]

            assert isinstance(attn_mask, torch.Tensor)
            input_embeds = model.model.embed_tokens(inputs['input_ids'])

        assert isinstance(input_embeds, torch.Tensor)

        if packet_wrapper is not None:
            filler_length = packet_wrapper.header_len + packet_wrapper.trailer_len

            wrapped_input_embeds = torch.zeros(
                (
                    input_embeds.size(0),
                    input_embeds.size(1) + filler_length,
                    input_embeds.size(2)
                ),
            ).to(input_embeds)

            for j in range(input_embeds.size(0)):
                attn_mask_j = attn_mask[j] # type: ignore
                seq_len = int(attn_mask_j.sum().item())
                wrapped_input_embeds[j, :seq_len + filler_length, :] = \
                    packet_wrapper.wrap(input_embeds[j, :seq_len, :])
                num_orig_tokens += seq_len
                num_wrapped_tokens += seq_len + filler_length

            # Adjust attention mask to (batch_size, seq_len + filler_length)
            attn_mask = torch.concat([
                torch.ones(
                    (input_embeds.size(0), filler_length),
                    dtype=attn_mask.dtype,
                    device=attn_mask.device
                ),
                attn_mask
            ], dim=1)

            input_embeds = wrapped_input_embeds

        if compressor is not None:
            # compress only supports single sample currently
            kv_caches: list[KVCache] = []
            for b_idx in range(input_embeds.size(0)):
                input_embed_b = input_embeds[b_idx:b_idx+1, :, :]
                attn_mask_b = attn_mask[b_idx:b_idx+1, :]
                seq_len = int(attn_mask_b.sum().item())
                input_embed_b = input_embed_b[:, :seq_len, :]

                if keep_filler_tokens:
                    assert packet_wrapper is not None, "keep_filler_tokens is only compatible with packet wrapper."
                    indices_to_keep = list(
                        range(packet_wrapper.header_len)
                    ) + list(
                        range(seq_len - packet_wrapper.trailer_len, seq_len)
                    )
                else:
                    indices_to_keep = None

                kv_cache = get_kv_caches(
                    model=model,
                    input_embeds=input_embed_b,
                    compressor=compressor,
                    indices_to_keep=indices_to_keep,
                )[0]
                kv_caches.append(kv_cache)
        else:
            kv_caches = get_kv_caches(
                model=model,
                input_embeds=input_embeds,
                attention_mask=attn_mask,
                compressor=compressor,
            )

        if quantization_config is not None:
            kv_caches = [
                KVCache.from_state_dict(
                    quantize_kv_cache_sd(
                        kv_cache.state_dict(),
                        num_bits=quantization_config['num_bits'],
                        axis=quantization_config['axis'],
                        q_group_size=quantization_config['group_size']
                    )
                )
                for kv_cache in kv_caches
            ]

        result = cache_comb_func(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            preamble=preamble,
            documents=documents,
            task_prompt=task_prompt,
            document_kvs=kv_caches,
            answer=gt_answer,
            answer_postprocess_func=answer_postprocess_func,
            kwargs=cache_comb_kwargs,
        )

        ttft = result['ttft']
        tp = result['tp']
        fp = result['fp']
        fn = result['fn']
        flops = result["flops"]

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_ttft += ttft
        total_flops += flops
        num_eval += 1
    
    precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
    avg_ttft = total_ttft / num_eval if num_eval > 0 else 0.0
    avg_flops = total_flops / num_eval if num_eval > 0 else 0.0

    f_result = EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        ttft=avg_ttft,
        flops=avg_flops,
        num_orig_tokens=num_orig_tokens,
        num_wrapped_tokens=num_wrapped_tokens
    )

    return f_result


def run_one_config(
    eval_config_file: str,
    eval_cache: dict[str, dict],
    eval_results: dict[str, dict],
    overwrite: bool = False,
    debug: bool = False,
):
    result_folder = os.path.join(
        os.path.dirname(eval_config_file),
        "eval_results"
    )
    result_file = os.path.splitext(os.path.basename(eval_config_file))[0] + "_result.json"
    result_path = os.path.join(result_folder, result_file)

    if not overwrite and os.path.exists(result_path):
        print(f"Skipping existing evaluation for config: {eval_config_file}")
        return

    eval_config_json = load_config_file(
        eval_config_file,
        default_config_file="_default.json"
    )
    eval_config = load_eval_config(eval_config_json)
    os.makedirs(result_folder, exist_ok=True)

    # Save a copy of the original config before preparation
    _eval_config = eval_config.copy()
    packet_wrapper_key = eval_config.get("packet_wrapper", None)

    # Try to load model, tokenizer from cache
    model_cache_key = (
        eval_config["model"]["model_path"],
        eval_config["model"]["dtype"],
        eval_config["model"]["device"],
    )

    packet_wrapper = eval_cache["packet_wrapper"].get(packet_wrapper_key, None)
    model = eval_cache["model"].get(model_cache_key, None)
    tokenizer = eval_cache["tokenizer"].get(model_cache_key, None)

    if packet_wrapper is None and packet_wrapper_key is not None:
        assert eval_config["cache_comb"]["method"] == "kv_packet", \
            "Packet wrapper is only compatible with 'kv_packet' cache_comb method."
        packet_wrapper = load_wrapper(
            packet_wrapper_key,
            device=torch.device(eval_config["model"]["device"])
        )
        print(f"Packet wrapper loaded {packet_wrapper}.")
        eval_cache["packet_wrapper"][packet_wrapper_key] = packet_wrapper

    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(
            eval_config["model"]["model_path"],
            dtype=eval_config["model"]["dtype"],
            device_map=torch.device(eval_config["model"]["device"]),
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            eval_config["model"]["model_path"]
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id + 2
        assert model.generation_config is not None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        eval_cache["model"][model_cache_key] = model
        eval_cache["tokenizer"][model_cache_key] = tokenizer

    # Prepare eval generator
    eval_generator = get_ret_eval_generator(
        name=eval_config["dataset"]["dataset_name"],
        num_samples=eval_config["dataset"]["num_samples"],
        num_data_strs=eval_config["dataset"]["num_data_strs"],
        num_shots=eval_config["dataset"]["num_shots"],
        subset=eval_config["dataset"]["subset"],
        split=eval_config["dataset"]["split"],
        seed=eval_config["dataset"]["seed"],
        data_kwargs=eval_config["dataset"]["data_kwargs"],
        template=eval_config["dataset"]["template"],
        template_kwargs=eval_config["dataset"]["template_kwargs"],
    )

    answer_postprocess_func = ANSWER_POSTPROCESS_DICT.get(
        eval_config["dataset"]["dataset_name"], None
    )
    # Set up generation config if provided
    if eval_config["model"]["generation_kwargs"]:
        generation_config: GenerationConfig|None = GenerationConfig(
            **eval_config["model"]["generation_kwargs"]
        )
    else:
        generation_config = None

    cache_comb_method = eval_config["cache_comb"]["method"]
    cache_comb_func = get_cache_comb_func(cache_comb_method)
    comb_kwargs = eval_config["cache_comb"].get("kwargs", {})

    compress_config = eval_config["compress"]
    if compress_config is not None:
        compression_ratio = compress_config['compression_ratio']
        compress_cls = PRESS_CLASSES.get(compress_config['method'], None)
        if compress_cls is None:
            raise ValueError(f"Unknown compression method: {compress_config['method']}")

        keep_filler_tokens = compress_config['keep_filler_tokens']
        compressor = compress_cls(
            compression_ratio=compression_ratio,
            **compress_config.get('kwargs', {})
        )
    else:
        compressor = None
        keep_filler_tokens = False

    # Run evaluation
    assert isinstance(model, SupportedModel), "Model type not supported."
    result = run_eval(
        model=model,
        tokenizer=tokenizer,
        eval_generator=eval_generator,
        cache_comb_func=cache_comb_func,
        cache_comb_kwargs=comb_kwargs,
        packet_wrapper=packet_wrapper,
        compressor=compressor,
        keep_filler_tokens=keep_filler_tokens,
        quantization_config=eval_config["quantization"],
        generation_config=generation_config,
        answer_postprocess_func=answer_postprocess_func,
        eval_config=eval_config,
        debug=debug,
    )

    eval_results[eval_config_file] = {
        "config": _eval_config,
        "result": result,
    }
    with open(result_path, "w") as f:
        json.dump(eval_results[eval_config_file], f, indent=4)



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to the training configuration file (glob pattern)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging."
    )
    args = parser.parse_args()

    config_files_or_paths: list[str] = args.config_files_or_paths
    overwrite: bool = args.overwrite
    debug: bool = args.debug
    assert isinstance(config_files_or_paths, list)

    all_config_files: set[str] = set()

    for pattern in config_files_or_paths:
        matched_paths = glob.glob(pattern, recursive=False)

        for path in matched_paths:
            try:
                configs = gather_config_files(
                    path,
                    pattern=r"\.json$",
                    skip_pattern=r"_default\.json"
                )
                for c in configs:
                    all_config_files.add(c)
            except ValueError as e:
                print(f"Warning: {e} Skipping path: {path}")


    sorted_config_files = sorted(list(all_config_files))

    if not sorted_config_files:
        print("No configuration files found. Please check the provided paths and patterns.")
        exit(1)

    else:
        print(f"Found {len(sorted_config_files)} configuration files:")
        for config_file in sorted_config_files:
            print(f"  {config_file}")

    eval_results: dict[str, dict] = {}

    ## Cache for models, tokenizers, corpora
    eval_cache: dict[str, dict] = {
        "model": {},
        "tokenizer": {},
        "packet_wrapper": {},
        "compressor": {}
    }

    print("\nStarting evaluation...\n")
    for eval_config_file in sorted_config_files:
        print(f"Running evaluation for config: {eval_config_file}...")
        run_one_config(
            eval_config_file,
            eval_cache,
            eval_results,
            overwrite=overwrite,
            debug=debug
        )
        print(f"Evaluation for config {eval_config_file} completed.")
