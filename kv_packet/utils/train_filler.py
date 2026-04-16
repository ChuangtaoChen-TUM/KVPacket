import torch
from warnings import warn
from typing import TypedDict
from alive_progress import alive_bar
from .generate import (
    get_generation,
    GenerationConfig,
    GenerationCache,
    TokenizerType
)
from ..model import SupportedModel
from ..packet_wrapper import PacketWrapper
from ..dataset.abc import RetEvalEntry


class TrainSample(RetEvalEntry):
    pass


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


class TrainConfig(TypedDict):
    total_epoch: int
    gen_batch_size: int
    batch_size: int
    forward_batch_size: int
    header_len: int
    trailer_len: int
    dtype: str|None
    use_logits: bool
    use_cache: bool
    model: ModelConfig
    cache_device: str
    cache_path: str|None
    seed: int
    save_path: str
    file_name: str
    ckpt_epoch: int
    resume: bool
    resume_epoch: int|None
    opt_config: dict
    scheduler_config: dict
    data_configs: list[DatasetConfig]


dtype_map: dict[str, torch.dtype] = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'double': torch.double,
    'int64': torch.int64,
    'long': torch.long,
    'bool': torch.bool
}


def load_train_config(config: dict) -> TrainConfig:
    model = ModelConfig(
        model_path=config["model"]["model_path"],
        dtype=config["model"].get("dtype", "bfloat16"),
        device=config["model"].get("device", "cuda:0"),
        generation_kwargs=config["model"].get("generation_kwargs", {}),
    )
    data_configs: list[DatasetConfig] = []
    for data_conf in config["data_configs"]:
        data_config = DatasetConfig(
            dataset_name=data_conf["dataset_name"],
            num_data_strs=data_conf["num_data_strs"],
            num_shots=data_conf["num_shots"],
            num_samples=data_conf["num_samples"],
            subset=data_conf["subset"],
            split=data_conf.get("split", "train"),
            seed=data_conf.get("seed", 42),
            data_kwargs=data_conf.get("data_kwargs", {}),
            template=data_conf.get("template", ""),
            template_kwargs=data_conf.get("template_kwargs", {}),
        )
        data_configs.append(data_config)
    seed = config.get("seed", 42)
    cache_device = config.get("cache_device", "cuda:0")
    train_config = TrainConfig(
        total_epoch=config["total_epoch"],
        gen_batch_size=config["gen_batch_size"],
        batch_size=config["batch_size"],
        forward_batch_size=config["forward_batch_size"],
        header_len=config["header_len"],
        trailer_len=config["trailer_len"],
        dtype=config.get("dtype", None),
        use_logits=config["use_logits"],
        use_cache=config.get("use_cache", False),
        model=model,
        cache_device=cache_device,
        cache_path=config.get("cache_path", None),
        seed=seed,
        save_path=config["save_path"],
        file_name=config["file_name"],
        ckpt_epoch=config["ckpt_epoch"],
        opt_config=config["opt_config"],
        scheduler_config=config["scheduler_config"],
        data_configs=data_configs,
        resume=config.get("resume", False),
        resume_epoch=config.get("resume_epoch", None),
    )
    return train_config


def packet_4d_mask(
    input_chunk_sizes: list[int],
    query_len: int,
    sliding_window: int|None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a 4D attention mask for packet-KV with 4D attention.

    Args:
        input_chunk_sizes (list[int]): List of chunk lengths in the input sequence.
        query_len (int): Length of the query tokens.
    Returns:
        torch.Tensor: The attention mask tensor of shape (1, 1, seq_len, seq_len).

    The attention mask is constructed such that:
    - The input chunks can only attend themselves.
    - The query tokens can attend to all input chunks and themselves.

    For example:
    If input_chunk_sizes = [3, 5, 2] and query_len = 4, the resulting attention mask will be:
    [[
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]]
    """
    total_chunk_len = sum(input_chunk_sizes)
    total_seq_len = total_chunk_len + query_len
    attn_mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

    start = 0
    chunks_sizes_w_query = input_chunk_sizes + [query_len]

    for size in chunks_sizes_w_query:
        end = start + size
        attn_mask[start:end, start:end] = torch.tril(
            torch.ones((size, size), dtype=torch.bool, device=device)
        )
        start = end

    attn_mask[-query_len:, :-query_len] = 1  # Query tokens can attend to all
    
    if sliding_window is not None:
        assert sliding_window > 0, "sliding_window must be positive."

        if total_chunk_len > sliding_window:
            warn(
                f"Total input chunk length ({total_chunk_len}) exceeds the sliding window "
                f"size ({sliding_window}). The query tokens will not be able to attend to "
                "the earliest chunks, which may cause information loss during retrieval.",
                UserWarning
            )
        
        swa_mask = torch.triu(
            torch.ones((total_seq_len, total_seq_len), dtype=torch.bool, device=device),
            diagonal=-sliding_window
        )
        attn_mask = attn_mask & swa_mask

    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    return attn_mask


def batched_packet_4d_mask(
    batch_input_chunk_sizes: list[list[int]],
    batch_query_len: list[int],
    sliding_window: int|None = None,
    padding_side: str = 'left',
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    mask_list: list[torch.Tensor] = [
        packet_4d_mask(
            input_chunk_sizes=input_chunk_sizes,
            query_len=query_len,
            sliding_window=sliding_window,
            device=device,
        )
        for input_chunk_sizes, query_len in zip(
            batch_input_chunk_sizes, batch_query_len
        )
    ]

    max_seq_len = max(mask.size(-1) for mask in mask_list)
    batch_size = len(mask_list)

    padded_masks = torch.zeros(
        (batch_size, 1, max_seq_len, max_seq_len),
        dtype=torch.bool, device=device
    )

    for i, mask in enumerate(mask_list):
        seq_len = mask.size(-1)
        if padding_side == 'left':
            padded_masks[i, :, -seq_len:, -seq_len:] = mask
        else:
            padded_masks[i, :, :seq_len, :seq_len] = mask
    return padded_masks


def batched_input_embed(
    input_embed_list: list[torch.Tensor],
    padding_side: str = 'left',
    padding_tensor: torch.Tensor|None = None,
) -> torch.Tensor:
    """ 
    Pad a batch of input embeddings to the same sequence length.
    Args:
        input_embed_list (list[torch.Tensor]): List of input embeddings of shape (1, seq_len, dim).
        padding_side (str): 'left' or 'right' padding.
        padding_tensor (torch.Tensor|None): Optional tensor (1, 1, dim) to use for padding instead of zeros.
    Returns:
        torch.Tensor: Padded input embeddings of shape (batch_size, max_seq_len, dim).
    """
    if len(input_embed_list) == 0:
        return torch.zeros((0, 0, 0))

    dim = input_embed_list[0].size(2)

    if padding_tensor is not None:
        assert padding_tensor.dim() == 3
        assert padding_tensor.size(0) == 1 and padding_tensor.size(1) == 1
        assert padding_tensor.size(2) == dim
    else:
        padding_tensor = torch.zeros((1, 1, dim)).to(input_embed_list[0])

    max_seq_len = max(embed.size(1) for embed in input_embed_list)
    batch_size = len(input_embed_list)

    padded_embeds = torch.zeros(
        (batch_size, max_seq_len, input_embed_list[0].size(2)),
        dtype=input_embed_list[0].dtype,
        device=input_embed_list[0].device
    )

    for i, embed in enumerate(input_embed_list):
        seq_len = embed.size(1)
        pad_len = max_seq_len - seq_len
        
        if pad_len > 0:
            pad_embed = padding_tensor.repeat(1, pad_len, 1)
            if padding_side == 'left':
                padded_embeds[i] = torch.cat([pad_embed, embed], dim=1)
            else:
                padded_embeds[i] = torch.cat([embed, pad_embed], dim=1)
        else:
            padded_embeds[i] = embed
    
    return padded_embeds


def sample_to_str(sample: TrainSample) -> str:
    return sample["preamble"] + " ".join(sample["documents"]) + " " + sample["task_prompt"]

def prepare_sample_input(
    sample: TrainSample,
    model: SupportedModel,
    generation_cache: GenerationCache,
    tokenizer: TokenizerType,
    packet_wrapper: PacketWrapper,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], int, int]:
    sample_str = sample_to_str(sample)
    generation = generation_cache.get(sample_str)
    assert generation is not None
    assert len(generation["sequences"]) == 1

    input_chunk_sizes: list[int] = []
    input_embed_list: list[torch.Tensor] = []

    if sample["preamble"]:
        context_input = tokenizer(
            [sample["preamble"]],
            add_special_tokens=False,
            return_tensors="pt",
        )
        context_ids = context_input["input_ids"]
        assert isinstance(context_ids, torch.Tensor)

        input_chunk_sizes.append(context_ids.size(1))
        context_embed = model.model.embed_tokens(context_ids.to(device))
        input_embed_list.append(context_embed)
    
    for data_str in sample["documents"]:
        data_input = tokenizer(
            [data_str],
            add_special_tokens=False,
            return_tensors="pt",
        )
        data_ids = data_input["input_ids"]
        assert isinstance(data_ids, torch.Tensor)

        data_embed = model.model.embed_tokens(data_ids.to(device))
        wrapped_data_embed = packet_wrapper.wrap(data_embed)

        input_chunk_sizes.append(wrapped_data_embed.size(1))
        input_embed_list.append(wrapped_data_embed)
    
    query_input = tokenizer(
        sample["task_prompt"],
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    query_ids = query_input["input_ids"]
    assert isinstance(query_ids, torch.Tensor)

    gen_seq = generation["sequences"][0].to(device)
    # We only need to predict up to the last token, thus the input is gen_seq[:-1]
    query_ids = torch.cat([query_ids, gen_seq[:-1].unsqueeze(0)], dim=1)

    query_len = query_ids.size(1)
    gen_seq_len = gen_seq.size(0)

    query_embed = model.model.embed_tokens(query_ids)
    input_embed_list.append(query_embed)

    input_embed = torch.cat(input_embed_list, dim=1)

    return (
        input_embed,
        input_chunk_sizes,
        query_len,
        gen_seq_len,
    )


def get_packed_logits(
    samples: list[TrainSample],
    generation_cache: GenerationCache,
    padding_side: str = 'left',
):
    """
    Retrieve and batch the logits from the generation cache for the given samples.

    Args:
        samples (list[TrainSample]): List of training samples.
        generation_cache (GenerationCache): Cache containing generation outputs.
        padding_side (str): 'left' or 'right' padding for batching.
    Returns:
        torch.Tensor: Batched logits tensor. [batch_size, max_gen_seq_len, vocab_size].
    """
    packed_logits_list: list[torch.Tensor] = []

    for sample in samples:
        sample_str = sample_to_str(sample)
        generation = generation_cache.get(sample_str)
        assert generation is not None
        assert len(generation["logits"]) == 1

        gen_logits = generation["logits"][0].unsqueeze(0) # [1, gen_seq_len, vocab_size]
        packed_logits_list.append(gen_logits)

    batched_logits = batched_input_embed(
        input_embed_list=packed_logits_list,
        padding_side=padding_side
    )

    return batched_logits


def get_packed_labels(
    samples: list[TrainSample],
    generation_cache: GenerationCache,
    padding_side: str = 'left',
) -> torch.Tensor:
    """
    Retrieve and batch the label sequences from the generation cache for the given samples.

    Args:
        samples (list[TrainSample]): List of training samples.
        generation_cache (GenerationCache): Cache containing generation outputs.
        padding_side (str): 'left' or 'right' padding for batching.
    Returns:
        torch.Tensor: Batched label tensor. [batch_size, max_gen_seq_len].
    """
    label_list: list[torch.Tensor] = []

    for sample in samples:
        sample_str = sample_to_str(sample)
        generation = generation_cache.get(sample_str)
        assert generation is not None
        assert len(generation["sequences"]) == 1

        gen_seq = generation["sequences"][0] # [gen_seq_len]
        label_list.append(gen_seq)

    max_seq_len = max(label.size(0) for label in label_list)
    batched_labels = torch.zeros(
        (len(label_list), max_seq_len),
        dtype=torch.long,
        device=label_list[0].device if label_list else torch.device("cpu"),
    )
    for i, label in enumerate(label_list):
        seq_len = label.size(0)
        pad_len = max_seq_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len,), dtype=torch.long).to(label.device)
            if padding_side == 'left':
                batched_labels[i] = torch.cat([pad_tensor, label], dim=0)
            else:
                batched_labels[i] = torch.cat([label, pad_tensor], dim=0)
        else:
            batched_labels[i] = label
    return batched_labels


def build_generation_cache(
    samples: list[TrainSample],
    batch_size: int,
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None = None,
    generation_cache: GenerationCache|None = None,
    store_logits: bool = True,
) -> GenerationCache:
    if generation_cache is None:
        generation_cache = GenerationCache()
    
    # Determine which samples need generation
    samples_to_gen = [
        sample for sample in samples
        if sample_to_str(sample) not in generation_cache
    ]

    if len(samples_to_gen) == 0:
        return generation_cache

    input_strs = [
        sample["preamble"] + " ".join(sample["documents"]) + 
        " " + sample["task_prompt"] for sample in samples_to_gen
    ]

    if batch_size <= 0:
        batch_size = len(input_strs)
    
    with alive_bar(total=len(input_strs), title="Building generation cache") as bar:
        for i in range(0, len(input_strs), batch_size):
            batch_input_strs = input_strs[i: i + batch_size]
            gen = get_generation(
                model,
                tokenizer,
                input_strs=batch_input_strs,
                generation_config=generation_config,
            )
            gen_len = len(batch_input_strs)
            bar(gen_len)

            for k in range(gen_len):
                sample_index = i + k
                sample_str = sample_to_str(samples_to_gen[sample_index])
                generation_cache.add(sample_str, {
                    "sequences": [gen["sequences"][k]],
                    "logits": [gen["logits"][k]] if store_logits else [],
                    "text": [gen["text"][k]],
                })

    return generation_cache
