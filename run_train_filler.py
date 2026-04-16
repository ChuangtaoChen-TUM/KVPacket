import torch
import argparse
import sys
import random
import os
import gc
import re
from alive_progress import alive_it, alive_bar
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from kv_packet.dataset import get_ret_eval_generator
from kv_packet.utils.generate import GenerationCache, TokenizerType
from kv_packet.packet_wrapper import PacketWrapper, WrapperStateDict
from kv_packet.model import SupportedModel
from kv_packet.utils.config import gather_config_files, load_config_file
from kv_packet.utils.train_filler import (
    TrainSample,
    TrainConfig,
    sample_to_str,
    load_train_config,
    prepare_sample_input,
    batched_input_embed,
    packet_4d_mask,
    batched_packet_4d_mask,
    get_packed_labels,
    get_packed_logits,
    build_generation_cache,
    dtype_map
)


def train_wrapper_4d_batch(
    samples: list[TrainSample],
    model: SupportedModel,
    tokenizer: TokenizerType,
    packet_wrapper: PacketWrapper,
    batch_size: int,
    gen_batch_size: int,
    use_logits: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    generation_cache: GenerationCache|None=None,
    generation_config: GenerationConfig|None=None,
    device: torch.device = torch.device("cuda:0"),
    forward_batch_size: int = -1,
    epoch: int = -1,
    epoch_indices: list[int]|None = None,
):
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if forward_batch_size <= 0:
        forward_batch_size = batch_size
    
    if batch_size % forward_batch_size != 0:
        raise ValueError("batch_size must be multiple of forward_batch_size")

    if len(samples) % batch_size != 0:
        raise ValueError("Number of samples must be multiple of batch_size")

    # Determine which samples need generation
    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )

    # Training loop
    train_step = 0
    eval_tokens = 0
    acc_loss = 0.0

    # Shuffle samples
    num_samples = len(samples)
    # rand_indices = list(range(num_samples))
    # random.shuffle(rand_indices)
    if epoch_indices is None:
        epoch_indices = list(range(num_samples))
    else:
        assert len(epoch_indices) == num_samples

    batched_indices_list: list[list[int]] = []
    for i in range(0, num_samples, forward_batch_size):
        batched_indices_list.append(epoch_indices[i: i + forward_batch_size])

    samples_bar = alive_it(batched_indices_list)
    samples_bar.title = f"Train epoch {epoch}" if epoch >= 0 else "Train" # type: ignore

    if hasattr(model.config, "sliding_window"):
        sliding_window = model.config.sliding_window
        assert isinstance(sliding_window, int|None)
    else:
        sliding_window = None

    with alive_bar(total=num_samples, title="Training") as bar:
        for batched_indices in batched_indices_list:
            batched_samples = [samples[idx] for idx in batched_indices]
            samples_bar.text = f"Train step {train_step}/{len(samples)}" # type: ignore
            batch_input_embeds: list[torch.Tensor] = []
            batch_input_chunk_sizes: list[list[int]] = []
            batch_query_lens: list[int] = []
            batch_gen_seq_lens: list[int] = []

            for sample in batched_samples:
                prepared_input = prepare_sample_input(
                    sample,
                    model,
                    generation_cache,
                    tokenizer,
                    packet_wrapper,
                    device,
                )
                batch_input_embeds.append(prepared_input[0])
                batch_input_chunk_sizes.append(prepared_input[1])
                batch_query_lens.append(prepared_input[2])
                batch_gen_seq_lens.append(prepared_input[3])

            input_embed = batched_input_embed(batch_input_embeds)
            packet_attn_mask = batched_packet_4d_mask(
                batch_input_chunk_sizes,
                batch_query_lens,
                sliding_window=sliding_window,
                device=device,
            )

            max_gen_seq_len = max(batch_gen_seq_lens)
            eval_mask = torch.zeros(
                (len(batched_samples), max_gen_seq_len),
                dtype=input_embed.dtype,
                device=device
            ) # [f_batch_size, max_gen_seq_len]

            for i, gen_seq_len in enumerate(batch_gen_seq_lens):
                eval_mask[i, -gen_seq_len:] = 1.0

            outputs = model(
                inputs_embeds=input_embed,
                attention_mask=packet_attn_mask,
            )

            logits = outputs.logits
            assert isinstance(logits, torch.Tensor) # [f_batch_size, seq_len, vocab_size]

            logits_to_eval = logits[:, -max_gen_seq_len:, :]
            num_tokens = int(eval_mask.sum().item())
            eval_tokens += num_tokens

            if use_logits:
                target_logits = get_packed_logits(
                    batched_samples,
                    generation_cache,
                    padding_side='left',
                ) # [f_batch_size, max_gen_seq_len, vocab_size]
                target_logits = target_logits.log_softmax(dim=-1)
                logits_to_eval = logits_to_eval.log_softmax(dim=-1)

                assert target_logits.size() == logits_to_eval.size()

                loss_fct: torch.nn.Module = torch.nn.KLDivLoss(
                    reduction="none",
                    log_target=True
                )

                loss = loss_fct(
                    input=logits_to_eval,
                    target=target_logits,
                ).sum(-1) # [f_batch_size, max_gen_seq_len]

                loss = (loss * eval_mask).sum()
            
            else:
                target_ids = get_packed_labels(
                    batched_samples,
                    generation_cache,
                    padding_side='left',
                ).reshape(-1) # [f_batch_size * max_gen_seq_len]

                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none"
                )

                loss = loss_fct(
                    logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                    target_ids,
                ) # [f_batch_size * max_gen_seq_len]

                loss = (loss * eval_mask.reshape(-1)).sum()
            
            loss.backward()
            acc_loss += loss.item()

            train_step += forward_batch_size
            bar(forward_batch_size)

            if train_step % batch_size == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr = scheduler.get_last_lr()[0]
                print(f"eval tokens {eval_tokens}, loss {acc_loss / eval_tokens:.4f}, lr {lr:.3e}")
                eval_tokens = 0
                acc_loss = 0.0


def train_wrapper_4d(
    samples: list[TrainSample],
    model: SupportedModel,
    tokenizer: TokenizerType,
    packet_wrapper: PacketWrapper,
    batch_size: int,
    gen_batch_size: int,
    use_logits: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    generation_cache: GenerationCache|None=None,
    generation_config: GenerationConfig|None=None,
    # cache_path: str|None=None,
    device: torch.device = torch.device("cuda:0"),
    epoch: int = -1,
    epoch_indices: list[int]|None = None,
):
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.gradient_checkpointing_enable()

    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )

    # Training loop
    train_step = 0
    eval_tokens = 0
    acc_loss = 0.0
    # Shuffle samples
    # rand_indices = list(range(len(samples)))
    # random.shuffle(rand_indices)

    if epoch_indices is None:
        epoch_indices = list(range(len(samples)))
    else:
        assert len(epoch_indices) == len(samples)

    samples_bar = alive_it(epoch_indices)
    samples_bar.title = f"Train epoch {epoch}" if epoch >= 0 else "Train" # type: ignore

    # for sample in samples_bar:
    for idx in samples_bar:
        sample = samples[idx]
        samples_bar.text = f"Train step {train_step}/{len(samples)}" # type: ignore
        gen = generation_cache.get(sample_to_str(sample), device=device)
        assert gen is not None
        assert len(gen["sequences"]) == 1
        assert len(gen["logits"]) == 1 or not use_logits

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
            with torch.no_grad():
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

            with torch.no_grad():
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

        gen_seq = gen["sequences"][0].to(device)

        # We only need to predict up to the last token, thus the input is gen_seq[:-1]
        query_ids = torch.cat([query_ids, gen_seq[:-1].unsqueeze(0)], dim=1)

        query_len = query_ids.size(1)
        gen_seq_len = gen_seq.size(0)

        if hasattr(model.config, "sliding_window"):
            sliding_window = model.config.sliding_window
            assert isinstance(sliding_window, int|None)
        else:
            sliding_window = None

        packet_attn_mask = packet_4d_mask(
            input_chunk_sizes=input_chunk_sizes,
            query_len=query_len,
            sliding_window=sliding_window,
            device=device,
        )

        with torch.no_grad():
            query_embed = model.model.embed_tokens(query_ids)
        input_embed_list.append(query_embed)

        input_embed = torch.cat(input_embed_list, dim=1)
    
        outputs = model(
            inputs_embeds=input_embed,
            attention_mask=packet_attn_mask,
        )
        logits = outputs.logits

        assert isinstance(logits, torch.Tensor)
        logits_to_eval = logits[:, -gen_seq_len:, :]

        num_tokens = logits_to_eval.size(1)
        eval_tokens += num_tokens

        if use_logits:
            loss_fct: torch.nn.Module = torch.nn.KLDivLoss(
                reduction="sum",
                log_target=True
            )
            gen_logits = gen["logits"][0].to(device)
            logits_to_eval = logits_to_eval.log_softmax(dim=-1)
            target_logits = torch.nn.functional.log_softmax(gen_logits, dim=-1)
            loss = loss_fct(
                input=logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                target=target_logits.reshape(-1, logits_to_eval.size(-1)),
            )
        else:
            target_ids = gen_seq.unsqueeze(0)
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="sum"
            )
            loss = loss_fct(
                input=logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                target=target_ids.reshape(-1),
            )

        del outputs, logits, input_embed, packet_attn_mask
        torch.cuda.empty_cache()
        loss.backward()
        acc_loss += loss.detach().item()
        train_step += 1

        if train_step % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            print(f"eval tokens {eval_tokens}, loss {acc_loss / eval_tokens:.4f}, lr {lr:.3e}")
            eval_tokens = 0
            acc_loss = 0.0
            gc.collect()
            torch.cuda.empty_cache()


    if train_step % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()


class TrainCache:
    def __init__(self):
        self.tokenizer_cache: dict[str, TokenizerType] = {}
        self.model_cache: dict[tuple[str, str, str], SupportedModel] = {}


def train_one_config(
    train_config: TrainConfig,
    train_cache: TrainCache,
):
    print("Training configuration:")
    pprint(train_config)

    if os.path.exists(train_config["save_path"]) is False:
        os.makedirs(train_config["save_path"], exist_ok=True)

    use_cache = train_config["use_cache"]

    if train_config["model"]["device"] == "auto":
        device_map: torch.device|str= "auto"
        model_device = torch.device("cuda:0")
    else:
        device_map = torch.device(train_config["model"]["device"])
        model_device = torch.device(train_config["model"]["device"])

    # if resume_checkpoint:
    #     assert os.path.exists(resume_checkpoint), f"Checkpoint {resume_checkpoint} does not exist."

    if not use_cache or train_cache.tokenizer_cache.get(train_config["model"]["model_path"]) is None:
        tokenizer: TokenizerType = AutoTokenizer.from_pretrained(
            train_config["model"]["model_path"],
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if use_cache:
            train_cache.tokenizer_cache[train_config["model"]["model_path"]] = tokenizer
    else:
        tokenizer = train_cache.tokenizer_cache[train_config["model"]["model_path"]]

    model_key: tuple[str, str, str] = (
        train_config["model"]["model_path"],
        train_config["model"]["dtype"],
        train_config["model"]["device"],
    )

    if train_cache.model_cache.get(model_key) is None:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            train_config["model"]["model_path"],
            dtype=train_config["model"]["dtype"],
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        assert isinstance(loaded_model, SupportedModel)
        model = loaded_model
        assert model.generation_config is not None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        train_cache.model_cache[model_key] = model
    else:
        model = train_cache.model_cache[model_key]

    cache_device = torch.device(train_config["cache_device"])

    mean = torch.mean(model.model.embed_tokens.weight).item()
    std = torch.std(model.model.embed_tokens.weight).item()
    print(f"Embedding mean: {mean:.3e}, std: {std:.3e}")

    wrapper_dtype = train_config["dtype"] if train_config["dtype"] is not None else train_config["model"]["dtype"]

    resume = train_config["resume"]
    resume_epoch: int|None = train_config["resume_epoch"]
    resume_checkpoint: str|None = None

    if resume:
        save_path = train_config["save_path"]
        file_name = train_config["file_name"]

        if resume_epoch is not None:
            # Try to find the specific epoch checkpoint
            target = os.path.join(save_path, f"{file_name}.epoch{resume_epoch}")
            if not os.path.exists(target):
                raise ValueError(f"Checkpoint for epoch {resume_epoch} not found: {target}")
            resume_checkpoint = target
        else:
            # Find the latest checkpoint in the save directory
            ckpt_files = [
                f for f in os.listdir(save_path)
                if re.search(rf"^{re.escape(file_name)}\.epoch(\d+)$", f)
            ]
            if ckpt_files:
                ckpt_files.sort(key=lambda f: int(m.group(1)) if (m := re.search(r"\.epoch(\d+)$", f)) else -1)
                resume_checkpoint = os.path.join(save_path, ckpt_files[-1])
            else:
                print(f"No checkpoints found in {save_path}, starting from scratch.")

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=model_device)
        header = checkpoint["header"]
        trailer = checkpoint["trailer"]

        assert isinstance(header, torch.Tensor)
        assert isinstance(trailer, torch.Tensor)

        packet_wrapper = PacketWrapper.from_state_dict(
            WrapperStateDict(
                header=header,
                trailer=trailer,
                train_config=checkpoint.get("train_config", None),
            ),
            device=model_device
        )
        assert packet_wrapper.dim == model.config.hidden_size, (
            f"Packet wrapper dim {packet_wrapper.dim} does not match model hidden size {model.config.hidden_size}"
        )
        assert packet_wrapper.header_len == train_config["header_len"], (
            f"Packet wrapper header len {packet_wrapper.header_len} does not match training config {train_config['header_len']}"
        )
        assert packet_wrapper.trailer_len == train_config["trailer_len"], (
            f"Packet wrapper trailer len {packet_wrapper.trailer_len} does not match training config {train_config['trailer_len']}"
        )
        epoch_match = re.search(r"\.epoch(\d+)$", resume_checkpoint)
        assert epoch_match, "Could not parse epoch number from checkpoint filename."
        start_epoch = int(epoch_match.group(1))
        print(f"Resuming from epoch: {start_epoch}")
    else:
        assert model.config.hidden_size is not None, "Model config must have hidden_size defined."
        packet_wrapper = PacketWrapper(
            header_len=train_config["header_len"],
            trailer_len=train_config["trailer_len"],
            dim=model.config.hidden_size,
            dtype=dtype_map[wrapper_dtype],
            mean=mean,
            std=std,
            device=model_device
        )
        start_epoch = 0

    optimizer_config = train_config["opt_config"]
    optimizer = torch.optim.AdamW(
        params=[
            packet_wrapper.header,
            packet_wrapper.trailer,
        ],
        **optimizer_config
    )

    samples: list[TrainSample] = []

    for data_config in train_config["data_configs"]:
        eval_generator = get_ret_eval_generator(
            name=data_config["dataset_name"],
            num_samples=data_config["num_samples"],
            num_data_strs=data_config["num_data_strs"],
            num_shots=data_config["num_shots"],
            subset=data_config["subset"],
            split=data_config["split"],
            seed=data_config["seed"],
            data_kwargs=data_config["data_kwargs"],
            template=data_config["template"],
            template_kwargs=data_config["template_kwargs"],
        )
        samples.extend([TrainSample(**sample,) for sample in eval_generator]) # type: ignore

    num_samples = len(samples)
    print(f"Total training samples: {num_samples}")

    if num_samples % train_config["batch_size"] != 0:
        print(f"Warning: number of samples {num_samples} is not divisible by batch size {train_config['batch_size']}.")

    iter_per_epoch = num_samples // train_config["batch_size"]
    total_iter = iter_per_epoch * train_config["total_epoch"]

    scheduler_config = train_config["scheduler_config"]

    if scheduler_config.get("total_iters", 0) == 0:
        scheduler_config["total_iters"] = total_iter

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        **scheduler_config
    )

    random.seed(train_config["seed"])
    epoch_indices = list(range(num_samples))

    if start_epoch > 0:
        steps_to_skip = start_epoch * iter_per_epoch
        print(f"Advancing scheduler {steps_to_skip} steps...")
        for _ in range(steps_to_skip):
            scheduler.step()
    
        for _ in range(start_epoch):
            random.shuffle(epoch_indices)

    cache_path = train_config["cache_path"]
    if cache_path is not None:
        if os.path.exists(cache_path):
            generation_cache = GenerationCache.load_from_file(cache_path, device=cache_device)
            print(f"Loaded generation cache from {cache_path}, size: {len(generation_cache.cache)}")
        else:
            generation_cache = GenerationCache(device=cache_device)
            print(f"Initialized new generation cache at {cache_path}")
    else:
        generation_cache = GenerationCache(device=cache_device)

    if train_config['model']["generation_kwargs"]:
        generation_config = GenerationConfig(
            **train_config["model"]["generation_kwargs"]
        )
    else:
        generation_config = None

    batch_size = train_config["batch_size"]
    gen_batch_size = train_config["gen_batch_size"]
    forward_batch_size = train_config["forward_batch_size"]
    total_epoch = train_config["total_epoch"]
    use_logits = train_config["use_logits"]
    cache_path = train_config["cache_path"]

    old_cache_len = len(generation_cache.cache)
    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )
    new_cache_len = len(generation_cache.cache)

    if new_cache_len > old_cache_len and cache_path is not None:
        torch.save(generation_cache.to_state_dict(), cache_path)
        print(f"Saved updated generation cache to {cache_path}, size: {new_cache_len}")

    try:
        for epoch in range(start_epoch, total_epoch):
            print(f"Epoch {epoch + 1}/{total_epoch}")
            random.shuffle(epoch_indices)
            if forward_batch_size == 1: # this is faster for batch size 1
                train_wrapper_4d(
                    samples=samples,
                    model=model,
                    batch_size=batch_size,
                    gen_batch_size=gen_batch_size,
                    use_logits=use_logits,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    packet_wrapper=packet_wrapper,
                    generation_cache=generation_cache,
                    generation_config=generation_config,
                    device=model_device,
                    epoch=epoch,
                    epoch_indices=epoch_indices,
                )
            else:
                train_wrapper_4d_batch(
                    samples=samples,
                    model=model,
                    batch_size=batch_size,
                    gen_batch_size=gen_batch_size,
                    use_logits=use_logits,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    packet_wrapper=packet_wrapper,
                    generation_cache=generation_cache,
                    generation_config=generation_config,
                    device=model_device,
                    forward_batch_size=forward_batch_size,
                    epoch=epoch,
                    epoch_indices=epoch_indices,
                )
            gc.collect()
            torch.cuda.empty_cache()
            if train_config["ckpt_epoch"] > 0 and (epoch + 1) % train_config["ckpt_epoch"] == 0 and (epoch + 1) < total_epoch:
                state_dict = packet_wrapper.state_dict()
                state_dict["train_config"] = train_config # type: ignore
                torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}.epoch{epoch + 1}")
                print(f"Saved checkpoint to {train_config['save_path']}/{train_config['file_name']}.epoch{epoch + 1}")

        state_dict = packet_wrapper.state_dict()
        state_dict["train_config"] = train_config # type: ignore
        torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}")
        print(f"Saved trained packet to {train_config['save_path']}/{train_config['file_name']}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving packet.")
        state_dict = packet_wrapper.state_dict()
        state_dict["train_config"] = train_config # type: ignore
        torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}.interrupt")
        print(f"Saved trained filler to {train_config['save_path']}/{train_config['file_name']}.interrupt")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to the training configuration file (JSON format)."
    )
    args = parser.parse_args()

    config_files_or_paths: list[str] = args.config_files_or_paths
    assert isinstance(config_files_or_paths, list)

    all_config_files: list[str] = []
    for file_or_path in config_files_or_paths:
        config_files = gather_config_files(file_or_path, pattern=r".*\.json$")
        all_config_files.extend(config_files)

    train_configs: list[TrainConfig] = [
        load_train_config(load_config_file(
            config_file, default_config_file="_default.json"
        )) for config_file in all_config_files
    ]

    print(f"Loaded {len(train_configs)} training configurations.")
    for config_file in all_config_files:
        print(f" - {config_file}")

    train_cache = TrainCache()

    for train_config in train_configs:
        train_one_config(train_config, train_cache)
