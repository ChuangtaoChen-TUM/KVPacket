import random
import warnings
from typing import Iterator
from datasets import load_dataset, Dataset
from .abc import RetEvalEntry
from .utils import split_with_nltk, clean_with_prefixes

TRAIN_RATIO = 0.5
NIAH_DATASET: dict[str, Dataset|None] = {}


def niah_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "8192",
    split: str = "test",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    cache_dataset = kwargs.pop("cache_dataset", True)
    chunk_size = kwargs.pop("chunk_size", 4096)
    max_len = kwargs.pop("max_len", -1)

    if kwargs:
        warnings.warn(f"Unused kwargs in niah_ret_eval_generator: {kwargs}")

    if num_data_strs > 0:
        raise ValueError("NIAH dataset requires num_data_strs to be 0 to use all passages.")

    if cache_dataset:
        cached_ds = NIAH_DATASET.get(subset)
    else:
        cached_ds = None

    if cached_ds is not None:
        ds = cached_ds
    else:
        ds = load_dataset("simonjegou/ruler", subset, split="test")
        assert isinstance(ds, Dataset)

    ds_len = len(ds)
    train_size = int(ds_len * TRAIN_RATIO)
    all_indices = list(range(ds_len))

    shuffle_rng = random.Random(42) # Fixed seed for dataset shuffling
    shuffle_rng.shuffle(all_indices)

    if split == "train":
        all_indices = all_indices[:train_size]
    else:
        all_indices = all_indices[train_size:]

    rng = random.Random(seed)
    rng.shuffle(all_indices)

    few_shot_indices = all_indices[-num_shots:] if num_shots > 0 else []
    
    if ds_len - num_shots < num_samples:
        raise ValueError(
            f"Not enough samples in the dataset after accounting for few-shots: "
            f"requested {num_samples} samples but only {ds_len - num_shots} available."
        )
    
    selected_indices = all_indices[: ds_len - num_shots]

    few_shot_strs: list[str] = []

    if num_shots > 0:
        warnings.warn("Do not recommend using few-shot for NIAH dataset.")

    for idx in few_shot_indices:
        sample = ds[idx]
        context = sample["context"]
        question = sample["question"]
        answers: list[str] = sample["answer"]
        few_shot_str = f"{context}\nQuestion: {question}\nShort Answer: {' '.join(answers)}"
    if few_shot_strs:
        few_shot_str = "\n".join(few_shot_strs) + "\n"
    else:
        few_shot_str = ""

    for _ in range(num_samples):
        sample_idx, selected_indices = next_valid_sample(
            ds,
            selected_indices,
            max_len,
        )
        sample = ds[sample_idx]
        context = sample["context"]
        question = sample["question"]
        answer = " ".join(sample["answer"])
        data_strs = split_with_nltk(
            context, max_chunk_size=chunk_size
        )
        question = f"{question}\nProvide a short answer directly separated by space. Short Answer:"
        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=data_strs,
            task_prompt=question,
            query=question,
            answer=answer,
        )


def next_valid_sample(
    ds: Dataset,
    indices: list[int],
    max_len: int,
) -> tuple[int, list[int]]:
    counter = 0
    while counter < len(indices):
        idx = indices[counter]
        sample = ds[idx]
        context = sample["context"]
        if len(context) <= max_len or max_len < 0:
            return idx, indices[counter + 1 :]
        counter += 1
    raise ValueError("No valid sample found within the given max_len.")


def niah_answer_postprocess(
    pred_answer: str,
    gold_answer: str
) -> tuple[str, str]:
    
    # 1. Normalize basic formatting
    pred_answer = pred_answer.lower().strip().rstrip(".")
    gold_answer = gold_answer.lower().strip().rstrip(".")

    # 2. List common chatty prefixes to remove
    # (Order matters: longer phrases first to avoid partial cuts)
    prefixes = [
        "is:",
        "are:",
        "is",
        "are",
        "in"
    ]
    pred_answer = clean_with_prefixes(pred_answer, prefixes).replace(",", " ")
    pred_answer = pred_answer.replace("and", " ")

    return pred_answer, gold_answer