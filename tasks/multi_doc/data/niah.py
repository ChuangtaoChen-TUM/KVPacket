"""RULER needle-in-a-haystack retrieval generator."""
from __future__ import annotations

import random
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from ._utils import split_sentences
from .base import RetrievalSample

__all__ = ["niah_generator", "niah_answer_postprocess"]

_NIAH_CACHE: dict[str, Sequence[Mapping[str, Any]]] = {}
_TRAIN_RATIO = 0.5


def _load_niah(subset: str, cache: bool) -> Sequence[Mapping[str, Any]]:
    if cache and subset in _NIAH_CACHE:
        return _NIAH_CACHE[subset]
    from datasets import Dataset, load_dataset

    dataset = load_dataset("simonjegou/ruler", subset, split="test")
    if not isinstance(dataset, Dataset):
        raise TypeError("RULER loader did not return a Dataset split")
    typed_dataset = cast(Sequence[Mapping[str, Any]], dataset)
    if cache:
        _NIAH_CACHE[subset] = typed_dataset
    return typed_dataset


def _answer(value: object) -> str:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, str) for item in value)
    ):
        return " ".join(cast(Sequence[str], value))
    raise ValueError("NIAH answer must be a string or sequence of strings")


def _row(dataset: Sequence[Mapping[str, Any]], index: int) -> tuple[str, str, str]:
    sample = dataset[index]
    context = sample.get("context")
    question = sample.get("question")
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError("NIAH sample has malformed context or question")
    return context, question, _answer(sample.get("answer"))


def niah_generator(
    num_samples: int,
    num_documents: int = 0,
    num_shots: int = 0,
    subset: str = "8192",
    split: str = "test",
    seed: int = 42,
    cache_dataset: bool = True,
    chunk_size: int = 4096,
    max_len: int = -1,
    **kwargs: object,
) -> Iterator[RetrievalSample]:
    if kwargs:
        warnings.warn(f"unused kwargs in niah_generator: {kwargs}")
    if num_samples < 0 or num_shots < 0:
        raise ValueError("num_samples and num_shots must be non-negative")
    if num_documents != 0:
        raise ValueError("NIAH uses every context chunk; num_documents must be 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_len == 0 or max_len < -1:
        raise ValueError("max_len must be -1 or positive")
    if split not in {"train", "test"}:
        raise ValueError("NIAH split must be 'train' or 'test'")

    dataset = _load_niah(subset, cache_dataset)
    indices = list(range(len(dataset)))
    partition_rng = random.Random(42)
    partition_rng.shuffle(indices)
    boundary = int(len(indices) * _TRAIN_RATIO)
    indices = indices[:boundary] if split == "train" else indices[boundary:]
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_len > 0:
        indices = [
            index for index in indices if len(_row(dataset, index)[0]) <= max_len
        ]
    if num_shots + num_samples > len(indices):
        raise ValueError(
            "not enough matching NIAH samples after accounting for few-shots: "
            f"requested {num_samples} samples and {num_shots} shots, "
            f"but only {len(indices)} are available"
        )

    shot_indices = indices[-num_shots:] if num_shots else []
    selected_indices = indices[:num_samples]
    if shot_indices:
        warnings.warn("few-shot prompting is not recommended for NIAH")
    shot_strings: list[str] = []
    for index in shot_indices:
        context, question, answer = _row(dataset, index)
        shot_strings.append(
            f"{context}\nQuestion: {question}\nShort Answer: {answer}"
        )
    preamble = ("\n".join(shot_strings) + "\n") if shot_strings else ""

    for index in selected_indices:
        context, question, answer = _row(dataset, index)
        documents = split_sentences(context, max_chunk_size=chunk_size)
        if not documents:
            raise ValueError("NIAH context produced no document chunks")
        task_prompt = (
            f"{question}\n"
            "Provide a short answer directly separated by space. Short Answer:"
        )
        yield RetrievalSample(
            preamble=preamble,
            documents=documents,
            task_prompt=task_prompt,
            query=question,
            answer=answer,
            meta={"index": index, "subset": subset},
        )


def niah_answer_postprocess(pred: str, gold: str) -> tuple[str, str]:
    cleaned = pred.lower().strip().rstrip(".")
    normalized_gold = gold.lower().strip().rstrip(".")
    answer_parts = re.split(r"(?i)(?:short\s+)?answer\s*:", cleaned)
    if len(answer_parts) > 1:
        cleaned = answer_parts[-1].strip()
    for prefix in ("is:", "are:", "is ", "are ", "in "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip(" :.,")
            break
    cleaned = " ".join(cleaned.replace(",", " ").replace(" and ", " ").split())
    return cleaned, normalized_gold
