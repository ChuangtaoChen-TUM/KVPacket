"""MultiNews multi-document long-form summarization generator."""
from __future__ import annotations

import random
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from .base import RetrievalSample

__all__ = ["multinews_generator", "multinews_answer_postprocess"]

_MULTINEWS_CACHE: dict[str, Sequence[Mapping[str, Any]]] = {}


def _load_multinews(split: str, cache: bool) -> Sequence[Mapping[str, Any]]:
    if cache and split in _MULTINEWS_CACHE:
        return _MULTINEWS_CACHE[split]
    from datasets import Dataset, load_dataset

    dataset = load_dataset(
        "Awesome075/multi_news_parquet",
        split=split,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("MultiNews loader did not return a Dataset split")
    typed = cast(Sequence[Mapping[str, Any]], dataset)
    if cache:
        _MULTINEWS_CACHE[split] = typed
    return typed


def _documents(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError("MultiNews sample has malformed document text")
    return [part.strip() for part in raw.split("|||||") if part.strip()]


def _prompt() -> str:
    return (
        "Write a self-contained 150-250-word news summary that synthesizes the "
        "provided articles. Integrate the important facts across sources, resolve "
        "overlap or differing emphasis, preserve factual attribution, and omit "
        "irrelevant details. Use coherent prose and do not mention these instructions."
    )


def multinews_generator(
    num_samples: int,
    num_documents: int = 0,
    num_shots: int = 0,
    subset: str = "default",
    split: str = "train",
    seed: int = 42,
    min_documents: int = 2,
    max_documents: int = 8,
    max_total_chars: int = 30_000,
    cache_dataset: bool = True,
    **kwargs: object,
) -> Iterator[RetrievalSample]:
    """Yield bounded MultiNews examples with one packet per source article."""
    if kwargs:
        warnings.warn(f"unused kwargs in multinews_generator: {kwargs}")
    if subset != "default":
        raise ValueError("MultiNews supports only subset='default'")
    if num_documents != 0:
        raise ValueError("MultiNews controls its document count; use num_documents=0")
    if num_shots != 0:
        raise ValueError("MultiNews long-form evaluation does not use few-shot examples")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if min_documents < 2 or max_documents < min_documents:
        raise ValueError("require 2 <= min_documents <= max_documents")
    if max_total_chars <= 0:
        raise ValueError("max_total_chars must be positive")

    dataset = _load_multinews(split, cache_dataset)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = 0
    for index in indices:
        sample = dataset[index]
        documents = _documents(sample.get("document"))
        summary = sample.get("summary")
        if not isinstance(summary, str):
            raise ValueError("MultiNews sample has malformed summary")
        if not min_documents <= len(documents) <= max_documents:
            continue
        if sum(len(document) for document in documents) > max_total_chars:
            continue
        yield RetrievalSample(
            preamble="",
            documents=documents,
            task_prompt=_prompt(),
            query="Synthesize the supplied news articles.",
            answer=summary.strip(),
            meta={
                "index": index,
                "num_documents": len(documents),
                "total_document_chars": sum(len(document) for document in documents),
            },
        )
        selected += 1
        if selected == num_samples:
            return
    if selected != num_samples:
        raise ValueError(
            f"only {selected} MultiNews examples satisfy the length constraints; "
            f"requested {num_samples}"
        )


def multinews_answer_postprocess(pred: str, gold: str) -> tuple[str, str]:
    return pred.strip(), gold.strip()
