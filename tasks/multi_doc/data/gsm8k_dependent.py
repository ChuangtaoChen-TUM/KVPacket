"""GSM8K retrieval task built from dependent contiguous reasoning chunks."""

from __future__ import annotations

import random
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from .base import RetrievalSample

__all__ = ["gsm8k_dependent_generator", "gsm8k_numeric_postprocess"]

_CACHE: dict[str, Sequence[Mapping[str, Any]]] = {}
_FINAL_MARKER = re.compile(r"####\s*(.+?)\s*$")
_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _load(split: str, cache: bool) -> Sequence[Mapping[str, Any]]:
    if cache and split in _CACHE:
        return _CACHE[split]
    from datasets import Dataset, load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError("GSM8K loader did not return a Dataset split")
    typed = cast(Sequence[Mapping[str, Any]], dataset)
    if cache:
        _CACHE[split] = typed
    return typed


def _gold_and_rationale(answer: str) -> tuple[str, str]:
    match = _FINAL_MARKER.search(answer)
    if match is None:
        raise ValueError("GSM8K answer has no final #### marker")
    gold = match.group(1).strip()
    rationale = answer[: match.start()].strip()
    if not gold or not rationale:
        raise ValueError("GSM8K answer has an empty gold or rationale")
    return gold, rationale


def _mask_gold(rationale: str, gold: str) -> str | None:
    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    if not lines:
        return None
    escaped = re.escape(gold)
    pattern = re.compile(rf"(?<![\d.])\$?{escaped}(?![\d.])")
    # Avoid leaking the answer from an earlier reasoning step. The final line is
    # retained but its result is masked, leaving the last operation to complete.
    if any(pattern.search(line) for line in lines[:-1]):
        return None
    masked, replacements = pattern.subn("[MASK]", lines[-1])
    if replacements == 0:
        normalized = gold.replace(",", "").lstrip("$")
        normalized_pattern = re.compile(rf"(?<![\d.])\$?{re.escape(normalized)}(?![\d.])")
        masked, replacements = normalized_pattern.subn("[MASK]", lines[-1])
    if replacements == 0:
        return None
    lines[-1] = masked
    return "\n".join(lines)


def _chunks(text: str, count: int) -> list[str]:
    words = text.split()
    if len(words) < count:
        raise ValueError("reasoning trace is too short for requested chunks")
    base, remainder = divmod(len(words), count)
    chunks: list[str] = []
    cursor = 0
    for index in range(count):
        size = base + int(index < remainder)
        part = " ".join(words[cursor : cursor + size])
        label = chr(ord("A") + index)
        chunks.append(f"Ordered reasoning chunk {label}: {part}")
        cursor += size
    return chunks


def gsm8k_dependent_generator(
    num_samples: int,
    num_documents: int = 4,
    num_shots: int = 0,
    subset: str = "main",
    split: str = "test",
    seed: int = 42,
    cache_dataset: bool = True,
    **kwargs: object,
) -> Iterator[RetrievalSample]:
    """Split one masked worked solution into contiguous, dependent packets."""
    if kwargs:
        warnings.warn(f"unused kwargs in gsm8k_dependent_generator: {kwargs}")
    if subset != "main":
        raise ValueError("GSM8K supports only subset='main'")
    if split not in {"train", "test"}:
        raise ValueError("GSM8K split must be 'train' or 'test'")
    if num_samples < 0 or num_shots != 0:
        raise ValueError("GSM8K dependent chunks require num_shots=0")
    if num_documents < 2:
        raise ValueError("dependent chunks require at least two documents")

    dataset = _load(split, cache_dataset)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    yielded = 0
    for index in indices:
        row = dataset[index]
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        gold, rationale = _gold_and_rationale(answer)
        masked = _mask_gold(rationale, gold)
        if masked is None:
            continue
        try:
            documents = _chunks(masked, num_documents)
        except ValueError:
            continue
        yield RetrievalSample(
            preamble=(
                "Math problem:\n"
                f"{question}\n\n"
                "A worked solution is provided in ordered, contiguous chunks. "
                "Later chunks may depend on quantities established earlier.\n"
            ),
            documents=documents,
            task_prompt=(
                "Complete the masked final calculation using the ordered "
                "reasoning chunks. Return only "
                "'Short Answer: <numeric answer>'.\n"
            ),
            query=question,
            answer=gold,
            meta={
                "index": index,
                "structure": "contiguous_dependent_reasoning",
                "num_chunks": num_documents,
            },
        )
        yielded += 1
        if yielded == num_samples:
            return
    if yielded != num_samples:
        raise ValueError(
            f"only {yielded} non-leaking GSM8K examples were available; requested {num_samples}"
        )


def _normalize_number(value: str) -> str:
    cleaned = value.strip().replace("$", "").replace(",", "")
    try:
        return str(Fraction(cleaned))
    except (ValueError, ZeroDivisionError):
        return cleaned.lower()


def gsm8k_numeric_postprocess(pred: str, gold: str) -> tuple[str, str]:
    markers = re.split(r"(?i)(?:short\s+)?answer\s*:", pred)
    candidate = markers[-1] if len(markers) > 1 else pred
    numbers = _NUMBER.findall(candidate)
    prediction = _normalize_number(numbers[-1]) if numbers else ""
    return prediction, _normalize_number(gold)
