"""HotpotQA multi-hop retrieval generator.

Each Hotpot context paragraph remains an independent document packet. The number of
documents therefore comes from the dataset row rather than ``num_documents``.
"""
from __future__ import annotations

import random
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from .base import RetrievalSample

__all__ = ["hotpot_qa_generator", "hotpot_qa_answer_postprocess"]

_HOTPOT_CACHE: dict[tuple[str, str], Sequence[Mapping[str, Any]]] = {}
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_ANSWER_MARKER = re.compile(
    r"(?i)\b(?:the\s+)?(?:(?:short|final|correct|best)\s+)?"
    r"answer\s*(?::|is\s*:?)\s*"
)


def _load_hotpot(
    subset: str, split: str, cache: bool
) -> Sequence[Mapping[str, Any]]:
    key = (subset, split)
    if cache and key in _HOTPOT_CACHE:
        return _HOTPOT_CACHE[key]
    from datasets import Dataset, load_dataset

    dataset = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError("HotpotQA loader did not return a Dataset split")
    typed_dataset = cast(Sequence[Mapping[str, Any]], dataset)
    if cache:
        _HOTPOT_CACHE[key] = typed_dataset
    return typed_dataset


def _documents(sample: Mapping[str, Any]) -> list[str]:
    context = sample.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("HotpotQA sample has malformed context")
    sentences = context.get("sentences")
    if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
        raise ValueError("HotpotQA context has malformed sentences")
    documents: list[str] = []
    for paragraph in sentences:
        if not isinstance(paragraph, Sequence) or isinstance(paragraph, (str, bytes)):
            raise ValueError("HotpotQA paragraph must be a sentence sequence")
        if not all(isinstance(sentence, str) for sentence in paragraph):
            raise ValueError("HotpotQA paragraph contains a non-string sentence")
        documents.append("".join(paragraph))
    if not documents:
        raise ValueError("HotpotQA sample contains no context documents")
    return documents


def _prompt(
    question: str,
    *,
    add_inst: bool,
    add_cot: bool,
    answer_only: bool,
) -> str:
    parts: list[str] = []
    if add_inst:
        parts.append("Answer the following question based on the provided context.")
    parts.append(question)
    if add_cot:
        parts.append("You should get the final answer by thinking step by step.")
    if add_inst:
        if answer_only:
            parts.append(
                "Return only: 'Short Answer: <your final answer>'. "
                "Do not include an explanation."
            )
        else:
            parts.append(
                "Your response should end with: 'Short Answer: <your final answer>'."
            )
    return "\n".join(parts) + "\n"


def hotpot_qa_generator(
    num_samples: int,
    num_documents: int = 0,
    num_shots: int = 0,
    subset: str = "fullwiki",
    split: str = "validation",
    seed: int = 42,
    add_inst: bool = True,
    add_cot: bool = True,
    answer_only: bool = False,
    difficulty: Sequence[str] | None = None,
    cache_dataset: bool = True,
    **kwargs: object,
) -> Iterator[RetrievalSample]:
    """Yield HotpotQA entries with one packetizable document per context paragraph."""
    if kwargs:
        warnings.warn(f"unused kwargs in hotpot_qa_generator: {kwargs}")
    if num_samples < 0 or num_shots < 0:
        raise ValueError("num_samples and num_shots must be non-negative")
    if num_documents != 0:
        raise ValueError(
            "HotpotQA controls its own document count; num_documents must be 0"
        )
    allowed_difficulties: frozenset[str] | None = None
    if difficulty is not None:
        if isinstance(difficulty, (str, bytes)):
            raise ValueError("difficulty must be a sequence of level names")
        allowed_difficulties = frozenset(difficulty)
        unknown = allowed_difficulties - _DIFFICULTIES
        if not allowed_difficulties or unknown:
            raise ValueError(
                f"difficulty must contain values from {sorted(_DIFFICULTIES)}"
            )

    dataset = _load_hotpot(subset, split, cache_dataset)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if allowed_difficulties is not None:
        indices = [
            index
            for index in indices
            if dataset[index].get("level") in allowed_difficulties
        ]
    if num_shots + num_samples > len(indices):
        raise ValueError(
            "not enough matching HotpotQA samples after accounting for few-shots: "
            f"requested {num_samples} samples and {num_shots} shots, "
            f"but only {len(indices)} are available"
        )

    shot_indices = indices[:num_shots]
    selected_indices = indices[num_shots : num_shots + num_samples]
    if shot_indices:
        warnings.warn("few-shot prompting is not recommended for HotpotQA")
    shot_strings: list[str] = []
    for index in shot_indices:
        sample = dataset[index]
        question = sample.get("question")
        answer = sample.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("HotpotQA sample has malformed question or answer")
        shot_strings.append(
            f"Context: {' '.join(_documents(sample))}\n"
            f"Question: {question}\nAnswer: {answer}\n"
        )
    preamble = "\n".join(shot_strings)

    for index in selected_indices:
        sample = dataset[index]
        question = sample.get("question")
        answer = sample.get("answer")
        level = sample.get("level")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("HotpotQA sample has malformed question or answer")
        yield RetrievalSample(
            preamble=preamble,
            documents=_documents(sample),
            task_prompt=_prompt(
                question,
                add_inst=add_inst,
                add_cot=add_cot,
                answer_only=answer_only,
            ),
            query=question,
            answer=answer,
            meta={"index": index, "difficulty": level},
        )


def hotpot_qa_answer_postprocess(pred: str, gold: str) -> tuple[str, str]:
    markers = list(_ANSWER_MARKER.finditer(pred))
    # The prompt requires an explicit final-answer marker.  Treat a missing
    # marker as a format failure instead of scoring the entire rationale as the
    # answer; this matches the paper's legacy evaluation contract.
    answer = pred[markers[-1].end() :] if markers else ""
    cleaned_pred = answer.lower().strip().rstrip(".")
    return cleaned_pred, gold.lower().strip().rstrip(".")
