"""MuSiQue multi-hop retrieval generator."""

from __future__ import annotations

import random
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from .base import RetrievalSample

__all__ = ["musique_generator", "musique_answer_postprocess"]

_MUSIQUE_CACHE: dict[tuple[str, str], Sequence[Mapping[str, Any]]] = {}
_ANSWER_MARKER = re.compile(
    r"(?i)\b(?:the\s+)?(?:(?:short|final|correct|best)\s+)?"
    r"answer\s*(?::|is\s*:?)\s*"
)


def _load_musique(subset: str, split: str, cache: bool) -> Sequence[Mapping[str, Any]]:
    key = (subset, split)
    if cache and key in _MUSIQUE_CACHE:
        return _MUSIQUE_CACHE[key]
    from datasets import Dataset, load_dataset

    dataset = load_dataset("dgslibisey/MuSiQue", subset, split=split)
    if not isinstance(dataset, Dataset):
        raise TypeError("MuSiQue loader did not return a Dataset split")
    typed_dataset = cast(Sequence[Mapping[str, Any]], dataset)
    if cache:
        _MUSIQUE_CACHE[key] = typed_dataset
    return typed_dataset


def _paragraphs(sample: Mapping[str, Any]) -> list[str]:
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, Sequence) or isinstance(paragraphs, (str, bytes)):
        raise ValueError("MuSiQue sample has malformed paragraphs")
    documents: list[str] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, Mapping):
            raise ValueError("MuSiQue paragraph must be an object")
        text = paragraph.get("paragraph_text")
        if not isinstance(text, str):
            raise ValueError("MuSiQue paragraph_text must be a string")
        documents.append(text)
    if not documents:
        raise ValueError("MuSiQue sample contains no paragraphs")
    return documents


def _select_support_preserving_documents(
    sample: Mapping[str, Any], num_documents: int
) -> tuple[list[str], list[int], list[int]]:
    """Select a deterministic MuSiQue context that retains every support.

    Every supporting paragraph is retained.  Remaining slots are filled by the
    earliest distractors in the dataset's original order, and the resulting
    documents are emitted in that original order as well.  This makes the
    selection deterministic and identical for the original and direct-final-hop
    question variants.
    """
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, Sequence) or isinstance(paragraphs, (str, bytes)):
        raise ValueError("MuSiQue sample has malformed paragraphs")
    if num_documents <= 0 or num_documents > len(paragraphs):
        raise ValueError(
            "support_preserving_num_documents must be between 1 and the number of "
            f"available paragraphs ({len(paragraphs)})"
        )

    support_indices: list[int] = []
    distractor_indices: list[int] = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, Mapping):
            raise ValueError("MuSiQue paragraph must be an object")
        target = support_indices if bool(paragraph.get("is_supporting")) else distractor_indices
        target.append(index)
    if not support_indices:
        raise ValueError("document selection requires annotated supporting paragraphs")
    if len(support_indices) > num_documents:
        raise ValueError(
            "document limit would remove supporting evidence: "
            f"{len(support_indices)} supports for {num_documents} slots"
        )

    selected_indices = sorted(
        support_indices + distractor_indices[: num_documents - len(support_indices)]
    )
    documents: list[str] = []
    for index in selected_indices:
        paragraph = paragraphs[index]
        assert isinstance(paragraph, Mapping)
        text = paragraph.get("paragraph_text")
        if not isinstance(text, str):
            raise ValueError("MuSiQue paragraph_text must be a string")
        documents.append(text)
    return documents, selected_indices, support_indices


_DECOMPOSITION_REFERENCE = re.compile(r"#(\d+)")


def _direct_final_hop_question(
    sample: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve gold intermediate entities in only the final decomposition step.

    The final answer is never inserted.  For example, ``#1 >> spouse`` with the
    first-step answer ``Steve Hillage`` becomes the auditable direct question
    ``What is the value of the relation \"spouse\" for \"Steve Hillage\"?``.
    """
    decomposition = sample.get("question_decomposition")
    if (
        not isinstance(decomposition, Sequence)
        or isinstance(decomposition, (str, bytes))
        or not decomposition
    ):
        raise ValueError("direct-final-hop control requires question_decomposition")
    steps: list[Mapping[str, Any]] = []
    for step in decomposition:
        if not isinstance(step, Mapping):
            raise ValueError("MuSiQue decomposition step must be an object")
        steps.append(step)

    final_step = steps[-1]
    symbolic_question = final_step.get("question")
    if not isinstance(symbolic_question, str):
        raise ValueError("MuSiQue final decomposition question must be a string")

    def replace_reference(match: re.Match[str]) -> str:
        reference = int(match.group(1))
        # MuSiQue references are one-indexed and must point to an earlier step.
        if reference < 1 or reference >= len(steps):
            raise ValueError(
                f"final decomposition contains an invalid or unresolved reference: #{reference}"
            )
        answer = steps[reference - 1].get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"MuSiQue decomposition answer for #{reference} is malformed")
        return answer.strip()

    resolved = _DECOMPOSITION_REFERENCE.sub(replace_reference, symbolic_question)
    if _DECOMPOSITION_REFERENCE.search(resolved):
        raise ValueError("final decomposition contains an unresolved reference")
    parts = resolved.split(">>", maxsplit=1)
    if len(parts) == 2 and all(part.strip() for part in parts):
        subject, relation = (part.strip() for part in parts)
        direct_question = f'What is the value of the relation "{relation}" for "{subject}"?'
    elif ">>" not in resolved and resolved.strip():
        # Many MuSiQue final decomposition steps are already natural-language
        # questions (for example, ``When did #1 die?``).  Substituting the
        # intermediate entities is sufficient and avoids an unnecessary rewrite.
        direct_question = resolved.strip()
        if direct_question[-1] not in "?!.":
            direct_question += "?"
    else:
        raise ValueError("MuSiQue final decomposition question is malformed")
    return direct_question, resolved


def _question_prompt(
    question: str,
    *,
    add_inst: bool,
    add_cot: bool,
    answer_only: bool = False,
) -> str:
    if add_inst:
        prompt = f"Answer the following question based on the provided context.\n{question}\n"
    else:
        prompt = f"Question: {question}\n"
    if add_cot:
        prompt += (
            "You should get the final answer by thinking step by step.\n"
            "The answer is a word or short phrase. Your response should end with: "
            "'Answer: <your final answer>'.\n"
        )
    elif add_inst:
        if answer_only:
            prompt += "Return only: 'Answer: <your final answer>'. Do not include an explanation.\n"
        else:
            prompt += (
                "The answer is a word or short phrase. Your response should end with: "
                "'Answer: <your final answer>'.\n"
            )
    return prompt


def musique_generator(
    num_samples: int,
    num_documents: int = 0,
    num_shots: int = 0,
    subset: str = "default",
    split: str = "train",
    seed: int = 42,
    add_inst: bool = True,
    add_cot: bool = True,
    answer_only: bool = False,
    support_preserving_num_documents: int | None = None,
    question_mode: str = "multi_hop",
    cache_dataset: bool = True,
    **kwargs: object,
) -> Iterator[RetrievalSample]:
    if kwargs:
        warnings.warn(f"unused kwargs in musique_generator: {kwargs}")
    if num_samples < 0 or num_shots < 0:
        raise ValueError("num_samples and num_shots must be non-negative")
    if num_documents != 0:
        raise ValueError("MuSiQue controls its own document count; num_documents must be 0")
    if question_mode not in {"multi_hop", "direct_final_hop"}:
        raise ValueError("question_mode must be 'multi_hop' or 'direct_final_hop'")
    transformed = support_preserving_num_documents is not None or question_mode != "multi_hop"
    if transformed and num_shots:
        raise ValueError("MuSiQue transformed views do not support few-shot prompts")

    dataset = _load_musique(subset, split, cache_dataset)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if num_shots + num_samples > len(indices):
        raise ValueError(
            "not enough MuSiQue samples after accounting for few-shots: "
            f"requested {num_samples} samples and {num_shots} shots, "
            f"but only {len(indices)} are available"
        )

    shot_indices = indices[-num_shots:] if num_shots else []
    selected_indices = indices[:num_samples]
    if shot_indices and add_cot:
        warnings.warn("few-shot prompting with CoT is not recommended for MuSiQue")
    shot_strings: list[str] = []
    for index in shot_indices:
        sample = dataset[index]
        question = sample.get("question")
        answer = sample.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("MuSiQue sample has malformed question or answer")
        prompt = _question_prompt(
            question,
            add_inst=add_inst,
            add_cot=False,
            answer_only=answer_only,
        )
        shot_strings.append(f"Context: {' '.join(_paragraphs(sample))}\n{prompt}Answer: {answer}\n")
    preamble = ("\n".join(shot_strings) + "\n") if shot_strings else ""

    for index in selected_indices:
        sample = dataset[index]
        original_question = sample.get("question")
        answer = sample.get("answer")
        if not isinstance(original_question, str) or not isinstance(answer, str):
            raise ValueError("MuSiQue sample has malformed question or answer")
        resolved_final_step: str | None = None
        if question_mode == "direct_final_hop":
            question, resolved_final_step = _direct_final_hop_question(sample)
        else:
            question = original_question
        if support_preserving_num_documents is None:
            documents = _paragraphs(sample)
            document_indices = list(range(len(documents)))
            paragraphs = sample.get("paragraphs")
            assert isinstance(paragraphs, Sequence)
            support_indices = [
                paragraph_index
                for paragraph_index, paragraph in enumerate(paragraphs)
                if isinstance(paragraph, Mapping) and bool(paragraph.get("is_supporting"))
            ]
        else:
            documents, document_indices, support_indices = _select_support_preserving_documents(
                sample, support_preserving_num_documents
            )
        yield RetrievalSample(
            preamble=preamble,
            documents=documents,
            task_prompt=_question_prompt(
                question,
                add_inst=add_inst,
                add_cot=add_cot,
                answer_only=answer_only,
            ),
            query=question,
            answer=answer,
            meta={
                "index": index,
                "original_question": original_question,
                "question_mode": question_mode,
                "document_indices": document_indices,
                "support_indices": support_indices,
                "resolved_final_step": resolved_final_step,
            },
        )


def musique_answer_postprocess(pred: str, gold: str) -> tuple[str, str]:
    # Native-thinking models may mention candidate "answers" inside <think>. Only
    # the assistant content after a completed reasoning block is eligible for
    # scoring. An unclosed block is a right-censored generation, not an answer.
    if "<think>" in pred:
        scoring_text = pred.rsplit("</think>", 1)[1] if "</think>" in pred else ""
    else:
        scoring_text = pred
    markers = list(_ANSWER_MARKER.finditer(scoring_text))
    # The prompt requires an explicit final-answer marker.  Treat a missing
    # marker as a format failure instead of scoring the entire rationale as the
    # answer; this matches the paper's legacy evaluation contract.
    answer = scoring_text[markers[-1].end() :] if markers else ""
    cleaned_pred = answer.lower().strip().rstrip(".")
    return cleaned_pred, gold.lower().strip().rstrip(".")
