"""Retrieval generator registry and optional tokenizer-native chat wrapping."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .base import AnswerPostprocess, RetrievalSample, SampleGenerator
from .biography import bio_generator, biography_answer_postprocess
from .gsm8k_dependent import (
    gsm8k_dependent_generator,
    gsm8k_numeric_postprocess,
)
from .hotpot_qa import hotpot_qa_answer_postprocess, hotpot_qa_generator
from .length_matching import length_match_samples
from .multinews import multinews_answer_postprocess, multinews_generator
from .musique import musique_answer_postprocess, musique_generator
from .niah import niah_answer_postprocess, niah_generator
from .synthetic import identity_postprocess, synthetic_generator
from .templates import TokenizerType, apply_chat_template

__all__ = ["DatasetSpec", "DATASETS", "get_dataset", "get_ret_eval_generator"]


@dataclass
class DatasetSpec:
    generator: SampleGenerator
    answer_postprocess: AnswerPostprocess


DATASETS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(synthetic_generator, identity_postprocess),
    "biography": DatasetSpec(bio_generator, biography_answer_postprocess),
    "gsm8k_dependent": DatasetSpec(
        gsm8k_dependent_generator,
        gsm8k_numeric_postprocess,
    ),
    "hotpot_qa": DatasetSpec(hotpot_qa_generator, hotpot_qa_answer_postprocess),
    "musique": DatasetSpec(musique_generator, musique_answer_postprocess),
    "multinews": DatasetSpec(multinews_generator, multinews_answer_postprocess),
    "niah": DatasetSpec(niah_generator, niah_answer_postprocess),
}


def _group_document_packets(
    sample: RetrievalSample,
    *,
    tokenizer: TokenizerType,
    num_packets: int | None = None,
    max_documents_per_packet: int | None = None,
) -> RetrievalSample:
    """Group consecutive documents without changing their token stream.

    Decoding each group's concatenated document-token IDs, then verifying an exact
    encode round trip, lets the packet-count control hold the model input fixed.
    Only packet boundaries change.
    """
    num_documents = len(sample.documents)
    if (num_packets is None) == (max_documents_per_packet is None):
        raise ValueError("exactly one of num_packets and max_documents_per_packet must be set")
    if max_documents_per_packet is not None:
        if max_documents_per_packet <= 0:
            raise ValueError("max_documents_per_packet must be positive")
        num_packets = math.ceil(num_documents / max_documents_per_packet)
        group_sizes = [
            min(max_documents_per_packet, num_documents - start)
            for start in range(0, num_documents, max_documents_per_packet)
        ]
    else:
        assert num_packets is not None
        if num_packets <= 0 or num_packets > num_documents:
            raise ValueError(
                "group_documents_into_packets must be between 1 and the number of "
                f"documents ({num_documents})"
            )
        quotient, remainder = divmod(num_documents, num_packets)
        group_sizes = [quotient + (1 if index < remainder else 0) for index in range(num_packets)]
    if num_packets == num_documents:
        return sample

    document_ids = [
        (
            list(document)
            if isinstance(document, list)
            else list(tokenizer(document, add_special_tokens=False)["input_ids"])
        )
        for document in sample.documents
    ]
    grouped_documents: list[list[int]] = []
    cursor = 0
    for size in group_sizes:
        group_ids = [token_id for ids in document_ids[cursor : cursor + size] for token_id in ids]
        cursor += size
        grouped_documents.append(group_ids)

    original_stream = [token_id for ids in document_ids for token_id in ids]
    grouped_stream = [token_id for document in grouped_documents for token_id in document]
    if grouped_stream != original_stream:
        raise AssertionError("grouping changed the concatenated document token stream")
    meta = dict(sample.meta)
    meta.update(
        {
            "source_document_count": num_documents,
            "packet_count": num_packets,
            "packet_group_sizes": group_sizes,
            "max_documents_per_packet": max_documents_per_packet,
            "packet_grouping_token_exact": True,
        }
    )
    return replace(sample, documents=grouped_documents, meta=meta)


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choices: {sorted(DATASETS)}")
    return DATASETS[name]


def get_ret_eval_generator(
    name: str,
    num_samples: int,
    num_documents: int,
    num_shots: int,
    subset: str,
    split: str,
    seed: int,
    tokenizer: TokenizerType | None = None,
    data_kwargs: Mapping[str, Any] | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    system_prompt: str = "",
) -> Iterator[RetrievalSample]:
    """Build a retrieval generator, optionally applying the tokenizer chat template.

    Chat formatting happens once as samples leave the dataset generator. Evaluators
    therefore consume a single, already-formatted ``RetrievalSample`` representation
    and need no template dispatch or configuration.
    """
    spec = get_dataset(name)

    def wrapped_generator() -> Iterator[RetrievalSample]:
        generator_kwargs = dict(data_kwargs or {})
        length_match = generator_kwargs.pop("length_match", None)
        group_documents_into_packets = generator_kwargs.pop("group_documents_into_packets", None)
        max_documents_per_packet = generator_kwargs.pop("max_documents_per_packet", None)
        if group_documents_into_packets is not None and max_documents_per_packet is not None:
            raise ValueError(
                "group_documents_into_packets and max_documents_per_packet are mutually exclusive"
            )
        selection_offset = int(generator_kwargs.pop("selection_offset", 0))
        if selection_offset < 0:
            raise ValueError("selection_offset must be non-negative")
        requested_samples = num_samples + selection_offset
        if length_match is not None:
            if tokenizer is None:
                raise ValueError("length matching requires a tokenizer")
            if num_shots:
                raise ValueError("length matching does not support few-shot prompts")
            if not isinstance(length_match, Mapping):
                raise TypeError("length_match must be an object")
            filler_samples = int(length_match.get("filler_samples", 64))
            if filler_samples <= 0:
                raise ValueError("length_match.filler_samples must be positive")
            requested_samples += filler_samples

        generated = list(
            spec.generator(
                num_samples=requested_samples,
                num_documents=num_documents,
                num_shots=num_shots,
                subset=subset,
                split=split,
                seed=seed,
                **generator_kwargs,
            )
        )
        if length_match is not None:
            assert tokenizer is not None
            generated = length_match_samples(
                generated[selection_offset : selection_offset + num_samples],
                generated[num_samples + selection_offset :],
                tokenizer=tokenizer,
                target_total_tokens=int(length_match.get("target_total_tokens", 7680)),
                num_packets=int(length_match.get("num_packets", 8)),
                seed=int(length_match.get("seed", seed)),
                filler_position=str(length_match.get("filler_position", "append")),
            )
        else:
            generated = generated[selection_offset : selection_offset + num_samples]

        for sample in generated[:num_samples]:
            if group_documents_into_packets is not None or max_documents_per_packet is not None:
                if tokenizer is None:
                    raise ValueError("document grouping requires a tokenizer")
                sample = _group_document_packets(
                    sample,
                    tokenizer=tokenizer,
                    num_packets=(
                        int(group_documents_into_packets)
                        if group_documents_into_packets is not None
                        else None
                    ),
                    max_documents_per_packet=(
                        int(max_documents_per_packet)
                        if max_documents_per_packet is not None
                        else None
                    ),
                )
            yield (
                apply_chat_template(
                    sample,
                    tokenizer,
                    system_prompt=system_prompt,
                    chat_template_kwargs=chat_template_kwargs,
                )
                if tokenizer is not None
                else sample
            )

    return wrapped_generator()
