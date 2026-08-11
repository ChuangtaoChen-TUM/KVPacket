"""Deterministically pad and repack retrieval samples to a target token length."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from .base import RetrievalSample

__all__ = ["length_match_samples"]


class _TextTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str | list[str]: ...


def _encode(tokenizer: _TextTokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("tokenizer.encode must return list[int]")
    return token_ids


def _text_documents(sample: RetrievalSample) -> list[str]:
    if not all(isinstance(document, str) for document in sample.documents):
        raise TypeError("length matching requires text documents")
    return [document for document in sample.documents if isinstance(document, str)]


def length_match_samples(
    targets: Sequence[RetrievalSample],
    fillers: Sequence[RetrievalSample],
    *,
    tokenizer: _TextTokenizer,
    target_total_tokens: int = 7680,
    num_packets: int = 8,
    seed: int = 42,
    filler_position: str = "append",
) -> list[RetrievalSample]:
    """Pad samples with unrelated passages and repack the document token stream.

    Target examples and their questions/answers are unchanged. Only their cached
    document region is transformed: unrelated documents from separately selected
    filler examples pad the answer-bearing stream, and the result is split into
    nearly equal packet-sized pieces. By default filler is appended; ``mixed``
    deterministically varies the target stream's beginning/middle/end placement.
    """
    if target_total_tokens <= 0 or num_packets <= 0:
        raise ValueError("target_total_tokens and num_packets must be positive")
    if target_total_tokens < num_packets:
        raise ValueError("target_total_tokens must be at least num_packets")
    if filler_position not in {"append", "mixed"}:
        raise ValueError("filler_position must be 'append' or 'mixed'")
    filler_texts = [
        document for sample in fillers for document in _text_documents(sample) if document.strip()
    ]
    if not filler_texts:
        raise ValueError("length matching requires non-empty unrelated filler passages")

    separator_ids = _encode(tokenizer, "\n\n")
    filler_ids = [_encode(tokenizer, text) for text in filler_texts]
    filler_ids = [ids for ids in filler_ids if ids]
    if not filler_ids:
        raise ValueError("filler passages produced no tokens")

    transformed: list[RetrievalSample] = []
    for target_index, sample in enumerate(targets):
        original_text = "\n\n".join(_text_documents(sample))
        original_stream = _encode(tokenizer, original_text)
        original_tokens = len(original_stream)
        if original_tokens > target_total_tokens:
            raise ValueError(
                "answer-bearing documents exceed the requested length-matched "
                f"budget: {original_tokens} > {target_total_tokens}"
            )

        required_filler = target_total_tokens - original_tokens
        order = list(range(len(filler_ids)))
        random.Random(seed + target_index).shuffle(order)
        filler_stream: list[int] = []
        filler_cursor = 0
        while len(filler_stream) < required_filler:
            if separator_ids:
                filler_stream.extend(separator_ids[: required_filler - len(filler_stream)])
            if len(filler_stream) >= required_filler:
                break
            ids = filler_ids[order[filler_cursor % len(order)]]
            filler_stream.extend(ids[: required_filler - len(filler_stream)])
            filler_cursor += 1
        if filler_position == "append":
            prefix_filler = 0
        else:
            # Cover beginning/middle/end placements without changing the target
            # content or total length. Five positions repeat deterministically.
            prefix_filler = required_filler * (target_index % 5) // 4
        stream = filler_stream[:prefix_filler] + original_stream + filler_stream[prefix_filler:]

        base, remainder = divmod(target_total_tokens, num_packets)
        packet_sizes = [
            base + (1 if packet_index < remainder else 0) for packet_index in range(num_packets)
        ]
        packets: list[str] = []
        offset = 0
        for size in packet_sizes:
            packet_ids = stream[offset : offset + size]
            packet = tokenizer.decode(
                packet_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(packet, str):
                raise TypeError("tokenizer.decode must return str for one token sequence")
            packets.append(packet)
            offset += size

        meta = dict(sample.meta)
        meta["length_match"] = {
            "target_total_tokens": target_total_tokens,
            "num_packets": num_packets,
            "original_document_tokens": original_tokens,
            "stream_packet_tokens": packet_sizes,
            "filler_pool_documents": len(filler_ids),
            "filler_position": filler_position,
            "prefix_filler_tokens": prefix_filler,
        }
        transformed.append(replace(sample, documents=packets, meta=meta))
    return transformed
