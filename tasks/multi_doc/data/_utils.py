"""Small text helpers shared by dataset generators (ported from the reference)."""
from __future__ import annotations

import re

__all__ = ["clean_with_prefixes", "split_sentences"]


def clean_with_prefixes(answer: str, prefixes: list[str]) -> str:
    """Strip the first matching chatty prefix and surrounding punctuation."""
    for prefix in prefixes:
        if prefix in answer:
            return answer.split(prefix, 1)[-1].strip(" :.,")
    return answer


def split_sentences(text: str, max_chunk_size: int = 512) -> list[str]:
    """Sentence-pack text without downloading tokenizer data as a side effect."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    try:
        import nltk

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text)
                if sentence.strip()
            ]
    except Exception:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

    if not sentences:
        return []

    pieces: list[str] = []
    for sentence in sentences:
        remainder = sentence.strip()
        while len(remainder) > max_chunk_size:
            boundary = remainder.rfind(" ", 0, max_chunk_size + 1)
            if boundary <= 0:
                boundary = max_chunk_size
            pieces.append(remainder[:boundary].strip())
            remainder = remainder[boundary:].strip()
        if remainder:
            pieces.append(remainder)
    if not pieces:
        return []

    chunks: list[str] = []
    current = pieces[0]
    for sentence in pieces[1:]:
        if len(current) + len(sentence) + 1 <= max_chunk_size:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    chunks.append(current)
    return chunks
