"""Typed sample + generator protocol for retrieval-augmented (multi-doc) tasks.

A ``RetrievalSample`` is laid out as ``[preamble][doc_1][doc_2]...[task_prompt]``:

* ``preamble``    — shared prefix (few-shot examples / system context); cached.
* ``documents``   — the retrieved passages, each cached independently as a packet.
* ``task_prompt`` — the trailing instruction/question that seeds generation.
* ``query``       — the raw query (bookkeeping; usually equals ``task_prompt``).
* ``answer``      — the gold answer, scored with token-F1.

This is the reference ``RetEvalEntry`` promoted from a ``TypedDict`` to a validated
dataclass.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

__all__ = ["Document", "RetrievalSample", "SampleGenerator", "AnswerPostprocess"]

Document = str | list[int]


@dataclass
class RetrievalSample:
    preamble: str
    documents: Sequence[Document]
    task_prompt: str
    query: str = ""
    answer: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(document, str)
            and not (
                isinstance(document, list)
                and all(isinstance(token_id, int) for token_id in document)
            )
            for document in self.documents
        ):
            raise TypeError("each document must be text or a list of token IDs")
        self.documents = list(self.documents)
        if not self.query:
            self.query = self.task_prompt


# (pred_answer, gold_answer) -> (cleaned_pred, cleaned_gold)
AnswerPostprocess = Callable[[str, str], "tuple[str, str]"]


class SampleGenerator(Protocol):
    def __call__(
        self,
        num_samples: int,
        num_documents: int = 4,
        num_shots: int = 0,
        subset: str = "default",
        split: str = "test",
        seed: int = 42,
        **kwargs,
    ) -> Iterator[RetrievalSample]:
        ...
