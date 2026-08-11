"""Biography multi-doc generator (``alex-karev/biographies``).

The paper's main RAG-completion task: each document is a one-line biography built
from a person's attributes; the task prompt is a completion/QA over one attribute.
Ported from the reference ``dataset/biography.py`` and re-typed onto
``RetrievalSample``.
"""
from __future__ import annotations

import random
import warnings
from typing import Iterator

from ._utils import clean_with_prefixes
from .base import RetrievalSample

__all__ = ["bio_generator", "biography_answer_postprocess", "BIO_ATTRIBUTES"]

_BIO_CACHE: dict[str, object] = {}
TRAIN_RATIO = 0.5
BIO_ATTRIBUTES = (
    "birth_date", "birth_place", "current_location", "university", "major", "company",
)


def _load_bio(subset: str, cache: bool):
    if cache and subset in _BIO_CACHE:
        return _BIO_CACHE[subset]
    from datasets import DatasetDict, load_dataset

    ds = load_dataset("alex-karev/biographies", subset)
    assert isinstance(ds, DatasetDict)
    ds = _filter_bio(ds)
    if cache:
        _BIO_CACHE[subset] = ds
    return ds


def _filter_bio(dataset):
    """Drop test entries whose ``birth_date`` labels are degenerate (all identical)."""
    import numpy as np
    from datasets import Dataset, DatasetDict

    test_pd = dataset["test"].to_pandas()
    train_pd = dataset["train"].to_pandas()

    def is_valid(item) -> bool:
        bd = item["birth_date"]
        assert isinstance(bd, np.ndarray)
        return len(bd) > 0 and not all(c == bd[0] for c in bd)

    keep = [i for i in range(len(test_pd)) if is_valid(test_pd.iloc[i])]
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_pd.iloc[keep]),
            "test": Dataset.from_pandas(test_pd.iloc[keep]),
        }
    )


def bio_generator(
    num_samples: int,
    num_documents: int = 4,
    num_shots: int = 0,
    subset: str = "1k",
    split: str = "test",
    seed: int = 42,
    question_type: str = "completion",
    cache_dataset: bool = True,
    **kwargs,
) -> Iterator[RetrievalSample]:
    if kwargs:
        warnings.warn(f"unused kwargs in bio_generator: {kwargs}")
    ds = _load_bio(subset, cache_dataset)
    ds_len = len(ds["test"])
    split_idx = int(ds_len * TRAIN_RATIO)
    if split == "train":
        indices = list(range(split_idx))
    elif split == "test":
        indices = list(range(split_idx, ds_len))
    else:
        raise ValueError(f"unknown split: {split}")

    rng = random.Random(seed)
    rng.shuffle(indices)
    shot_idx = indices[-num_shots:] if num_shots > 0 else []
    indices = indices[:-num_shots] if num_shots > 0 else indices

    def data_str(idx: int) -> str:
        item = ds["train"][idx]
        return " ".join(rng.choice(item[a]) for a in BIO_ATTRIBUTES)

    def qa(idx: int) -> tuple[str, str]:
        item = ds["test"][idx]
        key = rng.choice(BIO_ATTRIBUTES)
        if question_type == "completion":
            q = "## " + rng.choice(item[key]).rstrip()
        elif question_type == "QA":
            q = _format_question(key, item["name"], rng)
        else:
            raise ValueError(f"unknown question_type: {question_type}")
        return q, item["labels"][key]

    shots = []
    for si in shot_idx:
        q, a = qa(si)
        shots.append(f"{data_str(si)} {q} {a}.")
    preamble = ("\n".join(shots) + "\n") if shots else ""

    for i in range(min(num_samples, len(indices))):
        idx = indices[i]
        question, answer = qa(idx)
        doc_idx = rng.sample(indices, num_documents)
        if num_documents > 0 and idx not in doc_idx:
            doc_idx[0] = idx
        yield RetrievalSample(
            preamble=preamble,
            documents=[data_str(d) for d in doc_idx],
            task_prompt=question,
            query=question,
            answer=answer,
            meta={"index": idx, "question_type": question_type},
        )


def _format_question(attribute: str, name: str, rng: random.Random) -> str:
    templates = {
        "birth_date": [
            "What is the birth date of {name}?",
            "When was {name} born?",
            "When did {name}'s birth occur?",
            "When did {name} come into this world?",
        ],
        "birth_place": [
            "Where was {name} born?",
            "Which place is the birthplace of {name}?",
            "What is the birthplace of {name}?",
            "Where did {name} originate from?",
        ],
        "current_location": [
            "Where is {name} currently located?",
            "What is the current location of {name}?",
            "Where can I find {name} now?",
            "Where does {name} live currently?",
            "Where is {name} residing at present?",
        ],
        "university": [
            "Which university did {name} attend?",
            "What university did {name} go to?",
            "Where did {name} study in higher education?",
            "Where did {name} go for higher education?",
        ],
        "major": [
            "What was {name}'s major in university?",
            "What did {name} study in university?",
            "What field did {name} specialize in at university?",
            "In which subject did {name} major during university?",
        ],
        "company": [
            "Which company does {name} work for?",
            "Where is {name} employed?",
            "What is the name of the company where {name} works?",
            "At which company is {name} employed?",
        ],
    }[attribute]
    return rng.choice(templates).format(name=name) + " Respond with the answer only. Answer:"


def biography_answer_postprocess(pred: str, gold: str) -> tuple[str, str]:
    pred = pred.lower().strip().rstrip(".")
    gold = gold.lower().strip().rstrip(".")
    prefixes = [
        "studied", "attended", "works for", "university was", "employed by",
        " in ", " at ", " on ", " was ", " from ", " is ",
    ]
    pred = clean_with_prefixes(pred, prefixes).strip(" :.,")
    return pred, gold
