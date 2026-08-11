"""A self-contained synthetic multi-doc RAG generator (no downloads).

Mirrors the biography-completion structure with in-repo fictional facts, so the
engine can be verified end-to-end (all three methods) on any cached model without
network access — the CI-speed analogue of the real ``bio_generator``. The gold
answer always appears verbatim in exactly one document, so the ``full_recompute``
oracle should score high F1.
"""
from __future__ import annotations

import random
from typing import Iterator

from .base import RetrievalSample

__all__ = ["synthetic_generator", "identity_postprocess"]

_NAMES = [
    "Alice Bennett", "Marcus Reed", "Priya Nair", "Diego Alvarez", "Hana Kimura",
    "Omar Farouk", "Lena Petrova", "Tomas Novak", "Grace Okoro", "Yusuf Demir",
    "Clara Meyer", "Ravi Shankar", "Nadia Haddad", "Erik Larsson", "Mei Chen",
    "Pablo Sosa", "Ingrid Dahl", "Kwame Mensah", "Sofia Rossi", "Jonas Weber",
]
_CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Munich", "Bogota", "Helsinki", "Cairo"]
_UNIS = ["Rossmoor College", "Vale University", "Northgate Institute", "Aldbrook University"]
_COMPANIES = ["Halcyon Systems", "Brightpath Labs", "Ironwood Corp", "Meridian Analytics"]
_ATTRS = ("birth_year", "city", "university", "company")

_QUESTION = {
    "birth_year": "## {name} was born in the year",
    "city": "## {name} currently lives in the city of",
    "university": "## {name} studied at",
    "company": "## {name} works at",
}


def _profile(rng: random.Random, name: str) -> dict:
    return {
        "name": name,
        "birth_year": str(rng.randint(1960, 1998)),
        "city": rng.choice(_CITIES),
        "university": rng.choice(_UNIS),
        "company": rng.choice(_COMPANIES),
    }


def _document(p: dict) -> str:
    n = p["name"]
    return (
        f"{n} was born in {p['birth_year']}. {n} currently lives in {p['city']}. "
        f"{n} studied at {p['university']}. {n} works at {p['company']}."
    )


def synthetic_generator(
    num_samples: int,
    num_documents: int = 4,
    num_shots: int = 0,
    subset: str = "default",
    split: str = "test",
    seed: int = 0,
    **kwargs,
) -> Iterator[RetrievalSample]:
    rng = random.Random(seed)
    people = [_profile(rng, n) for n in _NAMES]
    n_docs = max(1, min(num_documents, len(people)))

    preamble = ""
    if num_shots > 0:
        shots = []
        for k in range(num_shots):
            p = people[k % len(people)]
            attr = _ATTRS[k % len(_ATTRS)]
            shots.append(_QUESTION[attr].format(name=p["name"]) + f" {p[attr]}.")
        preamble = "\n".join(shots) + "\n"

    for i in range(num_samples):
        target = people[i % len(people)]
        attr = _ATTRS[(i // len(people) + i) % len(_ATTRS)]
        pool = [p for p in people if p["name"] != target["name"]]
        distractors = rng.sample(pool, k=min(n_docs - 1, len(pool)))
        docs = distractors + [target]
        rng.shuffle(docs)
        yield RetrievalSample(
            preamble=preamble,
            documents=[_document(p) for p in docs],
            task_prompt=_QUESTION[attr].format(name=target["name"]),
            answer=target[attr],
            meta={"attribute": attr, "target": target["name"]},
        )


def identity_postprocess(pred: str, gold: str) -> tuple[str, str]:
    return pred.strip().strip(".").lower(), gold.strip().strip(".").lower()
