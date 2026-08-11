"""Focused tests for reusable-chunk occurrence and cover selection."""

import random

from kvpacket.runtime.matching import (
    Match,
    find_occurrences,
    string_cover,
    token_cover,
    weighted_interval_cover,
)


def _naive_token_cover(tokens: list[int], chunks: dict[str, list[int]]) -> list[Match]:
    candidates = [
        Match(chunk_id, start, end)
        for chunk_id, chunk_tokens in chunks.items()
        for start, end in find_occurrences(tokens, chunk_tokens)
    ]
    return weighted_interval_cover(candidates)


def test_token_cover_finds_prefix_suffix_and_overlapping_patterns():
    tokens = [1, 2, 1, 2, 3]
    chunks = {
        "pair": [1, 2],
        "suffix": [2],
        "long": [1, 2, 3],
    }

    cover = token_cover(tokens, chunks)

    assert cover == [Match("pair", 0, 2), Match("long", 2, 5)]


def test_token_cover_preserves_registration_order_for_duplicate_patterns():
    cover = token_cover([1, 2], {"first": [1, 2], "second": [1, 2]})

    assert cover == [Match("first", 0, 2)]


def test_token_cover_matches_naive_reference_on_random_inputs():
    generator = random.Random(7)
    for _ in range(100):
        tokens = [generator.randrange(5) for _ in range(generator.randrange(20))]
        chunks = {
            f"chunk-{index}": [generator.randrange(5) for _ in range(generator.randrange(5))]
            for index in range(generator.randrange(10))
        }
        assert token_cover(tokens, chunks) == _naive_token_cover(tokens, chunks)


def test_string_cover_maps_repeated_occurrences_to_token_spans():
    text = "cat cat"
    offsets = [(0, 3), (3, 4), (4, 7)]

    cover = string_cover(text, {"animal": "cat"}, offsets)

    assert cover == [Match("animal", 0, 1), Match("animal", 2, 3)]


def test_string_cover_expands_subtoken_occurrence_to_its_token():
    text = "uncatalogued"

    cover = string_cover(text, {"fragment": "cat"}, [(0, len(text))])

    assert cover == [Match("fragment", 0, 1)]


def test_string_cover_ignores_special_tokens_and_character_gaps():
    assert string_cover("abc", {"body": "abc"}, [(0, 0), (0, 3), (0, 0)]) == [Match("body", 1, 2)]
    assert string_cover("a--b", {"gap": "--"}, [(0, 1), (3, 4)]) == []
