"""Find reusable chunks inside pending session content.

Callers register chunks but do not label individual pushes with chunk IDs. The session
instead enumerates occurrences and selects the non-overlapping set covering the most
tokens. Token matching is the substitution-safe default; optional string matching maps
character occurrences back to token spans through tokenizer offsets.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "Match",
    "find_occurrences",
    "weighted_interval_cover",
    "token_cover",
    "string_cover",
    "find_chunk_cover",
]


@dataclass(frozen=True)
class Match:
    """One registered chunk occurrence in half-open token-index space.

    Attributes:
        chunk_id: Registration ID of the matched reusable chunk.
        start: Inclusive token offset in the pending stream.
        end: Exclusive token offset in the pending stream.
    """

    chunk_id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of tokens covered by the occurrence."""
        return self.end - self.start


def find_occurrences(haystack: Sequence, needle: Sequence) -> list[tuple[int, int]]:
    """Find every possibly overlapping occurrence of a sequence.

    Args:
        haystack: Sequence to scan.
        needle: Contiguous sequence to locate. An empty sequence never matches.

    Returns:
        Half-open ``(start, end)`` spans in ascending start order.
    """
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return []
    # Cheap membership prefilter on the first element, then verify.
    hay = list(haystack)
    ndl = list(needle)
    first = ndl[0]
    out: list[tuple[int, int]] = []
    i = 0
    limit = n - m
    while i <= limit:
        if hay[i] == first and hay[i : i + m] == ndl:
            out.append((i, i + m))
        i += 1
    return out


def weighted_interval_cover(matches: list[Match]) -> list[Match]:
    """Select the non-overlapping matches covering the most tokens.

    A weighted-interval dynamic program uses match length as weight. Equal-coverage
    choices favor the earlier cover.

    Args:
        matches: Candidate occurrences, which may overlap or be unordered.

    Returns:
        Selected matches in ascending position order.
    """
    if not matches:
        return []
    iv = sorted(matches, key=lambda m: (m.end, m.start))
    n = len(iv)
    ends = [m.end for m in iv]
    # p[i] = index of the last interval whose end <= iv[i].start (or -1).
    p = [bisect.bisect_right(ends, iv[i].start) - 1 for i in range(n)]

    dp = [0] * (n + 1)  # dp[k] = best coverage using iv[0..k-1]
    for i in range(1, n + 1):
        incl = iv[i - 1].length + dp[p[i - 1] + 1]
        dp[i] = max(incl, dp[i - 1])

    chosen: list[Match] = []
    i = n
    while i > 0:
        incl = iv[i - 1].length + dp[p[i - 1] + 1]
        if incl > dp[i - 1]:  # strict '>' => on a tie, prefer the earlier cover
            chosen.append(iv[i - 1])
            i = p[i - 1] + 1
        else:
            i -= 1
    chosen.reverse()
    return chosen


def token_cover(tokens: Sequence[int], chunks: Mapping[str, Sequence[int]]) -> list[Match]:
    """Find the largest token-exact cover by registered chunk sequences.

    A token-valued Aho-Corasick automaton finds all registered sequences in one scan.
    This avoids rescanning the pending stream separately for every chunk.

    Args:
        tokens: Pending token stream.
        chunks: Registered chunk IDs and their exact token sequences.

    Returns:
        Maximum-coverage non-overlapping matches.
    """
    transitions: list[dict[int, int]] = [{}]
    failures = [0]
    outputs: list[list[tuple[int, str, int]]] = [[]]

    for registration_order, (chunk_id, chunk_tokens) in enumerate(chunks.items()):
        pattern = [int(token) for token in chunk_tokens]
        if not pattern:
            continue
        state = 0
        for token in pattern:
            next_state = transitions[state].get(token)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][token] = next_state
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = next_state
        outputs[state].append((registration_order, chunk_id, len(pattern)))

    queue = deque(transitions[0].values())
    while queue:
        state = queue.popleft()
        for token, next_state in transitions[state].items():
            queue.append(next_state)
            fallback = failures[state]
            while fallback and token not in transitions[fallback]:
                fallback = failures[fallback]
            failures[next_state] = transitions[fallback].get(token, 0)
            outputs[next_state].extend(outputs[failures[next_state]])

    # Preserve registration order when duplicate patterns produce the same interval.
    for state_outputs in outputs:
        state_outputs.sort(key=lambda item: item[0])

    candidates: list[Match] = []
    state = 0
    for end, raw_token in enumerate(tokens, start=1):
        token = int(raw_token)
        while state and token not in transitions[state]:
            state = failures[state]
        state = transitions[state].get(token, 0)
        for _registration_order, chunk_id, length in outputs[state]:
            candidates.append(Match(chunk_id, end - length, end))
    return weighted_interval_cover(candidates)


def string_cover(
    text: str,
    chunks: Mapping[str, str],
    offset_mapping: list[tuple[int, int]],
) -> list[Match]:
    """Find a substring cover and map character occurrences to token spans.

    Each substring occurrence is mapped to the smallest token interval overlapping its
    characters before maximum-coverage selection.

    Args:
        text: Original pending text.
        chunks: Registered chunk IDs and source strings.
        offset_mapping: Character start and end offsets for every token in ``text``.

    Returns:
        Maximum-coverage non-overlapping token-space matches.
    """
    if not text or not offset_mapping:
        return []
    # Zero-width offsets belong to special tokens and cannot overlap source text.
    # Index only real source spans while retaining their original token positions.
    indexed_offsets = [
        (token_index, char_start, char_end)
        for token_index, (char_start, char_end) in enumerate(offset_mapping)
        if char_end > char_start
    ]
    token_indices = [token_index for token_index, _, _ in indexed_offsets]
    token_starts = [char_start for _, char_start, _ in indexed_offsets]
    token_ends = [char_end for _, _, char_end in indexed_offsets]
    candidates: list[Match] = []
    for cid, ctext in chunks.items():
        if not ctext:
            continue
        pos = text.find(ctext)
        while pos != -1:
            cs, ce = pos, pos + len(ctext)
            # Locate the first token ending after the occurrence starts and the last
            # token starting before it ends. Tokenizer offsets are monotonic, so both
            # boundaries are logarithmic lookups rather than a scan per occurrence.
            first = bisect.bisect_right(token_ends, cs)
            after_last = bisect.bisect_left(token_starts, ce)
            if first < after_last:
                candidates.append(
                    Match(
                        cid,
                        token_indices[first],
                        token_indices[after_last - 1] + 1,
                    )
                )
            pos = text.find(ctext, pos + 1)
    return weighted_interval_cover(candidates)


def find_chunk_cover(
    tokens: Sequence[int],
    chunks: dict,
    level: str = "token",
    *,
    text: str | None = None,
    offset_mapping: list[tuple[int, int]] | None = None,
) -> list[Match]:
    """Find registered chunks using token-exact or string matching.

    Args:
        tokens: Pending token stream.
        chunks: Chunk IDs mapped to token sequences for ``"token"`` matching or text
            for ``"string"`` matching.
        level: Matching representation: ``"token"`` or ``"string"``.
        text: Original text required for string matching.
        offset_mapping: Token character offsets required for string matching.

    Returns:
        Maximum-coverage non-overlapping matches.

    Raises:
        ValueError: If ``level`` is unknown or string matching lacks text or offsets.
    """
    if level == "token":
        return token_cover(tokens, chunks)
    if level == "string":
        if text is None or offset_mapping is None:
            raise ValueError("level='string' requires text= and offset_mapping=")
        return string_cover(text, chunks, offset_mapping)
    raise ValueError(f"unknown match level {level!r} (use 'token' or 'string')")
