"""Run stateful packet matching, cache assembly, generation, and teacher recording.

``PacketSession`` buffers text or token pushes and detects registered reusable chunks
inside them. In serving mode, matched chunks are substituted with precomputed caches
and unmatched spans are forwarded normally. In recording mode, the full prompt is
forwarded without substitution and the resulting teacher trajectory is retained as a
``TrainingSample``.

Pushes are lazy: model work occurs only when content is processed, generated from,
scored, or recorded. The session owns logical positions and uses backend-native
decoding when available; an explicit loop remains available for recording, diagnostics,
and forward-only backends.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Literal, Protocol, cast, overload

from time import perf_counter

import torch
from torch import Tensor

from ..backend.base import CacheStore, ComputeBackend, GenerationBackend
from ..errors import BackendCompatibilityError, SourceConflictError, UnsupportedModalityError
from ..kvcache.cache import KVCache
from ..runtime import (
    MixedPrefillPolicy,
    PacketCacheState,
    PacketDecodeState,
    PacketExecutor,
    SamplingConfig,
    select_next_token,
)
from ..runtime.matching import Match, string_cover
from .sample import TrainingSample
from .store import PacketStore

__all__ = [
    "Mode",
    "MixedPrefillPolicy",
    "GenerationResult",
    "ContinuationScore",
    "SessionTokenizer",
    "PacketSession",
]


class SessionTokenizer(Protocol):
    """Tokenizer operations required by a packet session.

    Hugging Face tokenizers satisfy this interface, but the protocol keeps the session
    independent of a concrete tokenizer framework.
    """

    @property
    def eos_token_id(self) -> int | None:
        """Configured end-of-sequence token ID, when available."""
        ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, object]:
        """Tokenize one string and optionally return character offsets."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str | list[str]:
        """Decode one token sequence."""
        ...


class Mode(Enum):
    """Whether a session records full-attention teachers or serves cached packets."""

    RECORD = "record"
    SERVE = "serve"


@dataclass
class GenerationResult:
    """Generated continuation and packet-serving diagnostics.

    Attributes:
        token_ids: Generated continuation IDs excluding the prompt.
        text: Decoded continuation, or an empty string without a tokenizer.
        prompt_len: Number of trailing prompt tokens used to seed decoding.
        flops: Re-rotation FLOPs spent assembling substituted packets.
        ttft: Seconds through the first host-visible generated token.
        cover: Matched reusable chunks in pending-stream token coordinates.
        prompt_logits: Optional first-forward logits for parity diagnostics.
    """

    token_ids: list[int]
    text: str
    prompt_len: int
    flops: int = 0
    ttft: float = 0.0
    cover: list[Match] = field(default_factory=list)
    prompt_logits: Tensor | None = None

    @property
    def matched_chunk_ids(self) -> list[str]:
        """Registered chunk IDs detected in the pending stream."""
        return [m.chunk_id for m in self.cover]

    @property
    def cached_tokens(self) -> int:
        """Number of source tokens matched to reusable packets.

        This counts matched source tokens rather than wrapper tokens in the substituted
        caches.
        """
        return sum(m.length for m in self.cover)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(new={len(self.token_ids)}, prompt_len={self.prompt_len}, "
            f"flops={self.flops:,}, ttft={self.ttft * 1e3:.1f}ms, "
            f"matched={self.matched_chunk_ids}, cached_tokens={self.cached_tokens})"
        )


@dataclass
class ContinuationScore:
    """Teacher-forced likelihood and serving diagnostics for a fixed continuation.

    Attributes:
        token_ids: Caller-supplied continuation tokens.
        token_nll: Per-token negative log-likelihoods in continuation order.
        prompt_len: Number of trailing prompt tokens used to begin scoring.
        flops: Re-rotation FLOPs spent assembling substituted packets.
        cover: Matched reusable chunks in pending-stream token coordinates.
    """

    token_ids: list[int]
    token_nll: list[float]
    prompt_len: int
    flops: int = 0
    cover: list[Match] = field(default_factory=list)

    @property
    def mean_nll(self) -> float:
        """Arithmetic mean of continuation token negative log-likelihoods."""
        if not self.token_nll:
            raise ValueError("cannot average an empty continuation")
        return sum(self.token_nll) / len(self.token_nll)

    @property
    def perplexity(self) -> float:
        """Exponentiated mean continuation negative log-likelihood."""
        return math.exp(self.mean_nll)


@dataclass
class _Push:
    """One buffered input and whether it participates in chunk matching."""

    payload: str | list[int]
    is_text: bool
    matching_enabled: bool


class PacketSession:
    """Stateful runtime for reusable-chunk registration and model interaction.

    Serving sessions substitute detected chunks with packets from ``store``. Recording
    sessions run full attention and retain teacher logits for adapter training. Session
    state consists of registered matching representations, buffered pushes, the
    committed prefix cache, its matching logical token trajectory, the next logical
    model position, and recorded samples.

    Args:
        backend: Framework-neutral model execution backend.
        mode: Packet-serving or teacher-recording behavior.
        match: Chunk matching representation: ``"token"`` or ``"string"``.
        store: Optional prebuilt packet store. An empty framework store is created when
            omitted.
        tokenizer: Tokenizer used for text pushes, matching, decoding, and chat render.
        prefill_chunk_size: Optional maximum tokens per unmatched prefill forward.
        mixed_prefill_policy: Sequential or fused handling of interleaved cached and
            uncached spans.

    Raises:
        TypeError: If ``backend`` does not implement ``ComputeBackend``.
        ValueError: If matching, chunk-size, or mixed-prefill options
            are invalid.
        RerotationError: If serving uses a positional encoding that cannot be shifted.
    """

    def __init__(
        self,
        *,
        backend: ComputeBackend,
        mode: Mode = Mode.SERVE,
        match: str = "token",
        store: CacheStore | None = None,
        tokenizer: SessionTokenizer | None = None,
        prefill_chunk_size: int | None = None,
        mixed_prefill_policy: MixedPrefillPolicy | str = MixedPrefillPolicy.SEQUENTIAL,
    ) -> None:
        if match not in ("token", "string"):
            raise ValueError("match must be 'token' or 'string'")
        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend must implement the ComputeBackend protocol")
        if mode is Mode.SERVE:
            # The backend owns model-family/capability knowledge. The session only
            # asks whether packet re-rotation is safe for this backend/model pair.
            backend.assert_rerotatable()
        self.backend = backend
        if prefill_chunk_size is not None and (
            not isinstance(prefill_chunk_size, int)
            or isinstance(prefill_chunk_size, bool)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        self.mode = mode
        self.match_level = match
        self.tokenizer: SessionTokenizer | None = tokenizer
        self.store = store if store is not None else PacketStore(backend=backend)
        self.executor = PacketExecutor(
            backend=backend,
            store=self.store,
            packet_ids=(),
            prefill_chunk_size=prefill_chunk_size,
            mixed_prefill_policy=mixed_prefill_policy,
        )
        self.prefill_chunk_size = self.executor.prefill_chunk_size
        self.mixed_prefill_policy = self.executor.mixed_prefill_policy

        # String matching needs original text. Token matching lives in the shared
        # executor so direct and stateful callers use the same catalog.
        self._chunk_text: dict[str, str] = {}

        # Pending pushes remain uncommitted until a processing operation consumes them.
        self._pending: list[_Push] = []
        self._cache: KVCache | None = None
        self._offset: int = 0
        self._trajectory_ids: list[int] = []
        self._trajectory_chunk_spans: dict[str, list[tuple[int, int]]] = {}

        self._samples: list[TrainingSample] = []

    def register_chunk(
        self, chunk_id: str, text: str | None = None, token_ids: list[int] | None = None
    ) -> None:
        """Register source content for matching against preprocessed packets.

        Text is tokenized with the session tokenizer. If a stored packet exists, the
        supplied matching content must reproduce its metadata. Serving requires packets
        to be constructed beforehand by ``PacketPreprocessor``.

        Args:
            chunk_id: Stable chunk and packet-store identifier.
            text: Optional source text used for tokenization and string matching.
            token_ids: Optional explicit token sequence used instead of tokenizing text.

        Raises:
            ValueError: If content is missing, disagrees with a packet, or a serving
                packet has not been preprocessed.
            UnsupportedModalityError: If a stored packet has no token representation.
        """
        toks = self._resolve_tokens(text, token_ids)
        if chunk_id in self.store:
            stored = self.store.get(chunk_id)
            if stored.token_ids is None:
                raise UnsupportedModalityError(
                    f"stored packet {chunk_id!r} has no token representation for matching"
                )
            if toks != stored.token_ids:
                raise ValueError(
                    f"registered content for {chunk_id!r} does not match its preprocessed packet"
                )
            if text is not None and stored.match_text is not None and text != stored.match_text:
                raise ValueError(
                    f"registered text for {chunk_id!r} does not match its preprocessed packet"
                )
        if self.mode is Mode.SERVE and chunk_id not in self.store:
            raise ValueError(f"packet {chunk_id!r} is missing; preprocess sources before serving")
        if self.mode is Mode.SERVE:
            self.executor._register_packet(chunk_id, toks)
        else:
            self.executor._register_token_pattern(chunk_id, toks)
        if text is not None:
            self._chunk_text[chunk_id] = text

    def register_chunks(self, chunks: dict[str, str]) -> None:
        """Register a mapping of chunk IDs to source text."""
        for cid, text in chunks.items():
            self.register_chunk(cid, text=text)

    def register_store_chunks(self, chunk_ids: list[str] | None = None) -> list[str]:
        """Register preprocessed packets with the library-side matcher.

        This copies framework-produced matching representations without rebuilding
        packets or resolving wrappers.

        Args:
            chunk_ids: Selected packet IDs, or every stored packet when omitted.

        Returns:
            Registered IDs in requested or store order.

        Raises:
            ValueError: If IDs repeat or an existing registration disagrees.
            KeyError: If a requested packet is absent.
            BackendCompatibilityError: If the session tokenizer does not reproduce
                preprocessed tokens.
            UnsupportedModalityError: If required token or text metadata is absent.
            SourceConflictError: If identical token sequences select different wrappers.
        """
        selected = self.store.ids() if chunk_ids is None else list(chunk_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("chunk_ids contains duplicates")

        candidates: list[tuple[str, list[int], str | None]] = []
        token_owners: dict[tuple[int, ...], tuple[str, str | None]] = {}
        for chunk_id in selected:
            if chunk_id not in self.store:
                raise KeyError(f"packet store has no chunk {chunk_id!r}")
            stored = self.store.get(chunk_id)
            if stored.token_ids is None:
                raise UnsupportedModalityError(
                    f"stored packet {chunk_id!r} has no framework-produced token "
                    "representation for automatic matching"
                )
            if self.match_level == "string" and stored.match_text is None:
                raise UnsupportedModalityError(
                    f"stored packet {chunk_id!r} has no framework-produced text "
                    "representation for string matching"
                )
            tokens = list(stored.token_ids)
            if stored.match_text is not None and self.tokenizer is not None:
                session_tokens = self._tokenize(stored.match_text)
                if session_tokens != tokens:
                    raise BackendCompatibilityError(
                        f"session tokenizer does not reproduce preprocessed tokens "
                        f"for packet {chunk_id!r}"
                    )
            wrapper_id = getattr(getattr(stored, "identity", None), "wrapper_id", None)
            previous = token_owners.get(tuple(tokens))
            if previous is not None and previous[1] != wrapper_id:
                raise SourceConflictError(
                    f"stored packets {previous[0]!r} and {chunk_id!r} have identical "
                    "matchable tokens but different wrappers"
                )
            token_owners[tuple(tokens)] = (chunk_id, wrapper_id)
            current = self.executor.registered_token_ids(chunk_id)
            if current is not None and current != tokens:
                raise ValueError(f"chunk {chunk_id!r} is already registered with different tokens")
            candidates.append((chunk_id, tokens, stored.match_text))

        for chunk_id, tokens, match_text in candidates:
            self.executor._register_packet(chunk_id, tokens)
            if match_text is not None:
                self._chunk_text[chunk_id] = match_text
        return selected

    def push(self, content: str | Iterable[int], match: bool | None = None) -> None:
        """Buffer text or token IDs without performing model work.

        Args:
            content: Text or an iterable of token IDs.
            match: Set to ``False`` to prevent chunk matching across this push.
        """
        is_text = isinstance(content, str)
        if not is_text:
            content: list[int] = [int(x) for x in content]
        self._pending.append(
            _Push(payload=content, is_text=is_text, matching_enabled=(match is not False))
        )

    def preview_matches(self) -> list[Match]:
        """Inspect pending chunk matches without committing or forwarding content.

        Returns:
            Detected chunk occurrences in pending-stream token coordinates.
        """
        if not self._pending:
            return []
        stream_ids, records = self._build_stream()
        return self._match(stream_ids, records)

    def process(self) -> list[Match]:
        """Commit pending content into the persistent prefix without generation.

        Returns:
            Committed chunk matches, or an empty list when no content is pending.
        """
        if not self._pending:
            return []
        stream_ids, records = self._build_stream()
        cover = self._match(stream_ids, records)
        result = self.executor.process(
            PacketCacheState(self._cache, self._offset),
            stream_ids,
            cover=cover,
            substitute=self.mode is Mode.SERVE,
        )
        self._cache = result.cache_state.cache
        self._offset = result.cache_state.next_position
        self._commit_trajectory(stream_ids, cover)
        self._pending.clear()
        return cover

    def generate(
        self,
        max_new_tokens: int = 64,
        capture_logits: bool | None = None,
        return_prompt_logits: bool = False,
        eos_token_id: int | list[int] | None = None,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> GenerationResult:
        """Commit pending content and generate a continuation.

        The final unmatched span is treated as the generation prompt. Prefix assembly
        and prompt prefill determine TTFT; decoding then updates the persistent cache.
        Greedy selection is the default, with temperature, top-p, top-k, and min-p
        filtering available when sampling is enabled. RECORD mode also stores a
        ``TrainingSample``.

        Args:
            max_new_tokens: Maximum continuation length.
            capture_logits: Whether to retain one teacher distribution per generated
                token. Defaults to enabled in RECORD mode.
            return_prompt_logits: Whether to retain logits for the first prompt forward.
            eos_token_id: Explicit stopping token or tokens, overriding tokenizer EOS.
            do_sample: Whether to sample instead of selecting the maximum logit.
            temperature: Positive sampling temperature.
            top_p: Cumulative-probability threshold in ``(0, 1]``.
            top_k: Maximum number of candidate tokens; zero disables this filter.
            min_p: Minimum probability relative to the most likely token.

        Returns:
            Generated tokens, decoded text, matches, and serving diagnostics.

        Raises:
            ValueError: If sampling parameters are invalid or no trailing prompt exists.
        """
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        sampling = SamplingConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        stream_ids, records = self._build_stream()
        cover = self._match(stream_ids, records)
        substitute = self.mode is Mode.SERVE

        if capture_logits is None:
            capture_logits = self.mode is Mode.RECORD
        eos = self._eos_ids(eos_token_id)

        use_native_continuation = isinstance(self.backend, GenerationBackend)

        # The framework owns one continuous TTFT interval through first-token
        # selection. A capable backend may natively decode only later tokens.
        self._sync()
        t0 = perf_counter()
        gen_logits: list[Tensor] = []
        prefill = self.executor.prefill(
            PacketCacheState(self._cache, self._offset),
            stream_ids,
            cover=cover,
            substitute=substitute,
            return_prompt_logits=return_prompt_logits,
        )
        prefix_flops = prefill.rerotation_flops
        prompt_len = prefill.prompt_len
        first_token_logits = prefill.first_token_logits
        cache = prefill.cache_state.cache
        assert cache is not None
        cur = prefill.cache_state.next_position
        prompt_logits = prefill.prompt_logits

        if use_native_continuation:
            gen_ids, gen_logits, cache, final_offset, ttft = self._continue_natively(
                cache,
                cur,
                first_token_logits,
                max_new_tokens,
                eos,
                eos_token_id,
                capture_logits=capture_logits,
                started_at=t0,
                sampling=sampling,
            )
        else:
            gen_ids, gen_logits, cache, final_offset, ttft = self._continue(
                cache,
                cur,
                first_token_logits,
                max_new_tokens,
                eos,
                capture_logits,
                started_at=t0,
                sampling=sampling,
            )
        self._cache, self._offset = cache, final_offset
        generated_start = len(self._trajectory_ids) + len(stream_ids)
        self._commit_trajectory(stream_ids + gen_ids, cover)

        self._pending.clear()
        if self.mode is Mode.RECORD:
            self._record_sample(generated_start, len(gen_ids), gen_logits)

        text = self._decode_text(gen_ids)
        return GenerationResult(
            token_ids=gen_ids,
            text=text,
            prompt_len=prompt_len,
            flops=prefix_flops if substitute else 0,
            ttft=ttft,
            cover=cover,
            prompt_logits=prompt_logits,
        )

    @torch.no_grad()
    def score_continuation(self, token_ids: list[int]) -> ContinuationScore:
        """Score a fixed continuation with teacher forcing without changing session state.

        Args:
            token_ids: Non-empty continuation scored in the supplied order.

        Returns:
            Per-token negative log-likelihoods and packet-serving diagnostics.

        Raises:
            ValueError: If the continuation is empty or no trailing prompt exists.
        """
        continuation = [int(token_id) for token_id in token_ids]
        if not continuation:
            raise ValueError("continuation must contain at least one token")

        stream_ids, records = self._build_stream()
        cover = self._match(stream_ids, records)
        prefill = self.executor.prefill(
            PacketCacheState(self._cache, self._offset),
            stream_ids,
            cover=cover,
            substitute=self.mode is Mode.SERVE,
            return_prompt_logits=False,
        )
        prefix_flops = prefill.rerotation_flops
        prompt_len = prefill.prompt_len
        logits = prefill.first_token_logits
        cache = prefill.cache_state.cache
        assert cache is not None
        cur = prefill.cache_state.next_position

        token_nll: list[float] = []
        device = self.backend.device
        for token_id in continuation:
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_nll.append(float(-log_probs[0, token_id].item()))
            step = torch.tensor([[token_id]], dtype=torch.long, device=device)
            position = torch.tensor([[cur]], dtype=torch.long, device=device)
            output = self.backend.forward(
                input_ids=step,
                position_ids=position,
                cache=cache,
            )
            cache = output.cache
            cur += 1
            logits = output.logits[:, -1, :]

        return ContinuationScore(
            token_ids=continuation,
            token_nll=token_nll,
            prompt_len=prompt_len,
            flops=prefix_flops if self.mode is Mode.SERVE else 0,
            cover=cover,
        )

    @torch.no_grad()
    def add_forced_continuation(
        self, token_ids: list[int], *, capture_logits: bool = True
    ) -> GenerationResult:
        """Immediately commit pending content and a caller-supplied continuation.

        Like ``process`` and ``generate``, this method executes and clears all pending
        pushes immediately. It teacher-forces the supplied tokens through the frozen
        full-attention model, optionally records one distribution per token, commits the
        cache and logical trajectory, and appends a ``TrainingSample``.

        Args:
            token_ids: Non-empty continuation to record and commit.
            capture_logits: Whether to retain full-vocabulary teacher logits.

        Returns:
            Recorded continuation and prompt timing metadata.

        Raises:
            RuntimeError: If the session is not in RECORD mode.
            ValueError: If the continuation is empty or no trailing prompt exists.
        """
        if self.mode is not Mode.RECORD:
            raise RuntimeError("add_forced_continuation requires RECORD mode")
        continuation = [int(token_id) for token_id in token_ids]
        if not continuation:
            raise ValueError("continuation must contain at least one token")

        stream_ids, records = self._build_stream()
        cover = self._match(stream_ids, records)

        self._sync()
        started = perf_counter()
        prefill = self.executor.prefill(
            PacketCacheState(self._cache, self._offset),
            stream_ids,
            cover=cover,
            substitute=False,
            return_prompt_logits=False,
        )
        logits = prefill.first_token_logits
        cache = prefill.cache_state.cache
        assert cache is not None
        cur = prefill.cache_state.next_position
        self._sync()
        ttft = perf_counter() - started

        teacher_logits: list[Tensor] = []
        device = self.backend.device
        for token_id in continuation:
            if capture_logits:
                teacher_logits.append(logits[0].detach().clone())
            step = torch.tensor([[token_id]], dtype=torch.long, device=device)
            position = torch.tensor([[cur]], dtype=torch.long, device=device)
            output = self.backend.forward(
                input_ids=step,
                position_ids=position,
                cache=cache,
            )
            cache = output.cache
            cur += 1
            logits = output.logits[:, -1, :]

        self._cache, self._offset = cache, cur
        generated_start = len(self._trajectory_ids) + len(stream_ids)
        self._commit_trajectory(stream_ids + continuation, cover)
        self._pending.clear()
        self._record_sample(
            generated_start,
            len(continuation),
            teacher_logits,
        )
        return GenerationResult(
            token_ids=continuation,
            text=self._decode_text(continuation),
            prompt_len=prefill.prompt_len,
            flops=0,
            ttft=ttft,
            cover=cover,
        )

    def export_sample(self) -> TrainingSample | None:
        """Most recently recorded sample, or ``None`` when none exists."""
        return self._samples[-1] if self._samples else None

    def export_samples(self) -> list[TrainingSample]:
        """Shallow copy of all samples recorded since the last reset."""
        return list(self._samples)

    def reset(self, keep_chunks: bool = True) -> None:
        """Clear pending content, committed cache state, and recorded samples.

        Args:
            keep_chunks: Whether matcher registrations survive the reset.
        """
        self._pending.clear()
        self._cache = None
        self._offset = 0
        self._trajectory_ids.clear()
        self._trajectory_chunk_spans.clear()
        self._samples.clear()
        if not keep_chunks:
            self.executor._clear_packet_registrations()
            self._chunk_text.clear()

    def _resolve_tokens(self, text: str | None, token_ids: list[int] | None) -> list[int]:
        """Resolve explicit token IDs or tokenize supplied text."""
        if token_ids is not None:
            return [int(x) for x in token_ids]
        if text is None:
            raise ValueError("provide text= or token_ids=")
        return self._tokenize(text)

    @overload
    def _tokenize(self, text: str, offsets: Literal[False] = False) -> list[int]: ...

    @overload
    def _tokenize(
        self, text: str, offsets: Literal[True]
    ) -> tuple[list[int], list[tuple[int, int]]]: ...

    def _tokenize(
        self, text: str, offsets: bool = False
    ) -> list[int] | tuple[list[int], list[tuple[int, int]]]:
        """Tokenize text, optionally returning character offsets."""
        if self.tokenizer is None:
            raise RuntimeError("this session needs a tokenizer for text input/chunks")
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=offsets)
        ids = [int(token_id) for token_id in cast(Iterable[int], enc["input_ids"])]
        if offsets:
            raw_offsets = cast(Iterable[tuple[int, int]], enc["offset_mapping"])
            return ids, [(int(start), int(end)) for start, end in raw_offsets]
        return ids

    def _build_stream(self) -> tuple[list[int], list[tuple[int, int, _Push]]]:
        """Flatten pending pushes and retain each push's half-open token span."""
        stream_ids: list[int] = []
        records: list[tuple[int, int, _Push]] = []
        for p in self._pending:
            ids = self._tokenize(p.payload) if isinstance(p.payload, str) else list(p.payload)
            start = len(stream_ids)
            stream_ids.extend(ids)
            records.append((start, start + len(ids), p))
        return stream_ids, records

    def _match(self, stream_ids: list[int], records: list[tuple[int, int, _Push]]) -> list[Match]:
        """Match registered chunks only across pushes that permit matching."""
        if self.executor.packet_count == 0:
            return []
        matches: list[Match] = []
        if self.match_level == "token":
            # token cover over each maximal run of consecutive matchable pushes
            for lo, hi in self._matchable_runs(records):
                for m in self.executor.match(stream_ids[lo:hi]):
                    matches.append(Match(m.chunk_id, m.start + lo, m.end + lo))
        elif self.match_level == "string":
            # string: match within each matchable text push individually
            for start, _, p in records:
                if not (isinstance(p.payload, str) and p.matching_enabled):
                    continue
                _, offs = self._tokenize(p.payload, offsets=True)
                for m in string_cover(p.payload, self._chunk_text, offs):
                    matches.append(Match(m.chunk_id, m.start + start, m.end + start))
        else:
            raise ValueError(f"unsupported match level {self.match_level!r}")
        matches.sort(key=lambda m: m.start)
        return matches

    @staticmethod
    def _matchable_runs(records: list[tuple[int, int, _Push]]) -> list[tuple[int, int]]:
        """Find maximal token intervals spanning consecutive matchable pushes."""
        runs: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for start, end, p in records:
            if p.matching_enabled:
                cur = (cur[0], end) if cur else (start, end)
            else:
                if cur:
                    runs.append(cur)
                    cur = None
        if cur:
            runs.append(cur)
        return runs

    @torch.no_grad()
    def _continue_natively(
        self,
        cache: KVCache,
        cur: int,
        first_token_logits: Tensor,
        max_new_tokens: int,
        eos: set[int],
        eos_token_id: int | list[int] | None,
        *,
        capture_logits: bool,
        started_at: float,
        sampling: SamplingConfig,
    ) -> tuple[list[int], list[Tensor], KVCache, int, float]:
        """Select the timed first token, then delegate only its continuation."""
        backend = self.backend
        if not isinstance(backend, GenerationBackend):
            raise TypeError("native continuation requires a GenerationBackend")

        first = select_next_token(first_token_logits, sampling)
        self._sync()
        ttft = perf_counter() - started_at
        generated = [first]
        generated_logits = [first_token_logits[0].detach().clone()] if capture_logits else []
        if first in eos or max_new_tokens == 1:
            committed = self.executor.forward_tokens(PacketCacheState(cache, cur), generated)
            assert committed.cache_state.cache is not None
            return (
                generated,
                generated_logits,
                committed.cache_state.cache,
                committed.cache_state.next_position,
                ttft,
            )

        native_eos = eos_token_id if eos_token_id is not None else (sorted(eos) if eos else None)
        native = backend.generate(
            input_ids=torch.tensor([[first]], dtype=torch.long, device=self.backend.device),
            position_ids=torch.tensor([[cur]], dtype=torch.long, device=self.backend.device),
            cache=cache,
            max_new_tokens=max_new_tokens - 1,
            eos_token_id=native_eos,
            capture_logits=capture_logits,
            do_sample=sampling.do_sample,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
        )
        generated.extend(native.token_ids)
        generated_logits.extend(native.logits)
        return generated, generated_logits, native.cache, native.final_offset, ttft

    @torch.no_grad()
    def _continue(
        self,
        cache: KVCache,
        cur: int,
        next_token_logits: Tensor,
        max_new_tokens: int,
        eos: set[int],
        capture_logits: bool,
        *,
        started_at: float,
        sampling: SamplingConfig,
    ) -> tuple[list[int], list[Tensor], KVCache, int, float]:
        """Decode tokens while retaining optional teacher logits and cache state."""
        gen_ids: list[int] = []
        gen_logits: list[Tensor] = []
        first_token_ttft: float | None = None
        state = PacketDecodeState(
            cache_state=PacketCacheState(cache, cur),
            next_token_logits=next_token_logits,
        )
        for _ in range(max_new_tokens):
            if capture_logits:
                gen_logits.append(state.next_token_logits[0].detach().clone())
            step = self.executor.decode_step(state, sampling)
            nxt = step.token_id
            state = step.state
            gen_ids.append(nxt)
            if first_token_ttft is None:
                self._sync()
                first_token_ttft = perf_counter() - started_at
            if nxt in eos:
                break
        assert first_token_ttft is not None
        assert state.cache_state.cache is not None
        return (
            gen_ids,
            gen_logits,
            state.cache_state.cache,
            state.cache_state.next_position,
            first_token_ttft,
        )

    def _sync(self) -> None:
        """Synchronize CUDA so measured TTFT excludes asynchronous spillover."""
        if self.backend.device.type == "cuda":
            torch.cuda.synchronize(self.backend.device)

    def _commit_trajectory(self, token_ids: list[int], cover: list[Match]) -> None:
        """Append committed source tokens and globally shifted packet spans."""
        start = len(self._trajectory_ids)
        self._trajectory_ids.extend(token_ids)
        for match in cover:
            self._trajectory_chunk_spans.setdefault(match.chunk_id, []).append(
                (start + match.start, start + match.end)
            )

    def _record_sample(
        self,
        generated_start: int,
        generated_count: int,
        generated_logits: list[Tensor],
    ) -> None:
        """Snapshot the committed logical trajectory for one RECORD target span."""
        generated_end = generated_start + generated_count
        if not 0 <= generated_start < generated_end <= len(self._trajectory_ids):
            raise ValueError("generated target span is outside the committed trajectory")
        if generated_logits and len(generated_logits) != generated_count:
            raise ValueError("generated logits must align one-to-one with target tokens")

        token_ids = torch.tensor(self._trajectory_ids, dtype=torch.long)
        generated_mask = torch.zeros(len(self._trajectory_ids), dtype=torch.bool)
        generated_mask[generated_start:generated_end] = True
        logits = torch.stack(generated_logits, dim=0) if generated_logits else None
        self._samples.append(
            TrainingSample(
                token_ids=token_ids,
                generated_mask=generated_mask,
                chunk_spans={
                    chunk_id: list(spans)
                    for chunk_id, spans in self._trajectory_chunk_spans.items()
                },
                logits=logits,
            )
        )

    def _eos_ids(self, eos_token_id) -> set[int]:
        """Resolve explicit or tokenizer-configured stopping token IDs."""
        if eos_token_id is not None:
            return {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
        tok = self.tokenizer
        if tok is not None and tok.eos_token_id is not None:
            return {int(tok.eos_token_id)}
        return set()

    def _decode_text(self, ids: list[int]) -> str:
        """Decode generated IDs when a tokenizer is available."""
        if self.tokenizer is None or not ids:
            return ""
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer must decode one token sequence to a string")
        return decoded

    def __repr__(self) -> str:
        return (
            f"PacketSession(mode={self.mode.value}, match={self.match_level}, "
            f"chunks={self.executor.packet_count}, offset={self._offset})"
        )
