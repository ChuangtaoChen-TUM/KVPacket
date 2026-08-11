"""Execute packet matching, cache substitution, and prompt prefill reentrantly.

``PacketExecutor`` contains shared serving configuration but no conversation state.
Callers pass a ``PacketCacheState`` into every operation and retain the returned state,
which lets an existing cache or session manager own cache lifetime without adopting
``PacketSession``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..backend.base import CacheStore, ComputeBackend
from ..errors import SourceConflictError, UnsupportedModalityError
from ..kvcache.cache import KVCache, concat_caches
from .matching import Match, token_cover
from .types import (
    MixedPrefillPolicy,
    PacketCacheState,
    PacketPrefillResult,
    PacketProcessResult,
    PacketTokenForwardResult,
)
from .generation import PacketDecodeState, PacketDecodeStep, SamplingConfig, select_next_token

__all__ = ["PacketExecutor"]


@dataclass(frozen=True)
class _FusedPrefill:
    """Physically reordered cache and online inputs for one fused prefill."""

    cache: KVCache
    input_ids: list[int]
    position_ids: Tensor
    attention_mask: Tensor
    final_position: int
    prompt_len: int
    rerotation_flops: int


class PacketExecutor:
    """Reentrant packet-aware forward engine.

    The executor owns a backend, packet store, and token-matching catalog. It never
    retains a request's cache or logical position. A caller can therefore share one
    executor across conversations while storing each returned ``PacketCacheState`` in
    its existing session manager.

    Args:
        backend: Framework-neutral model execution backend.
        store: Read-only store of preprocessed packet caches.
        packet_ids: Packet IDs available for automatic token matching. By default,
            use every packet in the store; pass an empty iterable when matches are
            supplied explicitly by a higher-level session.
        prefill_chunk_size: Optional maximum tokens per sequential online forward.
        mixed_prefill_policy: Sequential or fused handling of interleaved packets and
            online spans.

    Raises:
        TypeError: If ``backend`` does not implement ``ComputeBackend``.
        ValueError: If a prefill or mixed-prefill option is invalid.
    """

    def __init__(
        self,
        *,
        backend: ComputeBackend,
        store: CacheStore,
        packet_ids: Iterable[str] | None = None,
        prefill_chunk_size: int | None = None,
        mixed_prefill_policy: MixedPrefillPolicy | str = MixedPrefillPolicy.SEQUENTIAL,
    ) -> None:
        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend must implement the ComputeBackend protocol")
        if prefill_chunk_size is not None and (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer or None")
        try:
            policy = MixedPrefillPolicy(mixed_prefill_policy)
        except ValueError as error:
            choices = ", ".join(item.value for item in MixedPrefillPolicy)
            raise ValueError(f"mixed_prefill_policy must be one of: {choices}") from error

        self.backend = backend
        self.store = store
        self.prefill_chunk_size = prefill_chunk_size
        self.mixed_prefill_policy = policy
        self._packet_tokens: dict[str, list[int]] = {}
        self._register_store_packets(packet_ids)

    @property
    def packet_count(self) -> int:
        """Number of packet token patterns available to the matcher."""
        return len(self._packet_tokens)

    def registered_token_ids(self, chunk_id: str) -> list[int] | None:
        """Copy of one registered token pattern, if present."""
        tokens = self._packet_tokens.get(chunk_id)
        return None if tokens is None else list(tokens)

    def _register_packet(
        self,
        chunk_id: str,
        token_ids: Sequence[int] | None = None,
    ) -> None:
        """Register one stored packet for token-exact matching.

        Args:
            chunk_id: Packet-store identifier.
            token_ids: Expected source-body tokens. Stored metadata is used when
                omitted and verified when supplied.

        Raises:
            KeyError: If the packet is absent from the store.
            ValueError: If supplied tokens disagree with the stored packet.
            UnsupportedModalityError: If the packet has no token representation.
        """
        if chunk_id not in self.store:
            raise KeyError(f"packet store has no chunk {chunk_id!r}")
        stored = self.store.get(chunk_id)
        if stored.token_ids is None:
            raise UnsupportedModalityError(
                f"stored packet {chunk_id!r} has no token representation for matching"
            )
        stored_tokens = [int(token_id) for token_id in stored.token_ids]
        resolved = stored_tokens if token_ids is None else [int(token_id) for token_id in token_ids]
        if resolved != stored_tokens:
            raise ValueError(
                f"registered content for {chunk_id!r} does not match its preprocessed packet"
            )
        self._register_token_pattern(chunk_id, resolved)

    def _register_token_pattern(self, chunk_id: str, token_ids: Sequence[int]) -> None:
        """Register a token pattern without requiring a materialized packet.

        This supports full-attention recording, where matches label training spans but
        are not substituted. It is intentionally private; ``PacketSession`` owns
        dynamic recording registrations.

        Args:
            chunk_id: Stable label attached to detected spans.
            token_ids: Non-empty exact token pattern.
        """
        tokens = [int(token_id) for token_id in token_ids]
        if not tokens:
            raise ValueError("registered token patterns must be non-empty")
        current = self._packet_tokens.get(chunk_id)
        if current is not None and current != tokens:
            raise ValueError(f"chunk {chunk_id!r} is already registered with different tokens")
        self._packet_tokens[chunk_id] = tokens

    def _register_store_packets(self, chunk_ids: Iterable[str] | None = None) -> list[str]:
        """Register selected stored packets for token-exact matching.

        Duplicate token patterns are allowed only when they select the same wrapper.
        This prevents an arbitrary registration-order choice from silently changing
        packet semantics.

        Args:
            chunk_ids: Selected IDs, or every packet in store order when omitted.

        Returns:
            Registered IDs in requested or store order.
        """
        selected = self.store.ids() if chunk_ids is None else list(chunk_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("chunk_ids contains duplicates")

        token_owners: dict[tuple[int, ...], tuple[str, str | None]] = {}
        registrations: list[tuple[str, list[int]]] = []
        for chunk_id in selected:
            if chunk_id not in self.store:
                raise KeyError(f"packet store has no chunk {chunk_id!r}")
            stored = self.store.get(chunk_id)
            if stored.token_ids is None:
                raise UnsupportedModalityError(
                    f"stored packet {chunk_id!r} has no token representation for matching"
                )
            tokens = [int(token_id) for token_id in stored.token_ids]
            identity = getattr(stored, "identity", None)
            wrapper_id = getattr(identity, "wrapper_id", None)
            previous = token_owners.get(tuple(tokens))
            if previous is not None and previous[1] != wrapper_id:
                raise SourceConflictError(
                    f"stored packets {previous[0]!r} and {chunk_id!r} have identical "
                    "matchable tokens but different wrappers"
                )
            token_owners[tuple(tokens)] = (chunk_id, wrapper_id)
            registrations.append((chunk_id, tokens))

        for chunk_id, tokens in registrations:
            self._register_packet(chunk_id, tokens)
        return selected

    def _clear_packet_registrations(self) -> None:
        """Remove all token patterns without modifying the packet store."""
        self._packet_tokens.clear()

    def match(self, input_ids: Sequence[int]) -> list[Match]:
        """Find the maximum-coverage set of registered packet occurrences."""
        return token_cover(input_ids, self._packet_tokens)

    @torch.no_grad()
    def forward_tokens(
        self,
        cache_state: PacketCacheState,
        input_ids: Sequence[int],
    ) -> PacketTokenForwardResult:
        """Forward explicit online tokens without packet matching.

        This is the continuation-side counterpart of ``prefill``. It is useful when a
        caller or protocol adapter selects generation tokens incrementally and must
        commit them to a caller-owned cache.

        Args:
            cache_state: Caller-owned prefix cache and logical position.
            input_ids: Non-empty tokens to forward exactly as supplied.

        Returns:
            Extended cache state and logits for all supplied positions.
        """
        tokens = [int(token_id) for token_id in input_ids]
        if not tokens:
            raise ValueError("input_ids must contain at least one token")
        position = cache_state.next_position
        tensor = torch.tensor(tokens, dtype=torch.long, device=self.backend.device).unsqueeze(0)
        position_ids = torch.arange(
            position,
            position + len(tokens),
            device=self.backend.device,
        ).unsqueeze(0)
        output = self.backend.forward(
            input_ids=tensor,
            position_ids=position_ids,
            cache=cache_state.cache,
        )
        return PacketTokenForwardResult(
            cache_state=PacketCacheState(output.cache, position + len(tokens)),
            logits=output.logits,
        )

    @torch.no_grad()
    def decode_step(
        self,
        state: PacketDecodeState,
        sampling: SamplingConfig = SamplingConfig(),
    ) -> PacketDecodeStep:
        """Select and commit one token without retaining generation state."""
        token_id = select_next_token(state.next_token_logits, sampling)
        forwarded = self.forward_tokens(state.cache_state, [token_id])
        return PacketDecodeStep(
            token_id=token_id,
            state=PacketDecodeState(
                cache_state=forwarded.cache_state,
                next_token_logits=forwarded.logits[:, -1, :],
            ),
        )

    @torch.no_grad()
    def process(
        self,
        cache_state: PacketCacheState,
        input_ids: Sequence[int],
        *,
        cover: Sequence[Match] | None = None,
        substitute: bool = True,
    ) -> PacketProcessResult:
        """Commit input without generating or retaining request state.

        Args:
            cache_state: Caller-owned prefix cache and logical position.
            input_ids: New token stream to commit.
            cover: Optional precomputed matches. Token matching runs when omitted.
            substitute: Whether matches become stored packets. ``False`` forwards the
                complete stream and is used by full-attention teacher recording.

        Returns:
            Extended cache state, matches, and packet re-rotation cost.
        """
        tokens = [int(token_id) for token_id in input_ids]
        matches = self._resolve_cover(tokens, cover, require_packets=substitute)
        if not tokens:
            return PacketProcessResult(cache_state=cache_state, cover=matches)
        if substitute:
            self.backend.assert_rerotatable()

        if self._should_fuse(tokens, matches, substitute=substitute, trailing_prompt=False):
            fused = self._prepare_fused_prefill(
                cache_state,
                tokens,
                matches,
                trailing_prompt=False,
            )
            _, state, _ = self._execute_fused_prefill(fused, return_prompt_logits=False)
            flops = fused.rerotation_flops
        else:
            cache, position, _, flops = self._assemble(
                cache_state,
                tokens,
                matches,
                trailing_prompt=False,
                substitute=substitute,
            )
            state = PacketCacheState(cache=cache, next_position=position)
        return PacketProcessResult(
            cache_state=state,
            cover=matches,
            rerotation_flops=flops if substitute else 0,
        )

    @torch.no_grad()
    def prefill(
        self,
        cache_state: PacketCacheState,
        input_ids: Sequence[int],
        *,
        cover: Sequence[Match] | None = None,
        substitute: bool = True,
        return_prompt_logits: bool = False,
    ) -> PacketPrefillResult:
        """Assemble packet-aware input and prefill its trailing prompt once.

        The unmatched suffix after the final packet match is the prompt. With no
        matches, the complete input is the prompt. The returned cache includes both the
        assembled prefix and prompt; the executor itself retains neither.

        Args:
            cache_state: Caller-owned prefix cache and logical position.
            input_ids: New packet-bearing context and trailing prompt tokens.
            cover: Optional precomputed matches. Token matching runs when omitted.
            substitute: Whether matches become stored packets.
            return_prompt_logits: Whether to copy logits for every prompt position.

        Returns:
            First-token logits, extended cache state, matches, and diagnostics.

        Raises:
            ValueError: If no trailing unmatched prompt exists.
        """
        tokens = [int(token_id) for token_id in input_ids]
        matches = self._resolve_cover(tokens, cover, require_packets=substitute)
        if substitute:
            self.backend.assert_rerotatable()

        if self._should_fuse(tokens, matches, substitute=substitute, trailing_prompt=True):
            fused = self._prepare_fused_prefill(
                cache_state,
                tokens,
                matches,
                trailing_prompt=True,
            )
            first_logits, state, prompt_logits = self._execute_fused_prefill(
                fused,
                return_prompt_logits=return_prompt_logits,
            )
            prompt_len = fused.prompt_len
            flops = fused.rerotation_flops
        else:
            cache, position, prompt_ids, flops = self._assemble(
                cache_state,
                tokens,
                matches,
                trailing_prompt=True,
                substitute=substitute,
            )
            first_logits, state, prompt_logits = self._prefill(
                PacketCacheState(cache=cache, next_position=position),
                prompt_ids,
                return_prompt_logits=return_prompt_logits,
            )
            prompt_len = len(prompt_ids)
        return PacketPrefillResult(
            cache_state=state,
            first_token_logits=first_logits,
            prompt_logits=prompt_logits,
            prompt_len=prompt_len,
            cover=matches,
            rerotation_flops=flops if substitute else 0,
        )

    def _resolve_cover(
        self,
        input_ids: list[int],
        cover: Sequence[Match] | None,
        *,
        require_packets: bool,
    ) -> list[Match]:
        """Resolve and validate packet matches for one input stream."""
        matches = (
            self.match(input_ids) if cover is None else sorted(cover, key=lambda item: item.start)
        )
        previous_end = 0
        for match in matches:
            if require_packets and match.chunk_id not in self.store:
                raise KeyError(f"packet store has no chunk {match.chunk_id!r}")
            if match.start < previous_end or match.start < 0 or match.end > len(input_ids):
                raise ValueError("packet cover contains overlapping or out-of-bounds matches")
            if match.end <= match.start:
                raise ValueError("packet matches must cover at least one token")
            previous_end = match.end
        return list(matches)

    def _should_fuse(
        self,
        input_ids: list[int],
        cover: list[Match],
        *,
        substitute: bool,
        trailing_prompt: bool,
    ) -> bool:
        """Select fusion when an online prefix span precedes a later packet.

        Sequential assembly must forward an online gap before it can append a later
        packet. Fusion avoids that serial boundary by placing packets first physically
        and restoring the logical causal graph with a four-dimensional mask. A trailing
        generation prompt is excluded from this decision because it already requires
        the final prefill.
        """
        if not substitute or self.mixed_prefill_policy is not MixedPrefillPolicy.FUSED or not cover:
            return False
        prefix_end = max(item.end for item in cover) if trailing_prompt else len(input_ids)
        saw_gap = False
        for segment in self._segments(cover, 0, prefix_end):
            if isinstance(segment, Match):
                if saw_gap:
                    return True
            else:
                _, start, end = segment
                saw_gap = saw_gap or start < end
        return False

    def _prepare_fused_prefill(
        self,
        cache_state: PacketCacheState,
        input_ids: list[int],
        cover: list[Match],
        *,
        trailing_prompt: bool,
    ) -> _FusedPrefill:
        """Reorder packets and online tokens for one masked fused forward."""
        if not self._should_fuse(
            input_ids,
            cover,
            substitute=True,
            trailing_prompt=trailing_prompt,
        ):
            raise ValueError("fused prefill requested for a non-mixed input stream")

        prompt_start = max(item.end for item in cover) if trailing_prompt else len(input_ids)
        if trailing_prompt and prompt_start >= len(input_ids):
            raise ValueError(self._missing_prompt_message())

        device = self.backend.device
        logical_position = cache_state.next_position
        historical = cache_state.cache
        historical_len = 0 if historical is None else historical.seq_len
        shifted_packets: list[KVCache] = []
        packet_positions: list[Tensor] = []
        online_ids: list[int] = []
        online_positions: list[int] = []
        rerotation_flops = 0

        for segment in self._segments(cover, 0, len(input_ids)):
            if isinstance(segment, Match):
                stored = self.store.get(segment.chunk_id)
                shifted = self.backend.shift_cache(stored.packet.to(device), logical_position)
                shifted_packets.append(shifted)
                packet_positions.append(
                    torch.full(
                        (1, shifted.seq_len),
                        logical_position + stored.wrapped_len - 1,
                        dtype=torch.long,
                        device=device,
                    )
                )
                rerotation_flops += self.backend.shift_cache_flops(shifted)
                logical_position += stored.wrapped_len
            else:
                _, start, end = segment
                online_ids.extend(input_ids[start:end])
                online_positions.extend(range(logical_position, logical_position + end - start))
                logical_position += end - start

        cache = self._concat_cache_group(
            ([] if historical is None else [historical]) + shifted_packets
        )
        if cache is None or not online_ids:
            raise RuntimeError("mixed fused prefill needs cached and online tokens")

        online_pos = torch.tensor(online_positions, dtype=torch.long, device=device)
        past_len = cache.seq_len
        online_len = len(online_ids)
        allowed = torch.zeros(
            (online_len, past_len + online_len),
            dtype=torch.bool,
            device=device,
        )
        if historical_len:
            allowed[:, :historical_len] = True
        if packet_positions:
            packet_pos = torch.cat(packet_positions, dim=1).squeeze(0)
            allowed[:, historical_len:past_len] = packet_pos.unsqueeze(0) <= online_pos.unsqueeze(1)
        allowed[:, past_len:] = online_pos.unsqueeze(0) <= online_pos.unsqueeze(1)
        mask = torch.zeros(
            (online_len, past_len + online_len),
            dtype=cache.layers[0].key.dtype,
            device=device,
        )
        mask.masked_fill_(~allowed, -float("inf"))
        return _FusedPrefill(
            cache=cache,
            input_ids=online_ids,
            position_ids=online_pos.unsqueeze(0),
            attention_mask=mask.unsqueeze(0).unsqueeze(0),
            final_position=logical_position,
            prompt_len=len(input_ids) - prompt_start if trailing_prompt else 0,
            rerotation_flops=rerotation_flops,
        )

    def _execute_fused_prefill(
        self,
        fused: _FusedPrefill,
        *,
        return_prompt_logits: bool,
    ) -> tuple[Tensor, PacketCacheState, Tensor | None]:
        """Run a prepared fused forward and extract first-token state."""
        input_ids = torch.tensor(
            fused.input_ids,
            dtype=torch.long,
            device=self.backend.device,
        ).unsqueeze(0)
        output = self.backend.forward(
            input_ids=input_ids,
            position_ids=fused.position_ids,
            attention_mask=fused.attention_mask,
            cache=fused.cache,
        )
        prompt_logits = None
        if return_prompt_logits:
            if fused.prompt_len <= 0:
                raise ValueError("prompt logits require a trailing generation prompt")
            prompt_logits = output.logits[:, -fused.prompt_len :, :].detach().clone()
        return (
            output.logits[:, -1, :],
            PacketCacheState(output.cache, fused.final_position),
            prompt_logits,
        )

    def _assemble(
        self,
        cache_state: PacketCacheState,
        input_ids: list[int],
        cover: list[Match],
        *,
        trailing_prompt: bool,
        substitute: bool,
    ) -> tuple[KVCache | None, int, list[int], int]:
        """Extend a caller-owned prefix through forwarding or packet substitution."""
        if trailing_prompt:
            prompt_start = max((item.end for item in cover), default=0)
            if prompt_start >= len(input_ids):
                raise ValueError(self._missing_prompt_message())
            prefix_end = prompt_start
            prompt_ids = input_ids[prompt_start:]
        else:
            prefix_end = len(input_ids)
            prompt_ids = []

        cache = cache_state.cache
        position = cache_state.next_position
        flops = 0
        if not substitute:
            if prefix_end:
                cache, position = self._forward_run(cache, input_ids[:prefix_end], position)
            return cache, position, prompt_ids, flops

        pending_caches: list[KVCache] = []

        def flush_pending() -> None:
            nonlocal cache
            if not pending_caches:
                return
            cache = self._concat_cache_group(([] if cache is None else [cache]) + pending_caches)
            pending_caches.clear()

        for segment in self._segments(cover, 0, prefix_end):
            if isinstance(segment, Match):
                stored = self.store.get(segment.chunk_id)
                shifted = self.backend.shift_cache(stored.packet.to(self.backend.device), position)
                pending_caches.append(shifted)
                flops += self.backend.shift_cache_flops(shifted)
                position += stored.wrapped_len
            else:
                _, start, end = segment
                flush_pending()
                cache, position = self._forward_run(cache, input_ids[start:end], position)
        flush_pending()
        return cache, position, prompt_ids, flops

    @staticmethod
    def _segments(
        cover: list[Match],
        start: int,
        end: int,
    ) -> list[Match | tuple[Literal["gap"], int, int]]:
        """Partition a token interval into packet matches and online gaps."""
        segments: list[Match | tuple[Literal["gap"], int, int]] = []
        cursor = start
        for match in cover:
            if match.start >= end:
                break
            if match.start > cursor:
                segments.append(("gap", cursor, match.start))
            segments.append(match)
            cursor = match.end
        if cursor < end:
            segments.append(("gap", cursor, end))
        return segments

    @staticmethod
    def _concat_cache_group(caches: list[KVCache]) -> KVCache | None:
        """Concatenate a cache group once while avoiding unnecessary copies."""
        if not caches:
            return None
        if len(caches) == 1:
            return caches[0]
        return concat_caches(caches)

    def _forward_run(
        self,
        cache: KVCache | None,
        input_ids: list[int],
        position: int,
    ) -> tuple[KVCache | None, int]:
        """Forward an unmatched run, optionally in bounded-size chunks."""
        if not input_ids:
            return cache, position
        chunk_size = self.prefill_chunk_size or len(input_ids)
        for start in range(0, len(input_ids), chunk_size):
            chunk = input_ids[start : start + chunk_size]
            tokens = torch.tensor(chunk, dtype=torch.long, device=self.backend.device).unsqueeze(0)
            positions = torch.arange(
                position,
                position + len(chunk),
                device=self.backend.device,
            ).unsqueeze(0)
            output = self.backend.forward(
                input_ids=tokens,
                position_ids=positions,
                cache=cache,
            )
            cache = output.cache
            position += len(chunk)
        return cache, position

    def _prefill(
        self,
        cache_state: PacketCacheState,
        prompt_ids: list[int],
        *,
        return_prompt_logits: bool,
    ) -> tuple[Tensor, PacketCacheState, Tensor | None]:
        """Forward a trailing prompt and expose logits for its final position."""
        if not prompt_ids:
            raise ValueError(self._missing_prompt_message())
        position = cache_state.next_position
        input_ids = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=self.backend.device,
        ).unsqueeze(0)
        position_ids = torch.arange(
            position,
            position + len(prompt_ids),
            device=self.backend.device,
        ).unsqueeze(0)
        output = self.backend.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache_state.cache,
        )
        prompt_logits = output.logits.detach().clone() if return_prompt_logits else None
        return (
            output.logits[:, -1, :],
            PacketCacheState(output.cache, position + len(prompt_ids)),
            prompt_logits,
        )

    @staticmethod
    def _missing_prompt_message() -> str:
        """Diagnostic for a prefill with no unmatched trailing prompt."""
        return (
            "prefill needs a trailing non-packet prompt to produce first-token logits; "
            "append a query/instruction after the packets (or use process())"
        )
