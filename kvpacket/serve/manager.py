"""Optionally retain caller-owned packet cache states across serving requests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock, RLock
from time import monotonic
from typing import TypeVar

from ..runtime import PacketCacheState, PacketExecutor

__all__ = ["PacketStateManager", "SessionStateSnapshot"]

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class ManagedPacketState:
    """Cache state plus the logical external tokens represented by that cache.

    Packet wrappers add hidden soft tokens, so ``len(token_ids)`` need not equal the
    cache's logical or physical length. The history is retained only to verify whether
    a later self-contained prompt extends the cached conversation exactly.
    """

    cache_state: PacketCacheState = PacketCacheState()
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SessionStateSnapshot:
    """Read-only view of one manager-owned conversation state."""

    session_id: str
    cache_state: PacketCacheState
    token_ids: tuple[int, ...]
    last_accessed_at: float


@dataclass
class _Entry:
    """Mutable state and synchronization for one conversation."""

    state: ManagedPacketState = field(default_factory=ManagedPacketState)
    last_accessed_at: float = 0.0
    users: int = 0
    lock: Lock = field(default_factory=Lock)


class PacketStateManager:
    """Bounded optional owner of per-conversation packet cache state.

    Existing serving systems can bypass this class and store ``PacketCacheState``
    themselves. The supplied manager is useful for HTTP adapters or small deployments:
    it provides session IDs, idle expiry, least-recently-used eviction, and per-session
    serialization while delegating all model work to a shared ``PacketExecutor``.

    Args:
        executor: Shared reentrant packet executor.
        max_sessions: Maximum retained conversations.
        idle_timeout_s: Optional idle lifetime before an unused state is evicted.
        serialize_execution: Whether model calls across different sessions share one
            lock. This is the safe default for a single backend/model instance.
        clock: Monotonic clock override used by deterministic tests.
    """

    def __init__(
        self,
        executor: PacketExecutor,
        *,
        max_sessions: int = 128,
        idle_timeout_s: float | None = None,
        serialize_execution: bool = True,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        if idle_timeout_s is not None and idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be positive or None")
        self.executor = executor
        self.max_sessions = max_sessions
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._entries_lock = RLock()
        self._execution_lock = Lock() if serialize_execution else None

    def _transact(
        self,
        session_id: str,
        operation: Callable[[ManagedPacketState], tuple[_ResultT, ManagedPacketState]],
    ) -> _ResultT:
        """Run one engine-owned state transition under the conversation lock.

        This private boundary prevents callers from splitting prefill and decoding
        into separately locked operations. The completion engine is the sole owner of
        multi-step state transitions.
        """
        entry = self._checkout(session_id, create=True)
        try:
            with entry.lock:
                result, state = self._execute(lambda: operation(entry.state))
                entry.state = state
                return result
        finally:
            self._release(session_id, entry)

    def snapshot(self, session_id: str) -> SessionStateSnapshot:
        """Inspect one retained state without transferring ownership."""
        entry = self._checkout(session_id, create=False)
        try:
            with entry.lock:
                return SessionStateSnapshot(
                    session_id,
                    entry.state.cache_state,
                    entry.state.token_ids,
                    entry.last_accessed_at,
                )
        finally:
            self._release(session_id, entry)

    def reset(self, session_id: str) -> None:
        """Replace one retained conversation with an empty cache state."""
        entry = self._checkout(session_id, create=False)
        try:
            with entry.lock:
                entry.state = ManagedPacketState()
        finally:
            self._release(session_id, entry)

    def delete(self, session_id: str) -> bool:
        """Remove an idle conversation and release its cache references."""
        self._validate_session_id(session_id)
        with self._entries_lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return False
            if entry.users:
                raise RuntimeError(f"session {session_id!r} is currently in use")
            del self._entries[session_id]
            return True

    def prune(self) -> list[str]:
        """Evict idle expired conversations and return their IDs."""
        with self._entries_lock:
            return self._prune_locked(self._clock())

    def session_ids(self) -> list[str]:
        """Retained session IDs from least to most recently used."""
        with self._entries_lock:
            return list(self._entries)

    def _checkout(self, session_id: str, *, create: bool) -> _Entry:
        """Pin one entry so expiry or capacity eviction cannot remove it."""
        self._validate_session_id(session_id)
        now = self._clock()
        with self._entries_lock:
            self._prune_locked(now)
            entry = self._entries.get(session_id)
            if entry is None:
                if not create:
                    raise KeyError(f"unknown packet session {session_id!r}")
                self._make_capacity_locked()
                entry = _Entry(last_accessed_at=now)
                self._entries[session_id] = entry
            entry.users += 1
            entry.last_accessed_at = now
            self._entries.move_to_end(session_id)
            return entry

    def _release(self, session_id: str, entry: _Entry) -> None:
        """Unpin and touch an entry after a manager operation."""
        with self._entries_lock:
            entry.users -= 1
            entry.last_accessed_at = self._clock()
            if self._entries.get(session_id) is entry:
                self._entries.move_to_end(session_id)

    def _make_capacity_locked(self) -> None:
        """Evict the least-recently-used unpinned entry when full."""
        if len(self._entries) < self.max_sessions:
            return
        for session_id, entry in self._entries.items():
            if entry.users == 0:
                del self._entries[session_id]
                return
        raise RuntimeError("all managed packet sessions are currently in use")

    def _prune_locked(self, now: float) -> list[str]:
        """Evict expired unpinned entries while holding the manager lock."""
        if self.idle_timeout_s is None:
            return []
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if entry.users == 0 and now - entry.last_accessed_at >= self.idle_timeout_s
        ]
        for session_id in expired:
            del self._entries[session_id]
        return expired

    def _execute(self, operation):
        """Run one model operation under optional backend-wide serialization."""
        if self._execution_lock is None:
            return operation()
        with self._execution_lock:
            return operation()

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject empty or non-string external session keys."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
