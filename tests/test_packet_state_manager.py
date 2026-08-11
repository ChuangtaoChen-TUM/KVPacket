"""Optional managed cache lifetime tests."""

from __future__ import annotations

import pytest

from kvpacket.runtime import PacketCacheState, PacketExecutor
from kvpacket.serve import PacketStateManager
from kvpacket.serve.manager import ManagedPacketState
from tests.test_phase2 import _FakeBackend, _packet_store


def _set_position(manager: PacketStateManager, session_id: str, position: int) -> None:
    manager._transact(
        session_id,
        lambda _state: (
            None,
            ManagedPacketState(PacketCacheState(next_position=position), tuple(range(position))),
        ),
    )


def test_manager_keeps_conversation_states_separate() -> None:
    backend = _FakeBackend()
    executor = PacketExecutor(
        backend=backend,
        store=_packet_store(backend, {"doc": [10, 11]}),
    )
    manager = PacketStateManager(executor)
    assert not hasattr(manager, "prefill")
    assert not hasattr(manager, "decode_step")

    _set_position(manager, "first", 4)
    _set_position(manager, "second", 1)

    assert manager.snapshot("first").cache_state.next_position == 4
    assert manager.snapshot("second").cache_state.next_position == 1


def test_manager_applies_lru_capacity_and_idle_expiry() -> None:
    now = [0.0]
    backend = _FakeBackend()
    manager = PacketStateManager(
        PacketExecutor(backend=backend, store=_packet_store(backend, {})),
        max_sessions=2,
        idle_timeout_s=5.0,
        clock=lambda: now[0],
    )

    _set_position(manager, "a", 1)
    now[0] = 1.0
    _set_position(manager, "b", 1)
    manager.snapshot("a")
    _set_position(manager, "c", 1)
    assert manager.session_ids() == ["a", "c"]

    now[0] = 7.0
    assert manager.prune() == ["a", "c"]
    with pytest.raises(KeyError, match="unknown packet session"):
        manager.snapshot("a")


def test_manager_reset_and_delete_release_state() -> None:
    backend = _FakeBackend()
    manager = PacketStateManager(PacketExecutor(backend=backend, store=_packet_store(backend, {})))
    _set_position(manager, "request", 2)

    manager.reset("request")
    assert manager.snapshot("request").cache_state.cache is None
    assert manager.delete("request")
    assert not manager.delete("request")
