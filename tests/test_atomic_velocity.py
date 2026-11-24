"""Atomic snapshot-then-record: concurrent bursts on one account can't share a stale read."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest

from fraud_engine.config import Settings
from fraud_engine.scoring import RiskEngine
from fraud_engine.state import (
    AtomicVelocityBackend,
    VelocitySnapshot,
    VelocityStore,
    snapshot_and_record,
)
from fraud_engine.state_redis import RedisVelocityStore, VelocityContentionError

from .conftest import make_txn

N = 24


class SlowReadStore(VelocityStore):
    """Widens the read->write gap so a non-atomic implementation would interleave."""

    def _snapshot_locked(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        time.sleep(0.002)
        return super()._snapshot_locked(name_orig, name_dest, step)


class PlainBackend:
    """A custom backend with only the base protocol (no atomic method)."""

    def __init__(self) -> None:
        self.inner = VelocityStore()
        self.window_steps = self.inner.window_steps
        self.calls: list[str] = []

    def snapshot(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        self.calls.append("snapshot")
        return self.inner.snapshot(name_orig, name_dest, step)

    def record(self, name_orig: str, name_dest: str, step: int, amount: float) -> None:
        self.calls.append("record")
        self.inner.record(name_orig, name_dest, step, amount)

    def reset(self) -> None:
        self.inner.reset()


def _burst(fn, n: int = N) -> list[int]:
    barrier = threading.Barrier(n)

    def run(i: int) -> int:
        barrier.wait()
        return fn(i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return sorted(pool.map(run, range(n)))


def test_in_memory_matches_snapshot_then_record():
    a, b = VelocityStore(window_steps=5), VelocityStore(window_steps=5)
    events = [("A", "M", 1, 10.0), ("A", "M", 2, 5.0), ("B", "M", 3, 7.0), ("A", "N", 9, 1.0)]
    for orig, dest, step, amount in events:
        expected = a.snapshot(orig, dest, step)
        a.record(orig, dest, step, amount)
        assert b.snapshot_and_record(orig, dest, step, amount) == expected
    assert len(a) == len(b)
    assert a.snapshot("A", "M", 9) == b.snapshot("A", "M", 9)


def test_snapshot_excludes_the_event_itself():
    s = VelocityStore()
    first = s.snapshot_and_record("A", "B", 1, 100.0)
    assert first == VelocitySnapshot()
    assert s.snapshot("A", "B", 1).orig_txn_count == 1


def test_in_memory_burst_is_serialised():
    store = SlowReadStore()
    counts = _burst(lambda i: store.snapshot_and_record("ACC", f"D{i}", 7, 1.0).orig_txn_count)
    # Each event saw exactly the events committed before it: 0, 1, ..., N-1.
    assert counts == list(range(N))


def test_engine_burst_sees_increasing_velocity():
    engine = RiskEngine(Settings(), store=SlowReadStore())

    def score(i: int) -> int:
        txn = make_txn(nameOrig="C_BURST", nameDest=f"M{i}", transaction_id=f"t{i}")
        return int(engine.score_detailed(txn).features["orig_txn_count_window"])

    assert _burst(score) == list(range(N))


def test_what_if_scoring_never_records():
    store = VelocityStore()
    engine = RiskEngine(Settings(), store=store)
    engine.score(make_txn(), update_state=False)
    assert len(store) == 0
    engine.score(make_txn())
    assert store.snapshot("C100000001", "X", 10).orig_txn_count == 1


def test_plain_backend_falls_back_to_snapshot_then_record():
    backend = PlainBackend()
    assert not isinstance(backend, AtomicVelocityBackend)
    assert snapshot_and_record(backend, "A", "B", 1, 5.0) == VelocitySnapshot()
    assert backend.calls == ["snapshot", "record"]
    engine = RiskEngine(Settings(), store=backend)
    engine.score(make_txn())
    assert backend.calls[-2:] == ["snapshot", "record"]


def test_builtin_stores_are_atomic_backends():
    assert isinstance(VelocityStore(), AtomicVelocityBackend)
    redis_store = RedisVelocityStore(fakeredis.FakeRedis())
    assert isinstance(redis_store, AtomicVelocityBackend)


# --- Redis -----------------------------------------------------------------------------


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _redis(server, **kwargs) -> RedisVelocityStore:
    return RedisVelocityStore(fakeredis.FakeRedis(server=server), window_steps=24, **kwargs)


def test_redis_matches_in_memory(server):
    redis_store, mem = _redis(server), VelocityStore(window_steps=24)
    events = [("A", "M", 1, 10.0), ("A", "M", 2, 5.5), ("B", "M", 30, 7.0), ("A", "M", 30, 1.0)]
    for orig, dest, step, amount in events:
        assert redis_store.snapshot_and_record(orig, dest, step, amount) == mem.snapshot_and_record(
            orig, dest, step, amount
        )
    assert redis_store.contention_retries == 0


def test_redis_burst_across_replicas_is_serialised(server):
    # One store (and client) per "replica", all sharing the same Redis server.
    replicas = [_redis(server, max_atomic_retries=10_000) for _ in range(N)]
    counts = _burst(
        lambda i: replicas[i].snapshot_and_record("ACC", f"D{i}", 7, 1.0).orig_txn_count
    )
    assert counts == list(range(N))
    assert replicas[0].snapshot("ACC", "X", 7).orig_txn_count == N


def _interfere(store: RedisVelocityStore, server, times: int) -> list[int]:
    """Make another client write the watched key between WATCH and EXEC ``times`` times."""
    other = _redis(server)
    hits = [0]
    real_pipeline = store.client.pipeline

    def pipeline(*args, **kwargs):
        pipe = real_pipeline(*args, **kwargs)
        real_multi = pipe.multi

        def multi():
            if hits[0] < times:
                hits[0] += 1
                other.record("A", "ELSEWHERE", 1, 100.0)
            return real_multi()

        pipe.multi = multi
        return pipe

    store.client.pipeline = pipeline  # type: ignore[method-assign]
    return hits


def test_redis_retries_on_concurrent_write_and_sees_it(server):
    store = _redis(server)
    hits = _interfere(store, server, times=2)
    snap = store.snapshot_and_record("A", "B", 1, 5.0)
    assert hits[0] == 2
    assert store.contention_retries == 2
    # The committed snapshot includes both writes that raced it, never a stale read.
    assert snap.orig_txn_count == 2
    assert snap.orig_amount_sum == 200.0
    assert store.snapshot("A", "B", 1).orig_txn_count == 3


def test_redis_gives_up_after_max_retries(server):
    store = _redis(server, max_atomic_retries=3)
    _interfere(store, server, times=100)
    with pytest.raises(VelocityContentionError):
        store.snapshot_and_record("A", "B", 1, 5.0)
    assert store.contention_retries == 3
    # Nothing was recorded on a stale read.
    assert store.snapshot("A", "B", 1).orig_txn_count == 3
    assert store.snapshot("A", "B", 1).dest_inbound_count == 0


def test_redis_rejects_bad_retry_budget(server):
    with pytest.raises(ValueError, match="max_atomic_retries"):
        _redis(server, max_atomic_retries=0)
