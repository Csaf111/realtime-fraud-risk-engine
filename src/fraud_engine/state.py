"""Point-in-time velocity state.

The same ``VelocityStore`` is used online (inside the API) and offline (when building
training features), so the features a model is trained on are computed exactly the way
they are served. That removes a whole class of train/serve skew.

Leakage guarantee: ``snapshot()`` is always called *before* ``record()`` for the current
event, so a transaction never sees itself or anything that happened after it.

Atomic read-then-write: serving calls :func:`snapshot_and_record`, which takes the
point-in-time snapshot and records the event as one atomic step (a single lock acquisition
in memory, an optimistic ``WATCH``/``MULTI`` transaction in Redis). With separate
``snapshot`` and ``record`` calls, N concurrent requests for the same account could all
snapshot before any of them recorded, so every one of them would see the same (stale)
velocity: a burst of parallel transactions, exactly the pattern velocity rules exist to
catch, could slip under them. Atomicity serialises the burst so the k-th event sees the k-1
before it. The snapshot is still taken strictly before the event is added, so the leakage
guarantee and online/offline parity are unchanged (offline feature building uses the same
call).

Idle-key eviction: the in-memory store drops an account's key once every event it holds has
left the window behind the newest recorded step (the "high-water" step). Without this, every
account ever seen would keep an (empty) entry forever, and a long-running replica scoring a
PaySim-sized stream (millions of distinct accounts) would leak memory without bound. Eviction
runs each time the high-water step advances and uses a per-step expiry index, so it costs
O(1) amortised per record (it never scans live keys),
and it is exact for any snapshot at or after the high-water step (the normal, time-ordered
case): an evicted event is at or before ``high_water - window_steps`` and so is outside every
such snapshot's window. Like the per-key pruning in ``record``, it assumes events arrive in
(approximately) non-decreasing step order; an event far older than the window is out of scope.

The in-memory implementation is per-process. For multi-replica deployments use
:class:`fraud_engine.state_redis.RedisVelocityStore` (Redis sorted sets keyed by account),
which implements the same :class:`VelocityBackend` interface and produces identical
snapshots for the same event stream. :func:`build_velocity_store` picks the backend from
:class:`~fraud_engine.config.Settings`.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import Settings


@dataclass(frozen=True)
class VelocitySnapshot:
    orig_txn_count: int = 0
    orig_amount_sum: float = 0.0
    dest_inbound_count: int = 0
    dest_inbound_sum: float = 0.0


@runtime_checkable
class VelocityBackend(Protocol):
    """Interface every velocity store implements (in-memory, Redis, ...).

    Contract: ``snapshot`` returns only events with ``step - window_steps < ev_step <= step``
    that were ``record``-ed earlier; callers must snapshot *before* recording the event.
    """

    window_steps: int

    def snapshot(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot: ...

    def record(self, name_orig: str, name_dest: str, step: int, amount: float) -> None: ...

    def reset(self) -> None: ...


class VelocityStore:
    def __init__(
        self,
        window_steps: int = 24,
        max_events_per_key: int = 10_000,
        evict_idle: bool = True,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if max_events_per_key < 1:
            raise ValueError("max_events_per_key must be >= 1")
        self.window_steps = window_steps
        self.max_events_per_key = max_events_per_key
        self.evict_idle = evict_idle
        self._orig: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._dest: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._high_water: int | None = None
        self.evicted_keys = 0
        # Expiry index: newest event step per (side, key), and keys bucketed by that step,
        # so eviction touches only the buckets that fell out of the window (O(1) amortised
        # per record) instead of scanning every key.
        self._newest: dict[tuple[int, str], int] = {}
        self._buckets: dict[int, set[tuple[int, str]]] = defaultdict(set)

    def _window(self, events: deque[tuple[int, float]], step: int) -> tuple[int, float]:
        lower = step - self.window_steps  # exclusive
        count, total = 0, 0.0
        for ev_step, amount in events:
            if lower < ev_step <= step:
                count += 1
                total += amount
        return count, total

    def _prune(self, events: deque[tuple[int, float]], step: int) -> None:
        lower = step - self.window_steps
        while events and events[0][0] <= lower:
            events.popleft()
        while len(events) > self.max_events_per_key:
            events.popleft()

    def snapshot(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        """Return activity strictly before the current event, within the window."""
        with self._lock:
            return self._snapshot_locked(name_orig, name_dest, step)

    def record(self, name_orig: str, name_dest: str, step: int, amount: float) -> None:
        with self._lock:
            self._record_locked(name_orig, name_dest, step, amount)

    def snapshot_and_record(
        self, name_orig: str, name_dest: str, step: int, amount: float
    ) -> VelocitySnapshot:
        """Snapshot activity before this event, then record it, under one lock.

        Equivalent to ``snapshot`` followed by ``record``, except no other thread can
        record an event for the same accounts in between.
        """
        with self._lock:
            snap = self._snapshot_locked(name_orig, name_dest, step)
            self._record_locked(name_orig, name_dest, step, amount)
        return snap

    def _snapshot_locked(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        oc, osum = self._window(self._orig.get(name_orig, deque()), step)
        dc, dsum = self._window(self._dest.get(name_dest, deque()), step)
        return VelocitySnapshot(oc, osum, dc, dsum)

    def _record_locked(self, name_orig: str, name_dest: str, step: int, amount: float) -> None:
        if self._high_water is None or step > self._high_water:
            self._high_water = step
            if self.evict_idle:
                self.evicted_keys += self._evict_idle_locked()
        o = self._orig[name_orig]
        o.append((step, amount))
        self._prune(o, step)
        d = self._dest[name_dest]
        d.append((step, amount))
        self._prune(d, step)
        self._touch((0, name_orig), step)
        self._touch((1, name_dest), step)

    def _touch(self, side_key: tuple[int, str], step: int) -> None:
        newest = self._newest.get(side_key)
        if newest is not None and newest >= step:
            return  # a late event never makes a key expire sooner
        if newest is not None:
            bucket = self._buckets[newest]
            bucket.discard(side_key)
            if not bucket:
                del self._buckets[newest]
        self._newest[side_key] = step
        self._buckets[step].add(side_key)

    def _evict_idle_locked(self) -> int:
        """Drop keys whose newest event is outside the window behind the high-water step."""
        if self._high_water is None:
            return 0
        lower = self._high_water - self.window_steps  # events at or before this are dead
        sides = (self._orig, self._dest)
        evicted = 0
        for bucket_step in [b for b in self._buckets if b <= lower]:
            for side, key in self._buckets.pop(bucket_step):
                del self._newest[(side, key)]
                sides[side].pop(key, None)
                evicted += 1
        return evicted

    def sweep(self) -> int:
        """Evict idle keys now (also runs automatically when the high-water step advances).

        Returns the number of keys removed. Snapshots at or after the high-water step are
        unchanged by eviction.
        """
        with self._lock:
            removed = self._evict_idle_locked()
            self.evicted_keys += removed
            return removed

    @property
    def high_water_step(self) -> int | None:
        """Newest step recorded so far (``None`` before the first ``record``)."""
        return self._high_water

    def reset(self) -> None:
        with self._lock:
            self._orig.clear()
            self._dest.clear()
            self._newest.clear()
            self._buckets.clear()
            self._high_water = None
            self.evicted_keys = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._orig) + len(self._dest)


@runtime_checkable
class AtomicVelocityBackend(VelocityBackend, Protocol):
    """A backend that can snapshot-then-record one event atomically."""

    def snapshot_and_record(
        self, name_orig: str, name_dest: str, step: int, amount: float
    ) -> VelocitySnapshot: ...


def snapshot_and_record(
    store: VelocityBackend, name_orig: str, name_dest: str, step: int, amount: float
) -> VelocitySnapshot:
    """Return the point-in-time snapshot for an event, then record the event.

    Uses the backend's atomic ``snapshot_and_record`` when it has one (both built-in stores
    do); a custom backend without it falls back to ``snapshot`` then ``record``, which is
    correct for a single writer but not atomic under concurrency.
    """
    if isinstance(store, AtomicVelocityBackend):
        return store.snapshot_and_record(name_orig, name_dest, step, amount)
    snap = store.snapshot(name_orig, name_dest, step)
    store.record(name_orig, name_dest, step, amount)
    return snap


def build_velocity_store(settings: Settings) -> VelocityBackend:
    """Create the velocity store configured by ``settings.velocity_backend``.

    ``memory`` (default) is per-process; ``redis`` shares state across replicas and needs
    the optional ``redis`` extra (``pip install "realtime-fraud-risk-engine[redis]"``).
    """
    if settings.velocity_backend == "redis":
        from .state_redis import RedisVelocityStore

        return RedisVelocityStore.from_url(
            settings.redis_url,
            window_steps=settings.velocity_window_steps,
            key_prefix=settings.redis_key_prefix,
        )
    return VelocityStore(settings.velocity_window_steps)
