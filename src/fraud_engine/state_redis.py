"""Redis-backed point-in-time velocity state for multi-replica deployments.

Implements the same :class:`~fraud_engine.state.VelocityBackend` interface as the
in-memory :class:`~fraud_engine.state.VelocityStore`, so features are computed identically
online and offline. Each account has one sorted set per direction::

    <prefix>:orig:<nameOrig>   score = step, member = "<seq>|<amount>"
    <prefix>:dest:<nameDest>

``seq`` comes from a single ``INCR`` counter and is zero-padded, so members are unique and
events in the same step keep their insertion order (Redis orders equal scores by member).

Leakage guarantee (unchanged): callers ``snapshot()`` before ``record()``, and a snapshot
only reads events with ``step - window_steps < ev_step <= step``, so a transaction never
sees itself or any later event.

Concurrency: serving uses :meth:`RedisVelocityStore.snapshot_and_record`, an optimistic
``WATCH``/``MULTI``/``EXEC`` transaction over the event's two account keys. If another
replica writes either key between the read and the commit, the transaction aborts and is
retried with a fresh read, so concurrent events for the same account across replicas are
serialised: each sees every event committed before it. Retries are counted in
``contention_retries``; after ``max_atomic_retries`` aborts in a row it raises
:class:`VelocityContentionError` rather than recording on a stale read. (Plain ``snapshot``
and ``record`` remain separate round trips for what-if scoring and tests.) Concurrent events
in the *same* step still have no defined order between them; route each account to one
partition (e.g. Kafka keyed by ``nameOrig``) if you need a total order.

``redis`` is an optional dependency (``pip install "realtime-fraud-risk-engine[redis]"``),
imported only when :meth:`RedisVelocityStore.from_url` is used, so the default serving
image stays slim.
"""

from __future__ import annotations

from typing import Any

from .state import VelocitySnapshot

_SEQ_WIDTH = 20  # fits any 64-bit INCR value, keeps lexicographic == numeric order


def _escape_glob(text: str) -> str:
    return "".join(f"\\{c}" if c in "*?[]\\" else c for c in text)


class VelocityContentionError(RuntimeError):
    """The atomic snapshot-and-record kept losing races on the same account keys."""


def _parse(members: list[Any]) -> tuple[int, float]:
    total = 0.0
    for m in members:
        raw = m.decode() if isinstance(m, bytes) else str(m)
        total += float(raw.split("|", 1)[1])
    return len(members), total


class RedisVelocityStore:
    """Shared velocity store with the same semantics as ``VelocityStore``.

    Args:
        client: a ``redis.Redis``-compatible client (``fakeredis`` works in tests).
        window_steps: look-back window in PaySim steps.
        max_events_per_key: cap per account set; the oldest events are dropped first.
        key_prefix: namespace for all keys written by this store.
        ttl_seconds: optional idle expiry for account keys (wall-clock safety net so
            dormant accounts don't accumulate; window pruning still happens per event).
        max_atomic_retries: optimistic-transaction attempts per event in
            ``snapshot_and_record`` before giving up with ``VelocityContentionError``.
    """

    def __init__(
        self,
        client: Any,
        window_steps: int = 24,
        max_events_per_key: int = 10_000,
        key_prefix: str = "fre:velocity",
        ttl_seconds: int | None = None,
        max_atomic_retries: int = 16,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if max_events_per_key < 1:
            raise ValueError("max_events_per_key must be >= 1")
        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        if ttl_seconds is not None and ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1 when set")
        if max_atomic_retries < 1:
            raise ValueError("max_atomic_retries must be >= 1")
        self.client = client
        self.window_steps = window_steps
        self.max_events_per_key = max_events_per_key
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.max_atomic_retries = max_atomic_retries
        self.contention_retries = 0

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisVelocityStore:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "FRE_VELOCITY_BACKEND=redis requires the 'redis' extra: "
                'pip install "realtime-fraud-risk-engine[redis]"'
            ) from exc
        return cls(redis.Redis.from_url(url), **kwargs)

    def ping(self) -> None:
        """Round-trip to Redis; raises if it is unreachable (used by ``/ready``)."""
        self.client.ping()

    def _orig_key(self, name: str) -> str:
        return f"{self.key_prefix}:orig:{name}"

    def _dest_key(self, name: str) -> str:
        return f"{self.key_prefix}:dest:{name}"

    @property
    def _seq_key(self) -> str:
        return f"{self.key_prefix}:seq"

    def snapshot(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        """Return activity strictly before the current event, within the window."""
        lower = step - self.window_steps  # exclusive
        pipe = self.client.pipeline(transaction=False)
        pipe.zrangebyscore(self._orig_key(name_orig), f"({lower}", step)
        pipe.zrangebyscore(self._dest_key(name_dest), f"({lower}", step)
        orig_members, dest_members = pipe.execute()
        oc, osum = _parse(orig_members)
        dc, dsum = _parse(dest_members)
        return VelocitySnapshot(oc, osum, dc, dsum)

    def record(self, name_orig: str, name_dest: str, step: int, amount: float) -> None:
        seq = int(self.client.incr(self._seq_key))
        pipe = self.client.pipeline(transaction=True)
        self._queue_record(pipe, name_orig, name_dest, step, amount, seq)
        pipe.execute()

    def _queue_record(
        self, pipe: Any, name_orig: str, name_dest: str, step: int, amount: float, seq: int
    ) -> None:
        member = f"{seq:0{_SEQ_WIDTH}d}|{float(amount)!r}"
        lower = step - self.window_steps
        for key in (self._orig_key(name_orig), self._dest_key(name_dest)):
            pipe.zadd(key, {member: step})
            pipe.zremrangebyscore(key, "-inf", lower)  # same as in-memory: drop <= lower
            pipe.zremrangebyrank(key, 0, -(self.max_events_per_key + 1))
            if self.ttl_seconds is not None:
                pipe.expire(key, self.ttl_seconds)

    def snapshot_and_record(
        self, name_orig: str, name_dest: str, step: int, amount: float
    ) -> VelocitySnapshot:
        """Snapshot activity before this event, then record it, atomically across replicas.

        Optimistic concurrency: ``WATCH`` both account keys, read the window, then queue
        the write in ``MULTI``/``EXEC``. If another client changed either key meanwhile,
        ``EXEC`` aborts (``WatchError``) and the whole read-then-write is retried, so the
        returned snapshot is exactly the state the event was recorded on top of.
        """
        from redis.exceptions import WatchError

        okey, dkey = self._orig_key(name_orig), self._dest_key(name_dest)
        lower = step - self.window_steps  # exclusive
        for _ in range(self.max_atomic_retries):
            with self.client.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(okey, dkey)
                    # In WATCH mode commands run immediately and return their results.
                    orig_members = pipe.zrangebyscore(okey, f"({lower}", step)
                    dest_members = pipe.zrangebyscore(dkey, f"({lower}", step)
                    seq = int(pipe.incr(self._seq_key))  # unwatched key: never aborts us
                    pipe.multi()
                    self._queue_record(pipe, name_orig, name_dest, step, amount, seq)
                    pipe.execute()
                except WatchError:
                    self.contention_retries += 1
                    continue
            oc, osum = _parse(orig_members)
            dc, dsum = _parse(dest_members)
            return VelocitySnapshot(oc, osum, dc, dsum)
        raise VelocityContentionError(
            f"velocity update for {name_orig!r}/{name_dest!r} lost "
            f"{self.max_atomic_retries} optimistic-transaction races in a row"
        )

    def _keys(self) -> list[Any]:
        pattern = f"{_escape_glob(self.key_prefix)}:*"
        return list(self.client.scan_iter(match=pattern, count=500))

    def reset(self) -> None:
        """Delete every key under ``key_prefix`` (never touches other namespaces)."""
        keys = self._keys()
        for i in range(0, len(keys), 500):
            self.client.delete(*keys[i : i + 500])

    def __len__(self) -> int:
        """Number of tracked account keys (orig + dest), matching ``VelocityStore``."""
        seq_key = self._seq_key
        return sum(
            1 for k in self._keys() if (k.decode() if isinstance(k, bytes) else k) != seq_key
        )
