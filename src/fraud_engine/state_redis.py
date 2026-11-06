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

Known limitation: ``snapshot`` and ``record`` are separate round trips. Two replicas scoring
events for the same account concurrently may each miss the other's in-flight event, which
is the same ordering ambiguity a single replica has between concurrent HTTP requests.
Route each account to one partition (e.g. Kafka keyed by ``nameOrig``) if you need a total
order.

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
    """

    def __init__(
        self,
        client: Any,
        window_steps: int = 24,
        max_events_per_key: int = 10_000,
        key_prefix: str = "fre:velocity",
        ttl_seconds: int | None = None,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if max_events_per_key < 1:
            raise ValueError("max_events_per_key must be >= 1")
        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        if ttl_seconds is not None and ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1 when set")
        self.client = client
        self.window_steps = window_steps
        self.max_events_per_key = max_events_per_key
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

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
        member = f"{seq:0{_SEQ_WIDTH}d}|{float(amount)!r}"
        lower = step - self.window_steps
        pipe = self.client.pipeline(transaction=True)
        for key in (self._orig_key(name_orig), self._dest_key(name_dest)):
            pipe.zadd(key, {member: step})
            pipe.zremrangebyscore(key, "-inf", lower)  # same as in-memory: drop <= lower
            pipe.zremrangebyrank(key, 0, -(self.max_events_per_key + 1))
            if self.ttl_seconds is not None:
                pipe.expire(key, self.ttl_seconds)
        pipe.execute()

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
