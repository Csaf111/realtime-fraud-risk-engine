"""Idempotent scoring: a retried ``transaction_id`` returns the original decision.

Clients retry. A gateway times out, a mobile app resends, a load balancer replays a request
whose response was lost. Without protection every retry is scored *again*, and scoring is
not side-effect free: it records the event in the velocity store. A transaction retried
three times then looks like four transactions to its own account, the ``orig_txn_count`` /
``orig_amount_sum`` velocity features inflate, rule R004 can fire on a burst that never
happened, and the retry can get a *different* (harsher) decision than the original. The
online state also drifts away from the offline replay, which scores each PaySim row
exactly once, so the online/offline feature parity the model was trained under breaks.

:class:`IdempotencyCache` fixes that for requests that carry a ``transaction_id``:

* **First sight** of an id: the caller scores the transaction (recording velocity once) and
  stores the result with a fingerprint of the full request payload.
* **Replay** (same id, byte-identical payload after canonicalisation): the stored
  :class:`~fraud_engine.schemas.RiskResult` is returned unchanged. Velocity state, metrics,
  drift windows and the label tracker are not touched again.
* **Conflict** (same id, *different* payload): rejected. Silently returning the old result
  would hide a client bug; silently re-scoring would let one id carry two decisions.
* **Concurrent duplicates**: the first request claims the id; others wait for it to finish
  and then get its result, so two simultaneous retries can't both record velocity. If the
  claimant fails, the claim is released and a waiter scores instead.

The cache is bounded (oldest ids evicted first, ``FRE_IDEMPOTENCY_CAPACITY``). An evicted id
is scored again if it is replayed, which is the pre-existing behaviour, never worse.
Requests without a ``transaction_id`` and what-if scoring (``update_state=false``) bypass it.

Leakage: the cache keys on ``transaction_id`` only to *deduplicate requests*. No feature,
rule or model input ever reads the id, and the cached result was computed from the
point-in-time state of the first request, so nothing from the future can leak in.

State is per replica and in memory by default. With several replicas, either route retries
with the same id to the same replica (for example consistent hashing on ``transaction_id``),
or set ``FRE_IDEMPOTENCY_BACKEND=redis`` to share the claims and results across replicas
(:class:`~fraud_engine.idempotency_redis.RedisIdempotencyStore`, same
:class:`IdempotencyBackend` interface).

Pure Python and stdlib only, so it adds nothing to the serving image.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .schemas import RiskResult

if TYPE_CHECKING:
    from .config import Settings

__all__ = [
    "IdempotencyBackend",
    "IdempotencyCache",
    "IdempotencyConflictError",
    "Lookup",
    "LookupStatus",
    "build_idempotency_store",
    "fingerprint",
]


class IdempotencyConflictError(Exception):
    """A ``transaction_id`` was reused with a different payload."""

    def __init__(self, transaction_id: str) -> None:
        super().__init__(
            f"transaction_id {transaction_id!r} was already scored with a different payload"
        )
        self.transaction_id = transaction_id


class LookupStatus(StrEnum):
    CLAIMED = "claimed"  # caller must score, then complete() or release()
    REPLAY = "replay"  # result holds the original decision


@dataclass(frozen=True)
class Lookup:
    status: LookupStatus
    result: RiskResult | None = None


def fingerprint(payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 of a request payload (key order and float repr independent)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class IdempotencyBackend(Protocol):
    """What the API needs from an idempotency store (in-memory or shared)."""

    def claim(self, transaction_id: str, payload_fingerprint: str) -> Lookup: ...

    def complete(self, transaction_id: str, result: RiskResult) -> None: ...

    def release(self, transaction_id: str) -> None: ...

    def render(self) -> list[str]: ...


@dataclass
class _Entry:
    fingerprint: str
    result: RiskResult | None = None  # None while the claimant is still scoring
    done: threading.Event | None = None


class IdempotencyCache:
    """Bounded, thread-safe map of ``transaction_id -> (payload fingerprint, result)``.

    Usage::

        lookup = cache.claim(txn_id, fp)          # may raise IdempotencyConflictError
        if lookup.status is LookupStatus.REPLAY:
            return lookup.result
        try:
            result = score(...)
        except BaseException:
            cache.release(txn_id)
            raise
        cache.complete(txn_id, result)
    """

    def __init__(self, capacity: int, wait_timeout_s: float = 5.0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be > 0")
        self.capacity = capacity
        self.wait_timeout_s = wait_timeout_s
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.replays = 0
        self.conflicts = 0
        self.evicted = 0

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if e.result is not None)

    def claim(self, transaction_id: str, payload_fingerprint: str) -> Lookup:
        """Claim an unseen id, or return the stored result for a replay.

        Blocks (up to ``wait_timeout_s``) while another request is scoring the same id.
        Raises :class:`IdempotencyConflictError` if the id was seen with another payload.
        """
        while True:
            with self._lock:
                entry = self._entries.get(transaction_id)
                if entry is None:
                    self._entries[transaction_id] = _Entry(
                        payload_fingerprint, done=threading.Event()
                    )
                    return Lookup(LookupStatus.CLAIMED)
                if entry.fingerprint != payload_fingerprint:
                    self.conflicts += 1
                    raise IdempotencyConflictError(transaction_id)
                if entry.result is not None:
                    self.replays += 1
                    self._entries.move_to_end(transaction_id)
                    return Lookup(LookupStatus.REPLAY, entry.result)
                done = entry.done
            # In flight elsewhere: wait outside the lock, then re-check (the claimant may
            # have completed, or released the claim on failure so we can take it over).
            assert done is not None
            if not done.wait(self.wait_timeout_s):
                raise TimeoutError(
                    f"transaction_id {transaction_id!r} is still being scored by another request"
                )

    def complete(self, transaction_id: str, result: RiskResult) -> None:
        """Store the result of a claimed id and wake any waiting duplicates."""
        with self._lock:
            entry = self._entries.get(transaction_id)
            if entry is None:  # pragma: no cover - only if release() raced a complete()
                return
            entry.result = result
            self._entries.move_to_end(transaction_id)
            done, entry.done = entry.done, None
            self._evict()
        if done is not None:
            done.set()

    def release(self, transaction_id: str) -> None:
        """Drop an in-flight claim after a failure so a retry can score the id."""
        with self._lock:
            entry = self._entries.get(transaction_id)
            if entry is None or entry.result is not None:
                return
            del self._entries[transaction_id]
            done = entry.done
        if done is not None:
            done.set()

    def _evict(self) -> None:
        # Evict oldest *completed* entries only; in-flight claims are never dropped.
        if len(self._entries) <= self.capacity:
            return
        for key in list(self._entries):
            if len(self._entries) <= self.capacity:
                break
            if self._entries[key].result is not None:
                del self._entries[key]
                self.evicted += 1

    def render(self) -> list[str]:
        """Prometheus lines for ``/metrics``."""
        size = len(self)
        with self._lock:
            replays, conflicts, evicted = self.replays, self.conflicts, self.evicted
        return [
            "# HELP fre_idempotent_replays_total Retried transaction_ids answered from the "
            "idempotency cache (not re-scored, velocity not re-recorded).",
            "# TYPE fre_idempotent_replays_total counter",
            f"fre_idempotent_replays_total {replays}",
            "# HELP fre_idempotency_conflicts_total transaction_ids reused with a different "
            "payload (rejected with 409).",
            "# TYPE fre_idempotency_conflicts_total counter",
            f"fre_idempotency_conflicts_total {conflicts}",
            "# HELP fre_idempotency_keys transaction_ids currently held for replay.",
            "# TYPE fre_idempotency_keys gauge",
            f"fre_idempotency_keys {size}",
            "# HELP fre_idempotency_evicted_total Oldest transaction_ids evicted at capacity.",
            "# TYPE fre_idempotency_evicted_total counter",
            f"fre_idempotency_evicted_total {evicted}",
        ]


def build_idempotency_store(settings: Settings) -> IdempotencyBackend | None:
    """Create the idempotency store configured by ``settings`` (``None`` when disabled).

    ``memory`` (default) is a bounded per-process cache; ``redis`` shares claims and
    results across replicas and needs the optional ``redis`` extra.
    """
    if settings.idempotency_capacity == 0:
        return None
    if settings.idempotency_backend == "redis":
        from .idempotency_redis import RedisIdempotencyStore

        return RedisIdempotencyStore.from_url(
            settings.redis_url,
            ttl_seconds=settings.idempotency_ttl_seconds,
            key_prefix=settings.idempotency_key_prefix,
        )
    return IdempotencyCache(settings.idempotency_capacity)
