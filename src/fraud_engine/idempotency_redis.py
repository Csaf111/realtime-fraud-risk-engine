"""Redis-backed idempotency store, so a retry that lands on *another* replica also replays.

:class:`~fraud_engine.idempotency.IdempotencyCache` is per process. Behind a load balancer
a retried request often reaches a different replica than the original, which then scores
it again and, with the shared Redis velocity backend, records it in velocity a second time:
exactly the inflation idempotency exists to prevent. :class:`RedisIdempotencyStore`
implements the same :class:`~fraud_engine.idempotency.IdempotencyBackend` interface on top
of one Redis string per id::

    <prefix>:<transaction_id>   {"fp": <payload sha256>, "token": <claim token>,
                                 "result": null | <RiskResult JSON>}

* **Claim**: ``SET key <pending> NX PX lease``. Exactly one request across all replicas
  wins and scores the id. The claim is a *lease* (``lease_seconds``), so a replica that
  dies mid-request can't wedge an id forever: once the lease expires another request can
  claim it and score it.
* **Replay / conflict**: a losing request ``GET``s the entry. Same fingerprint with a
  result: the stored result is returned unchanged. Different fingerprint: rejected (409).
  Same fingerprint still pending: poll every ``poll_interval_s`` until the claimant
  completes or releases, or ``wait_timeout_s`` passes (503 + ``Retry-After``).
* **Complete / release** are compare-and-set on the claim token (``WATCH``/``MULTI``), so
  a claimant whose lease expired and was taken over never overwrites or deletes the new
  claimant's entry. That case is counted in ``fre_idempotency_lease_lost_total``.
* **Retention**: completed results expire after ``ttl_seconds`` (wall clock) instead of
  the in-memory store's capacity bound. An expired id is scored again if replayed, which
  is the pre-existing behaviour, never worse.

Leakage: unchanged. The id is only a deduplication key; no feature, rule or model input
reads it, and a replayed result was computed from the point-in-time state of the original
request.

``redis`` is the same optional extra as the Redis velocity store and is imported only by
:meth:`RedisIdempotencyStore.from_url`, so the default serving image stays slim.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from .idempotency import IdempotencyConflictError, Lookup, LookupStatus
from .schemas import RiskResult

__all__ = ["RedisIdempotencyStore"]

_CAS_ATTEMPTS = 8


def _decode(raw: Any) -> dict[str, Any]:
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    entry = json.loads(text)
    if not isinstance(entry, dict) or not isinstance(entry.get("fp"), str):
        raise ValueError("malformed idempotency entry in Redis")
    return entry


class RedisIdempotencyStore:
    """Shared ``transaction_id -> (payload fingerprint, result)`` store in Redis.

    Args:
        client: a ``redis.Redis``-compatible client (``fakeredis`` works in tests).
        ttl_seconds: how long a completed result is kept for replay.
        key_prefix: namespace for every key written by this store.
        lease_seconds: how long a claim stays valid while its request is being scored.
        wait_timeout_s: how long a concurrent duplicate waits for the claimant.
        poll_interval_s: how often a waiting duplicate re-reads the entry.
    """

    def __init__(
        self,
        client: Any,
        ttl_seconds: int = 86_400,
        key_prefix: str = "fre:idempotency",
        lease_seconds: float = 30.0,
        wait_timeout_s: float = 5.0,
        poll_interval_s: float = 0.01,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        if wait_timeout_s <= 0:
            raise ValueError("wait_timeout_s must be > 0")
        if not 0 < poll_interval_s <= wait_timeout_s:
            raise ValueError("poll_interval_s must be > 0 and <= wait_timeout_s")
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.lease_seconds = lease_seconds
        self.wait_timeout_s = wait_timeout_s
        self.poll_interval_s = poll_interval_s
        # Claims held by *this* process: transaction_id -> (token, fingerprint).
        self._claims: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()
        self.replays = 0
        self.conflicts = 0
        self.lease_lost = 0

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisIdempotencyStore:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "FRE_IDEMPOTENCY_BACKEND=redis requires the 'redis' extra: "
                'pip install "realtime-fraud-risk-engine[redis]"'
            ) from exc
        return cls(redis.Redis.from_url(url), **kwargs)

    def _key(self, transaction_id: str) -> str:
        return f"{self.key_prefix}:{transaction_id}"

    def claim(self, transaction_id: str, payload_fingerprint: str) -> Lookup:
        """Claim an unseen id across all replicas, or return the stored result.

        Raises :class:`~fraud_engine.idempotency.IdempotencyConflictError` if the id was
        seen with another payload and ``TimeoutError`` if a concurrent claimant is still
        scoring it after ``wait_timeout_s``.
        """
        key = self._key(transaction_id)
        lease_ms = max(1, int(self.lease_seconds * 1000))
        deadline = time.monotonic() + self.wait_timeout_s
        while True:
            token = uuid.uuid4().hex
            pending = json.dumps({"fp": payload_fingerprint, "token": token, "result": None})
            if self.client.set(key, pending, nx=True, px=lease_ms):
                with self._lock:
                    self._claims[transaction_id] = (token, payload_fingerprint)
                return Lookup(LookupStatus.CLAIMED)
            raw = self.client.get(key)
            if raw is not None:
                entry = _decode(raw)
                if entry["fp"] != payload_fingerprint:
                    with self._lock:
                        self.conflicts += 1
                    raise IdempotencyConflictError(transaction_id)
                if entry.get("result") is not None:
                    with self._lock:
                        self.replays += 1
                    return Lookup(LookupStatus.REPLAY, RiskResult.model_validate(entry["result"]))
            # Pending elsewhere (or expired between SET and GET): wait, then retry the claim.
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"transaction_id {transaction_id!r} is still being scored by another request"
                )
            time.sleep(self.poll_interval_s)

    def _compare_and_set(self, transaction_id: str, result: RiskResult | None) -> None:
        """Write ``result`` (or delete, if ``None``) only while we still own the claim."""
        from redis.exceptions import WatchError

        with self._lock:
            claim = self._claims.pop(transaction_id, None)
        if claim is None:
            return
        token, fp = claim
        key = self._key(transaction_id)
        for _ in range(_CAS_ATTEMPTS):
            with self.client.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current = None if raw is None else _decode(raw)
                    ours = current is not None and current.get("token") == token
                    if result is None:
                        if not ours:
                            return  # expired or taken over: nothing of ours to release
                        pipe.multi()
                        pipe.delete(key)
                    else:
                        if current is not None and not ours:
                            with self._lock:
                                self.lease_lost += 1
                            return
                        done = {"fp": fp, "token": token, "result": result.model_dump(mode="json")}
                        pipe.multi()
                        pipe.set(key, json.dumps(done), ex=self.ttl_seconds)
                    pipe.execute()
                    return
                except WatchError:
                    continue
        # Persistent contention on a single id: leave the lease to expire rather than
        # overwrite an entry we can't prove is ours.
        with self._lock:
            self.lease_lost += 1

    def complete(self, transaction_id: str, result: RiskResult) -> None:
        """Store the result of an id this process claimed (kept for ``ttl_seconds``)."""
        self._compare_and_set(transaction_id, result)

    def release(self, transaction_id: str) -> None:
        """Drop this process's pending claim after a failure so a retry can score the id."""
        self._compare_and_set(transaction_id, None)

    def render(self) -> list[str]:
        """Prometheus lines for ``/metrics`` (counters are per replica).

        No key-count gauge: counting Redis keys needs a ``SCAN``, too expensive per scrape;
        retention is bounded by ``ttl_seconds`` instead.
        """
        with self._lock:
            replays, conflicts, lost = self.replays, self.conflicts, self.lease_lost
        return [
            "# HELP fre_idempotent_replays_total Retried transaction_ids answered from the "
            "idempotency cache (not re-scored, velocity not re-recorded).",
            "# TYPE fre_idempotent_replays_total counter",
            f"fre_idempotent_replays_total {replays}",
            "# HELP fre_idempotency_conflicts_total transaction_ids reused with a different "
            "payload (rejected with 409).",
            "# TYPE fre_idempotency_conflicts_total counter",
            f"fre_idempotency_conflicts_total {conflicts}",
            "# HELP fre_idempotency_lease_lost_total Claims whose lease expired and was taken "
            "over before the result could be stored (raise the lease if this grows).",
            "# TYPE fre_idempotency_lease_lost_total counter",
            f"fre_idempotency_lease_lost_total {lost}",
        ]
