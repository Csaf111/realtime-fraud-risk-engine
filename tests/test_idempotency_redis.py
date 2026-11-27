"""Shared (Redis) idempotency: a retry that lands on another replica also replays."""

from __future__ import annotations

import json
import threading
import time

import fakeredis
import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.idempotency import (
    IdempotencyBackend,
    IdempotencyCache,
    IdempotencyConflictError,
    LookupStatus,
    build_idempotency_store,
    fingerprint,
)
from fraud_engine.idempotency_redis import RedisIdempotencyStore
from fraud_engine.scoring import RiskEngine
from fraud_engine.state_redis import RedisVelocityStore

from .conftest import make_txn
from .test_idempotency import _metric, _result


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _store(server, **kwargs) -> RedisIdempotencyStore:
    kwargs.setdefault("poll_interval_s", 0.005)
    return RedisIdempotencyStore(fakeredis.FakeRedis(server=server), **kwargs)


# --- store unit behaviour ----------------------------------------------------------------


def test_implements_backend_protocol(server):
    assert isinstance(_store(server), IdempotencyBackend)
    assert isinstance(IdempotencyCache(10), IdempotencyBackend)


def test_claim_complete_replay_across_replicas(server):
    a, b = _store(server), _store(server)
    assert a.claim("t1", "fp").status is LookupStatus.CLAIMED
    a.complete("t1", _result("t1", 0.42))
    lookup = b.claim("t1", "fp")  # another replica, same Redis
    assert lookup.status is LookupStatus.REPLAY
    assert lookup.result == _result("t1", 0.42)
    assert b.replays == 1 and a.replays == 0


def test_conflicting_payload_raises_on_any_replica(server):
    a, b = _store(server), _store(server)
    a.claim("t1", "fp-a")
    a.complete("t1", _result("t1"))
    with pytest.raises(IdempotencyConflictError):
        b.claim("t1", "fp-b")
    assert b.conflicts == 1


def test_conflict_detected_while_pending(server):
    a, b = _store(server), _store(server)
    a.claim("t1", "fp-a")
    with pytest.raises(IdempotencyConflictError):
        b.claim("t1", "fp-b")


def test_release_lets_another_replica_claim(server):
    a, b = _store(server), _store(server)
    a.claim("t1", "fp")
    a.release("t1")
    assert b.claim("t1", "fp").status is LookupStatus.CLAIMED


def test_release_and_complete_without_claim_are_noops(server):
    s = _store(server)
    s.release("never-claimed")
    s.complete("never-claimed", _result("never-claimed"))
    assert s.client.get(s._key("never-claimed")) is None


def test_release_after_complete_keeps_result(server):
    s = _store(server)
    s.claim("t1", "fp")
    s.complete("t1", _result("t1"))
    s.release("t1")
    assert s.claim("t1", "fp").status is LookupStatus.REPLAY


def test_concurrent_duplicate_on_other_replica_waits(server):
    a, b = _store(server), _store(server, wait_timeout_s=2.0)
    a.claim("t1", "fp")
    got: list = []
    waiter = threading.Thread(target=lambda: got.append(b.claim("t1", "fp")))
    waiter.start()
    time.sleep(0.05)
    assert not got  # still waiting on the claimant
    a.complete("t1", _result("t1", 0.3))
    waiter.join(2)
    assert got[0].status is LookupStatus.REPLAY
    assert got[0].result.risk_score == 0.3


def test_waiter_takes_over_after_claimant_releases(server):
    a, b = _store(server), _store(server, wait_timeout_s=2.0)
    a.claim("t1", "fp")
    got: list = []
    waiter = threading.Thread(target=lambda: got.append(b.claim("t1", "fp")))
    waiter.start()
    time.sleep(0.03)
    a.release("t1")
    waiter.join(2)
    assert got[0].status is LookupStatus.CLAIMED


def test_waiter_times_out(server):
    a, b = _store(server), _store(server, wait_timeout_s=0.05)
    a.claim("t1", "fp")
    with pytest.raises(TimeoutError):
        b.claim("t1", "fp")


def test_expired_lease_can_be_reclaimed_and_old_claimant_cannot_overwrite(server):
    a = _store(server, lease_seconds=0.05)
    b = _store(server)
    a.claim("t1", "fp")
    time.sleep(0.1)  # a's replica "died": the lease expires
    assert b.claim("t1", "fp").status is LookupStatus.CLAIMED
    a.complete("t1", _result("t1", 0.9))  # late, stale claimant
    assert a.lease_lost == 1
    a.release("t1")  # no-op: it no longer owns the claim
    b.complete("t1", _result("t1", 0.2))
    lookup = _store(server).claim("t1", "fp")
    assert lookup.status is LookupStatus.REPLAY
    assert lookup.result.risk_score == 0.2


def test_complete_after_lease_expired_unclaimed_still_stores(server):
    a = _store(server, lease_seconds=0.05)
    a.claim("t1", "fp")
    time.sleep(0.1)
    a.complete("t1", _result("t1"))
    assert a.lease_lost == 0
    assert _store(server).claim("t1", "fp").status is LookupStatus.REPLAY


def test_results_expire_after_ttl(server):
    s = _store(server, ttl_seconds=60)
    s.claim("t1", "fp")
    s.complete("t1", _result("t1"))
    ttl = s.client.ttl(s._key("t1"))
    assert 0 < ttl <= 60


def test_keys_are_namespaced(server):
    s = _store(server, key_prefix="ns:idem")
    s.claim("t1", "fp")
    assert s.client.exists("ns:idem:t1")
    entry = json.loads(s.client.get("ns:idem:t1"))
    assert entry["fp"] == "fp" and entry["result"] is None


def test_malformed_entry_is_an_error(server):
    s = _store(server)
    s.client.set(s._key("t1"), json.dumps({"no": "fp"}))
    with pytest.raises(ValueError, match="malformed"):
        s.claim("t1", "fp")


def test_render_metrics(server):
    s = _store(server)
    s.claim("t1", "fp")
    s.complete("t1", _result("t1"))
    s.claim("t1", "fp")
    text = "\n".join(s.render())
    assert _metric(text, "fre_idempotent_replays_total") == 1
    assert _metric(text, "fre_idempotency_conflicts_total") == 0
    assert _metric(text, "fre_idempotency_lease_lost_total") == 0
    assert "fre_idempotency_keys" not in text  # no per-scrape SCAN


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl_seconds": 0},
        {"key_prefix": ""},
        {"lease_seconds": 0},
        {"wait_timeout_s": 0},
        {"poll_interval_s": 0},
        {"poll_interval_s": 10.0, "wait_timeout_s": 1.0},
    ],
)
def test_invalid_store_config(server, kwargs):
    with pytest.raises(ValueError):
        RedisIdempotencyStore(fakeredis.FakeRedis(server=server), **kwargs)


# --- settings / builder ------------------------------------------------------------------


def test_settings_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        Settings(idempotency_backend="memcached")
    with pytest.raises(ValueError):
        Settings(idempotency_ttl_seconds=0)
    with pytest.raises(ValueError):
        Settings(idempotency_key_prefix="")
    with pytest.raises(ValueError):
        Settings(
            idempotency_backend="redis",
            velocity_backend="redis",
            idempotency_key_prefix="fre:shared",
            redis_key_prefix="fre:shared",
        )
    monkeypatch.setenv("FRE_IDEMPOTENCY_BACKEND", " Redis ")
    monkeypatch.setenv("FRE_IDEMPOTENCY_TTL_SECONDS", "3600")
    monkeypatch.setenv("FRE_IDEMPOTENCY_KEY_PREFIX", "x:idem")
    s = Settings.from_env()
    assert s.idempotency_backend == "redis"
    assert s.idempotency_ttl_seconds == 3600
    assert s.idempotency_key_prefix == "x:idem"


def test_builder_selects_backend(monkeypatch):
    assert build_idempotency_store(Settings(idempotency_capacity=0)) is None
    assert isinstance(build_idempotency_store(Settings()), IdempotencyCache)
    store = build_idempotency_store(
        Settings(idempotency_backend="redis", idempotency_ttl_seconds=120)
    )
    assert isinstance(store, RedisIdempotencyStore)  # from_url connects lazily
    assert store.ttl_seconds == 120 and store.key_prefix == "fre:idempotency"


# --- API: two replicas sharing Redis velocity + idempotency ------------------------------


def _replica(server, monkeypatch) -> TestClient:
    settings = Settings(idempotency_backend="redis", velocity_backend="redis")
    velocity = RedisVelocityStore(fakeredis.FakeRedis(server=server), key_prefix="fre:velocity")
    engine = RiskEngine(settings, store=velocity)
    monkeypatch.setattr(
        RedisIdempotencyStore,
        "from_url",
        classmethod(lambda cls, url, **kw: cls(fakeredis.FakeRedis(server=server), **kw)),
    )
    return TestClient(create_app(settings, engine=engine))


def test_retry_on_other_replica_replays_without_recording_velocity(server, monkeypatch):
    with _replica(server, monkeypatch) as a, _replica(server, monkeypatch) as b:
        txn = make_txn(transaction_id="tx-1", step=5)
        first = a.post("/v1/score", json=txn)
        assert first.status_code == 200
        retry = b.post("/v1/score", json=txn)  # the load balancer picked another replica
        assert retry.status_code == 200
        assert retry.headers["x-idempotent-replay"] == "true"
        assert retry.json() == first.json()

        conflict = b.post("/v1/score", json={**txn, "amount": txn["amount"] + 1})
        assert conflict.status_code == 409

        velocity = a.app.state.engine.store
        snap = velocity.snapshot(txn["nameOrig"], txn["nameDest"], txn["step"])
        assert snap.orig_txn_count == 1  # recorded once across both replicas
        assert _metric(b.get("/metrics").text, "fre_idempotent_replays_total") == 1


def test_fingerprint_matches_across_replicas():
    # Both replicas fingerprint the parsed Transaction, so field order on the wire can't
    # turn a retry into a conflict.
    txn = make_txn(transaction_id="tx-2")
    assert fingerprint(dict(reversed(list(txn.items())))) == fingerprint(txn)
