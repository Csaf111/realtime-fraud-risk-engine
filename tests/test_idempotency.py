"""Idempotent scoring: retried transaction_ids replay the original decision."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.idempotency import (
    IdempotencyCache,
    IdempotencyConflictError,
    LookupStatus,
    fingerprint,
)
from fraud_engine.schemas import Decision, RiskResult, Transaction
from fraud_engine.scoring import RiskEngine

from .conftest import make_txn


def _result(txn_id: str, score: float = 0.1) -> RiskResult:
    return RiskResult(
        transaction_id=txn_id,
        risk_score=score,
        decision=Decision.APPROVE,
        rules_score=score,
        model_score=None,
        reasons=[],
        rule_hits=[],
        top_features=[],
        model_version="test",
        latency_ms=0.1,
    )


def _metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    raise AssertionError(f"{name} not in metrics")


# --- fingerprint -------------------------------------------------------------------------


def test_fingerprint_is_key_order_independent():
    assert fingerprint({"a": 1, "b": 2.5}) == fingerprint({"b": 2.5, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_fingerprint_rejects_non_finite():
    with pytest.raises(ValueError):
        fingerprint({"a": float("nan")})


# --- cache unit behaviour ----------------------------------------------------------------


def test_claim_then_replay_returns_stored_result():
    cache = IdempotencyCache(10)
    assert cache.claim("t1", "fp").status is LookupStatus.CLAIMED
    cache.complete("t1", _result("t1"))
    lookup = cache.claim("t1", "fp")
    assert lookup.status is LookupStatus.REPLAY
    assert lookup.result == _result("t1")
    assert cache.replays == 1
    assert len(cache) == 1


def test_conflicting_payload_raises():
    cache = IdempotencyCache(10)
    cache.claim("t1", "fp-a")
    cache.complete("t1", _result("t1"))
    with pytest.raises(IdempotencyConflictError) as exc:
        cache.claim("t1", "fp-b")
    assert exc.value.transaction_id == "t1"
    assert cache.conflicts == 1


def test_release_lets_a_retry_claim_again():
    cache = IdempotencyCache(10)
    cache.claim("t1", "fp")
    cache.release("t1")
    assert cache.claim("t1", "fp").status is LookupStatus.CLAIMED
    assert len(cache) == 0  # in-flight claims are not counted as stored results


def test_release_after_complete_is_a_noop():
    cache = IdempotencyCache(10)
    cache.claim("t1", "fp")
    cache.complete("t1", _result("t1"))
    cache.release("t1")
    cache.release("unknown")
    assert cache.claim("t1", "fp").status is LookupStatus.REPLAY


def test_capacity_evicts_oldest_completed():
    cache = IdempotencyCache(2)
    for i in range(3):
        cache.claim(f"t{i}", "fp")
        cache.complete(f"t{i}", _result(f"t{i}"))
    assert len(cache) == 2
    assert cache.evicted == 1
    assert cache.claim("t0", "fp").status is LookupStatus.CLAIMED  # evicted -> scored again
    assert cache.claim("t2", "fp").status is LookupStatus.REPLAY


def test_eviction_never_drops_in_flight_claims():
    cache = IdempotencyCache(1)
    cache.claim("inflight", "fp")
    cache.claim("t1", "fp")
    cache.complete("t1", _result("t1"))
    # t1 is the only completed entry, so it is the one evicted; the claim survives.
    assert cache.evicted == 1
    cache.complete("inflight", _result("inflight"))
    assert cache.claim("inflight", "fp").status is LookupStatus.REPLAY


def test_concurrent_duplicate_waits_for_claimant():
    cache = IdempotencyCache(10)
    cache.claim("t1", "fp")
    seen: list[LookupStatus] = []

    def waiter() -> None:
        seen.append(cache.claim("t1", "fp").status)

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)
    assert not seen  # blocked while the claimant is scoring
    cache.complete("t1", _result("t1"))
    th.join(timeout=2)
    assert seen == [LookupStatus.REPLAY]


def test_waiter_takes_over_after_claimant_fails():
    cache = IdempotencyCache(10)
    cache.claim("t1", "fp")
    seen: list[LookupStatus] = []
    th = threading.Thread(target=lambda: seen.append(cache.claim("t1", "fp").status))
    th.start()
    time.sleep(0.05)
    cache.release("t1")
    th.join(timeout=2)
    assert seen == [LookupStatus.CLAIMED]


def test_waiter_times_out():
    cache = IdempotencyCache(10, wait_timeout_s=0.01)
    cache.claim("t1", "fp")
    with pytest.raises(TimeoutError):
        cache.claim("t1", "fp")


@pytest.mark.parametrize("kwargs", [{"capacity": 0}, {"capacity": 1, "wait_timeout_s": 0}])
def test_invalid_cache_config(kwargs):
    with pytest.raises(ValueError):
        IdempotencyCache(**kwargs)


def test_settings_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        Settings(idempotency_capacity=-1)
    monkeypatch.setenv("FRE_IDEMPOTENCY_CAPACITY", "0")
    assert Settings.from_env().idempotency_capacity == 0


# --- API ---------------------------------------------------------------------------------


@pytest.fixture
def idem_client() -> TestClient:
    app = create_app(Settings(max_batch_size=10))
    with TestClient(app) as c:
        yield c


def _burst_txn(i: int, **kw):
    # Same account, rising amounts: every recorded event raises the velocity features.
    return make_txn(**{"transaction_id": f"b{i}", "amount": 100.0 + i, **kw})


def test_retry_replays_without_recording_velocity(idem_client):
    first = idem_client.post("/v1/score", json=_burst_txn(1))
    assert first.status_code == 200
    assert "x-idempotent-replay" not in first.headers
    for _ in range(5):
        retry = idem_client.post("/v1/score", json=_burst_txn(1))
        assert retry.status_code == 200
        assert retry.headers["x-idempotent-replay"] == "true"
        assert retry.json() == first.json()

    # Velocity saw exactly one event: a fresh transaction from the same account scores as if
    # only b1 had happened, identical to an engine that never saw the retries.
    nxt = idem_client.post("/v1/score", json=_burst_txn(2)).json()
    ref = RiskEngine(Settings())
    ref.score(_burst_txn(1))
    expected = ref.score(_burst_txn(2))
    assert nxt["risk_score"] == expected.risk_score
    assert nxt["rule_hits"] == [h.model_dump() for h in expected.rule_hits]

    metrics = idem_client.get("/metrics").text
    assert _metric(metrics, "fre_idempotent_replays_total") == 5
    assert _metric(metrics, "fre_scored_total") == 2  # replays are not re-observed
    assert _metric(metrics, "fre_idempotency_keys") == 2


def test_velocity_not_inflated_by_retries_vs_disabled():
    """Without the cache, retries inflate velocity; with it they don't."""

    def run(capacity: int) -> float:
        app = create_app(Settings(idempotency_capacity=capacity))
        with TestClient(app) as c:
            for _ in range(10):
                c.post("/v1/score", json=_burst_txn(1))
            return c.post("/v1/score", json=_burst_txn(2)).json()["risk_score"]

    with_cache, without_cache = run(100), run(0)
    ref = RiskEngine(Settings())
    ref.score(_burst_txn(1))
    assert with_cache == ref.score(_burst_txn(2)).risk_score
    assert without_cache >= with_cache


def test_same_id_different_payload_is_409(idem_client):
    assert idem_client.post("/v1/score", json=_burst_txn(1)).status_code == 200
    r = idem_client.post("/v1/score", json=_burst_txn(1, nameDest="M999"))
    assert r.status_code == 409
    assert "different payload" in r.json()["detail"]
    assert _metric(idem_client.get("/metrics").text, "fre_idempotency_conflicts_total") == 1


def test_requests_without_id_are_always_scored(idem_client):
    txn = make_txn()
    for _ in range(3):
        assert "x-idempotent-replay" not in idem_client.post("/v1/score", json=txn).headers
    assert _metric(idem_client.get("/metrics").text, "fre_scored_total") == 3


def test_disabled_cache_has_no_metrics():
    app = create_app(Settings(idempotency_capacity=0))
    with TestClient(app) as c:
        c.post("/v1/score", json=_burst_txn(1))
        r = c.post("/v1/score", json=_burst_txn(1))
        assert "x-idempotent-replay" not in r.headers
        assert "fre_idempotent_replays_total" not in c.get("/metrics").text


def test_batch_replays_ids_seen_before_and_within_batch(idem_client):
    single = idem_client.post("/v1/score", json=_burst_txn(1)).json()
    body = idem_client.post(
        "/v1/score/batch",
        json={"transactions": [_burst_txn(1), _burst_txn(2), _burst_txn(2)]},
    ).json()
    assert body["count"] == 3
    assert body["results"][0] == single
    assert body["results"][1] == body["results"][2]
    metrics = idem_client.get("/metrics").text
    assert _metric(metrics, "fre_idempotent_replays_total") == 2
    assert _metric(metrics, "fre_scored_total") == 2


def test_batch_with_conflicting_ids_is_rejected_before_scoring(idem_client):
    r = idem_client.post(
        "/v1/score/batch",
        json={"transactions": [_burst_txn(1), _burst_txn(1, amount=5.0)]},
    )
    assert r.status_code == 409
    # Nothing in the rejected batch was scored or cached.
    assert _metric(idem_client.get("/metrics").text, "fre_scored_total") == 0
    assert "x-idempotent-replay" not in idem_client.post("/v1/score", json=_burst_txn(1)).headers


def test_what_if_batch_bypasses_cache(idem_client):
    idem_client.post("/v1/score", json=_burst_txn(1))
    # Different payload under the same id is fine for what-if scoring: no state, no cache.
    body = idem_client.post(
        "/v1/score/batch",
        json={"transactions": [_burst_txn(1, amount=9.0)], "update_state": False},
    )
    assert body.status_code == 200
    assert _metric(idem_client.get("/metrics").text, "fre_idempotent_replays_total") == 0


class _BoomEngine(RiskEngine):
    def __init__(self) -> None:
        super().__init__(Settings())
        self.calls = 0

    def score_detailed(self, txn, update_state=True):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return super().score_detailed(txn, update_state=update_state)


def test_failed_scoring_releases_claim():
    eng = _BoomEngine()
    app = create_app(Settings(), engine=eng)
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.post("/v1/score", json=_burst_txn(1)).status_code == 500
        r = c.post("/v1/score", json=_burst_txn(1))
        assert r.status_code == 200
        assert "x-idempotent-replay" not in r.headers  # scored for real on the retry


def test_concurrent_duplicate_in_flight_returns_503():
    app = create_app(Settings())
    with TestClient(app) as c:
        cache: IdempotencyCache = app.state.idempotency
        cache.wait_timeout_s = 0.01
        cache.claim("b1", fingerprint(Transaction(**_burst_txn(1)).model_dump()))
        r = c.post("/v1/score", json=_burst_txn(1))
        assert r.status_code == 503
        assert r.headers["retry-after"] == "1"
