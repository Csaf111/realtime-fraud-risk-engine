"""Readiness probe: ``/ready`` pings every shared backend and returns 503 when one is down."""

from __future__ import annotations

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.idempotency_redis import RedisIdempotencyStore
from fraud_engine.scoring import RiskEngine
from fraud_engine.state_redis import RedisVelocityStore

from .conftest import make_txn


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _client(server, monkeypatch, *, velocity: str = "redis", idempotency: str = "redis"):
    settings = Settings(idempotency_backend=idempotency, velocity_backend=velocity)
    store = RedisVelocityStore(fakeredis.FakeRedis(server=server)) if velocity == "redis" else None
    monkeypatch.setattr(
        RedisIdempotencyStore,
        "from_url",
        classmethod(lambda cls, url, **kw: cls(fakeredis.FakeRedis(server=server), **kw)),
    )
    return TestClient(create_app(settings, engine=RiskEngine(settings, store=store)))


def test_in_memory_backends_report_no_dependencies(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert "dependencies" not in body


def test_redis_backends_ready_when_reachable(server, monkeypatch):
    with _client(server, monkeypatch) as c:
        resp = c.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["dependencies"] == {"velocity_store": "ok", "idempotency_store": "ok"}


def test_redis_outage_makes_replica_not_ready(server, monkeypatch, caplog):
    with _client(server, monkeypatch) as c:
        server.connected = False
        resp = c.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["dependencies"] == {
            "velocity_store": "error: ConnectionError",
            "idempotency_store": "error: ConnectionError",
        }
        assert body["mode"] == "rules-only"  # the rest of the payload is still reported
        assert "Readiness check failed for velocity_store" in caplog.text

        # Recovery: once Redis is back the same replica is ready again, with no restart.
        server.connected = True
        assert c.get("/ready").status_code == 200
        assert c.post("/v1/score", json=make_txn(transaction_id="t1")).status_code == 200


def test_only_configured_redis_backends_are_checked(server, monkeypatch):
    with _client(server, monkeypatch, velocity="memory", idempotency="redis") as c:
        body = c.get("/ready").json()
    assert body["dependencies"] == {"idempotency_store": "ok"}

    with _client(server, monkeypatch, velocity="redis", idempotency="memory") as c:
        body = c.get("/ready").json()
    assert body["dependencies"] == {"velocity_store": "ok"}


def test_failure_detail_never_leaks_exception_text(monkeypatch):
    class Boom:
        def ping(self) -> None:
            raise RuntimeError("redis://:s3cret@internal-host:6379 refused")

    settings = Settings()
    engine = RiskEngine(settings)
    with TestClient(create_app(settings, engine=engine)) as c:
        c.app.state.idempotency = Boom()  # type: ignore[attr-defined]
        resp = c.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["dependencies"] == {"idempotency_store": "error: RuntimeError"}
    assert "s3cret" not in resp.text and "internal-host" not in resp.text


def test_store_ping_round_trips(server):
    RedisVelocityStore(fakeredis.FakeRedis(server=server)).ping()
    RedisIdempotencyStore(fakeredis.FakeRedis(server=server)).ping()
    server.connected = False
    with pytest.raises(redis.exceptions.ConnectionError):
        RedisVelocityStore(fakeredis.FakeRedis(server=server)).ping()
