import random

import fakeredis
import pytest

from fraud_engine.config import Settings
from fraud_engine.scoring import RiskEngine
from fraud_engine.state import VelocityBackend, VelocityStore, build_velocity_store
from fraud_engine.state_redis import RedisVelocityStore

from .conftest import make_txn


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def store(server):
    return RedisVelocityStore(fakeredis.FakeRedis(server=server), window_steps=24)


def test_implements_backend_protocol(store):
    assert isinstance(store, VelocityBackend)
    assert isinstance(VelocityStore(), VelocityBackend)


def test_snapshot_excludes_current_event(store):
    assert store.snapshot("A", "B", 1).orig_txn_count == 0
    store.record("A", "B", 1, 100.0)
    snap = store.snapshot("A", "B", 1)
    assert snap.orig_txn_count == 1
    assert snap.dest_inbound_sum == 100.0


def test_window_expires_old_events(server):
    s = RedisVelocityStore(fakeredis.FakeRedis(server=server), window_steps=5)
    s.record("A", "B", 1, 10.0)
    assert s.snapshot("A", "B", 5).orig_txn_count == 1
    assert s.snapshot("A", "B", 6).orig_txn_count == 0


def test_future_events_not_visible_to_earlier_query(store):
    store.record("A", "B", 10, 10.0)
    assert store.snapshot("A", "B", 9).orig_txn_count == 0


def test_dest_and_orig_tracked_separately(store):
    store.record("A", "MULE", 1, 5.0)
    store.record("X", "MULE", 1, 5.0)
    snap = store.snapshot("Z", "MULE", 2)
    assert snap.orig_txn_count == 0
    assert snap.dest_inbound_count == 2
    assert snap.dest_inbound_sum == 10.0


def test_identical_amounts_same_step_are_distinct_events(store):
    for _ in range(3):
        store.record("A", "B", 7, 50.0)
    assert store.snapshot("A", "B", 7).orig_txn_count == 3


def test_max_events_cap_drops_oldest(server):
    s = RedisVelocityStore(
        fakeredis.FakeRedis(server=server), window_steps=1000, max_events_per_key=3
    )
    for i in range(10):
        s.record("A", "B", 1, float(i))
    snap = s.snapshot("A", "B", 1)
    assert snap.orig_txn_count == 3
    assert snap.orig_amount_sum == 7.0 + 8.0 + 9.0


def test_prunes_expired_events_on_write(server):
    client = fakeredis.FakeRedis(server=server)
    s = RedisVelocityStore(client, window_steps=2)
    s.record("A", "B", 1, 1.0)
    s.record("A", "B", 10, 1.0)
    assert client.zcard("fre:velocity:orig:A") == 1


def test_ttl_applied_to_account_keys(server):
    client = fakeredis.FakeRedis(server=server)
    s = RedisVelocityStore(client, ttl_seconds=3600)
    s.record("A", "B", 1, 1.0)
    assert 0 < client.ttl("fre:velocity:orig:A") <= 3600
    assert 0 < client.ttl("fre:velocity:dest:B") <= 3600


def test_reset_and_len_only_touch_own_namespace(server):
    client = fakeredis.FakeRedis(server=server)
    client.set("other:key", "keep")
    s = RedisVelocityStore(client, key_prefix="fre:test")
    s.record("A", "B", 1, 1.0)
    assert len(s) == 2
    s.reset()
    assert len(s) == 0
    assert client.get("other:key") == b"keep"
    assert s.snapshot("A", "B", 1).orig_txn_count == 0


def test_works_with_decoded_responses(server):
    s = RedisVelocityStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    s.record("A", "B", 1, 12.5)
    assert s.snapshot("A", "B", 2).orig_amount_sum == 12.5
    assert len(s) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_steps": 0},
        {"max_events_per_key": 0},
        {"key_prefix": ""},
        {"ttl_seconds": 0},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        RedisVelocityStore(fakeredis.FakeRedis(), **kwargs)


def test_parity_with_in_memory_store(store):
    """Same event stream -> identical snapshots, so online/offline features match."""
    rng = random.Random(7)
    mem = VelocityStore(window_steps=24)
    accounts = [f"C{i}" for i in range(6)]
    step = 0
    for _ in range(400):
        step += rng.choice([0, 0, 1, 2, 5, 30])  # PaySim replays are step-sorted
        orig, dest = rng.choice(accounts), rng.choice(accounts)
        amount = round(rng.uniform(1, 50_000), 2)
        assert store.snapshot(orig, dest, step) == mem.snapshot(orig, dest, step)
        store.record(orig, dest, step, amount)
        mem.record(orig, dest, step, amount)


def test_replicas_share_state(server):
    """Two engines (replicas) on the same Redis see each other's history."""
    settings = Settings()
    a = RiskEngine(settings, store=RedisVelocityStore(fakeredis.FakeRedis(server=server)))
    b = RiskEngine(settings, store=RedisVelocityStore(fakeredis.FakeRedis(server=server)))
    for i in range(5):
        a.score(make_txn(nameOrig=f"C{i}", nameDest="MMULE", amount=1000.0, step=3))
    r = b.score(make_txn(nameOrig="C99", nameDest="MMULE", amount=1000.0, step=4))
    assert "R007" in {h.code for h in r.rule_hits}


def test_build_velocity_store_selects_backend():
    assert isinstance(build_velocity_store(Settings()), VelocityStore)
    s = build_velocity_store(
        Settings(
            velocity_backend="redis",
            redis_url="redis://localhost:6399/0",
            redis_key_prefix="fre:x",
            velocity_window_steps=12,
        )
    )
    assert isinstance(s, RedisVelocityStore)
    assert s.window_steps == 12
    assert s.key_prefix == "fre:x"


def test_engine_keeps_injected_empty_store():
    store = VelocityStore()
    assert RiskEngine(store=store).store is store


def test_settings_validate_backend(monkeypatch):
    with pytest.raises(ValueError):
        Settings(velocity_backend="memcached")
    with pytest.raises(ValueError):
        Settings(redis_key_prefix="")
    monkeypatch.setenv("FRE_VELOCITY_BACKEND", " Redis ")
    monkeypatch.setenv("FRE_REDIS_URL", "redis://cache:6379/2")
    s = Settings.from_env()
    assert s.velocity_backend == "redis"
    assert s.redis_url == "redis://cache:6379/2"
    assert s.redis_key_prefix == "fre:velocity"
