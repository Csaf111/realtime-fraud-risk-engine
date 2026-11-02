import pytest

from fraud_engine.state import VelocityStore


def test_snapshot_excludes_current_event():
    s = VelocityStore(window_steps=24)
    assert s.snapshot("A", "B", 1).orig_txn_count == 0
    s.record("A", "B", 1, 100.0)
    snap = s.snapshot("A", "B", 1)
    assert snap.orig_txn_count == 1
    assert snap.dest_inbound_sum == 100.0


def test_window_expires_old_events():
    s = VelocityStore(window_steps=5)
    s.record("A", "B", 1, 10.0)
    assert s.snapshot("A", "B", 5).orig_txn_count == 1
    assert s.snapshot("A", "B", 6).orig_txn_count == 0


def test_future_events_not_visible_to_earlier_query():
    s = VelocityStore(window_steps=24)
    s.record("A", "B", 10, 10.0)
    assert s.snapshot("A", "B", 9).orig_txn_count == 0


def test_dest_and_orig_tracked_separately():
    s = VelocityStore()
    s.record("A", "MULE", 1, 5.0)
    s.record("X", "MULE", 1, 5.0)
    snap = s.snapshot("Z", "MULE", 2)
    assert snap.orig_txn_count == 0
    assert snap.dest_inbound_count == 2


def test_reset_and_len():
    s = VelocityStore()
    s.record("A", "B", 1, 1.0)
    assert len(s) == 2
    s.reset()
    assert len(s) == 0


def test_invalid_window():
    with pytest.raises(ValueError):
        VelocityStore(window_steps=0)


def test_max_events_cap():
    s = VelocityStore(window_steps=1000, max_events_per_key=3)
    for i in range(10):
        s.record("A", "B", 1, float(i))
    assert s.snapshot("A", "B", 1).orig_txn_count == 3
