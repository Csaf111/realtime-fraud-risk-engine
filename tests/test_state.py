import random

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


def test_invalid_max_events_per_key():
    with pytest.raises(ValueError):
        VelocityStore(max_events_per_key=0)


def test_idle_keys_are_evicted_once_outside_window():
    s = VelocityStore(window_steps=5)
    s.record("A", "B", 1, 10.0)
    s.record("C", "D", 5, 10.0)  # high water 5: A/B newest at 1 > 5 - 5, still live
    assert len(s) == 4
    assert s.evicted_keys == 0
    s.record("E", "F", 6, 10.0)  # high water 6: step 1 <= 6 - 5 -> A and B evicted
    assert len(s) == 4
    assert s.evicted_keys == 2
    assert s.snapshot("A", "B", 6) == s.snapshot("never", "seen", 6)


def test_eviction_only_runs_when_high_water_advances():
    s = VelocityStore(window_steps=1)
    s.record("A", "B", 1, 1.0)
    s.record("C", "D", 3, 1.0)
    assert s.evicted_keys == 2
    # Same step again: no new sweep, nothing else to evict anyway.
    s.record("E", "F", 3, 1.0)
    assert s.evicted_keys == 2
    assert s.high_water_step == 3


def test_active_keys_survive_eviction():
    s = VelocityStore(window_steps=3)
    for step in range(1, 50):
        s.record("ACTIVE", "MULE", step, 1.0)
        s.record(f"ONE_OFF_{step}", "MULE", step, 1.0)
    snap = s.snapshot("ACTIVE", "MULE", 50)
    assert snap.orig_txn_count == 2  # steps 48, 49 (window (47, 50])
    assert snap.dest_inbound_count == 4
    # Only accounts active in the last window remain: memory is bounded.
    assert len(s) <= 2 + 3


def test_late_event_keeps_key_alive():
    s = VelocityStore(window_steps=5)
    s.record("A", "B", 10, 1.0)
    s.record("A", "B", 8, 1.0)  # slightly out of order, still inside the window
    s.record("X", "Y", 14, 1.0)  # lower bound 9: A/B newest is 10 -> kept
    assert s.snapshot("A", "B", 14).orig_txn_count == 1
    s.record("X", "Y", 15, 1.0)  # lower bound 10: A/B now dead
    assert "A" not in s._orig


def test_manual_sweep_and_disabled_eviction():
    s = VelocityStore(window_steps=2, evict_idle=False)
    s.record("A", "B", 1, 1.0)
    s.record("C", "D", 10, 1.0)
    assert len(s) == 4
    assert s.evicted_keys == 0
    assert s.sweep() == 2
    assert s.evicted_keys == 2
    assert len(s) == 2
    assert VelocityStore().sweep() == 0  # nothing recorded yet


def test_reset_clears_high_water_and_eviction_count():
    s = VelocityStore(window_steps=1)
    s.record("A", "B", 1, 1.0)
    s.record("C", "D", 5, 1.0)
    assert s.evicted_keys == 2
    s.reset()
    assert s.high_water_step is None
    assert s.evicted_keys == 0
    s.record("A", "B", 1, 1.0)  # stepping "back in time" after reset is a fresh stream
    assert s.snapshot("A", "B", 1).orig_txn_count == 1


def test_eviction_does_not_change_snapshots_for_time_ordered_stream():
    """Online/offline parity: eviction is invisible to every point-in-time snapshot."""
    rng = random.Random(7)
    with_evict = VelocityStore(window_steps=4)
    without = VelocityStore(window_steps=4, evict_idle=False)
    step = 0
    for _ in range(3000):
        step += rng.choice([0, 0, 0, 1, 2])
        orig = f"C{rng.randrange(200)}"
        dest = f"D{rng.randrange(60)}"
        assert with_evict.snapshot(orig, dest, step) == without.snapshot(orig, dest, step)
        amount = round(rng.uniform(1, 1000), 2)
        with_evict.record(orig, dest, step, amount)
        without.record(orig, dest, step, amount)
    assert with_evict.evicted_keys > 0
    assert len(with_evict) < len(without)
