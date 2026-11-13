from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.metrics import Metrics
from fraud_engine.outcomes import LabelStatus, OutcomeTracker

from .conftest import fraud_txn, make_txn

# --- tracker unit tests -------------------------------------------------------------------


def test_label_joins_stored_decision():
    t = OutcomeTracker(pending_capacity=10, window=10)
    t.record_decision("a", 0.9, "BLOCK")
    assert t.pending == 1
    assert t.record_label("a", True) is LabelStatus.MATCHED
    assert t.pending == 0
    assert t.outcomes[("BLOCK", True)] == 1
    assert t.window_count == 1


def test_unknown_and_duplicate_labels_are_not_counted_as_outcomes():
    t = OutcomeTracker(pending_capacity=10, window=10)
    assert t.record_label("never-scored", True) is LabelStatus.UNKNOWN
    t.record_decision("a", 0.1, "APPROVE")
    assert t.record_label("a", False) is LabelStatus.MATCHED
    assert t.record_label("a", True) is LabelStatus.DUPLICATE
    assert sum(t.outcomes.values()) == 1
    assert t.label_status == {"unknown": 1, "matched": 1, "duplicate": 1}


def test_decisions_without_id_are_ignored():
    t = OutcomeTracker(pending_capacity=10, window=10)
    t.record_decision(None, 0.9, "BLOCK")
    t.record_decision("", 0.9, "BLOCK")
    assert t.pending == 0


def test_pending_buffer_is_bounded_and_evicts_oldest():
    t = OutcomeTracker(pending_capacity=3, window=10)
    for i in range(5):
        t.record_decision(f"t{i}", 0.2, "APPROVE")
    assert t.pending == 3
    assert t.evicted == 2
    assert t.record_label("t0", True) is LabelStatus.UNKNOWN
    assert t.record_label("t4", True) is LabelStatus.MATCHED


def test_rescored_id_keeps_latest_decision():
    t = OutcomeTracker(pending_capacity=10, window=10)
    t.record_decision("a", 0.1, "APPROVE")
    t.record_decision("a", 0.95, "BLOCK")
    assert t.pending == 1
    t.record_label("a", True)
    assert t.outcomes == {("BLOCK", True): 1}


def test_capacity_zero_disables_tracking():
    t = OutcomeTracker(pending_capacity=0, window=10)
    assert not t.enabled
    t.record_decision("a", 0.9, "BLOCK")
    assert t.pending == 0


@pytest.mark.parametrize("kwargs", [{"pending_capacity": -1}, {"window": 0}])
def test_invalid_tracker_arguments(kwargs):
    with pytest.raises(ValueError):
        OutcomeTracker(**kwargs)


def test_rolling_precision_and_recall():
    t = OutcomeTracker(pending_capacity=100, window=100)
    # (decision, label): 2 BLOCK frauds, 1 BLOCK legit, 1 REVIEW fraud, 1 REVIEW legit,
    # 1 APPROVE fraud (missed), 4 APPROVE legit.
    cases = [
        ("BLOCK", True),
        ("BLOCK", True),
        ("BLOCK", False),
        ("REVIEW", True),
        ("REVIEW", False),
        ("APPROVE", True),
        *[("APPROVE", False)] * 4,
    ]
    for i, (d, y) in enumerate(cases):
        t.record_decision(f"t{i}", 0.5, d)
        t.record_label(f"t{i}", y)
    rates = t.rates()
    assert rates["alert"].precision == pytest.approx(3 / 5)
    assert rates["alert"].recall == pytest.approx(3 / 4)
    assert rates["block"].precision == pytest.approx(2 / 3)
    assert rates["block"].recall == pytest.approx(2 / 4)


def test_window_rolls_but_cumulative_counters_do_not():
    t = OutcomeTracker(pending_capacity=10, window=2)
    for i, (d, y) in enumerate([("BLOCK", False), ("BLOCK", True), ("BLOCK", True)]):
        t.record_decision(f"t{i}", 0.9, d)
        t.record_label(f"t{i}", y)
    assert t.window_count == 2
    assert t.rates()["block"].precision == 1.0  # the early false positive rolled out
    assert sum(t.outcomes.values()) == 3


def test_rates_undefined_without_data():
    rates = OutcomeTracker().rates()
    assert rates["alert"].precision is None and rates["alert"].recall is None


def test_render_exports_fixed_labels_only():
    t = OutcomeTracker(pending_capacity=10, window=10)
    t.record_decision("secret-txn-id", 0.9, "BLOCK")
    t.record_label("secret-txn-id", True)
    t.record_decision("other", 0.1, "APPROVE")
    text = "\n".join(t.render())
    assert "secret-txn-id" not in text
    assert 'fre_labelled_outcomes_total{decision="BLOCK",label="fraud"} 1' in text
    assert 'fre_labels_received_total{status="matched"} 1' in text
    assert "fre_label_pending 1" in text
    assert 'fre_label_precision{level="block"} 1.000000' in text
    assert 'fre_label_recall{level="alert"} 1.000000' in text


def test_render_omits_undefined_rates():
    text = "\n".join(OutcomeTracker().render())
    assert "fre_label_precision" not in text
    assert "fre_label_recall" not in text
    assert "fre_label_window_count 0" in text


def test_metrics_without_tracker_has_no_label_series():
    assert "fre_label" not in Metrics().render()


@pytest.mark.parametrize("kwargs", [{"label_pending_capacity": -1}, {"label_window": 0}])
def test_settings_validate_label_config(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_read_label_env(monkeypatch):
    monkeypatch.setenv("FRE_LABEL_PENDING_CAPACITY", "7")
    monkeypatch.setenv("FRE_LABEL_WINDOW", "9")
    s = Settings.from_env()
    assert (s.label_pending_capacity, s.label_window) == (7, 9)


# --- API ----------------------------------------------------------------------------------


def test_labels_endpoint_joins_and_reports(client: TestClient):
    client.post("/v1/score", json=fraud_txn(transaction_id="f1"))
    client.post("/v1/score", json=make_txn(transaction_id="b1"))
    r = client.post(
        "/v1/labels",
        json={
            "labels": [
                {"transaction_id": "f1", "is_fraud": True},
                {"transaction_id": "b1", "is_fraud": False},
                {"transaction_id": "nope", "is_fraud": True},
            ]
        },
    )
    assert r.status_code == 200
    assert r.json() == {"matched": 2, "unknown": 1, "duplicate": 0}
    again = client.post("/v1/labels", json={"labels": [{"transaction_id": "f1", "is_fraud": True}]})
    assert again.json() == {"matched": 0, "unknown": 0, "duplicate": 1}

    text = client.get("/metrics").text
    assert 'fre_labelled_outcomes_total{decision="BLOCK",label="fraud"} 1' in text
    assert 'fre_labelled_outcomes_total{decision="APPROVE",label="legit"} 1' in text
    assert 'fre_label_recall{level="block"} 1.000000' in text
    assert 'fre_requests_total{endpoint="labels"} 2' in text


def test_labels_do_not_change_later_scores(client: TestClient):
    """Leakage guard: labels reach only the monitor, never features, rules or state."""
    ref_app = create_app(Settings(max_batch_size=5))
    with TestClient(ref_app) as ref:
        seq = [fraud_txn(transaction_id=f"t{i}", step=10 + i) for i in range(3)]
        for txn in seq:
            a = client.post("/v1/score", json=txn).json()
            b = ref.post("/v1/score", json=txn).json()
            client.post(
                "/v1/labels",
                json={"labels": [{"transaction_id": txn["transaction_id"], "is_fraud": True}]},
            )
            a.pop("latency_ms"), b.pop("latency_ms")
            assert a == b
        probe = fraud_txn(transaction_id="probe", step=20)
        a, b = client.post("/v1/score", json=probe).json(), ref.post("/v1/score", json=probe).json()
        a.pop("latency_ms"), b.pop("latency_ms")
        assert a == b


def test_what_if_batch_scores_are_not_tracked(client: TestClient):
    client.post(
        "/v1/score/batch",
        json={"transactions": [fraud_txn(transaction_id="w1")], "update_state": False},
    )
    r = client.post("/v1/labels", json={"labels": [{"transaction_id": "w1", "is_fraud": True}]})
    assert r.json()["unknown"] == 1
    client.post("/v1/score/batch", json={"transactions": [fraud_txn(transaction_id="w2")]})
    r = client.post("/v1/labels", json={"labels": [{"transaction_id": "w2", "is_fraud": True}]})
    assert r.json()["matched"] == 1


def test_labels_validation_and_limits(client: TestClient):
    assert client.post("/v1/labels", json={"labels": []}).status_code == 422
    bad = {"labels": [{"transaction_id": "a", "is_fraud": True, "amount": 5}]}
    assert client.post("/v1/labels", json=bad).status_code == 422
    assert (
        client.post(
            "/v1/labels", json={"labels": [{"transaction_id": "", "is_fraud": True}]}
        ).status_code
        == 422
    )
    too_many = {"labels": [{"transaction_id": f"t{i}", "is_fraud": False} for i in range(6)]}
    assert client.post("/v1/labels", json=too_many).status_code == 413


def test_labels_endpoint_disabled():
    app = create_app(Settings(label_pending_capacity=0))
    with TestClient(app) as c:
        c.post("/v1/score", json=fraud_txn(transaction_id="x"))
        r = c.post("/v1/labels", json={"labels": [{"transaction_id": "x", "is_fraud": True}]})
        assert r.status_code == 404
        assert "fre_label" not in c.get("/metrics").text


def test_labelled_id_memory_is_bounded():
    t = OutcomeTracker(pending_capacity=2, window=10)
    for i in range(3):
        t.record_decision(f"t{i}", 0.1, "APPROVE")
        t.record_label(f"t{i}", False)
    # t0 has aged out of the duplicate memory, so a retry is now "unknown", not re-counted.
    assert t.record_label("t0", False) is LabelStatus.UNKNOWN
    assert t.record_label("t2", False) is LabelStatus.DUPLICATE
    assert sum(t.outcomes.values()) == 3
