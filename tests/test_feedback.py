from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.feedback import (
    FeedbackLog,
    FeedbackRecord,
    feedback_dataset,
    read_feedback,
)
from fraud_engine.outcomes import LabelStatus, OutcomeTracker
from fraud_engine.scoring import RiskEngine
from fraud_engine.state import VelocitySnapshot

from .conftest import fraud_txn, make_txn


def _features(**overrides: float) -> dict[str, float]:
    feats = compute_features(make_txn(), VelocitySnapshot())
    feats.update(overrides)
    return feats


def _record(tid: str = "t1", step: int = 3, is_fraud: bool = True) -> FeedbackRecord:
    return FeedbackRecord(
        transaction_id=tid,
        step=step,
        model_version="rules-v0+test",
        risk_score=0.9,
        decision="BLOCK",
        is_fraud=is_fraud,
        features=_features(),
    )


# --- record + log -----------------------------------------------------------------------


def test_record_round_trips_with_features_in_contract_order():
    rec = _record()
    raw = json.loads(rec.to_json())
    assert tuple(raw["features"]) == FEATURE_NAMES
    assert raw["schema"] == 1
    assert FeedbackRecord.from_json(rec.to_json()) == rec


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda r: r.update(schema=99), "schema"),
        (lambda r: r["features"].pop("log_amount"), "FEATURE_NAMES"),
        (lambda r: r["features"].update(extra=1.0), "FEATURE_NAMES"),
        (lambda r: r["features"].update(log_amount="nan"), "finite"),
        (lambda r: r.update(is_fraud=1), "boolean"),
    ],
)
def test_from_json_rejects_contract_violations(mutate, msg):
    raw = json.loads(_record().to_json())
    mutate(raw)
    with pytest.raises(ValueError, match=msg):
        FeedbackRecord.from_json(json.dumps(raw))


def test_from_json_rejects_non_object():
    with pytest.raises(ValueError, match="object"):
        FeedbackRecord.from_json("[1, 2]")


def test_log_appends_and_reads_back(tmp_path: Path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    assert log.append(_record("a")) and log.append(_record("b", is_fraud=False))
    got = list(read_feedback(log.path))
    assert [r.transaction_id for r in got] == ["a", "b"]
    assert log.written == 2 and log.errors == 0
    assert "fre_feedback_records_total 2" in log.render()


def test_log_write_errors_are_counted_not_raised(tmp_path: Path):
    log = FeedbackLog(tmp_path / "missing-dir" / "fb.jsonl")
    assert log.append(_record()) is False
    assert log.errors == 1 and log.written == 0
    assert "fre_feedback_write_errors_total 1" in log.render()


def test_read_feedback_reports_bad_line_number(tmp_path: Path):
    path = tmp_path / "fb.jsonl"
    path.write_text(_record().to_json() + "\n\n" + "{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"fb\.jsonl:3"):
        list(read_feedback(path))


def test_dataset_is_time_ordered_deduped_and_excludes_metadata():
    recs = [
        _record("late", step=9, is_fraud=False),
        _record("early", step=1, is_fraud=True),
        _record("late", step=9, is_fraud=True),  # re-delivered: last one wins
    ]
    X, y, steps = feedback_dataset(recs)
    assert steps == [1, 9]
    assert y == [1, 1]
    assert all(len(row) == len(FEATURE_NAMES) for row in X)
    # Leakage guard: ids, step, score and decision are never model inputs.
    for forbidden in ("transaction_id", "step", "risk_score", "decision", "is_fraud"):
        assert forbidden not in FEATURE_NAMES


# --- tracker integration ------------------------------------------------------------------


def test_tracker_writes_record_on_match(tmp_path: Path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    t = OutcomeTracker(pending_capacity=10, window=10, feedback=log)
    t.record_decision("a", 0.9, "BLOCK", features=_features(), step=4, model_version="m1")
    assert t.record_label("a", True) is LabelStatus.MATCHED
    assert t.record_label("a", True) is LabelStatus.DUPLICATE  # not written twice
    assert t.record_label("nope", True) is LabelStatus.UNKNOWN
    (rec,) = list(read_feedback(log.path))
    assert (rec.transaction_id, rec.step, rec.model_version, rec.is_fraud) == ("a", 4, "m1", True)
    assert "fre_feedback_records_total 1" in t.render()


def test_tracker_without_log_drops_features(tmp_path: Path):
    t = OutcomeTracker(pending_capacity=10, window=10)
    t.record_decision("a", 0.9, "BLOCK", features=_features(), step=4)
    assert t._pending["a"].features is None
    assert t.record_label("a", True) is LabelStatus.MATCHED
    assert not any(line.startswith("fre_feedback") for line in t.render())


def test_tracker_skips_log_when_decision_had_no_features(tmp_path: Path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    t = OutcomeTracker(pending_capacity=10, window=10, feedback=log)
    t.record_decision("a", 0.9, "BLOCK")
    assert t.record_label("a", True) is LabelStatus.MATCHED
    assert log.written == 0 and not log.path.exists()


# --- engine + API end to end --------------------------------------------------------------


def test_score_detailed_exposes_point_in_time_features():
    eng = RiskEngine()
    first = eng.score_detailed(make_txn(nameOrig="C1", step=2))
    second = eng.score_detailed(make_txn(nameOrig="C1", step=3))
    # Velocity is a snapshot BEFORE each event: the first sees no history.
    assert first.features["orig_txn_count_window"] == 0.0
    assert second.features["orig_txn_count_window"] == 1.0
    assert second.step == 3
    assert eng.score_with_shadow(make_txn(nameOrig="C2"))[0].decision is not None


def test_api_feedback_loop_matches_served_features(tmp_path: Path):
    path = tmp_path / "fb.jsonl"
    settings = Settings(feedback_log_path=str(path))
    txns = [
        make_txn(nameOrig="C7", step=5, transaction_id="p1"),
        fraud_txn(nameOrig="C7", step=6, transaction_id="p2"),
        make_txn(nameOrig="C8", step=6),  # no id: never joinable
    ]
    with TestClient(create_app(settings)) as c:
        c.post("/v1/score/batch", json={"transactions": txns}).raise_for_status()
        # What-if scores are not served decisions and must not overwrite p1's features.
        what_if = {"transactions": [make_txn(transaction_id="p1", step=99)], "update_state": False}
        c.post("/v1/score/batch", json=what_if).raise_for_status()
        resp = c.post(
            "/v1/labels",
            json={
                "labels": [
                    {"transaction_id": "p2", "is_fraud": True},
                    {"transaction_id": "p1", "is_fraud": False},
                ]
            },
        )
        assert resp.json()["matched"] == 2
        assert "fre_feedback_records_total 2" in c.get("/metrics").text

    by_id = {r.transaction_id: r for r in read_feedback(path)}
    # Parity: the logged vector is exactly what a fresh engine computes online.
    ref = RiskEngine()
    expected = [ref.score_detailed(t).features for t in txns[:2]]
    assert dict(by_id["p1"].features) == expected[0]
    assert dict(by_id["p2"].features) == expected[1]
    assert by_id["p1"].step == 5 and by_id["p2"].is_fraud is True
    _X, y, steps = feedback_dataset(read_feedback(path))
    assert steps == [5, 6] and y == [0, 1]


def test_settings_feedback_env_and_validation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FRE_FEEDBACK_LOG_PATH", str(tmp_path / "x.jsonl"))
    assert Settings.from_env().feedback_log_path == str(tmp_path / "x.jsonl")
    with pytest.raises(ValueError, match="feedback_log_path"):
        Settings(feedback_log_path="x.jsonl", label_pending_capacity=0)
