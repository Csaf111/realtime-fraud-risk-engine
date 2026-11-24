"""Retraining a candidate model from the delayed-label feedback log (``train --feedback``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_engine.artifacts import load_model
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.feedback import FeedbackLog, FeedbackRecord
from fraud_engine.preprocessing import LABEL, build_feature_frame, load_paysim
from fraud_engine.scoring import RiskEngine
from fraud_engine.synthetic import write_csv
from fraud_engine.train import SERVED_SCORE, feedback_frame, main, train_from_feedback
from fraud_engine.trees import GradientBoostedTrees

from .conftest import fraud_txn, make_txn


def _write_log(tmp_path: Path, n: int = 3000, fraud_rate: float = 0.04) -> Path:
    """A feedback log as the engine would write it: served features + delayed labels.

    Offline features equal online ones (parity tests), so the PaySim feature frame stands in
    for served vectors; the rules score stands in for the live model's served score.
    """
    csv = write_csv(tmp_path / "s.csv", n=n, fraud_rate=fraud_rate)
    frame = build_feature_frame(load_paysim(csv))
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    for i, row in enumerate(frame.to_dict(orient="records")):
        assert log.append(
            FeedbackRecord(
                transaction_id=f"t-{i}",
                step=int(row["step"]),
                model_version="live-1",
                risk_score=float(row["rules_score"]),
                decision="APPROVE",
                is_fraud=bool(row[LABEL]),
                features={name: float(row[name]) for name in FEATURE_NAMES},
            )
        )
    return log.path


def _rec(tid: str, step: int, is_fraud: bool, score: float = 0.5) -> FeedbackRecord:
    return FeedbackRecord(
        transaction_id=tid,
        step=step,
        model_version="live-1",
        risk_score=score,
        decision="REVIEW",
        is_fraud=is_fraud,
        features=dict.fromkeys(FEATURE_NAMES, 0.0),
    )


def test_feedback_frame_is_deduped_time_ordered_and_drops_ids(tmp_path: Path):
    log = FeedbackLog(tmp_path / "f.jsonl")
    log.append(_rec("b", 5, False))
    log.append(_rec("a", 2, False))
    log.append(_rec("b", 5, True, score=0.9))  # later label for the same id wins
    frame = feedback_frame(log.path)
    assert frame["step"].tolist() == [2, 5]
    assert frame[LABEL].tolist() == [0, 1]
    assert frame[SERVED_SCORE].tolist() == [0.5, 0.9]
    assert list(frame.columns) == [*FEATURE_NAMES, "step", LABEL, SERVED_SCORE]


def test_empty_feedback_log_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no records"):
        feedback_frame(path)


def test_rejects_bad_arguments(tmp_path: Path):
    with pytest.raises(ValueError, match="model_type"):
        train_from_feedback(tmp_path / "x.jsonl", tmp_path / "m.json", model_type="svm")
    with pytest.raises(ValueError, match="validation_frac"):
        train_from_feedback(tmp_path / "x.jsonl", tmp_path / "m.json", validation_frac=1.0)


@pytest.mark.slow
def test_retrain_from_feedback_end_to_end(tmp_path: Path):
    path = _write_log(tmp_path)
    out = tmp_path / "candidate.json"
    m = train_from_feedback(path, out, gap_steps=5, version="fb")

    assert m["source"] == "feedback" and m["score_reference"] is None
    assert m["feedback_records"] > 0
    assert set(m["served_baseline"]) == set(m["model"])
    report = m["leakage_report"]
    assert report["train_steps"][1] + 5 < report["test_steps"][0]  # embargo gap held
    assert out.with_suffix(".metrics.json").exists()
    assert out.with_suffix(".feature_reference.json").exists()
    assert not out.with_suffix(".score_reference.json").exists()

    model = load_model(out)
    # The served score is a baseline, never an input; nor are ids or step.
    assert list(model.feature_names) == list(FEATURE_NAMES)
    # Deployed as meant: a shadow via FRE_SHADOW_MODEL_PATH next to the live (rules) path.
    eng = RiskEngine(Settings(shadow_model_path=str(out)))
    assert eng.shadow_model_version is not None and "fb" in eng.shadow_model_version
    bad = eng.score_detailed(fraud_txn(), update_state=False).shadow
    good = eng.score_detailed(make_txn(), update_state=False).shadow
    assert bad is not None and good is not None
    assert bad.risk_score is not None and good.risk_score is not None
    assert bad.risk_score > good.risk_score


@pytest.mark.slow
def test_retrain_gbt_from_feedback(tmp_path: Path):
    path = _write_log(tmp_path)
    out = tmp_path / "gbt.json"
    m = train_from_feedback(path, out, gap_steps=5, model_type="gbt", n_estimators=10)
    assert m["model_type"] == "gbt"
    assert isinstance(load_model(out), GradientBoostedTrees)


def test_single_class_feedback_fails_leakage_checks(tmp_path: Path):
    log = FeedbackLog(tmp_path / "f.jsonl")
    for i in range(50):
        log.append(_rec(f"t{i}", i, False))
    with pytest.raises(AssertionError, match="single class"):
        train_from_feedback(log.path, tmp_path / "m.json", gap_steps=2)


@pytest.mark.slow
def test_cli_feedback(tmp_path: Path, capsys):
    path = _write_log(tmp_path)
    out = tmp_path / "cand.json"
    main(["--feedback", str(path), "--out", str(out), "--gap-steps", "5", "--version", "c"])
    printed = capsys.readouterr().out
    assert json.loads(out.read_text())["version"] == "c"
    assert "served_baseline" in printed
    assert f"FRE_SHADOW_MODEL_PATH={out}" in printed


@pytest.mark.parametrize(
    "extra",
    [["--review-budget", "0.01", "--block-budget", "0.002"], ["--rules-floor", "0.5"]],
)
def test_cli_feedback_rejects_options_needing_raw_transactions(tmp_path: Path, extra):
    with pytest.raises(SystemExit):
        main(["--feedback", str(tmp_path / "f.jsonl"), *extra])
