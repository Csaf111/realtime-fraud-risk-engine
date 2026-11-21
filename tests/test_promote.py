"""Offline promotion gate: candidate vs live on the unseen tail of the feedback log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_engine.artifacts import load_model
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.feedback import FeedbackLog, FeedbackRecord
from fraud_engine.model import LogisticBaseline
from fraud_engine.preprocessing import LeakageError
from fraud_engine.promote import GateCriteria, compare_on_feedback, main, training_horizon
from fraud_engine.train import train_from_feedback

from .test_train_feedback import _write_log

GAP = 5


def _constant_model(path: Path, intercept: float = -5.0, version: str = "const") -> Path:
    """A live model with no signal: same probability for every transaction."""
    n = len(FEATURE_NAMES)
    LogisticBaseline(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=[0.0] * n,
        intercept=intercept,
        version=version,
    ).save(path)
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("promote")
    log = _write_log(tmp)
    cand = tmp / "candidate.json"
    train_from_feedback(log, cand, gap_steps=GAP, version="cand")
    return {"log": log, "candidate": cand, "const": _constant_model(tmp / "const.json")}


def _rec(tid: str, step: int, is_fraud: bool) -> FeedbackRecord:
    return FeedbackRecord(
        transaction_id=tid,
        step=step,
        model_version="live-1",
        risk_score=0.5,
        decision="REVIEW",
        is_fraud=is_fraud,
        features=dict.fromkeys(FEATURE_NAMES, 0.0),
    )


def test_criteria_validation():
    for bad in (
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"max_pr_auc_drop": -0.1},
        {"max_recall_drop": 1.5},
        {"min_positives": 0},
    ):
        with pytest.raises(ValueError):
            GateCriteria(**bad)


def test_training_horizon(tmp_path: Path, trained):
    # PaySim-trained / untagged artifacts live on another clock: no horizon.
    assert training_horizon(load_model(trained["const"])) is None
    cand = load_model(trained["candidate"])
    split = cand.metadata["metrics"]["split"]
    assert training_horizon(cand) == split["cutoff_step"] + GAP
    # A feedback-trained artifact that lost its split cannot prove the window is unseen.
    cand.metadata = {"metrics": {"source": "feedback"}}
    with pytest.raises(ValueError, match="no split"):
        training_horizon(cand)


@pytest.mark.slow
def test_better_candidate_passes_and_default_window_is_its_unseen_test_window(trained):
    live, cand = load_model(trained["const"]), load_model(trained["candidate"])
    r = compare_on_feedback(live, cand, trained["log"])
    assert r["passed"] and r["reasons"] == []
    assert r["delta"]["pr_auc"] > 0
    split = cand.metadata["metrics"]["split"]
    # Default window = strictly after the candidate's training window + embargo, i.e.
    # exactly the test rows it was evaluated on at training time and never fitted on.
    assert r["window"]["after_step"] == split["cutoff_step"] + GAP
    assert r["window"]["steps"][0] > split["train_max_step"] + GAP
    assert r["window"]["rows"] == split["test_rows"]
    assert r["candidate"]["pr_auc"] == pytest.approx(cand.metadata["metrics"]["model"]["pr_auc"])
    assert r["newly_missed_fraud"] == 0  # the constant live model flags nothing
    assert 0.0 <= r["agreement"] <= 1.0


@pytest.mark.slow
def test_worse_candidate_fails_with_reasons(trained):
    good, bad = load_model(trained["candidate"]), load_model(trained["const"])
    r = compare_on_feedback(good, bad, trained["log"])
    assert not r["passed"]
    assert any("PR-AUC" in reason for reason in r["reasons"])
    assert any("recall" in reason for reason in r["reasons"])
    # A generous enough tolerance turns the same evidence into a pass.
    lax = GateCriteria(max_pr_auc_drop=1.0, max_recall_drop=1.0)
    assert compare_on_feedback(good, bad, trained["log"], lax)["passed"]


@pytest.mark.slow
def test_window_overlapping_training_data_is_a_leakage_error(trained):
    live, cand = load_model(trained["const"]), load_model(trained["candidate"])
    horizon = training_horizon(cand)
    assert horizon is not None
    with pytest.raises(LeakageError, match="overlaps"):
        compare_on_feedback(live, cand, trained["log"], after_step=horizon - 1)
    # The horizon applies whichever side the feedback-trained model is on.
    with pytest.raises(LeakageError):
        compare_on_feedback(cand, live, trained["log"], after_step=0)
    # Later than the horizon is fine.
    r = compare_on_feedback(live, cand, trained["log"], GateCriteria(min_positives=1), horizon)
    assert r["window"]["after_step"] == horizon


@pytest.mark.slow
def test_labels_never_change_scores(trained, tmp_path: Path):
    """Flipping every label changes the metrics but not a single score or flag."""
    live, cand = load_model(trained["const"]), load_model(trained["candidate"])
    flipped = tmp_path / "flipped.jsonl"
    lines = []
    for line in trained["log"].read_text().splitlines():
        raw = json.loads(line)
        raw["is_fraud"] = not raw["is_fraud"]
        lines.append(json.dumps(raw))
    flipped.write_text("\n".join(lines) + "\n")
    a = compare_on_feedback(live, cand, trained["log"], GateCriteria(min_positives=1))
    b = compare_on_feedback(live, cand, flipped, GateCriteria(min_positives=1))
    assert a["agreement"] == b["agreement"]
    assert a["window"]["rows"] == b["window"]["rows"]
    assert a["candidate"]["pr_auc"] != b["candidate"]["pr_auc"]


def test_thin_evidence_is_refused_not_passed(tmp_path: Path):
    log = FeedbackLog(tmp_path / "f.jsonl")
    for i in range(20):
        log.append(_rec(f"t{i}", i, is_fraud=i == 19))  # a single fraud
    m = LogisticBaseline.load(_constant_model(tmp_path / "m.json"))
    with pytest.raises(ValueError, match="not enough labelled evidence"):
        compare_on_feedback(m, m, log.path)
    with pytest.raises(ValueError, match="not enough"):
        compare_on_feedback(m, m, log.path, after_step=100)  # empty window
    only_fraud = FeedbackLog(tmp_path / "g.jsonl")
    for i in range(10):
        only_fraud.append(_rec(f"g{i}", i, is_fraud=True))
    with pytest.raises(ValueError, match="some legitimate"):
        compare_on_feedback(m, m, only_fraud.path)


def test_cli_blocks_on_thin_evidence(tmp_path: Path, capsys):
    log = FeedbackLog(tmp_path / "f.jsonl")
    log.append(_rec("a", 1, is_fraud=True))
    model = _constant_model(tmp_path / "m.json")
    args = ["--live", str(model), "--candidate", str(model), "--feedback", str(log.path)]
    assert main(args) == 1
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_rejects_bad_criteria(tmp_path: Path):
    model = _constant_model(tmp_path / "m.json")
    args = ["--live", str(model), "--candidate", str(model), "--feedback", "x"]
    with pytest.raises(SystemExit) as exc:
        main([*args, "--threshold", "2"])
    assert exc.value.code == 2


@pytest.mark.slow
def test_cli_pass_fail_and_report(trained, tmp_path: Path, capsys):
    base = ["--feedback", str(trained["log"])]
    fwd = [*base, "--live", str(trained["const"]), "--candidate", str(trained["candidate"])]
    report = tmp_path / "reports" / "promotion.json"
    code = main([*fwd, "--report", str(report)])
    out = capsys.readouterr().out
    assert code == 0 and "PASS: promote cand" in out
    assert json.loads(report.read_text())["passed"] is True

    code = main([*base, "--live", str(trained["candidate"]), "--candidate", str(trained["const"])])
    assert code == 1 and "FAIL:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main([*fwd, "--after-step", "0"])
    assert exc.value.code == 2
