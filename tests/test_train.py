import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from fraud_engine.artifacts import load_model
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.model import LogisticBaseline
from fraud_engine.scoring import RiskEngine
from fraud_engine.synthetic import write_csv
from fraud_engine.train import main, train
from fraud_engine.trees import GradientBoostedTrees

from .conftest import fraud_txn, make_txn


@pytest.mark.slow
def test_train_end_to_end(tmp_path):
    data = write_csv(tmp_path / "s.csv", n=4000, fraud_rate=0.03)
    out = tmp_path / "model.json"
    metrics = train(data, out, gap_steps=5, version="test")

    assert out.exists() and out.with_suffix(".metrics.json").exists()
    assert set(metrics) >= {"model", "rules_only_baseline", "split", "leakage_report"}
    assert 0.0 <= metrics["model"]["pr_auc"] <= 1.0

    model = LogisticBaseline.load(out)
    eng = RiskEngine(Settings(), model=model)
    assert eng.score(fraud_txn()).risk_score > eng.score(make_txn()).risk_score


@pytest.mark.slow
def test_cli_synthetic(tmp_path, capsys):
    out = tmp_path / "m.json"
    main(["--synthetic", "3000", "--out", str(out), "--gap-steps", "5", "--version", "cli"])
    assert json.loads(out.read_text())["version"] == "cli"
    assert "Saved model" in capsys.readouterr().out


@pytest.mark.slow
def test_train_gbt_end_to_end(tmp_path):
    data = write_csv(tmp_path / "s.csv", n=4000, fraud_rate=0.03)
    out = tmp_path / "gbt.json"
    metrics = train(
        data, out, gap_steps=5, version="gbt", model_type="gbt", n_estimators=20, max_depth=3
    )
    assert metrics["model_type"] == "gbt"
    assert metrics["hyperparams"]["n_estimators"] == 20
    assert 0.0 <= metrics["model"]["pr_auc"] <= 1.0

    model = load_model(out)
    assert isinstance(model, GradientBoostedTrees) and len(model.trees) == 20
    assert out.with_suffix(".score_reference.json").exists()
    eng = RiskEngine(Settings(), model=model)
    res = eng.score(fraud_txn(), update_state=False)
    assert res.model_version.endswith("+gbt-gbt")
    assert eng.score(fraud_txn()).risk_score > eng.score(make_txn()).risk_score


def test_train_rejects_unknown_model_type(tmp_path):
    with pytest.raises(ValueError, match="model_type"):
        train(tmp_path / "missing.csv", tmp_path / "m.json", model_type="forest")


@pytest.mark.slow
def test_cli_gbt(tmp_path, capsys):
    out = tmp_path / "g.json"
    main(
        [
            "--synthetic", "3000", "--out", str(out), "--gap-steps", "5",
            "--model", "gbt", "--n-estimators", "10", "--max-depth", "2",
        ]
    )  # fmt: skip
    assert json.loads(out.read_text())["model_type"] == "gradient_boosted_trees"
    assert "Saved model" in capsys.readouterr().out


def test_export_parity_check_raises_on_mismatch():
    from fraud_engine.train import _check_export_parity

    class FakeClf:
        def decision_function(self, rows):
            return np.full(len(rows), 123.0)

    clf = GradientBoostingClassifier(n_estimators=2, max_depth=1, random_state=0)
    X = np.random.default_rng(0).normal(size=(40, len(FEATURE_NAMES)))
    clf.fit(X, (X[:, 0] > 0).astype(int))
    model = GradientBoostedTrees.from_sklearn(clf, FEATURE_NAMES, X)
    _check_export_parity(model, clf, X)  # exact export passes
    with pytest.raises(RuntimeError, match="disagree"):
        _check_export_parity(model, FakeClf(), X)
