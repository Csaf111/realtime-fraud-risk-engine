import json

import pytest

from fraud_engine.config import Settings
from fraud_engine.model import LogisticBaseline
from fraud_engine.scoring import RiskEngine
from fraud_engine.synthetic import write_csv
from fraud_engine.train import main, train

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
