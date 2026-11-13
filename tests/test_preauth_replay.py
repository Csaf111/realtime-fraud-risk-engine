"""Regression tests for chronological pre-authorisation replay."""
import csv

import pytest

from fraud_engine.preauth_replay import replay

FIELDS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "isFraud"]


def dataset(tmp_path, rows):
    path = tmp_path / "paysim.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(step, label, **changes):
    result = dict(step=step, type="TRANSFER", amount=100, nameOrig="C1",
                  oldbalanceOrg=1000, nameDest="C2", isFraud=label)
    result.update(changes)
    return result


def test_replay_counts_alerts_and_preserves_stable_ties(tmp_path):
    path = dataset(tmp_path, [row(1, 1), row(2, 0), row(3, 1), row(4, 0)])
    result = replay(path, budgets=(0.25, 0.5))
    assert result["records"] == 4
    assert result["fraud_records"] == 2
    assert result["alert_budgets"]["0.25"]["alerts"] == 1
    assert result["alert_budgets"]["0.25"]["true_positives"] == 1


def test_rejects_out_of_order_steps(tmp_path):
    with pytest.raises(ValueError, match="out-of-order"):
        replay(dataset(tmp_path, [row(2, 0), row(1, 1)]))


def test_rejects_invalid_labels(tmp_path):
    with pytest.raises(ValueError, match="invalid fraud label"):
        replay(dataset(tmp_path, [row(1, 2)]))


def test_empty_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        replay(dataset(tmp_path, []))


def test_zero_positive_population(tmp_path):
    result = replay(dataset(tmp_path, [row(1, 0)]), budgets=(1.0,))
    assert result["alert_budgets"]["1.0"]["recall"] == 0.0


def test_invalid_budget_rejected(tmp_path):
    with pytest.raises(ValueError, match="budgets"):
        replay(dataset(tmp_path, [row(1, 1)]), budgets=(0.0,))
