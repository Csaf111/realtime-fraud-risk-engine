"""Regression tests for chronological pre-authorisation replay."""
import csv

import pytest

from fraud_engine.preauth_replay import export_scores, replay

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
    path = dataset(tmp_path, [row(1, 1, nameOrig="A1", nameDest="B1"), row(2, 0, nameOrig="A2", nameDest="B2"), row(3, 1, nameOrig="A3", nameDest="B3"), row(4, 0, nameOrig="A4", nameDest="B4")])
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


def test_export_scores_preserves_order_and_omits_identifiers(tmp_path):
    source = dataset(tmp_path, [row(1, 0), row(1, 1), row(2, 0)])
    output = tmp_path / "scores.csv"
    export_scores(source, output)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["step", "score", "isFraud"]
        records = list(reader)
    assert [r["step"] for r in records] == ["1", "1", "2"]
    assert [r["isFraud"] for r in records] == ["0", "1", "0"]
    assert all(0 <= float(r["score"]) <= 1 for r in records)


def test_export_does_not_overwrite_source(tmp_path):
    source = dataset(tmp_path, [row(1, 0)])
    original = source.read_bytes()
    with pytest.raises(FileExistsError):
        export_scores(source, source)
    assert source.read_bytes() == original


def test_failed_export_removes_partial_output(tmp_path):
    source = dataset(tmp_path, [row(2, 0), row(1, 1)])
    output = tmp_path / "scores.csv"
    with pytest.raises(ValueError, match="out-of-order"):
        export_scores(source, output)
    assert not output.exists()


def test_labels_do_not_change_exported_scores(tmp_path):
    source = dataset(tmp_path, [row(1, 0), row(2, 1)])
    first = tmp_path / "first.csv"
    export_scores(source, first)
    source = dataset(tmp_path, [row(1, 1), row(2, 0)])
    second = tmp_path / "second.csv"
    export_scores(source, second)
    def scores(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return [r["score"] for r in csv.DictReader(stream)]
    assert scores(first) == scores(second)
