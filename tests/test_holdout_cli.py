"""End-to-end CSV holdout reporting, including adversarial inputs."""
import csv

import pytest

from fraud_engine.holdout_cli import evaluate_csv


def write_rows(tmp_path, rows):
    path = tmp_path / "scores.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "score", "isFraud"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_temporal_partitions_exclude_embargo(tmp_path):
    path = write_rows(tmp_path, [
        {"step": i, "score": i / 40, "isFraud": str(i % 4 == 0)}
        for i in range(1, 41)
    ])
    report = evaluate_csv(path, budgets=(0.1,))
    assert report["source_rows"] == 40
    assert report["embargoed_records"] > 0
    assert sum(report[key]["records"] for key in ("train", "validation", "test")) + report["embargoed_records"] == 40
    assert report["test"]["alert_budgets"]["0.1"]["alerts"] >= 1


@pytest.mark.parametrize("score,label", [
    ("nan", "0"), ("inf", "0"), ("-0.1", "0"), ("1.1", "0"),
    ("0.5", "2"), ("0.5", "1.0"),
])
def test_invalid_score_or_label_rejected(tmp_path, score, label):
    path = write_rows(tmp_path, [
        {"step": i, "score": score if i == 1 else "0.5",
         "isFraud": label if i == 1 else "0"}
        for i in range(1, 41)
    ])
    with pytest.raises(ValueError, match="invalid row"):
        evaluate_csv(path)


def test_out_of_order_rejected(tmp_path):
    path = write_rows(tmp_path, [
        {"step": 41 - i, "score": "0.5", "isFraud": "0"}
        for i in range(1, 41)
    ])
    with pytest.raises(ValueError, match="ordered"):
        evaluate_csv(path)
