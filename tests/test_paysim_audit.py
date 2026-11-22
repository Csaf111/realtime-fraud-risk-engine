import csv

import pytest

from fraud_engine.paysim_audit import REQUIRED, audit


def write_rows(tmp_path, rows, headers=REQUIRED):
    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(**overrides):
    item = dict.fromkeys(REQUIRED, "0")
    item.update(step="1", type="TRANSFER", amount="42.50", nameOrig="C1",
                nameDest="C2", isFraud="1", isFlaggedFraud="0")
    item.update(overrides)
    return item


def test_aggregate_report_excludes_identifiers(tmp_path):
    report = audit(write_rows(tmp_path, [row(), row(step="2", isFraud="0")]))
    assert report["rows"] == 2
    assert report["fraud_prevalence"] == 0.5
    assert report["distinct_steps"] == 2
    assert "C1" not in str(report)


@pytest.mark.parametrize("overrides", [
    {"step": "-1"}, {"step": "1.5"}, {"type": "UNKNOWN"},
    {"amount": "-1"}, {"amount": "nan"}, {"amount": "inf"},
    {"isFraud": "2"}, {"isFlaggedFraud": "yes"}, {"nameOrig": ""},
])
def test_rejects_invalid_values(tmp_path, overrides):
    with pytest.raises(ValueError):
        audit(write_rows(tmp_path, [row(**overrides)]))


def test_rejects_out_of_order(tmp_path):
    with pytest.raises(ValueError, match="nonchronological"):
        audit(write_rows(tmp_path, [row(step="3"), row(step="2")]))


def test_rejects_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        audit(write_rows(tmp_path, []))


def test_rejects_duplicate_headers(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("step,step\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        audit(path)


def test_rejects_extra_csv_fields(tmp_path):
    path = write_rows(tmp_path, [row()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(["0"] * (len(REQUIRED) + 1)) + "\n")
    with pytest.raises(ValueError, match="malformed"):
        audit(path)


def test_distinct_steps_counts_runs_not_rows(tmp_path):
    report = audit(write_rows(tmp_path, [
        row(step="0", isFraud="0"), row(step="0"),
        row(step="4", isFraud="0"), row(step="4", isFraud="0"),
        row(step="100", isFraud="0"),
    ]))
    assert report["rows"] == 5
    assert report["min_step"] == 0
    assert report["max_step"] == 100
    assert report["distinct_steps"] == 3


def test_single_step_report(tmp_path):
    report = audit(write_rows(tmp_path, [row(step="7"), row(step="7")]))
    assert report["min_step"] == report["max_step"] == 7
    assert report["distinct_steps"] == 1
