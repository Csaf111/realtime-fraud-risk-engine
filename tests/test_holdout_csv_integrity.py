"""CSV ingestion integrity regression tests for temporal holdout reports."""
from pathlib import Path

import pytest

from fraud_engine.holdout_cli import evaluate_csv


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        ("step,score,isFraud,score", "1,0.1,0,0.2", "duplicate CSV column"),
        ("step,score,isFraud", "1,0.1,0,unexpected", "more values"),
        ("step,score,isFraud", "1,0.1", "missing CSV field"),
        ("step,score,isFraud", "-1,0.1,0", "nonnegative decimal"),
        ("step,score,isFraud", "1.0,0.1,0", "nonnegative decimal"),
        ("step,score,isFraud", "1,inf,0", "finite"),
        ("step,score,isFraud", "1,0.1,True", "label must"),
        ("step,score,isFraud", "1,0.1,0\n2,0.2,1\n3,0.3,0\n4,0.4,1\n5,0.5,0\n6,0.6,1\n7,0.7,0\n8,0.8,1\n9,0.9,0\n10,0.1,1\n11,0.2,0\n12,0.3,1\n13,0.4,0\n14,0.5,1\n15,0.6,0\n16,0.7,1\n17,0.8,0\n18,0.9,1\n19,0.1,0\n20,0.2,1", ""),
    ],
)
def test_csv_integrity(tmp_path: Path, header: str, rows: str, message: str) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(header + "\n" + rows + "\n", encoding="utf-8")
    if message:
        with pytest.raises(ValueError, match=message):
            evaluate_csv(path, embargo_steps=0)
    else:
        report = evaluate_csv(path, embargo_steps=0)
        assert report["source_rows"] == 20
        assert report["embargoed_records"] == 0


def test_utf8_bom_header_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    rows = [f"{step},{step / 20}, {step % 2}".replace(", ", ",") for step in range(1, 21)]
    path.write_text("step,score,isFraud\n" + "\n".join(rows), encoding="utf-8-sig")
    assert evaluate_csv(path, embargo_steps=0)["source_rows"] == 20
