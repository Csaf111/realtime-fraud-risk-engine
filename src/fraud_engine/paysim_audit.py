"""Streaming PaySim dataset integrity audit; never uses labels as model features."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

REQUIRED = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)
MONEY = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
TYPES = frozenset({"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"})


def audit(path: str | Path) -> dict:
    """Validate the complete CSV and return an aggregate, identifier-free report.

    Raises ValueError on the first invalid row; no partial success report is returned.
    """
    count = fraud = flagged = 0
    previous_step = -1
    by_type: Counter[str] = Counter()
    fraud_by_type: Counter[str] = Counter()
    flagged_by_type: Counter[str] = Counter()
    flagged_true_fraud = 0
    distinct_steps = 0
    first_step: int | None = None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, restkey="__extra__", restval=None)
        headers = reader.fieldnames
        if headers is None or len(headers) != len(set(headers)):
            raise ValueError("missing or duplicate CSV headers")
        if set(headers) != set(REQUIRED):
            raise ValueError("PaySim columns differ from expected schema")
        for line, row in enumerate(reader, start=2):
            if "__extra__" in row or any(value is None for value in row.values()):
                raise ValueError(f"line {line}: malformed CSV record")
            raw_step = row["step"]
            if not raw_step.isascii() or not raw_step.isdecimal():
                raise ValueError(f"line {line}: step must be a nonnegative integer")
            step = int(raw_step)
            if step < previous_step:
                raise ValueError(f"line {line}: nonchronological step")
            if step != previous_step:
                distinct_steps += 1
                if first_step is None:
                    first_step = step
            previous_step = step
            if row["type"] not in TYPES:
                raise ValueError(f"line {line}: unknown transaction type")
            if not row["nameOrig"] or not row["nameDest"]:
                raise ValueError(f"line {line}: empty account identifier")
            for field in MONEY:
                try:
                    value = float(row[field])
                except ValueError as exc:
                    raise ValueError(f"line {line}: invalid {field}") from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"line {line}: invalid {field}")
            for field in ("isFraud", "isFlaggedFraud"):
                if row[field] not in ("0", "1"):
                    raise ValueError(f"line {line}: invalid {field}")
            count += 1
            fraud += int(row["isFraud"])
            flagged += int(row["isFlaggedFraud"])
            by_type[row["type"]] += 1
            if row["isFraud"] == "1":
                fraud_by_type[row["type"]] += 1
            if row["isFlaggedFraud"] == "1":
                flagged_by_type[row["type"]] += 1
                flagged_true_fraud += int(row["isFraud"])
    if count == 0:
        raise ValueError("empty dataset")
    return {
        "rows": count, "fraud_rows": fraud, "fraud_prevalence": fraud / count,
        "flagged_rows": flagged, "min_step": first_step, "max_step": previous_step,
        "distinct_steps": distinct_steps, "transactions_by_type": dict(sorted(by_type.items())),
        "fraud_by_type": {kind: fraud_by_type[kind] for kind in sorted(TYPES)},
        "flagged_by_type": {kind: flagged_by_type[kind] for kind in sorted(TYPES)},
        "flagged_true_fraud": flagged_true_fraud,
        "flagged_false_positive": flagged - flagged_true_fraud,
        "unflagged_fraud": fraud - flagged_true_fraud,
        "schema": "PaySim-v1", "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PaySim CSV before modelling")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.csv_path)
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        parser.exit(2, f"PaySim audit failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
