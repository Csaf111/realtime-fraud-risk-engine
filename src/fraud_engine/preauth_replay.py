"""Chronological PaySim replay for the pre-authorisation rules baseline.

No future balances or fraud labels enter scoring. Labels are read only after
scoring for offline evaluation. Input must be ordered by nondecreasing step.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections.abc import Iterator
from math import ceil

from .preauth import PreAuthTransaction, score_preauth
from .state import VelocityStore

REQUIRED = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "isFraud"}


def iter_scores(path: Path, *, window_steps: int = 24) -> Iterator[tuple[int, float, int]]:
    """Yield step, causal rules score and offline label in source order."""
    store = VelocityStore(window_steps=window_steps)
    last_step = -1
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"missing PaySim columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        for index, row in enumerate(reader):
            step = int(row["step"])
            if step < last_step:
                raise ValueError(f"out-of-order step at data row {index + 1}")
            last_step = step
            label = int(row["isFraud"])
            if label not in (0, 1):
                raise ValueError(f"invalid fraud label at data row {index + 1}")
            txn = PreAuthTransaction(
                step=step, type=row["type"], amount=float(row["amount"]),
                nameOrig=row["nameOrig"], oldbalanceOrg=float(row["oldbalanceOrg"]),
                nameDest=row["nameDest"],
            )
            snapshot = store.snapshot(txn.nameOrig, txn.nameDest, txn.step)
            score = score_preauth(txn, snapshot).risk_score
            store.record(txn.nameOrig, txn.nameDest, txn.step, txn.amount)
            yield step, score, label


def replay(path: Path, *, window_steps: int = 24, budgets: tuple[float, ...] = (0.001, 0.005, 0.01)) -> dict:
    if not budgets or any(not 0 < b <= 1 for b in budgets):
        raise ValueError("budgets must be in (0, 1]")
    ranked = [(score, index, label) for index, (_, score, label) in enumerate(
        iter_scores(path, window_steps=window_steps)
    )]
    if not ranked:
        raise ValueError("empty dataset")
    ranked.sort(key=lambda x: (-x[0], x[1]))
    positives = sum(label for _, _, label in ranked)
    results = {}
    for budget in budgets:
        # Ceiling ensures nonzero review capacity for a small sample.
        capacity = ceil(len(ranked) * budget)
        caught = sum(label for _, _, label in ranked[:capacity])
        results[str(budget)] = {
            "alerts": capacity, "true_positives": caught,
            "precision": caught / capacity,
            "recall": caught / positives if positives else 0.0,
        }
    return {
        "records": len(ranked), "fraud_records": positives,
        "scorer": "preauth-rules-v2.0", "window_steps": window_steps,
        "alert_budgets": results,
    }


def export_scores(path: Path, output: Path, *, window_steps: int = 24) -> None:
    """Write holdout-compatible scores without account identifiers.

    Exclusive creation prevents overwriting the source or an existing report.
    Remove partial output if replay fails.
    """
    with output.open("x", newline="", encoding="utf-8") as stream:
        try:
            writer = csv.writer(stream)
            writer.writerow(["step", "score", "isFraud"])
            count = 0
            for record in iter_scores(path, window_steps=window_steps):
                writer.writerow(record)
                count += 1
            if not count:
                raise ValueError("empty dataset")
        except BaseException:
            stream.close()
            output.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--window-steps", type=int, default=24)
    parser.add_argument("--export-scores", type=Path)
    args = parser.parse_args()
    if args.export_scores is not None:
        export_scores(args.csv_path, args.export_scores, window_steps=args.window_steps)
        return
    print(json.dumps(replay(args.csv_path, window_steps=args.window_steps), indent=2))


if __name__ == "__main__":
    main()
