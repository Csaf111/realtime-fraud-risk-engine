"""CLI to evaluate *existing* point-in-time scores on a temporal PaySim holdout.

Input is a CSV with step, score, isFraud columns. Scores must have been generated
without labels, post-event fields, or future velocity. This utility cannot verify
the provenance of supplied scores.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from pathlib import Path

from .holdout_metrics import evaluate_holdout
from .temporal_holdout import chronological_holdout


def evaluate_csv(
    path: Path, *, train_fraction: float = 0.6,
    validation_fraction: float = 0.2, embargo_steps: int = 1,
    budgets: tuple[float, ...] = (0.001, 0.005, 0.01),
) -> dict:
    steps: list[int] = []
    scores: list[float] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "score", "isFraud"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                step = int(row["step"])
                score = float(row["score"])
                label_text = row["isFraud"]
                if label_text not in ("0", "1"):
                    raise ValueError("label must be 0 or 1")
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("score must be finite and in [0, 1]")
                steps.append(step)
                scores.append(score)
                labels.append(int(label_text))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row {row_number}: {exc}") from exc
    split = chronological_holdout(
        steps, train_fraction=train_fraction,
        validation_fraction=validation_fraction, embargo_steps=embargo_steps,
    )
    result = evaluate_holdout(scores, labels, split, budgets=budgets)
    result["source_rows"] = len(steps)
    result["provenance_warning"] = (
        "Supplied scores are not audited for leakage. Generate causally, "
        "freeze model selection before inspecting test labels."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--embargo-steps", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(evaluate_csv(
        args.csv_path, train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        embargo_steps=args.embargo_steps,
    ), indent=2))


if __name__ == "__main__":
    main()
