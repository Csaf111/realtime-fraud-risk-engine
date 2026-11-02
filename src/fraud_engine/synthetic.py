"""Small PaySim-shaped synthetic generator for tests, demos and CI.

This is NOT PaySim and models trained on it mean nothing for real performance. It exists
so that the full pipeline (features -> split -> train -> serve) can be exercised in CI
without downloading the 470 MB dataset.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

HEADER = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
_LEGIT_TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
_LEGIT_WEIGHTS = [0.34, 0.22, 0.35, 0.08, 0.01]


def _legit(rng: random.Random, step: int, i: int) -> list:
    t = rng.choices(_LEGIT_TYPES, _LEGIT_WEIGHTS)[0]
    amount = round(rng.lognormvariate(10.5, 1.2), 2)
    old_o = round(rng.lognormvariate(10.8, 1.5), 2)
    if t == "CASH_IN":
        new_o = round(old_o + amount, 2)
    else:
        amount = min(amount, old_o) if rng.random() < 0.9 else amount
        new_o = round(max(old_o - amount, 0.0), 2)
    if t in ("PAYMENT", "DEBIT"):
        dest = f"M{rng.randrange(10**9)}"
        old_d = new_d = 0.0
    else:
        dest = f"C{rng.randrange(10**6)}"
        old_d = round(rng.lognormvariate(11, 1.5), 2)
        new_d = round(old_d + amount, 2) if t != "CASH_IN" else round(max(old_d - amount, 0), 2)
    return [step, t, round(amount, 2), f"C{i:09d}", old_o, new_o, dest, old_d, new_d, 0, 0]


def _fraud(rng: random.Random, step: int, i: int, mule: str) -> list:
    t = rng.choice(["TRANSFER", "CASH_OUT"])
    old_o = round(rng.lognormvariate(12, 1.0), 2)
    amount = old_o  # account takeover: drain everything
    old_d, new_d = (0.0, 0.0) if rng.random() < 0.7 else (round(rng.lognormvariate(10, 1), 2),) * 2
    flagged = int(t == "TRANSFER" and amount > 200_000 and rng.random() < 0.01)
    return [step, t, amount, f"C{i:09d}", old_o, 0.0, mule, old_d, new_d, 1, flagged]


def generate_rows(
    n: int = 5000, fraud_rate: float = 0.02, steps: int = 200, seed: int = 7
) -> list[list]:
    rng = random.Random(seed)
    mules = [f"C9{rng.randrange(10**6):06d}" for _ in range(25)]
    rows = []
    for i in range(n):
        step = 1 + int(i * steps / n)
        rows.append(
            _fraud(rng, step, i, rng.choice(mules))
            if rng.random() < fraud_rate
            else _legit(rng, step, i)
        )
    return rows


def write_csv(path: str | Path, n: int = 5000, fraud_rate: float = 0.02, seed: int = 7) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(generate_rows(n, fraud_rate, seed=seed))
    return p


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Write a PaySim-shaped synthetic CSV")
    ap.add_argument("--out", default="data/raw/synthetic_paysim.csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--fraud-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(write_csv(a.out, a.rows, a.fraud_rate, a.seed))
