"""Leakage-safe offline preprocessing for PaySim.

The plan (see docs/leakage-safe-preprocessing.md for the reasoning):

1. Validate the schema and drop columns that leak the answer (``isFlaggedFraud``).
2. Sort chronologically by ``step`` (stable, so within-step file order is kept).
3. Replay the stream through the SAME ``VelocityStore`` + ``compute_features`` the API
   uses. Each row only sees rows before it. No labels are read while building features.
4. Split by time: train = early steps, a gap of ``gap_steps`` is discarded, test = later
   steps. Never a random row split.
5. Fit standardisation (mean/scale) on the training rows only.
6. Run ``leakage_checks`` and fail loudly if any invariant is broken.

Requires the ``train`` extra (pandas, numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, compute_features
from .rules import RuleContext, evaluate_rules
from .state import VelocityStore

PAYSIM_COLUMNS: tuple[str, ...] = (
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
)
LABEL = "isFraud"
# Columns that must never reach the model.
LEAKY_COLUMNS: tuple[str, ...] = ("isFlaggedFraud",)
ID_COLUMNS: tuple[str, ...] = ("nameOrig", "nameDest")
VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


class LeakageError(AssertionError):
    """Raised when a leakage invariant is violated."""


@dataclass(frozen=True)
class SplitInfo:
    train_rows: int
    gap_rows: int
    test_rows: int
    train_max_step: int
    test_min_step: int
    cutoff_step: int
    gap_steps: int


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in PAYSIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PaySim columns: {missing}")
    bad_types = set(df["type"].astype(str).unique()) - VALID_TYPES
    if bad_types:
        raise ValueError(f"Unknown transaction types: {sorted(bad_types)}")
    money = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    if (df[money] < 0).any().any():
        raise ValueError("Negative monetary values found")
    if df[LABEL].isna().any():
        raise ValueError("Missing labels found")


def load_paysim(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=_DTYPES, nrows=nrows)
    validate_schema(df)
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("step", kind="mergesort").reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame, window_steps: int = 24) -> pd.DataFrame:
    """Replay the stream in time order and compute point-in-time features.

    Returns a frame with FEATURE_NAMES, ``rules_score``, ``step`` and the label.
    The label column is copied through but is never read while computing features.
    """
    df = sort_chronologically(df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]))
    store = VelocityStore(window_steps)
    ctx = RuleContext()
    feature_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    rows: list[list[float]] = []
    rules_scores: list[float] = []
    for rec in df[feature_cols].itertuples(index=False):
        txn = rec._asdict()
        txn["type"] = str(txn["type"])
        snap = store.snapshot(txn["nameOrig"], txn["nameDest"], int(txn["step"]))
        feats = compute_features(txn, snap)
        rows.append([feats[f] for f in FEATURE_NAMES])
        rules_scores.append(evaluate_rules(txn, snap, ctx)[0])
        store.record(txn["nameOrig"], txn["nameDest"], int(txn["step"]), float(txn["amount"]))

    out = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    out["rules_score"] = rules_scores
    out["step"] = df["step"].to_numpy()
    out[LABEL] = df[LABEL].to_numpy()
    return out


def temporal_split(
    frame: pd.DataFrame, test_frac: float = 0.2, gap_steps: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Split by time with a discarded gap between train and test.

    The gap emulates label delay (chargebacks arrive days later) and stops the last
    training hours from bleeding into the first test hours.
    """
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    frame = frame.sort_values("step", kind="mergesort").reset_index(drop=True)
    cutoff_idx = int(len(frame) * (1 - test_frac))
    cutoff_step = int(frame["step"].iloc[max(cutoff_idx - 1, 0)])
    train = frame[frame["step"] <= cutoff_step]
    test = frame[frame["step"] > cutoff_step + gap_steps]
    gap_rows = len(frame) - len(train) - len(test)
    if train.empty or test.empty:
        raise ValueError("Split produced an empty partition; adjust test_frac / gap_steps")
    info = SplitInfo(
        train_rows=len(train),
        gap_rows=gap_rows,
        test_rows=len(test),
        train_max_step=int(train["step"].max()),
        test_min_step=int(test["step"].min()),
        cutoff_step=cutoff_step,
        gap_steps=gap_steps,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), info


def fit_standardiser(X_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean/scale from TRAINING rows only."""
    mean = X_train.to_numpy(dtype=float).mean(axis=0)
    scale = X_train.to_numpy(dtype=float).std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def leakage_checks(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], gap_steps: int
) -> dict[str, object]:
    """Fail loudly on leakage; return an informational report otherwise."""
    forbidden = {LABEL, *LEAKY_COLUMNS, *ID_COLUMNS, "step"}
    bad = forbidden.intersection(feature_cols)
    if bad:
        raise LeakageError(f"Forbidden columns used as features: {sorted(bad)}")
    if int(train["step"].max()) + gap_steps >= int(test["step"].min()):
        raise LeakageError("Test period overlaps the training period (or its gap)")
    for name, part in (("train", train), ("test", test)):
        if part[LABEL].nunique() < 2:
            raise LeakageError(f"{name} partition has a single class; metrics would be invalid")
    return {
        "train_fraud_rate": float(train[LABEL].mean()),
        "test_fraud_rate": float(test[LABEL].mean()),
        "train_steps": [int(train["step"].min()), int(train["step"].max())],
        "test_steps": [int(test["step"].min()), int(test["step"].max())],
    }
