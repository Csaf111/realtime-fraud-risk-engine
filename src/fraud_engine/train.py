"""Train the explainable logistic-regression baseline.

Examples:
    # Real data (download PaySim first, see docs/dataset-paysim.md)
    python -m fraud_engine.train --data data/raw/PS_20174392719_1491204439457_log.csv

    # Quick run on a sample
    python -m fraud_engine.train --data data/raw/paysim.csv --nrows 500000

    # Pipeline smoke test on synthetic data (what CI does)
    python -m fraud_engine.train --synthetic 5000 --out models/synthetic.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from .features import FEATURE_NAMES
from .model import LogisticBaseline
from .preprocessing import (
    LABEL,
    build_feature_frame,
    fit_standardiser,
    leakage_checks,
    load_paysim,
    temporal_split,
)
from .synthetic import write_csv


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": threshold,
    }


def train(
    data_path: str | Path,
    out_path: str | Path,
    nrows: int | None = None,
    test_frac: float = 0.2,
    gap_steps: int = 24,
    window_steps: int = 24,
    C: float = 1.0,
    version: str | None = None,
) -> dict[str, Any]:
    raw = load_paysim(data_path, nrows=nrows)
    frame = build_feature_frame(raw, window_steps=window_steps)
    train_df, test_df, split = temporal_split(frame, test_frac=test_frac, gap_steps=gap_steps)
    feature_cols = list(FEATURE_NAMES)
    report = leakage_checks(train_df, test_df, feature_cols, gap_steps)

    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    X_te = test_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df[LABEL].to_numpy()
    y_te = test_df[LABEL].to_numpy()

    mean, scale = fit_standardiser(train_df[feature_cols])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((X_tr - mean) / scale, y_tr)

    p_te = clf.predict_proba((X_te - mean) / scale)[:, 1]
    version = version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    model = LogisticBaseline(
        feature_names=feature_cols,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        version=version,
    )

    metrics = {
        "model": _metrics(y_te, p_te, 0.5),
        "rules_only_baseline": _metrics(y_te, test_df["rules_score"].to_numpy(), 0.5),
        "split": split.__dict__,
        "leakage_report": report,
        "data": str(data_path),
        "nrows": nrows,
        "window_steps": window_steps,
        "C": C,
    }
    model.metadata = {"trained_at": datetime.now(UTC).isoformat(), "metrics": metrics}
    model.save(out_path)
    Path(out_path).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Path to the PaySim CSV")
    src.add_argument("--synthetic", type=int, metavar="N", help="Generate N synthetic rows instead")
    ap.add_argument("--out", default="models/baseline.json")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--gap-steps", type=int, default=24)
    ap.add_argument("--window-steps", type=int, default=24)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)

    if a.synthetic:
        tmp = Path(tempfile.mkdtemp()) / "synthetic.csv"
        data = write_csv(tmp, n=a.synthetic)
    else:
        data = Path(a.data)

    m = train(data, a.out, a.nrows, a.test_frac, a.gap_steps, a.window_steps, a.C, a.version)
    print(
        json.dumps({"model": m["model"], "rules_only_baseline": m["rules_only_baseline"]}, indent=2)
    )
    print(f"Saved model to {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
