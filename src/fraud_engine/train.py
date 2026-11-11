"""Train the explainable logistic-regression baseline.

Examples:
    # Real data (download PaySim first, see docs/dataset-paysim.md)
    python -m fraud_engine.train --data data/raw/PS_20174392719_1491204439457_log.csv

    # Quick run on a sample
    python -m fraud_engine.train --data data/raw/paysim.csv --nrows 500000

    # Pipeline smoke test on synthetic data (what CI does)
    python -m fraud_engine.train --synthetic 5000 --out models/synthetic.json

    # Recommend thresholds for an alert budget: review <= 1% of traffic, block <= 0.2%
    python -m fraud_engine.train --data data/raw/paysim.csv --review-budget 0.01 \
        --block-budget 0.002
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

from .drift import ScoreReference
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
from .thresholds import blend_risk, evaluate_thresholds, select_thresholds


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": threshold,
    }


def _served_scores(model: LogisticBaseline, frame: Any, model_weight: float) -> list[float]:
    """Blended risk scores for a feature frame, computed exactly as the API serves them."""
    feats = frame[list(FEATURE_NAMES)].to_dict(orient="records")
    rules = frame["rules_score"].tolist()
    return [
        blend_risk(model.predict_proba(f), float(r), model_weight)
        for f, r in zip(feats, rules, strict=True)
    ]


def _validation_slice(train_df: Any, validation_frac: float) -> Any:
    """Most recent ``validation_frac`` of the (chronologically sorted) training window."""
    n_val = max(1, round(len(train_df) * validation_frac))
    return train_df.iloc[-n_val:]


def reference_path_for(model_path: str | Path) -> Path:
    """Where the score reference for a model artifact is written."""
    return Path(model_path).with_suffix(".score_reference.json")


def _score_reference(
    model: LogisticBaseline, train_df: Any, validation_frac: float, model_weight: float
) -> ScoreReference:
    """Served-score distribution on the most recent slice of the training window.

    Label-free and never touches the test window, like threshold selection.
    """
    val_df = _validation_slice(train_df, validation_frac)
    return ScoreReference.from_scores(
        _served_scores(model, val_df, model_weight),
        model_weight=model_weight,
        steps=[int(val_df["step"].min()), int(val_df["step"].max())],
        model_version=model.version,
    )


def _alert_budget(
    model: LogisticBaseline,
    train_df: Any,
    test_df: Any,
    review_budget: float,
    block_budget: float,
    validation_frac: float,
    model_weight: float,
) -> dict[str, Any]:
    """Pick thresholds on the most recent slice of the training window, report on test.

    Selection is label-free (score quantiles only) and never sees the test window.
    """
    val_df = _validation_slice(train_df, validation_frac)
    selection = select_thresholds(
        _served_scores(model, val_df, model_weight), review_budget, block_budget
    )
    test_scores = _served_scores(model, test_df, model_weight)
    return {
        **selection.as_dict(),
        "model_weight": model_weight,
        "validation_rows": len(val_df),
        "validation_steps": [int(val_df["step"].min()), int(val_df["step"].max())],
        "test": evaluate_thresholds(
            test_scores,
            test_df[LABEL].astype(int).tolist(),
            selection.review_threshold,
            selection.block_threshold,
        ),
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
    review_budget: float | None = None,
    block_budget: float | None = None,
    validation_frac: float = 0.2,
    model_weight: float = 0.6,
) -> dict[str, Any]:
    if (review_budget is None) != (block_budget is None):
        raise ValueError("review_budget and block_budget must be given together")
    if not 0.0 < validation_frac < 1.0:
        raise ValueError("validation_frac must be between 0 and 1")
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
    if review_budget is not None and block_budget is not None:
        metrics["alert_budget"] = _alert_budget(
            model, train_df, test_df, review_budget, block_budget, validation_frac, model_weight
        )
    reference = _score_reference(model, train_df, validation_frac, model_weight)
    ref_path = reference_path_for(out_path)
    metrics["score_reference"] = {
        "path": str(ref_path),
        "rows": reference.n,
        "steps": reference.steps,
        "model_weight": model_weight,
    }
    model.metadata = {"trained_at": datetime.now(UTC).isoformat(), "metrics": metrics}
    model.save(out_path)
    reference.save(ref_path)
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
    ap.add_argument(
        "--review-budget",
        type=float,
        default=None,
        help="Max share of traffic sent to REVIEW or BLOCK (e.g. 0.01); needs --block-budget",
    )
    ap.add_argument("--block-budget", type=float, default=None, help="Max share of traffic BLOCKed")
    ap.add_argument(
        "--validation-frac",
        type=float,
        default=0.2,
        help="Most recent share of the training window used for thresholds and the drift "
        "score reference",
    )
    ap.add_argument(
        "--model-weight", type=float, default=0.6, help="Blend weight, as FRE_MODEL_WEIGHT"
    )
    a = ap.parse_args(argv)
    if (a.review_budget is None) != (a.block_budget is None):
        ap.error("--review-budget and --block-budget must be given together")

    if a.synthetic:
        tmp = Path(tempfile.mkdtemp()) / "synthetic.csv"
        data = write_csv(tmp, n=a.synthetic)
    else:
        data = Path(a.data)

    m = train(
        data,
        a.out,
        a.nrows,
        a.test_frac,
        a.gap_steps,
        a.window_steps,
        a.C,
        a.version,
        review_budget=a.review_budget,
        block_budget=a.block_budget,
        validation_frac=a.validation_frac,
        model_weight=a.model_weight,
    )
    summary = {"model": m["model"], "rules_only_baseline": m["rules_only_baseline"]}
    if "alert_budget" in m:
        summary["alert_budget"] = m["alert_budget"]
    print(json.dumps(summary, indent=2))
    if "alert_budget" in m:
        ab = m["alert_budget"]
        if not ab["budget_met"]:
            print("WARNING: alert budget not attainable on the validation window (score ties)")
        print(
            f"Recommended: FRE_REVIEW_THRESHOLD={ab['review_threshold']} "
            f"FRE_BLOCK_THRESHOLD={ab['block_threshold']} FRE_MODEL_WEIGHT={a.model_weight}"
        )
    print(f"Saved model to {a.out}")
    print(f"Saved drift score reference to {m['score_reference']['path']}")


if __name__ == "__main__":  # pragma: no cover
    main()
