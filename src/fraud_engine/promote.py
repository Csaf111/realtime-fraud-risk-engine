"""Offline promotion gate: does a candidate model beat the live one on unseen labelled traffic?

Shadow metrics (``fre_shadow_*``) say how often a candidate *disagrees* with the live model;
they cannot say which one is right. Once delayed labels have arrived, the feedback log
(``FRE_FEEDBACK_LOG_PATH``) can: it holds the exact feature vectors the engine served, each
joined to its eventual label. This module replays that log through both artifacts and
returns a pass/fail verdict with reasons, so promotion is a reviewed, reproducible decision
rather than a judgement call on a dashboard.

Examples:
    # After `train --feedback ... --out models/candidate.json` and a shadow period
    python -m fraud_engine.promote --live models/baseline.json \
        --candidate models/candidate.json --feedback data/feedback.jsonl

    # Allow a small PR-AUC drop (e.g. a simpler model), write the report for the PR
    python -m fraud_engine.promote --live models/baseline.json \
        --candidate models/candidate.json --feedback data/feedback.jsonl \
        --max-pr-auc-drop 0.01 --report reports/promotion.json

Exit code 0 means the candidate passed, 1 means it failed (or the evidence was too thin
to decide), 2 is a usage error.

Leakage guarantees:

* Both models score the *served* feature vectors only; labels, transaction ids, ``step``,
  decisions and logged scores never reach either model.
* The evaluation window must lie strictly after every step a model could have been fitted
  on. For an artifact trained with ``train --feedback`` that is its split's
  ``cutoff_step + gap_steps`` (read from the artifact metadata); an explicit ``--after-step``
  inside that range raises :class:`~fraud_engine.preprocessing.LeakageError` instead of
  quietly grading a model on its own training data.
* A window without enough labelled fraud is rejected rather than "passed": no evidence is
  not evidence of no regression.

Scores are model probabilities (no rules blend): the feedback log has no raw transaction, so
rules cannot be re-run. This keeps the comparison apples-to-apples with ``train --feedback``.
Offline tooling only; it needs the ``train`` extra, never the serving image.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ScoringModel, load_model
from .features import FEATURE_NAMES
from .preprocessing import LABEL, LeakageError
from .train import SERVED_SCORE, _metrics, feedback_frame

EXIT_PASS = 0
EXIT_FAIL = 1


@dataclass(frozen=True)
class GateCriteria:
    """How much worse than live a candidate may be and still be promoted."""

    threshold: float = 0.5  # probability cut-off for precision/recall
    max_pr_auc_drop: float = 0.0  # candidate PR-AUC >= live PR-AUC - this
    max_recall_drop: float = 0.0  # candidate recall >= live recall - this
    min_positives: int = 5  # labelled frauds required in the window to decide at all

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not (0.0 <= self.max_pr_auc_drop <= 1.0 and 0.0 <= self.max_recall_drop <= 1.0):
            raise ValueError("max_pr_auc_drop and max_recall_drop must be between 0 and 1")
        if self.min_positives < 1:
            raise ValueError("min_positives must be at least 1")


def training_horizon(model: ScoringModel) -> int | None:
    """Last feedback-log step a model may have been fitted on (embargo included).

    Returns ``None`` for artifacts not trained from the feedback log (e.g. PaySim-trained
    models, whose steps live on a different clock).
    """
    metrics = model.metadata.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("source") != "feedback":
        return None
    split = metrics.get("split")
    if not isinstance(split, Mapping) or "cutoff_step" not in split:
        raise ValueError(
            f"model {model.version!r} was trained from feedback but records no split; "
            "cannot prove the evaluation window is unseen"
        )
    return int(split["cutoff_step"]) + int(split.get("gap_steps", 0))


def _scores(model: ScoringModel, rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([model.predict_proba(r) for r in rows], dtype=float)


def compare_on_feedback(
    live: ScoringModel,
    candidate: ScoringModel,
    feedback_path: str | Path,
    criteria: GateCriteria | None = None,
    after_step: int | None = None,
) -> dict[str, Any]:
    """Score ``live`` and ``candidate`` on the unseen tail of a feedback log and gate.

    The window is every (deduplicated) record with ``step > after_step``. ``after_step``
    defaults to the later of the two models' :func:`training_horizon` (the whole log if
    neither was trained from feedback) and may not be set below it.
    """
    criteria = criteria or GateCriteria()
    horizons = [h for h in (training_horizon(live), training_horizon(candidate)) if h is not None]
    floor = max(horizons) if horizons else None
    if after_step is None:
        after_step = floor if floor is not None else -1
    elif floor is not None and after_step < floor:
        raise LeakageError(
            f"after_step={after_step} overlaps a model's training window (ends at step {floor})"
        )

    frame = feedback_frame(feedback_path)
    window = frame[frame["step"] > after_step]
    positives = int(window[LABEL].sum()) if len(window) else 0
    if len(window) == 0 or positives < criteria.min_positives or positives == len(window):
        raise ValueError(
            f"not enough labelled evidence after step {after_step}: {len(window)} records, "
            f"{positives} fraud (need >= {criteria.min_positives} fraud and some legitimate)"
        )

    rows = window[list(FEATURE_NAMES)].to_dict(orient="records")
    y = window[LABEL].to_numpy(dtype=int)
    p_live = _scores(live, rows)
    p_cand = _scores(candidate, rows)
    live_m = _metrics(y, p_live, criteria.threshold)
    cand_m = _metrics(y, p_cand, criteria.threshold)
    served_m = _metrics(y, window[SERVED_SCORE].to_numpy(dtype=float), criteria.threshold)

    reasons: list[str] = []
    if cand_m["pr_auc"] < live_m["pr_auc"] - criteria.max_pr_auc_drop:
        reasons.append(
            f"PR-AUC {cand_m['pr_auc']:.4f} < live {live_m['pr_auc']:.4f} "
            f"- allowed drop {criteria.max_pr_auc_drop}"
        )
    if cand_m["recall"] < live_m["recall"] - criteria.max_recall_drop:
        reasons.append(
            f"recall@{criteria.threshold} {cand_m['recall']:.4f} < live {live_m['recall']:.4f} "
            f"- allowed drop {criteria.max_recall_drop}"
        )

    flagged_live = p_live >= criteria.threshold
    flagged_cand = p_cand >= criteria.threshold
    return {
        "passed": not reasons,
        "reasons": reasons,
        "live_version": live.version,
        "candidate_version": candidate.version,
        "window": {
            "after_step": after_step,
            "steps": [int(window["step"].min()), int(window["step"].max())],
            "rows": len(window),
            "positives": positives,
        },
        "criteria": asdict(criteria),
        "live": live_m,
        "candidate": cand_m,
        "served_baseline": served_m,
        "delta": {
            "pr_auc": cand_m["pr_auc"] - live_m["pr_auc"],
            "recall": cand_m["recall"] - live_m["recall"],
            "precision": cand_m["precision"] - live_m["precision"],
        },
        "agreement": float(np.mean(flagged_live == flagged_cand)),
        "newly_caught_fraud": int(np.sum(flagged_cand & ~flagged_live & (y == 1))),
        "newly_missed_fraud": int(np.sum(flagged_live & ~flagged_cand & (y == 1))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", required=True, help="Live model artifact (FRE_MODEL_PATH)")
    ap.add_argument("--candidate", required=True, help="Candidate artifact (shadowed model)")
    ap.add_argument("--feedback", required=True, help="Feedback log (FRE_FEEDBACK_LOG_PATH)")
    ap.add_argument(
        "--after-step",
        type=int,
        default=None,
        help="Evaluate records with step > this (default: end of the models' training window)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-pr-auc-drop", type=float, default=0.0)
    ap.add_argument("--max-recall-drop", type=float, default=0.0)
    ap.add_argument("--min-positives", type=int, default=5)
    ap.add_argument("--report", default=None, help="Also write the JSON report here")
    a = ap.parse_args(argv)
    try:
        criteria = GateCriteria(
            threshold=a.threshold,
            max_pr_auc_drop=a.max_pr_auc_drop,
            max_recall_drop=a.max_recall_drop,
            min_positives=a.min_positives,
        )
    except ValueError as exc:
        ap.error(str(exc))
    live = load_model(a.live)
    candidate = load_model(a.candidate)
    try:
        report = compare_on_feedback(live, candidate, a.feedback, criteria, a.after_step)
    except LeakageError as exc:  # grading a model on its own training data is a usage error
        ap.error(str(exc))
    except ValueError as exc:  # thin evidence: refuse to promote, but say why
        print(f"BLOCKED: {exc}")
        return EXIT_FAIL
    text = json.dumps(report, indent=2)
    print(text)
    if a.report:
        out = Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    if report["passed"]:
        print(f"PASS: promote {candidate.version} (FRE_MODEL_PATH={a.candidate})")
        return EXIT_PASS
    print("FAIL: " + "; ".join(report["reasons"]))
    return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
