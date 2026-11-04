import math

import pytest

from fraud_engine.config import Settings
from fraud_engine.thresholds import (
    alert_rate,
    blend_risk,
    evaluate_thresholds,
    select_thresholds,
    threshold_for_alert_rate,
)
from fraud_engine.train import main, train


def test_blend_matches_engine_formula():
    assert blend_risk(0.5, 1.0, 0.6) == pytest.approx(0.7)
    assert blend_risk(2.0, 2.0, 0.5) == 1.0
    assert blend_risk(-1.0, 0.0, 0.5) == 0.0
    assert blend_risk(1 / 3, 0.0, 1.0) == 0.333333


def test_threshold_respects_budget_on_distinct_scores():
    scores = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    t = threshold_for_alert_rate(scores, 0.05)
    assert t == pytest.approx(0.95)
    assert alert_rate(scores, t) == pytest.approx(0.05)


def test_threshold_is_lowest_within_budget():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = threshold_for_alert_rate(scores, 0.35)
    assert t == pytest.approx(0.8)  # 3 alerts; 0.7 would give 4 > 3.5


def test_ties_are_all_or_nothing():
    scores = [0.1] * 6 + [0.9] * 4
    # 0.9 tie group is 40% of traffic: a 30% budget cannot include it.
    t = threshold_for_alert_rate(scores, 0.3)
    assert t > 0.9
    assert alert_rate(scores, t) == 0.0
    assert threshold_for_alert_rate(scores, 0.4) == pytest.approx(0.9)


def test_zero_and_full_budget():
    scores = [0.2, 0.4, 0.6]
    assert alert_rate(scores, threshold_for_alert_rate(scores, 0.0)) == 0.0
    assert threshold_for_alert_rate(scores, 1.0) == pytest.approx(0.2)


def test_threshold_capped_at_one_when_scores_saturate():
    scores = [1.0] * 5 + [0.0] * 5
    t = threshold_for_alert_rate(scores, 0.1)
    assert t == 1.0
    sel = select_thresholds(scores, 0.2, 0.1)
    assert sel.budget_met is False


@pytest.mark.parametrize("rate", [-0.1, 1.5, math.nan])
def test_invalid_rate_rejected(rate):
    with pytest.raises(ValueError):
        threshold_for_alert_rate([0.1], rate)


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        threshold_for_alert_rate([], 0.1)
    with pytest.raises(ValueError):
        threshold_for_alert_rate([0.1, math.inf], 0.1)
    with pytest.raises(ValueError):
        alert_rate([], 0.5)


def test_select_thresholds_ordering_and_settings_compatible():
    scores = [i / 1000 for i in range(1000)]
    sel = select_thresholds(scores, review_budget=0.02, block_budget=0.005)
    assert sel.review_threshold <= sel.block_threshold <= 1.0
    assert sel.validation_review_rate <= 0.02
    assert sel.validation_block_rate <= 0.005
    assert sel.budget_met
    Settings(review_threshold=sel.review_threshold, block_threshold=sel.block_threshold)
    assert set(sel.as_dict()) >= {"review_threshold", "block_threshold", "budget_met"}


def test_select_thresholds_rejects_block_above_review():
    with pytest.raises(ValueError):
        select_thresholds([0.1, 0.2], review_budget=0.01, block_budget=0.02)


def test_evaluate_thresholds():
    scores = [0.1, 0.2, 0.6, 0.7, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 0]
    out = evaluate_thresholds(scores, labels, review_threshold=0.6, block_threshold=0.9)
    assert out["review_alert_rate"] == pytest.approx(4 / 6)
    assert out["review_precision"] == pytest.approx(0.5)
    assert out["review_recall"] == pytest.approx(1.0)
    assert out["block_alert_rate"] == pytest.approx(2 / 6)
    assert out["block_precision"] == pytest.approx(0.5)
    assert out["block_recall"] == pytest.approx(0.5)


def test_evaluate_thresholds_edge_cases():
    out = evaluate_thresholds([0.1, 0.2], [0, 0], 0.5, 0.9)
    assert out["review_precision"] == 0.0 and out["review_recall"] == 0.0
    with pytest.raises(ValueError):
        evaluate_thresholds([0.1], [0, 1], 0.5, 0.9)
    with pytest.raises(ValueError):
        evaluate_thresholds([], [], 0.5, 0.9)


def test_train_rejects_half_budget(tmp_path):
    with pytest.raises(ValueError, match="together"):
        train(tmp_path / "x.csv", tmp_path / "m.json", review_budget=0.01)
    with pytest.raises(ValueError, match="validation_frac"):
        train(tmp_path / "x.csv", tmp_path / "m.json", validation_frac=1.0)


def test_cli_rejects_half_budget(tmp_path):
    with pytest.raises(SystemExit):
        main(["--synthetic", "100", "--out", str(tmp_path / "m.json"), "--review-budget", "0.01"])


@pytest.mark.slow
def test_train_with_alert_budget(tmp_path):
    from fraud_engine.synthetic import write_csv

    data = write_csv(tmp_path / "s.csv", n=4000, fraud_rate=0.03)
    out = tmp_path / "model.json"
    metrics = train(data, out, gap_steps=5, version="ab", review_budget=0.05, block_budget=0.01)
    ab = metrics["alert_budget"]
    assert ab["review_threshold"] <= ab["block_threshold"] <= 1.0
    assert ab["validation_review_rate"] <= 0.05 + 1e-9
    assert ab["validation_block_rate"] <= 0.01 + 1e-9
    # Validation window lies strictly inside the training period, before the test window.
    assert ab["validation_steps"][1] <= metrics["split"]["train_max_step"]
    assert ab["validation_steps"][1] < metrics["split"]["test_min_step"]
    assert set(ab["test"]) >= {"review_alert_rate", "review_recall", "block_precision"}


@pytest.mark.slow
def test_cli_prints_recommended_thresholds(tmp_path, capsys):
    out = tmp_path / "m.json"
    main(
        [
            "--synthetic",
            "3000",
            "--out",
            str(out),
            "--gap-steps",
            "5",
            "--review-budget",
            "0.05",
            "--block-budget",
            "0.01",
        ]
    )
    printed = capsys.readouterr().out
    assert "FRE_REVIEW_THRESHOLD=" in printed and "FRE_BLOCK_THRESHOLD=" in printed
