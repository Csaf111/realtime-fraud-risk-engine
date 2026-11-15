"""Rules as a hard score floor under the model blend (``FRE_RULES_FLOOR``)."""

from __future__ import annotations

import json
import math

import pytest

from fraud_engine.config import Settings
from fraud_engine.drift import ScoreReference
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.model import LogisticBaseline
from fraud_engine.scoring import RiskEngine
from fraud_engine.synthetic import write_csv
from fraud_engine.thresholds import blend_risk, rules_floor_binds
from fraud_engine.train import main, train

from .conftest import fraud_txn, make_txn


def _constant_model(intercept: float, version: str = "const") -> LogisticBaseline:
    n = len(FEATURE_NAMES)
    return LogisticBaseline(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=[0.0] * n,
        intercept=intercept,
        version=version,
    )


# A model that is (wrongly) near-certain every transaction is legitimate.
CONFIDENT_LEGIT = _constant_model(-20.0, "legit")


def test_default_floor_is_plain_blend():
    assert blend_risk(0.0, 0.9, 0.6) == blend_risk(0.0, 0.9, 0.6, 0.0) == 0.36


@pytest.mark.parametrize(
    ("model", "rules", "weight", "floor", "expected"),
    [
        (0.0, 0.9, 0.6, 1.0, 0.9),  # floor binds: the model cannot pull below the rules
        (0.0, 0.9, 0.6, 0.5, 0.45),  # partial floor
        (1.0, 0.2, 0.6, 1.0, 0.68),  # blend above floor: floor is inert
        (0.0, 0.0, 0.6, 1.0, 0.0),  # no rules fired: nothing to floor
        (0.0, 0.9, 1.0, 1.0, 0.9),  # pure model weight still floored
        (2.0, 2.0, 0.5, 1.0, 1.0),  # still clipped to [0, 1]
    ],
)
def test_blend_risk_with_floor(model, rules, weight, floor, expected):
    assert blend_risk(model, rules, weight, floor) == pytest.approx(expected)


def test_floor_is_monotone_and_never_lowers_scores():
    grid = [i / 10 for i in range(11)]
    for m in grid:
        for r in grid:
            plain = blend_risk(m, r, 0.6)
            prev = plain
            for f in grid:
                cur = blend_risk(m, r, 0.6, f)
                assert cur >= prev >= plain
                assert cur >= round(f * r, 6) - 1e-12
                prev = cur


def test_rules_floor_binds():
    assert rules_floor_binds(0.0, 0.9, 0.6, 1.0)
    assert not rules_floor_binds(0.0, 0.9, 0.6, 0.0)
    assert not rules_floor_binds(1.0, 0.2, 0.6, 1.0)
    assert not rules_floor_binds(0.0, 0.0, 0.6, 1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_settings_reject_invalid_floor(bad):
    with pytest.raises(ValueError, match="rules_floor"):
        Settings(rules_floor=bad)


def test_settings_from_env(monkeypatch):
    assert Settings.from_env().rules_floor == 0.0
    monkeypatch.setenv("FRE_RULES_FLOOR", "0.75")
    assert Settings.from_env().rules_floor == 0.75


def test_confident_wrong_model_cannot_approve_rule_flagged_fraud():
    txn = fraud_txn()
    plain = RiskEngine(Settings(model_weight=0.8), model=CONFIDENT_LEGIT).score(dict(txn))
    floored = RiskEngine(Settings(model_weight=0.8, rules_floor=1.0), model=CONFIDENT_LEGIT).score(
        dict(txn)
    )
    assert plain.rules_score == floored.rules_score >= 0.8
    assert plain.decision.value == "APPROVE"  # the model alone overrides the rules
    assert floored.risk_score == pytest.approx(floored.rules_score)
    assert floored.decision.value == "BLOCK"
    assert any(r.startswith("FLOOR:") for r in floored.reasons)
    assert not any(r.startswith("FLOOR:") for r in plain.reasons)


def test_floor_inert_when_blend_is_higher():
    eng = RiskEngine(Settings(rules_floor=1.0), model=_constant_model(20.0, "fraudy"))
    r = eng.score(make_txn(type="TRANSFER"))
    assert r.risk_score >= r.rules_score
    assert not any(x.startswith("FLOOR:") for x in r.reasons)


def test_rules_only_mode_unaffected_by_floor():
    base = RiskEngine().score(fraud_txn())
    floored = RiskEngine(Settings(rules_floor=1.0)).score(fraud_txn())
    assert base.risk_score == floored.risk_score
    assert floored.reasons == base.reasons


def test_shadow_uses_same_floor():
    eng = RiskEngine(
        Settings(model_weight=0.8, rules_floor=1.0),
        model=CONFIDENT_LEGIT,
        shadow_model=_constant_model(-20.0, "legit-shadow"),
    )
    result, shadow = eng.score_with_shadow(fraud_txn())
    assert shadow is not None
    assert shadow.risk_score == pytest.approx(result.risk_score)
    assert shadow.agrees


def test_train_rejects_invalid_floor(tmp_path):
    with pytest.raises(ValueError, match="rules_floor"):
        train(tmp_path / "missing.csv", tmp_path / "m.json", rules_floor=1.5)


@pytest.mark.slow
def test_train_threads_floor_into_thresholds_and_reference(tmp_path, capsys):
    data = write_csv(tmp_path / "s.csv", n=3000, fraud_rate=0.03)
    out = tmp_path / "m.json"
    main(
        [
            "--data",
            str(data),
            "--out",
            str(out),
            "--gap-steps",
            "5",
            "--version",
            "floor",
            "--review-budget",
            "0.05",
            "--block-budget",
            "0.01",
            "--rules-floor",
            "1.0",
        ]
    )
    printed = capsys.readouterr().out
    assert "FRE_RULES_FLOOR=1.0" in printed
    metrics = json.loads(out.with_suffix(".metrics.json").read_text())
    assert metrics["alert_budget"]["rules_floor"] == 1.0
    assert metrics["score_reference"]["rules_floor"] == 1.0
    ref = ScoreReference.load(out.with_suffix(".score_reference.json"))
    assert ref.rules_floor == 1.0
