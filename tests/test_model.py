import json
import math

import pytest

from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.model import LogisticBaseline, sigmoid

from .conftest import fraud_txn, make_txn


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(0) == 0.5


def test_contributions_sum_to_logit(toy_model):
    f = compute_features(fraud_txn())
    contribs = toy_model.contributions(f)
    p = toy_model.predict_proba(f)
    assert math.log(p / (1 - p)) == pytest.approx(toy_model.intercept + sum(contribs.values()))


def test_fraud_scores_higher_than_benign(toy_model):
    assert toy_model.predict_proba(compute_features(fraud_txn())) > toy_model.predict_proba(
        compute_features(make_txn())
    )


def test_explain_returns_positive_drivers(toy_model):
    p, top = toy_model.explain(compute_features(fraud_txn()))
    names = [t[0] for t in top]
    assert names[0] == "orig_drained"
    assert all(c > 0 for _, _, c in top)
    assert 0 < p < 1


def test_roundtrip(tmp_path, toy_model):
    path = tmp_path / "m.json"
    toy_model.save(path)
    loaded = LogisticBaseline.load(path)
    f = compute_features(fraud_txn())
    assert loaded.predict_proba(f) == pytest.approx(toy_model.predict_proba(f))
    assert loaded.version == "toy"


def test_rejects_mismatched_features():
    n = len(FEATURE_NAMES)
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        LogisticBaseline(list(reversed(FEATURE_NAMES)), [0] * n, [1] * n, [0] * n, 0.0)


def test_rejects_wrong_lengths():
    with pytest.raises(ValueError):
        LogisticBaseline(list(FEATURE_NAMES), [0.0], [1.0], [0.0], 0.0)


def test_rejects_unknown_schema(tmp_path, toy_model):
    d = toy_model.to_dict()
    d["schema_version"] = 99
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema_version"):
        LogisticBaseline.load(p)


def test_zero_scale_is_guarded():
    n = len(FEATURE_NAMES)
    m = LogisticBaseline(list(FEATURE_NAMES), [0.0] * n, [0.0] * n, [0.0] * n, 0.0)
    assert all(s == 1.0 for s in m.scale)
