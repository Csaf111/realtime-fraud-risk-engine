"""Gradient-boosted trees: pure-Python inference, exact TreeSHAP, artifact validation."""

import itertools
import json
import math
import random

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from fraud_engine.artifacts import load_model, model_from_dict
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.model import LogisticBaseline, sigmoid
from fraud_engine.scoring import RiskEngine
from fraud_engine.trees import MAX_TREE_DEPTH, GradientBoostedTrees, RegressionTree

from .conftest import fraud_txn, make_txn

N = len(FEATURE_NAMES)
DRAINED = FEATURE_NAMES.index("orig_drained")
TRANSFER = FEATURE_NAMES.index("type_TRANSFER")
LOG_AMOUNT = FEATURE_NAMES.index("log_amount")


def _toy_tree() -> RegressionTree:
    """Depth-2 tree: drained? -> (transfer? | log_amount > 10?)."""
    #        0: drained <= 0.5
    #     1: transfer<=0.5      2: log_amount<=10
    #   3:-1.0   4:0.5        5:0.8   6:2.0
    return RegressionTree(
        feature=[DRAINED, TRANSFER, LOG_AMOUNT, -1, -1, -1, -1],
        threshold=[0.5, 0.5, 10.0, 0.0, 0.0, 0.0, 0.0],
        left=[1, 3, 5, -1, -1, -1, -1],
        right=[2, 4, 6, -1, -1, -1, -1],
        value=[0.0, 0.0, 0.0, -1.0, 0.5, 0.8, 2.0],
        cover=[100.0, 80.0, 20.0, 60.0, 20.0, 5.0, 15.0],
    )


def _repeat_feature_tree() -> RegressionTree:
    """The same feature split twice on one path exercises TreeSHAP's unwind step."""
    return RegressionTree(
        feature=[LOG_AMOUNT, LOG_AMOUNT, DRAINED, -1, -1, -1, -1],
        threshold=[10.0, 5.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        left=[1, 3, 5, -1, -1, -1, -1],
        right=[2, 4, 6, -1, -1, -1, -1],
        value=[0.0, 0.0, 0.0, -0.7, 0.1, 0.3, 1.9],
        cover=[50.0, 30.0, 20.0, 10.0, 20.0, 12.0, 8.0],
    )


@pytest.fixture
def toy_gbt() -> GradientBoostedTrees:
    return GradientBoostedTrees(
        feature_names=list(FEATURE_NAMES),
        base_score=-1.5,
        trees=[_toy_tree(), _repeat_feature_tree()],
        version="toy-gbt",
    )


def _cond_expectation(tree: RegressionTree, x: list[float], known: set[int], j: int = 0) -> float:
    """E[f(x) | x_S] under the path-dependent (cover-weighted) definition."""
    if tree.left[j] == -1:
        return tree.value[j]
    f = tree.feature[j]
    lo, hi = tree.left[j], tree.right[j]
    if f in known:
        return _cond_expectation(tree, x, known, lo if x[f] <= tree.threshold[j] else hi)
    return (
        tree.cover[lo] * _cond_expectation(tree, x, known, lo)
        + tree.cover[hi] * _cond_expectation(tree, x, known, hi)
    ) / tree.cover[j]


def _brute_force_shapley(tree: RegressionTree, x: list[float]) -> list[float]:
    used = sorted({f for f, lo in zip(tree.feature, tree.left, strict=True) if lo != -1})
    m = len(used)
    phi = [0.0] * N
    for i in used:
        others = [f for f in used if f != i]
        for r in range(m):
            weight = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
            for subset in itertools.combinations(others, r):
                s = set(subset)
                gain = _cond_expectation(tree, x, s | {i}) - _cond_expectation(tree, x, s)
                phi[i] += weight * gain
    return phi


def _random_x(rng: random.Random) -> list[float]:
    x = [0.0] * N
    x[DRAINED] = rng.choice([0.0, 1.0])
    x[TRANSFER] = rng.choice([0.0, 1.0])
    x[LOG_AMOUNT] = rng.uniform(0.0, 15.0)
    return x


@pytest.mark.parametrize("make_tree", [_toy_tree, _repeat_feature_tree])
def test_tree_shap_matches_brute_force_shapley(make_tree):
    tree = make_tree()
    rng = random.Random(7)
    for _ in range(25):
        x = _random_x(rng)
        phi = [0.0] * N
        tree.shap_into(x, phi)
        expected = _brute_force_shapley(tree, x)
        assert phi == pytest.approx(expected, abs=1e-12)
        # Local accuracy: contributions sum to prediction minus the expected value.
        assert sum(phi) == pytest.approx(tree.predict(x) - tree.expected_value, abs=1e-12)


def test_expected_value_is_cover_weighted_leaf_mean():
    tree = _toy_tree()
    want = (60 * -1.0 + 20 * 0.5 + 5 * 0.8 + 15 * 2.0) / 100
    assert tree.expected_value == pytest.approx(want)
    assert tree.depth == 2


def test_contributions_sum_to_log_odds(toy_gbt):
    for txn in (make_txn(), fraud_txn()):
        features = compute_features(txn)
        contribs = toy_gbt.contributions(features)
        total = toy_gbt.expected_log_odds + sum(contribs.values())
        assert total == pytest.approx(toy_gbt.log_odds(features), abs=1e-12)
        assert toy_gbt.predict_proba(features) == pytest.approx(sigmoid(total))


def test_explain_ranks_risk_raising_features(toy_gbt):
    features = compute_features(fraud_txn())
    p, top = toy_gbt.explain(features, top_k=2)
    assert 0.0 < p < 1.0
    assert len(top) <= 2
    assert all(c > 0 for _, _, c in top)
    assert [c for _, _, c in top] == sorted((c for _, _, c in top), reverse=True)
    assert top[0][0] == "orig_drained"


def test_round_trip_and_loader_dispatch(toy_gbt, toy_model, tmp_path):
    path = tmp_path / "gbt.json"
    toy_gbt.save(path)
    loaded = load_model(path)
    assert isinstance(loaded, GradientBoostedTrees)
    assert loaded.to_dict() == toy_gbt.to_dict()
    features = compute_features(fraud_txn())
    assert loaded.predict_proba(features) == toy_gbt.predict_proba(features)

    lr_path = tmp_path / "lr.json"
    toy_model.save(lr_path)
    assert isinstance(load_model(lr_path), LogisticBaseline)
    legacy = toy_model.to_dict()
    del legacy["model_type"]  # artifacts predating model_type are logistic baselines
    assert isinstance(model_from_dict(legacy), LogisticBaseline)


def test_loader_rejects_unknown_model_type(toy_gbt):
    artifact = toy_gbt.to_dict()
    artifact["model_type"] = "neural_net"
    with pytest.raises(ValueError, match="model_type"):
        model_from_dict(artifact)
    with pytest.raises(ValueError, match="Not a"):
        GradientBoostedTrees.from_dict({**artifact})


def _tree_dict() -> dict:
    return _toy_tree().to_dict()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t["cover"].__setitem__(0, 0.0), "cover"),
        (lambda t: t["cover"].__setitem__(0, float("nan")), "cover"),
        (lambda t: t["cover"].__setitem__(0, 99.0), "sum of its children"),
        (lambda t: t["value"].__setitem__(3, float("inf")), "leaf values"),
        (lambda t: t["threshold"].__setitem__(0, float("nan")), "thresholds"),
        (lambda t: t["feature"].__setitem__(0, N), "unknown feature"),
        (lambda t: t["left"].__setitem__(1, 0), "child indices"),  # cycle back to root
        (lambda t: t["right"].__setitem__(0, 1), "child indices"),  # both children equal
        (lambda t: t["left"].__setitem__(0, 99), "child indices"),
        (lambda t: t["value"].pop(), "same length"),
    ],
)
def test_rejects_corrupt_trees(mutate, match):
    t = _tree_dict()
    mutate(t)
    with pytest.raises(ValueError, match=match):
        RegressionTree.from_dict(t)


def test_rejects_empty_and_too_deep_trees():
    with pytest.raises(ValueError, match="at least one node"):
        RegressionTree([], [], [], [], [], [])
    depth = MAX_TREE_DEPTH + 1
    # A left-leaning chain: node 2k splits into leaf 2k+1 and node 2k+2.
    n = 2 * depth + 1
    feature, left, right = [-1] * n, [-1] * n, [-1] * n
    cover = [0.0] * n
    for k in range(depth):
        j = 2 * k
        feature[j], left[j], right[j] = LOG_AMOUNT, j + 1, j + 2
    cover[n - 1] = 1.0
    for j in range(n - 2, -1, -1):
        cover[j] = 1.0 if left[j] == -1 else cover[left[j]] + cover[right[j]]
    with pytest.raises(ValueError, match="depth"):
        RegressionTree(feature, [0.0] * n, left, right, [0.0] * n, cover)


def test_rejects_bad_ensembles(toy_gbt):
    artifact = toy_gbt.to_dict()
    with pytest.raises(ValueError, match="base_score"):
        GradientBoostedTrees.from_dict({**artifact, "base_score": float("nan")})
    with pytest.raises(ValueError, match="at least one tree"):
        GradientBoostedTrees.from_dict({**artifact, "trees": []})
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        GradientBoostedTrees.from_dict({**artifact, "feature_names": ["x"]})
    with pytest.raises(ValueError, match="schema_version"):
        GradientBoostedTrees.from_dict({**artifact, "schema_version": 99})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_rejects_nonfinite_inference_features(toy_gbt, invalid):
    features = compute_features(make_txn())
    features[FEATURE_NAMES[0]] = invalid
    with pytest.raises(ValueError, match="finite"):
        toy_gbt.predict_proba(features)


def test_sklearn_export_matches_decision_function_and_shap_is_additive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, N))
    y = (X[:, DRAINED] + 0.5 * X[:, LOG_AMOUNT] + rng.normal(scale=0.5, size=400)) > 0.8
    clf = GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=0)
    clf.fit(X, y.astype(int), sample_weight=rng.uniform(0.5, 2.0, size=400))
    model = GradientBoostedTrees.from_sklearn(clf, FEATURE_NAMES, X, version="sk")
    # JSON round trip must not lose float precision on thresholds or leaves.
    model = GradientBoostedTrees.from_dict(json.loads(json.dumps(model.to_dict())))
    want = clf.decision_function(X[:50])
    for row, w in zip(X[:50], want, strict=True):
        feats = dict(zip(FEATURE_NAMES, row.tolist(), strict=True))
        assert model.log_odds(feats) == pytest.approx(float(w), abs=1e-9)
        contribs = model.contributions(feats)
        assert model.expected_log_odds + sum(contribs.values()) == pytest.approx(float(w), abs=1e-9)


def test_sklearn_export_rejects_multiclass():
    X = np.random.default_rng(1).normal(size=(60, N))
    y = np.arange(60) % 3
    clf = GradientBoostingClassifier(n_estimators=2, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        GradientBoostedTrees.from_sklearn(clf, FEATURE_NAMES, X)


def test_engine_serves_gbt_with_explanations(toy_gbt, tmp_path):
    path = tmp_path / "gbt.json"
    toy_gbt.save(path)
    eng = RiskEngine(Settings(model_path=str(path)))
    assert eng.model_version.endswith("+gbt-toy-gbt")
    res = eng.score(fraud_txn(), update_state=False)
    assert res.model_score is not None
    assert res.top_features and res.top_features[0].feature == "orig_drained"
    assert any(r.startswith("MODEL: orig_drained") for r in res.reasons)
    assert (
        eng.score(fraud_txn(), update_state=False).risk_score
        > eng.score(make_txn(), update_state=False).risk_score
    )


def test_gbt_can_shadow_a_logistic_primary(toy_gbt, toy_model, tmp_path):
    path = tmp_path / "gbt.json"
    toy_gbt.save(path)
    eng = RiskEngine(Settings(shadow_model_path=str(path)), model=toy_model)
    assert eng.model_version.endswith("+lr-toy")
    assert eng.shadow_model_version is not None and eng.shadow_model_version.endswith(
        "+gbt-toy-gbt"
    )
    _, shadow = eng.score_with_shadow(fraud_txn(), update_state=False)
    assert shadow is not None and not shadow.failed
