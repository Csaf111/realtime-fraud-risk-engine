"""Regression tests for malformed model artifacts and inference inputs."""

import math

import pytest

from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.model import LogisticBaseline

from .conftest import make_txn


@pytest.mark.parametrize("field", ["mean", "scale", "coef"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_model_vectors(toy_model, field, invalid):
    artifact = toy_model.to_dict()
    artifact[field][0] = invalid
    with pytest.raises(ValueError, match=field):
        LogisticBaseline.from_dict(artifact)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_intercept(toy_model, invalid):
    artifact = toy_model.to_dict()
    artifact["intercept"] = invalid
    with pytest.raises(ValueError, match="intercept"):
        LogisticBaseline.from_dict(artifact)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_inference_features(toy_model, invalid):
    features = compute_features(make_txn())
    features[FEATURE_NAMES[0]] = invalid
    with pytest.raises(ValueError, match="finite"):
        toy_model.predict_proba(features)


def test_valid_model_still_predicts(toy_model):
    features = compute_features(make_txn())
    assert math.isfinite(toy_model.predict_proba(features))
