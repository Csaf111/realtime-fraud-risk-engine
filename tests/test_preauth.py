"""Regression tests for the independent pre-authorisation feature contract."""
import math

import pytest

from fraud_engine.preauth import PREAUTH_FEATURE_NAMES, compute_preauth_features


@pytest.fixture
def txn():
    return {
        "step": 25,
        "type": "TRANSFER",
        "amount": 200.0,
        "available_balance": 500.0,
        "nameDest": "C123",
    }


def test_order_and_finite_features(txn):
    features = compute_preauth_features(txn)
    assert tuple(features) == PREAUTH_FEATURE_NAMES
    assert all(math.isfinite(value) for value in features.values())


def test_post_event_fields_cannot_affect_features(txn):
    baseline = compute_preauth_features(txn)
    modified = {
        **txn,
        "newbalanceOrig": 0,
        "newbalanceDest": 999999,
        "isFraud": 1,
        "isFlaggedFraud": 1,
    }
    assert compute_preauth_features(modified) == baseline


@pytest.mark.parametrize("field,value", [
    ("amount", float("nan")),
    ("amount", float("inf")),
    ("available_balance", float("-inf")),
    ("available_balance", -1.0),
    ("amount", -1.0),
])
def test_invalid_money_rejected(txn, field, value):
    txn[field] = value
    with pytest.raises(ValueError):
        compute_preauth_features(txn)


def test_missing_available_balance_rejected(txn):
    del txn["available_balance"]
    with pytest.raises(ValueError, match="available_balance"):
        compute_preauth_features(txn)


def test_invalid_step_rejected(txn):
    txn["step"] = -1
    with pytest.raises(ValueError):
        compute_preauth_features(txn)
