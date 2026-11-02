import math

import pytest

from fraud_engine.features import (
    FEATURE_NAMES,
    compute_features,
    dest_balance_error,
    is_merchant,
    orig_balance_error,
    signed_log1p,
    to_vector,
)
from fraud_engine.state import VelocitySnapshot

from .conftest import fraud_txn, make_txn


def test_feature_names_and_order_are_stable():
    feats = compute_features(make_txn())
    assert tuple(feats) == FEATURE_NAMES
    assert len(to_vector(feats)) == len(FEATURE_NAMES)


def test_consistent_payment_has_zero_balance_error():
    assert orig_balance_error(make_txn()) == 0.0


def test_cash_in_credits_origin():
    txn = make_txn(type="CASH_IN", oldbalanceOrg=100.0, amount=50.0, newbalanceOrig=150.0)
    assert orig_balance_error(txn) == 0.0


def test_dest_balance_error_detects_missing_credit():
    txn = fraud_txn()
    assert dest_balance_error(txn) == pytest.approx(250_000.0)


def test_drained_and_zero_dest_flags():
    f = compute_features(fraud_txn())
    assert f["orig_drained"] == 1.0
    assert f["dest_zero_balances"] == 1.0
    assert f["type_TRANSFER"] == 1.0
    assert f["dest_is_merchant"] == 0.0


def test_benign_flags_off():
    f = compute_features(make_txn())
    assert f["orig_drained"] == 0.0
    assert f["dest_zero_balances"] == 0.0  # merchants are excluded
    assert f["dest_is_merchant"] == 1.0


def test_ratio_is_capped():
    f = compute_features(make_txn(amount=1e9, oldbalanceOrg=0.0, newbalanceOrig=0.0))
    assert f["amount_to_orig_balance"] == 10.0


def test_hour_encoding_is_cyclic():
    a = compute_features(make_txn(step=0))
    b = compute_features(make_txn(step=24))
    assert a["hour_sin"] == pytest.approx(b["hour_sin"])
    assert a["hour_cos"] == pytest.approx(1.0)


def test_velocity_snapshot_is_used():
    snap = VelocitySnapshot(
        orig_txn_count=3, orig_amount_sum=10.0, dest_inbound_count=7, dest_inbound_sum=0.0
    )
    f = compute_features(make_txn(), snap)
    assert f["orig_txn_count_window"] == 3
    assert f["dest_inbound_count_window"] == 7
    assert f["log_orig_amount_window"] == pytest.approx(math.log1p(10.0))


def test_signed_log_keeps_sign():
    assert signed_log1p(-5.0) == pytest.approx(-math.log1p(5.0))
    assert signed_log1p(0.0) == 0.0


def test_merchant_prefix():
    assert is_merchant("M123")
    assert not is_merchant("C123")
