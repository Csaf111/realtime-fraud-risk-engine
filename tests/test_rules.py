from fraud_engine.rules import DEFAULT_RULES, RuleContext, evaluate_rules
from fraud_engine.state import VelocitySnapshot

from .conftest import fraud_txn, make_txn

EMPTY = VelocitySnapshot()


def codes(txn, snap=EMPTY, ctx=None):
    return {r.code for r in evaluate_rules(txn, snap, ctx)[1]}


def test_rule_codes_unique():
    assert len({r.code for r in DEFAULT_RULES}) == len(DEFAULT_RULES)


def test_benign_payment_fires_nothing():
    score, hits = evaluate_rules(make_txn(), EMPTY)
    assert score == 0.0 and hits == []


def test_account_takeover_pattern():
    c = codes(fraud_txn())
    assert {"R001", "R002", "R003", "R005"} <= c


def test_score_is_capped_at_one():
    snap = VelocitySnapshot(orig_txn_count=10, dest_inbound_count=10)
    txn = fraud_txn(amount=500_000.0, oldbalanceOrg=250_000.0)  # also exceeds balance
    score, _ = evaluate_rules(txn, snap)
    assert score == 1.0


def test_hits_sorted_by_points():
    _, hits = evaluate_rules(fraud_txn(), EMPTY)
    pts = [h.points for h in hits]
    assert pts == sorted(pts, reverse=True)


def test_high_amount_threshold_configurable():
    txn = make_txn(amount=500.0, oldbalanceOrg=1000.0, newbalanceOrig=500.0)
    assert "R003" in codes(txn, ctx=RuleContext(high_amount_threshold=400))
    assert "R003" not in codes(txn)


def test_inconsistent_origin_balance():
    assert "R004" in codes(make_txn(newbalanceOrig=1000.0))


def test_exceeds_balance_not_for_cash_in():
    assert "R006" in codes(make_txn(amount=5000.0, oldbalanceOrg=10.0, newbalanceOrig=0.0))
    assert "R006" not in codes(
        make_txn(type="CASH_IN", amount=5000.0, oldbalanceOrg=10.0, newbalanceOrig=5010.0)
    )


def test_velocity_rules():
    snap = VelocitySnapshot(orig_txn_count=3, dest_inbound_count=5)
    assert {"R007", "R008"} <= codes(make_txn(), snap)


def test_dest_unchanged_ignores_merchants():
    assert "R005" not in codes(make_txn())
