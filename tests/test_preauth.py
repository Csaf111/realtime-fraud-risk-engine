"""Pre-authorisation contract and scoring regression tests."""
import pytest
from pydantic import ValidationError

from fraud_engine.preauth import PreAuthTransaction, score_preauth
from fraud_engine.schemas import Decision
from fraud_engine.state import VelocitySnapshot


def transaction(**overrides):
    data = dict(
        transaction_id="txn-1", step=10, type="TRANSFER", amount=100,
        nameOrig="C1", oldbalanceOrg=1000, nameDest="C2",
    )
    data.update(overrides)
    return PreAuthTransaction(**data)


@pytest.mark.parametrize("field,value", [
    ("newbalanceOrig", 0), ("newbalanceDest", 0),
    ("oldbalanceDest", 0), ("isFraud", 1), ("isFlaggedFraud", 0),
])
def test_rejects_post_event_and_label_fields(field, value):
    with pytest.raises(ValidationError):
        transaction(**{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_amount(value):
    with pytest.raises(ValidationError):
        transaction(amount=value)


def test_rule_scoring_is_deterministic_and_explainable():
    txn = transaction(amount=1000, oldbalanceOrg=1000)
    snapshot = VelocitySnapshot()
    a = score_preauth(txn, snapshot)
    b = score_preauth(txn, snapshot)
    assert a.risk_score == b.risk_score
    assert a.decision == Decision.REVIEW
    assert [h.code for h in a.rule_hits] == ["P001", "P004"]
    assert a.model_score is None
    assert a.model_version == "preauth-rules-v2.0"


def test_prior_velocity_only():
    base = score_preauth(transaction(), VelocitySnapshot())
    frequent = score_preauth(transaction(), VelocitySnapshot(orig_txn_count=4))
    assert frequent.risk_score > base.risk_score
    assert "P005" in [h.code for h in frequent.rule_hits]


def test_overspend_blocks_at_default_threshold_with_other_signals():
    txn = transaction(amount=250000, oldbalanceOrg=100)
    result = score_preauth(txn, VelocitySnapshot(orig_txn_count=3))
    assert result.decision == Decision.BLOCK
    assert result.risk_score == 0.9

def test_v2_api_rejects_post_event_fields_and_scores_valid_payload():
    from fastapi.testclient import TestClient
    from fraud_engine.api import create_app

    with TestClient(create_app()) as client:
        payload = transaction().model_dump(mode="json")
        response = client.post("/v2/score", json=payload)
        assert response.status_code == 200
        assert response.json()["model_version"] == "preauth-rules-v2.0"
        assert response.json()["model_score"] is None

        payload["newbalanceOrig"] = 0
        rejected = client.post("/v2/score", json=payload)
        assert rejected.status_code == 422
