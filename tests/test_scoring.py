import pytest

from fraud_engine.config import Settings
from fraud_engine.schemas import Decision, Transaction
from fraud_engine.scoring import RiskEngine

from .conftest import fraud_txn, make_txn


def test_rules_only_mode():
    eng = RiskEngine()
    assert eng.mode == "rules-only"
    r = eng.score(make_txn())
    assert r.decision == Decision.APPROVE
    assert r.model_score is None
    assert r.reasons == []


def test_fraud_is_blocked_with_reasons():
    r = RiskEngine().score(fraud_txn())
    assert r.decision == Decision.BLOCK
    assert r.risk_score >= 0.8
    assert any(reason.startswith("R002") for reason in r.reasons)


def test_accepts_pydantic_transaction():
    r = RiskEngine().score(Transaction(**fraud_txn(transaction_id="t-1")))
    assert r.transaction_id == "t-1"


def test_thresholds_drive_decision():
    eng = RiskEngine(Settings(review_threshold=0.1, block_threshold=0.9))
    assert eng.decide(0.05) == Decision.APPROVE
    assert eng.decide(0.1) == Decision.REVIEW
    assert eng.decide(0.95) == Decision.BLOCK


def test_model_blend(toy_model):
    eng = RiskEngine(Settings(model_weight=0.5), model=toy_model)
    r = eng.score(fraud_txn())
    assert eng.mode == "rules+model"
    assert r.model_score is not None
    assert r.risk_score == pytest.approx(0.5 * r.model_score + 0.5 * r.rules_score, abs=1e-6)
    assert r.top_features[0].feature == "orig_drained"
    assert "lr-toy" in r.model_version
    assert any(reason.startswith("MODEL:") for reason in r.reasons)


def test_velocity_builds_up_across_calls():
    eng = RiskEngine()
    for i in range(5):
        eng.score(
            make_txn(
                nameOrig=f"C{i}",
                nameDest="CMULE",
                type="TRANSFER",
                oldbalanceDest=0.0,
                newbalanceDest=100.0,
            )
        )
    r = eng.score(make_txn(nameOrig="Cnew", nameDest="CMULE"))
    assert "R007" in {h.code for h in r.rule_hits}


def test_update_state_false_does_not_record():
    eng = RiskEngine()
    eng.score(make_txn(), update_state=False)
    assert len(eng.store) == 0


def test_loads_model_from_path(tmp_path, toy_model):
    p = tmp_path / "m.json"
    toy_model.save(p)
    eng = RiskEngine(Settings(model_path=str(p)))
    assert eng.model is not None and eng.model.version == "toy"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(review_threshold=0.9, block_threshold=0.5),
        dict(model_weight=2.0),
        dict(velocity_window_steps=0),
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FRE_REVIEW_THRESHOLD", "0.3")
    monkeypatch.setenv("FRE_BLOCK_THRESHOLD", "0.7")
    monkeypatch.setenv("FRE_ENVIRONMENT", "staging")
    s = Settings.from_env()
    assert (s.review_threshold, s.block_threshold, s.environment) == (0.3, 0.7, "staging")
    assert s.model_path is None
