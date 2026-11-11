"""Shadow-model scoring: compare a candidate on live traffic without affecting decisions."""

from __future__ import annotations

import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.metrics import Metrics
from fraud_engine.model import LogisticBaseline
from fraud_engine.schemas import Decision
from fraud_engine.scoring import RiskEngine

from .conftest import fraud_txn, make_txn


def _model(intercept: float, version: str) -> LogisticBaseline:
    n = len(FEATURE_NAMES)
    return LogisticBaseline(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=[0.0] * n,
        intercept=intercept,
        version=version,
    )


class _ExplodingModel(LogisticBaseline):
    def predict_proba(self, features: Any) -> float:
        raise RuntimeError("boom")


class _NaNModel(LogisticBaseline):
    def predict_proba(self, features: Any) -> float:
        return math.nan


def test_no_shadow_by_default(benign):
    engine = RiskEngine(Settings())
    result, shadow = engine.score_with_shadow(benign)
    assert shadow is None
    assert engine.shadow_model_version is None
    assert result.decision == Decision.APPROVE


def test_shadow_does_not_change_primary_result(benign, fraudulent, toy_model):
    always_block = _model(intercept=50.0, version="always-block")
    plain = RiskEngine(Settings(), model=toy_model)
    shadowed = RiskEngine(Settings(), model=toy_model, shadow_model=always_block)
    for txn in (benign, fraudulent, benign):
        a = plain.score(txn)
        b, shadow = shadowed.score_with_shadow(txn)
        assert shadow is not None
        assert a.model_dump(exclude={"latency_ms"}) == b.model_dump(exclude={"latency_ms"})
        assert shadow.primary_decision == b.decision
    assert shadow.model_version.endswith("+lr-always-block")


def test_shadow_records_state_exactly_once(benign, toy_model):
    plain = RiskEngine(Settings())
    shadowed = RiskEngine(Settings(), shadow_model=toy_model)
    for _ in range(3):
        plain.score(benign)
        shadowed.score_with_shadow(benign)
    args = (benign["nameOrig"], benign["nameDest"], benign["step"] + 1)
    assert shadowed.store.snapshot(*args) == plain.store.snapshot(*args)
    assert shadowed.store.snapshot(*args).orig_txn_count == 3


def test_shadow_disagreement_is_reported(benign):
    # Rules-only primary approves the benign payment; a candidate with a huge intercept
    # would push the blended score to 0.6 * 1 + 0.4 * 0 = 0.6 -> REVIEW.
    engine = RiskEngine(Settings(), shadow_model=_model(intercept=50.0, version="hot"))
    result, shadow = engine.score_with_shadow(benign)
    assert shadow is not None
    assert result.decision == Decision.APPROVE
    assert shadow.decision == Decision.REVIEW
    assert shadow.risk_score == pytest.approx(0.6)
    assert not shadow.agrees and not shadow.failed


def test_shadow_uses_same_blend_as_primary(fraudulent, toy_model):
    # The same model as primary and shadow must produce identical scores.
    engine = RiskEngine(Settings(), model=toy_model, shadow_model=toy_model)
    result, shadow = engine.score_with_shadow(fraudulent)
    assert shadow is not None
    assert shadow.risk_score == result.risk_score
    assert shadow.agrees


@pytest.mark.parametrize("cls", [_ExplodingModel, _NaNModel])
def test_shadow_failure_is_swallowed(benign, cls, caplog):
    base = _model(intercept=0.0, version="bad")
    bad = cls(**{k: getattr(base, k) for k in base.__dataclass_fields__})
    engine = RiskEngine(Settings(), shadow_model=bad)
    with caplog.at_level("ERROR"):
        result, shadow = engine.score_with_shadow(benign)
    assert result.decision == Decision.APPROVE
    assert shadow is not None and shadow.failed
    assert shadow.risk_score is None
    assert "Shadow model" in caplog.text


def test_shadow_model_loaded_from_settings(tmp_path, toy_model):
    path = tmp_path / "candidate.json"
    toy_model.save(path)
    engine = RiskEngine(Settings(shadow_model_path=str(path)))
    assert engine.shadow_model is not None
    assert engine.shadow_model_version is not None
    assert engine.shadow_model_version.endswith("+lr-toy")
    assert engine.mode == "rules-only"


def test_bad_shadow_artifact_fails_fast(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 999}')
    with pytest.raises(ValueError):
        RiskEngine(Settings(shadow_model_path=str(path)))


def test_shadow_setting_from_env(monkeypatch):
    monkeypatch.setenv("FRE_SHADOW_MODEL_PATH", "/models/candidate.json")
    assert Settings.from_env().shadow_model_path == "/models/candidate.json"
    monkeypatch.setenv("FRE_SHADOW_MODEL_PATH", "")
    assert Settings.from_env().shadow_model_path is None


def test_shadow_metrics_render():
    m = Metrics()
    assert "fre_shadow" not in m.render()
    m.observe_shadow("APPROVE", 0.1, "REVIEW", 0.6, 0.2)
    m.observe_shadow("BLOCK", 0.9, "BLOCK", 0.85, 0.1)
    m.observe_shadow("APPROVE", 0.1, None, None, 0.05)
    text = m.render()
    assert "fre_shadow_scored_total 2" in text
    assert "fre_shadow_errors_total 1" in text
    assert 'fre_shadow_decisions_total{decision="BLOCK"} 1' in text
    assert 'fre_shadow_decisions_total{decision="REVIEW"} 1' in text
    assert 'fre_shadow_disagreements_total{primary="APPROVE",shadow="REVIEW"} 1' in text
    assert 'primary="BLOCK"' not in text
    assert "fre_shadow_abs_score_delta_sum 0.550000" in text
    assert "fre_shadow_latency_ms_sum 0.350" in text


@pytest.fixture
def shadow_client() -> TestClient:
    settings = Settings()
    engine = RiskEngine(settings, shadow_model=_model(intercept=50.0, version="hot"))
    app = create_app(settings, engine=engine)
    with TestClient(app) as c:
        yield c


def test_api_shadow_end_to_end(shadow_client):
    ready = shadow_client.get("/ready").json()
    assert ready["mode"] == "rules-only"
    assert ready["shadow_model_version"].endswith("+lr-hot")

    body = shadow_client.post("/v1/score", json=make_txn()).json()
    assert body["decision"] == "APPROVE"
    assert body["model_score"] is None
    assert "shadow" not in str(body).lower()

    batch = {"transactions": [make_txn(transaction_id="a"), fraud_txn(transaction_id="b")]}
    assert shadow_client.post("/v1/score/batch", json=batch).status_code == 200

    text = shadow_client.get("/metrics").text
    assert "fre_shadow_scored_total 3" in text
    assert 'fre_shadow_disagreements_total{primary="APPROVE",shadow="REVIEW"} 2' in text
    assert "nameOrig" not in text


def test_ready_omits_shadow_when_unset(client):
    assert "shadow_model_version" not in client.get("/ready").json()
