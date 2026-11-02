from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.features import FEATURE_NAMES
from fraud_engine.model import LogisticBaseline
from fraud_engine.scoring import RiskEngine


def make_txn(**overrides: Any) -> dict[str, Any]:
    """A benign PAYMENT to a merchant, with consistent balances."""
    base: dict[str, Any] = {
        "step": 10,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C100000001",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "nameDest": "M200000001",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    base.update(overrides)
    return base


def fraud_txn(**overrides: Any) -> dict[str, Any]:
    """Classic PaySim account-takeover pattern: drain via TRANSFER to a mule."""
    base = make_txn(
        type="TRANSFER",
        amount=250_000.0,
        oldbalanceOrg=250_000.0,
        newbalanceOrig=0.0,
        nameDest="C999999999",
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
    )
    base.update(overrides)
    return base


@pytest.fixture
def benign() -> dict[str, Any]:
    return make_txn()


@pytest.fixture
def fraudulent() -> dict[str, Any]:
    return fraud_txn()


@pytest.fixture
def toy_model() -> LogisticBaseline:
    """Hand-built model: risk driven only by orig_drained and type_TRANSFER."""
    n = len(FEATURE_NAMES)
    coef = [0.0] * n
    coef[FEATURE_NAMES.index("orig_drained")] = 3.0
    coef[FEATURE_NAMES.index("type_TRANSFER")] = 1.0
    return LogisticBaseline(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * n,
        scale=[1.0] * n,
        coef=coef,
        intercept=-2.0,
        version="toy",
    )


@pytest.fixture
def client() -> TestClient:
    app = create_app(Settings(max_batch_size=5))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def model_client(toy_model: LogisticBaseline) -> TestClient:
    settings = Settings()
    app = create_app(settings, engine=RiskEngine(settings, model=toy_model))
    with TestClient(app) as c:
        yield c
