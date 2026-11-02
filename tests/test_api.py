"""HTTP contract and validation tests."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_endpoint() -> None:
    response = client.post(
        "/v1/score",
        json={
            "transaction_id": "txn-demo-001",
            "transaction_type": "TRANSFER",
            "amount": 250000,
            "origin_balance": 100000,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": "txn-demo-001",
        "risk_score": 75,
        "risk_band": "high",
        "reasons": [
            {
                "code": "HIGH_RISK_TYPE",
                "points": 25,
                "description": "Transfer or cash-out transaction",
            },
            {
                "code": "VERY_LARGE_AMOUNT",
                "points": 25,
                "description": "Transaction amount is at least 100000",
            },
            {
                "code": "EXCEEDS_ORIGIN_BALANCE",
                "points": 25,
                "description": "Amount exceeds known origin balance",
            },
        ],
        "model_version": "heuristic-v1",
    }


def test_invalid_amount_returns_422() -> None:
    response = client.post(
        "/v1/score",
        json={"transaction_id": "bad", "transaction_type": "PAYMENT", "amount": -5},
    )
    assert response.status_code == 422


def test_post_event_balance_is_rejected() -> None:
    response = client.post(
        "/v1/score",
        json={
            "transaction_id": "bad",
            "transaction_type": "PAYMENT",
            "amount": 5,
            "newbalanceOrig": 0,
        },
    )
    assert response.status_code == 422


def test_fraud_label_is_rejected() -> None:
    response = client.post(
        "/v1/score",
        json={
            "transaction_id": "bad",
            "transaction_type": "PAYMENT",
            "amount": 5,
            "isFraud": 1,
        },
    )
    assert response.status_code == 422


def test_unknown_transaction_type_returns_422() -> None:
    response = client.post(
        "/v1/score",
        json={"transaction_id": "bad", "transaction_type": "WIRE", "amount": 5},
    )
    assert response.status_code == 422


def test_openapi_exposes_score_endpoint() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/score" in response.json()["paths"]
