from .conftest import fraud_txn, make_txn


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_mode(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["mode"] == "rules-only"


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert client.get("/health").headers["x-request-id"]


def test_score_benign(client):
    r = client.post("/v1/score", json=make_txn(transaction_id="t1"))
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "APPROVE"
    assert body["transaction_id"] == "t1"
    assert 0 <= body["risk_score"] <= 1
    assert body["latency_ms"] >= 0


def test_score_fraud_is_explained(client):
    body = client.post("/v1/score", json=fraud_txn()).json()
    assert body["decision"] == "BLOCK"
    assert {h["code"] for h in body["rule_hits"]} >= {"R001", "R002"}
    assert body["reasons"]


def test_score_with_model(model_client):
    body = model_client.post("/v1/score", json=fraud_txn()).json()
    assert body["model_score"] is not None
    assert body["top_features"][0]["feature"] == "orig_drained"
    assert model_client.get("/ready").json()["mode"] == "rules+model"


def test_label_columns_rejected(client):
    r = client.post("/v1/score", json={**make_txn(), "isFraud": 1})
    assert r.status_code == 422


def test_validation_errors(client):
    assert client.post("/v1/score", json=make_txn(amount=-5)).status_code == 422
    assert client.post("/v1/score", json=make_txn(type="WIRE")).status_code == 422
    bad = make_txn()
    del bad["nameDest"]
    assert client.post("/v1/score", json=bad).status_code == 422


def test_batch_scores_in_order(client):
    txns = [make_txn(transaction_id=f"t{i}") for i in range(3)] + [fraud_txn(transaction_id="f")]
    r = client.post("/v1/score/batch", json={"transactions": txns})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    assert [x["transaction_id"] for x in body["results"]] == ["t0", "t1", "t2", "f"]
    assert body["results"][-1]["decision"] == "BLOCK"


def test_batch_too_large(client):
    r = client.post("/v1/score/batch", json={"transactions": [make_txn()] * 6})
    assert r.status_code == 413


def test_batch_empty_rejected(client):
    assert client.post("/v1/score/batch", json={"transactions": []}).status_code == 422


def test_rules_endpoint(client):
    rules = client.get("/v1/rules").json()
    assert rules[0]["code"] == "R001"
    assert all({"code", "description", "points"} <= set(r) for r in rules)


def test_metrics_endpoint_counts_decisions(client):
    client.post("/v1/score", json=make_txn())
    client.post("/v1/score", json=fraud_txn())
    text = client.get("/metrics").text
    assert 'fre_decisions_total{decision="APPROVE"} 1' in text
    assert 'fre_decisions_total{decision="BLOCK"} 1' in text
    assert "fre_scoring_latency_ms_count 2" in text
    assert 'fre_requests_total{endpoint="score"} 2' in text


def test_openapi_available(client):
    assert client.get("/openapi.json").json()["info"]["title"] == "Realtime Fraud Risk Engine"
