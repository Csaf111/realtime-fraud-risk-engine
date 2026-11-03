# HTTP API reference

Base URL (local): `http://localhost:8000`. Interactive OpenAPI UI: `/docs`. Raw OpenAPI schema: `/openapi.json`.

## GET /health

Liveness endpoint. Response `200 OK`:

```json
{"status":"ok"}
```

This does not verify external dependencies; there are currently none.

## POST /v1/score

Score one transaction using only fields available before the transaction completes.

Request example:

```json
{
  "transaction_id": "txn-demo-001",
  "transaction_type": "TRANSFER",
  "amount": 250000,
  "origin_balance": 100000
}
```

| Field | Type | Required | Constraints |
| --- | --- | --- | --- |
| `transaction_id` | string | yes | 1–128 characters, letters/digits/period/underscore/colon/hyphen |
| `transaction_type` | string | yes | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | number | yes | finite and greater than 0 |
| `origin_balance` | number or null | no | finite and nonnegative if present |

Unexpected fields are rejected, including `isFraud`, `isFlaggedFraud`, `newbalanceOrig`, and `newbalanceDest`.

Response `200 OK`:

```json
{
  "transaction_id": "txn-demo-001",
  "risk_score": 75,
  "risk_band": "high",
  "reasons": [
    {"code": "HIGH_RISK_TYPE", "points": 25, "description": "Transfer or cash-out transaction"},
    {"code": "VERY_LARGE_AMOUNT", "points": 25, "description": "Transaction amount is at least 100000"},
    {"code": "EXCEEDS_ORIGIN_BALANCE", "points": 25, "description": "Amount exceeds known origin balance"}
  ],
  "model_version": "heuristic-v1"
}
```

### Rule table (heuristic-v1)

| Rule | Condition | Points |
| --- | --- | --- |
| HIGH_RISK_TYPE | TRANSFER or CASH_OUT | 25 |
| LARGE_AMOUNT | 10,000 ≤ amount < 100,000 | 15 |
| VERY_LARGE_AMOUNT | amount ≥ 100,000 | 25 |
| EXCEEDS_ORIGIN_BALANCE | balance provided and amount > origin balance | 25 |

Rules are additive and capped at 100. Bands: low 0–29, medium 30–59, high 60–100. Missing origin balance contributes **no** exceeds-balance points; it is not treated as zero. These are illustrative thresholds, not empirically learned, and the score is **not** a calibrated probability.

### Errors

`422 Unprocessable Entity` is returned for invalid types, missing required fields, invalid enum values, nonpositive/nonfinite amounts, negative balances and unexpected fields. Errors use FastAPI's standard validation response.

## Contract versioning

`/v1/score` is the first HTTP contract. `model_version` identifies scoring logic independently of API version. Any breaking request/response contract change requires a new API version or a documented migration.
