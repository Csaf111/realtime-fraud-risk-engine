# Pre-authorisation v2 rules-only endpoint

`POST /v2/score` is a deliberately isolated, **rules-only** baseline. It
accepts the transaction type, amount, origin available balance, account IDs,
step and optional transaction ID. It rejects PaySim post-event fields
(`newbalanceOrig`, `newbalanceDest`), labels and any other unknown fields
with HTTP 422. The old v1 endpoint is unchanged for compatibility and must
not be treated as an authorisation-grade fraud model.

## Example

```bash
curl -X POST localhost:8000/v2/score -H 'Content-Type: application/json' \
  -d '{"transaction_id":"demo","step":10,"type":"TRANSFER","amount":1000,"nameOrig":"C1","oldbalanceOrg":1000,"nameDest":"C2"}'
```

Response follows the existing `RiskResult` contract, with
`model_version=preauth-rules-v2.0`, `model_score=null` and P-prefixed
reason codes. The score is a **heuristic priority score, not a calibrated
probability**. Rules use only the request and the velocity snapshot taken
before recording the transaction. The existing in-memory/Redis store is
shared with v1, so concurrent ordering and idempotency limitations remain.

## Engineering trade-offs and next gates

- No destination available balance is required; no post-debit reconciliation
  or PaySim fraud label is consulted.
- Rule thresholds and high-amount limits are inherited from the application
  settings. Default scoring is not empirically calibrated to a loss budget.
- A debit exceeding the origin balance is a fraud-review signal here, not a
  replacement for the payment ledger's insufficient-funds enforcement.
- No ML artifact is loaded into v2. Train a **new** model using only v2
  pre-authorisation features and chronological, gapped evaluation before
  adding model inference.
- Add authenticated callers, transaction idempotency, atomic velocity
  ordering and measured latency before any payment-authorisation pilot.
- PaySim is synthetic; its fraud labels and balance artifacts limit
  generalisation. Do not report synthetic metrics as live fraud performance.
