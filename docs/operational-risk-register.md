# Input integrity and operational risk register

This is a synthetic-data research system, **not** a payment authorisation service. No score or rule should block real customers without calibration, governance and human oversight.

## Input integrity

The API rejects non-finite monetary values (NaN, positive infinity, negative infinity) with HTTP 422. Without this boundary, floating-point comparisons and arithmetic can silently corrupt scores, explanations and monitoring metrics. Negative balances and amounts are also rejected. Regression coverage lives in `tests/test_api.py`.

## Known limitations and required follow-up

| Priority | Risk | Remediation |
| --- | --- | --- |
| Critical before live deployment | Current rules and feature engineering consume `newbalanceOrig` and `newbalanceDest`, which may only be known **after** transaction execution. This is a point-in-time feature leakage risk despite chronological train/test splitting. | Define a pre-authorisation request contract; remove post-event balances from online scoring and retrain/evaluate on strictly pre-decision features. Treat current offline results as exploratory only. |
| High | In-memory velocity state is process-local and not transactionally coupled to scoring; concurrent requests can interleave snapshot/record operations. | Introduce atomic per-account state operations and idempotency keys, then test concurrent and replayed requests. Use shared state for multi-worker deployment. |
| High | No authentication, authorisation or rate limiting on scoring and metrics endpoints. | Require service-to-service identity, isolate operational endpoints and enforce gateway quotas. |
| High | PaySim is synthetic and may encode generator artefacts; rules explicitly reference fraud-prone PaySim transaction types. | Evaluate on separate time periods and independent datasets; report PR-AUC, precision at review capacity, false-positive costs and subgroup drift where permitted. |
| Medium | No retention/privacy controls for raw account identifiers at ingestion. | Minimise identifiers, define retention and access policies, and document audit logging. |

## Release gates

Before any claim of production readiness: establish an event-time feature contract, demonstrate no look-ahead under replay and concurrency, run reproducible evaluation against a held-out future window, validate failure modes, add authentication and observability, and document model governance and rollback.

This document distinguishes **implemented** safeguards from planned controls; it is not a security certification.
