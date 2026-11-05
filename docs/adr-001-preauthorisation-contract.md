# ADR-001: Separate pre-authorisation feature contract

**Status:** Proposed implementation, not yet activated in serving or training  
**Date:** 2026-11-05

## Problem

The current PaySim-compatible API and model use `newbalanceOrig` and
`newbalanceDest`, which represent balances after transaction processing.
Rules R002, R004 and R005 also inspect these fields. They cannot be assumed
available before authorisation. Training and serving with them risks target
leakage and inflated offline metrics.

## Decision

Introduce `src/fraud_engine/preauth.py` as a separate, versionable
pre-authorisation feature contract. It accepts a pre-debit
`available_balance` (from an authoritative ledger), transaction amount/type,
destination identifier, event time, and a prior-only velocity snapshot.

Do **not** silently replace the existing feature vector: previously trained
model coefficients and standardisation statistics are positional and would
be invalid. The existing API remains a PaySim research/demo interface until
the migration below is complete.

## Migration acceptance criteria

1. Define a distinct `PreAuthorisationTransaction` request schema and
   `/v2/score` endpoint; reject labels and post-event balance fields.
2. Replace rules R002/R004/R005 with pre-decision signals. Do not simply
   rename post-event values as pre-event values.
3. Retrain a new model using exactly `PREAUTH_FEATURE_NAMES`, preserving
   a schema fingerprint in the model artefact and checking it on load.
4. In PaySim experiments, `oldbalanceOrg` can serve as a *proxy* for
   pre-debit balance, not evidence of live ledger availability.
5. Replay events in time order with prior-only velocity state; split by
   chronological steps and allow for label maturation delays.
6. Report PR-AUC, precision/recall at operational alert budgets, confusion
   matrix, calibration, class prevalence, and time-slice stability.
7. Validate concurrent requests, state persistence, idempotency and
   out-of-order events before describing the service as production-ready.

## Trade-offs

Removing balance deltas may reduce synthetic-data performance while improving
decision-time validity. A live integration needs authenticated access to
ledger balances, durable feature state, and data-retention controls. PaySim
is synthetic; strong metrics on it do not establish real-world fraud efficacy.

## Verification

`tests/test_preauth.py` asserts stable ordering, finite values, required
fields, and invariance to changes in post-event fields and labels. These
tests protect the new contract only; they do not certify the legacy API.
