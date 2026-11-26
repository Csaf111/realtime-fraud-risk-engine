# Pre-authorisation feature contract: executable audit

**Status: v1 is not suitable for a live authorisation decision.**

The PaySim v1 API accepts `newbalanceOrig` and `newbalanceDest`, and its
rules and model features reference those fields. These are *post-event* balances
in the dataset, not reliable inputs available at the time a payment is authorised.
A temporal train/test split does not correct this kind of feature availability leakage.

## Reproduce the finding

```bash
python scripts/audit_preauth_contract.py
python scripts/audit_preauth_contract.py --strict  # intentionally non-zero on v1
pytest -q tests/test_preauth_contract_audit.py
```

The default command emits a machine-readable JSON report with filename, line
and forbidden field. It exits successfully to support exploratory audits; strict
mode is intended as a **future v2 release gate**, not a current v1 CI gate.
Do not present the current model's offline metrics as deployable fraud performance.

## Migration acceptance criteria

1. Define a new versioned request contract using only payment-initiation facts
   (amount, transaction type, sender available balance, known recipient metadata)
   and point-in-time prior activity. Never accept fraud labels or post-event balances.
2. Replace the rule set and feature vector together; assign a new feature schema
   identifier and reject legacy model artifacts rather than silently reusing them.
3. Retrain the baseline from chronological PaySim windows, preserving prior-event
   velocity semantics and a gap between training and evaluation.
4. Add online/offline parity tests, label-independence tests, future-event invariance
   tests and tests that changing any post-event balance cannot affect a v2 decision.
5. Run the audit with `--strict` against the **v2** scoring modules as a CI gate.
   The current scanner targets v1 modules and must be reconfigured during migration.
6. Publish holdout PR-AUC, precision/recall at fixed review capacity, calibration,
   latency and limitations, with no claims of real-world effectiveness from PaySim alone.

## Limitations

This is an AST-based literal-key scan. It catches direct dictionary indexing
and `.get()` calls but not aliases, dynamic key construction, imported helpers,
attribute access, SQL feature queries or time-travel errors in state stores.
It must supplement, not replace, behavioral and integration tests.
