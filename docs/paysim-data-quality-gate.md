# PaySim data-quality gate

Run before training or offline scoring:

```bash
python -m fraud_engine.paysim_audit /path/to/PS_20174392719_1491204439457_log.csv > audit.json
pytest -q tests/test_paysim_audit.py
```

The tool streams CSV rows without loading the full dataset into RAM. It validates exact column names, duplicate headers, complete records, nondecreasing integer time steps, finite nonnegative monetary values, transaction types, binary labels and nonempty account IDs. On failure it exits with status 2 and emits no success JSON. The report contains aggregate counts, fraud prevalence, step range and type frequencies, never account identifiers.

**Source:** PaySim synthetic mobile-money transactions, López-Rojas, Elmir and Axelsson (2016), publicly distributed at https://www.kaggle.com/datasets/ealaxi/paysim1 . Download separately and respect the dataset licence. No records are committed.

**Threat model and limitations:** Passing this gate establishes basic schema and row integrity, not fraud-label correctness, dataset representativeness, uniqueness of events or absence of leakage. `newbalanceOrig`, `newbalanceDest`, `isFraud` and `isFlaggedFraud` must not be used as pre-authorisation predictors. The gate reads those fields only for integrity checks and aggregate auditing. Temporal holdout and point-in-time feature generation are separate requirements. Do not claim measured model quality from an audit report.

**Engineering decision:** Fail closed on malformed rows rather than silently dropping them; dropping rows could alter fraud prevalence and evaluation ordering. Numeric values are validated as finite floats for practicality, not used for financial arithmetic. Production payment processing would require decimal-safe amounts and authenticated, privacy-controlled ingestion.

**Memory bound:** Because time steps are validated in nondecreasing order, the audit counts distinct steps by detecting transitions, rather than retaining a counter for every step. Aside from the fixed five transaction-type counters and CSV parser buffers, the audit's application-level aggregation uses O(1) memory with respect to dataset rows and distinct steps. This does not bound individual CSV field lengths or Python integer size; constrain input size and field lengths at a trusted ingestion boundary for untrusted uploads. Sparse and repeated-step regression tests verify the count semantics.

## Flag/label diagnostics

The report now includes `fraud_by_type` and `flagged_by_type` (all five transaction
categories, including zero counts), plus `flagged_true_fraud`,
`flagged_false_positive`, and `unflagged_fraud`. These are descriptive
cross-tabulations of PaySim's *dataset-provided* `isFlaggedFraud` field versus
`isFraud`, **not** the engine's precision/recall, a trained-model evaluation,
or an unbiased live detection rate. The counters are computed in one pass with
fixed memory and can help identify suspicious dataset composition before
training. Their identities are checked in tests:
`flagged_true_fraud + flagged_false_positive == flagged_rows` and
`flagged_true_fraud + unflagged_fraud == fraud_rows`.

Do not use either label or the dataset's flag as an input feature for
pre-authorisation scoring. Do not infer predictive effectiveness from these
aggregates without a point-in-time holdout and a separately defined detector.
