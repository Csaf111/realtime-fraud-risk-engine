# Leakage-safe preprocessing plan

Fraud models are easy to over-score offline and disappointing in production. Almost always
the cause is leakage: the model saw information at training time that it won't have at
decision time. This plan lists every leakage route that applies to PaySim and the control
that blocks it. Each control is enforced in code and covered by a test.

## Principles

1. **Point-in-time correctness.** A feature for a transaction at time *t* may only use the
   transaction payload and events strictly before it.
2. **One feature implementation.** Training and serving call the same
   `compute_features()` and `VelocityStore`. Nothing is re-implemented in pandas.
3. **Split by time, never at random.** Test data is always in the future relative to
   training data, with a gap.
4. **Fit on train only.** Every fitted statistic (scaler mean/std, thresholds, encoders)
   comes from training rows.
5. **Fail loudly.** `leakage_checks()` raises `LeakageError` and stops training.

## Leakage routes and controls

| # | Route | Why it leaks | Control | Test |
|---|---|---|---|---|
| L1 | `isFraud` used as a feature | It's the answer | Excluded from `FEATURE_NAMES`. API schema is `extra="forbid"`, so posting it gives 422 | `test_leakage_checks_reject_forbidden_features`, `test_label_columns_rejected` |
| L2 | `isFlaggedFraud` used as a feature | Output of another decision system, correlated with the label and unavailable at authorisation | Dropped in `build_feature_frame` (`LEAKY_COLUMNS`) | `test_feature_frame_excludes_leaky_columns` |
| L3 | Random row split | Neighbouring rows from the same hour or the same mule end up in both train and test, so the model learns the period instead of the pattern | `temporal_split()` by `step` | `test_temporal_split_has_gap` |
| L4 | No gap between train and test | Labels arrive late in reality (chargebacks take days), and late-train behaviour bleeds into early-test | `gap_steps` (default 24h) dropped between partitions | `test_temporal_split_has_gap`, `leakage_checks` overlap rule |
| L5 | Velocity features using future rows | Rolling or group stats computed over the whole frame look ahead | Stream replay: `snapshot()` before `record()` | `test_future_rows_do_not_change_past_features`, `test_snapshot_excludes_current_event` |
| L6 | Self-inclusion in aggregates | A transaction counts itself in its own velocity | `record()` happens after scoring/feature building | `test_snapshot_excludes_current_event` |
| L7 | Label-aware features (target encoding, "past fraud count for this mule") | Uses labels that wouldn't be known yet | No feature reads `isFraud`. Building features with every label flipped gives identical output | `test_features_do_not_depend_on_labels` |
| L8 | Scaler fitted on all data | Test distribution leaks into train | `fit_standardiser(train)` only | `test_standardiser_uses_train_only` |
| L9 | Raw IDs as features | `nameOrig`/`nameDest` memorise specific accounts and don't generalise | IDs are only velocity keys, never values. `leakage_checks` rejects them | `test_leakage_checks_reject_forbidden_features` |
| L10 | `step` as a raw feature | Lets the model learn "fraud happened late in the month" | Only cyclic hour-of-day (`hour_sin/cos`) is used | `test_leakage_checks_reject_forbidden_features` |
| L11 | Train/serve skew | Offline pandas features differ subtly from online code | Shared code path. Parity test replays rows online and compares | `test_offline_features_match_online` |
| L12 | Single-class partition | Metrics silently become undefined or misleading | `leakage_checks` asserts both classes in train and test | `test_leakage_checks_reject_single_class` |

## Known risks that remain (be honest about them)

- **Post-transaction balances.** `newbalanceOrig` / `newbalanceDest` describe the state
  *after* the transaction. In a real authorisation flow you'd know the *proposed* new
  balance. You might not know the destination's post-balance (it lives at another
  institution). The engine keeps these features because PaySim provides them and the API
  receives them in the payload. Before production use, confirm each one is actually
  available at decision time, and drop it if not. The `dest_balance_error` and
  `dest_zero_balances` features are the main suspects.
- **Simulator artefacts.** The balance inconsistencies in PaySim are unusually predictive
  of fraud (see `dataset-paysim.md`). High offline scores mostly measure how well the model
  learned the simulator. Report results with that caveat, and compare against the
  rules-only baseline that `train.py` always reports.
- **Within-step ordering.** PaySim gives hour-level timestamps. Rows inside the same step
  are replayed in file order, which is assumed to be event order.
- **Per-process state.** The in-memory `VelocityStore` is per replica. With several
  replicas, set `FRE_VELOCITY_BACKEND=redis`: `RedisVelocityStore` applies the same
  `(step - window, step]` filter and is parity-tested against the in-memory store, so
  offline (training replay, always in-memory) and online features stay identical.
  Concurrent requests for the same account are still unordered between `snapshot` and
  `record`; partition by account if strict ordering matters.

## Pipeline order (what `fraud_engine.train` does)

```
load_paysim ── validate_schema
     │
build_feature_frame        (drop isFlaggedFraud → sort by step → stream replay)
     │
temporal_split             (train ≤ cutoff | gap dropped | test > cutoff+gap)
     │
leakage_checks             (raise on any violation)
     │
fit_standardiser(train)    → LogisticRegression(class_weight="balanced")
     │
evaluate on test           (model vs rules-only: ROC-AUC, PR-AUC, precision, recall)
     │
[optional] alert budget    (thresholds from score quantiles on the train-window tail,
     │                      label-free; then alert rate/precision/recall on test)
models/baseline.json + baseline.metrics.json
```

## Evaluation guidance

- Lead with **PR-AUC**. ROC-AUC looks flattering at 0.1% prevalence.
- Choose operating thresholds from an **alert budget** (e.g. "review at most 0.5% of
  traffic"), then report recall at that budget.
- When tuning thresholds, use a validation window *between* train and test. Never tune on test.
  `--review-budget/--block-budget` follow this: thresholds come from the most recent slice
  of the training window and read only the score distribution, never labels. Caveat: the
  model was fitted on that slice, so its scores there can be slightly more confident than
  on unseen data. The realised alert rate on the test window (`alert_budget.test`) is the
  honest check; if it overshoots the budget, tighten it.
- Always report the **rules-only baseline** next to the model, so you can see what the
  model actually adds.
