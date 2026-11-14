# Point-in-time temporal holdout protocol

This module supplies a deterministic split planner for ordered PaySim transaction
steps. It is intentionally independent of the current v1 model pipeline and
does **not** make the existing v1 scoring API pre-authorisation safe.

## Usage

```python
from fraud_engine.temporal_holdout import chronological_holdout

split = chronological_holdout(
    steps, train_fraction=0.6, validation_fraction=0.2, embargo_steps=1
)
train_rows = [rows[i] for i in split.train]
validation_rows = [rows[i] for i in split.validation]
test_rows = [rows[i] for i in split.test]
```

Inputs must be in nondecreasing PaySim `step` order. All events at the same
step remain together; indices are partitioned into train, validation, test and
explicitly excluded embargo rows. The split is based solely on time, not
labels or amounts. Fractions refer to distinct time steps, not row counts.

## Evaluation contract

1. Fit scalers, imputers, encoders, thresholds and model parameters on the
   training partition only. Tune thresholds/hyperparameters on validation.
2. Freeze the complete pipeline, then report PR-AUC, ROC-AUC, recall,
   precision and alert-budget recall on test **once**. Include counts and
   fraud prevalence for each partition, plus uncertainty intervals.
3. Compute historical velocity causally in a chronological replay, never
   using future transactions. Do not allow replay state from validation/test
   to contaminate training. Document whether earlier history is carried
   forward at inference, as it would be in a deployed service.
4. Do not use PaySim `newbalanceOrig`, `newbalanceDest`, `isFraud` or
   `isFlaggedFraud` as pre-authorisation model inputs. A chronological split
   cannot repair post-event feature leakage.
5. Choose the embargo duration based on the actual feature lookback and
   label-availability assumptions. A one-step embargo is merely a default,
   **not** proof against leakage. Audit cross-boundary state explicitly.
6. Compare the rules-only and learned models on the same held-out rows.
   Include false-positive burden at fixed investigator capacity, and
   calibration if interpreting scores as probabilities.

## Engineering trade-offs and limitations

The planner uses distinct-step cut points, so actual row fractions can differ
substantially from requested fractions when event density varies. The embargo
excludes all records within the configured numeric step distance of either
cut point. This deliberately sacrifices samples to reduce boundary leakage.

This utility does not train a model, calculate metrics, verify feature parity,
or validate that a historical balance was observable at authorisation time.
PaySim is synthetic and contains simulation artefacts; results cannot be
extrapolated directly to real banking traffic.

Dataset attribution: PaySim, López-Rojas et al., *PaySim: A financial mobile
money simulator for fraud detection*, 2016. See `docs/dataset-paysim.md`
for repository dataset handling and provenance.
