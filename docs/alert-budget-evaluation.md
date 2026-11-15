# Evaluating fraud detection against review capacity

Fraud is rare, so accuracy and ROC-AUC alone can hide an unusable alert queue.
Use `fraud_engine.alert_budget.evaluate_alert_budgets` to report precision and
recall for an operationally fixed share of transactions sent for investigation.

## Example

```python
from fraud_engine.alert_budget import evaluate_alert_budgets

# Scores/labels must be from the untouched future test interval.
rows = evaluate_alert_budgets(
    scores=[0.1, 0.9, 0.8, 0.2],
    labels=[0, 1, 0, 1],
    budgets=[0.25, 0.5],
)
for row in rows:
    print(row.as_dict())
```

The implementation sorts scores descending, breaking ties by original row
position. At budget `b`, it selects exactly `floor(n*b)` transactions. It
returns actual alert volume, precision, recall, captured fraud, fraud prevalence
denominator and the lowest selected score. An empty alert queue has precision
and recall reported as zero by convention.

## Evaluation protocol

1. Train and tune on earlier chronological windows only; do not use test labels
   to select features, hyperparameters, score thresholds or alert budgets.
2. Freeze the model, reference features and operational review capacities.
3. Generate scores on a later untouched holdout using only features available
   at authorisation time. Record test interval, population and class prevalence.
4. Report results for each precommitted budget, with PR-AUC and subgroup
   diagnostics. Compare rules and model on the *same* holdout.
5. Re-evaluate under realistic delayed labels, missing features, load and drift.

## Limitations

This is an offline *ranking* metric, not a production rate limiter or calibrated
probability. A score threshold at the selected rank may admit more than the
budget when many events share that score. Production enforcement needs
deterministic tie-breaking, idempotent event handling and an atomic queue
capacity policy. A tiny holdout may have zero selected alerts after flooring.
Uncertainty intervals and financial loss weighting are not implemented.

No dataset is included. For PaySim provenance, synthetic-data caveats and
licence details, see [dataset-paysim.md](dataset-paysim.md). The existing v1
scoring path uses post-event balance fields and **must not be presented as a
live authorisation model**. This utility does not fix that leakage.
