# Partition-aware alert-budget evaluation

`fraud_engine.holdout_metrics.evaluate_holdout(scores, labels, split)` reports
train, validation, and test precision/recall at fixed investigation capacities
(0.1%, 0.5%, 1% by default). It excludes embargoed records, uses each
partition's own denominator, and breaks equal-score ties by original row index.

Example (illustrative, **not** a real model evaluation):

```python
from fraud_engine.temporal_holdout import chronological_holdout
from fraud_engine.holdout_metrics import evaluate_holdout

steps = list(range(1, 31))
split = chronological_holdout(steps, embargo_steps=1)
scores = [i / 30 for i in range(30)]  # replace with causal model scores
labels = [int(i % 3 == 0) for i in range(30)]
report = evaluate_holdout(scores, labels, split, budgets=(0.1, 0.5))
print(report["test"])
```

## Evaluation contract

- Generate **point-in-time** features and velocity state using only preceding
  transactions. Never use `newbalanceOrig`, `newbalanceDest`, labels or future
  aggregates as inference features.
- Fit scalers and models on the training partition only. Select hyperparameters
  and any alert threshold on validation only. Freeze the system before test.
- Report fraud prevalence and capacity-specific precision/recall separately for
  each partition; never combine test with training or embargoed rows.
- Capacity is `ceil(partition_rows * budget)`; small partitions therefore
  receive at least one alert. Stable ties use original record order.
- The utility **does not** calculate PR-AUC, train models, enforce production
  review queues, or guarantee causal scoring. No measured PaySim result is claimed.
- The source dataset is PaySim (synthetic mobile-money transactions); follow
  `docs/dataset-paysim.md` for attribution and known simulation artifacts.

## Next integration

Combine this branch with the v2 pre-authorisation replay in PR #19 after
resolving merge conflicts. Stream chronological transactions through the
same scoring implementation as the API, retain the resulting score/label/step
triples, and evaluate the held-out test partition exactly once. Add PR-AUC
and uncertainty intervals separately, without tuning on test.
