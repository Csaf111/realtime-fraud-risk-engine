# Tie-aware average precision

`fraud_engine.average_precision.average_precision(scores, labels, indices)`
computes non-interpolated average precision (AP) on one fixed partition.
It sorts descending by score and groups tied scores into a single threshold,
preventing row-order dependence. AP sums precision at each distinct-score
threshold weighted by the increment in recall. This is **not** trapezoidal
area under a precision-recall curve and must be labelled accordingly.

A partition with no fraud returns `None` (undefined), not a misleading zero.
Evaluate validation and untouched chronological test partitions separately;
never choose a model, feature set or threshold based on test labels.

This metric accepts already-generated scores and does **not** establish
point-in-time feature availability, model calibration or absence of leakage.
Use the embargo-aware partitions from `temporal_holdout.py`, generate
predictions causally, and report fraud prevalence and alert-budget metrics
alongside AP. The PaySim dataset is synthetic; results must not be presented
as evidence of real-world fraud detection accuracy.

Example:

```python
from fraud_engine.average_precision import average_precision

ap = average_precision([0.9, 0.2, 0.1], [1, 0, 1], [0, 1, 2])
print(ap)  # 0.8333333333333333
```

The example is synthetic, not a benchmark result.
