# Reproducible temporal holdout score evaluation

Run after installing the package:

```bash
python -m fraud_engine.holdout_cli path/to/causal_scores.csv > holdout-report.json
```

The CSV must be ordered by nondecreasing PaySim `step` and contain `step,score,isFraud` headers. Scores must be finite values in [0,1], and labels exactly 0 or 1. The script partitions **distinct** time steps into train/validation/test, excludes an embargo around both boundaries, and reports prevalence plus precision/recall at 0.1%, 0.5%, and 1% alert budgets **separately per partition**. Ties use original row order. No threshold is selected from the test labels.

**Critical limitation:** this command evaluates externally supplied scores; it does not train models or prove the scores were generated without future information. Produce each score with the same point-in-time features available at authorisation, including historical velocity strictly before the transaction. Do not use `newbalanceOrig`, `newbalanceDest`, `isFraud`, or test-derived tuning in scoring. Freeze feature definitions, model weights, hyperparameters and threshold policy before test inspection. Do not report these scores as real-world performance without independent out-of-time validation.

Data attribution: PaySim is a synthetic mobile-money transaction simulator; see [dataset-paysim.md](dataset-paysim.md). Never commit raw PaySim data or sensitive customer information.

The CLI is deliberately dependency-light (standard-library CSV/JSON); it is a reproducible **evaluation adapter**, not a live decision service or a replacement for PR #19's causal replay. Integration with the v2 scorer and model training remains outstanding.
