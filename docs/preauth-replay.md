# Pre-authorisation rules replay (PaySim)

Run `python -m fraud_engine.preauth_replay /path/to/PS_20174392719_1491204439457_log.csv`
after `pip install -e .`. The input is the publicly described **PaySim synthetic
mobile-money dataset**; obtain it from the dataset publisher, and do not commit
the CSV to Git. See [dataset attribution](dataset-paysim.md).

The replay requires PaySim's step, type, amount, nameOrig, oldbalanceOrg,
nameDest and isFraud columns, in nondecreasing step order. It snapshots
velocity **before** recording each event and scores without using labels,
post-event balances or destination balances. The fraud label is used only
after scoring for metrics. Equal scores are ranked by original row order.

Outputs are descriptive **in-sample** precision/recall at fixed alert budgets
(0.1%, 0.5%, 1%). Capacity uses ceiling rounding; zero-positive recall is
defined as zero. These are *not* holdout estimates, calibrated probabilities,
or evidence of live performance. Because the rules were selected with PaySim
in mind, their reported metrics are exploratory. A credible comparison requires
a held-out future time interval, an explicit temporal gap, threshold selection
on an earlier validation interval, and a report of class prevalence.

Limitations: this replay uses an unbounded in-memory list for ranking, a
single-process velocity store, PaySim's synthetic ordering and its step clock.
Do not use it as a production fraud processor. API authentication,
idempotency, concurrent event ordering and online/offline feature parity
remain separate release gates.
