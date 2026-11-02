# PaySim dataset guide

## Source and attribution

PaySim is a **synthetic** mobile-money transaction simulator/dataset used in fraud research. Dataset distribution: [Kaggle — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1). Research reference: López-Rojas, E. A., Elmir, A., and Axelsson, S. (2016), *PaySim: A financial mobile money simulator for fraud detection*. Verify the original publication and dataset licensing/usage terms before redistribution or publication of derived assets.

**The dataset is not bundled, downloaded automatically or committed to this repository.** The current API is a rules-only demonstration; no claims are made about model performance on PaySim.

## Typical PaySim fields

| Field | Meaning | Use at decision time? |
| --- | --- | --- |
| `step` | Simulation time step (commonly an hour) | Split ordering and possible known time feature, with care |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | Yes |
| `amount` | Transaction amount | Yes |
| `nameOrig` | Origin account identifier | Avoid raw ID as feature |
| `oldbalanceOrg` | Origin pre-transaction balance | Yes, only if actually available |
| `newbalanceOrig` | Origin post-transaction balance | **No: post-event leakage** |
| `nameDest` | Destination account identifier | Avoid raw ID as feature |
| `oldbalanceDest` | Destination pre-transaction balance | Only if available at decision time; not used by baseline |
| `newbalanceDest` | Destination post-transaction balance | **No: post-event leakage** |
| `isFraud` | Fraud label | **Target only** |
| `isFlaggedFraud` | Simulator/legacy detection flag | **No: existing rule/label proxy** |

The exact semantics and availability of fields must be validated against the selected dataset version and real-time system requirements. Do not silently interpret simulated post-event state as a production feature.

## Acquisition and local storage

1. Read the Kaggle dataset card, citation requirements and license/usage conditions.
2. Download it manually through Kaggle's supported interface.
3. Store it under `data/raw/` locally. `data/` and CSV/Parquet files are gitignored.
4. Record a checksum, file size, download date, source URL and dataset version in a **local** manifest (do not commit sensitive data).
5. Keep raw data immutable; write derived data under `data/processed/`.

Example local verification:

```bash
sha256sum data/raw/PS_20174392719_1491204439457_log.csv
```

## Data-quality checks before modelling

Validate schema, numeric ranges, missingness, duplicates, `step` ordering, class prevalence, class-by-transaction-type distribution, extreme balances and impossible transaction values. Document whether identifiers repeat across time and whether simulated scenarios differ from real payment traffic.

## Limitations

PaySim is synthetic, not representative of all geographies, rails, customers or real fraud typologies. A high offline score on this dataset does not imply real-world fraud detection performance. Evaluate drift, false-positive burden, fairness, operational latency and review capacity separately before any production use.
