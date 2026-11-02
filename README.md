# Realtime Fraud Risk Engine

An explainable, API-first fraud-risk scoring starter built with Python and FastAPI. It provides **deterministic, auditable heuristic scoring** for synthetic payment events, with a clear path to leakage-safe PaySim modelling. **It is not a trained fraud detector or a production payment decisioning system.**

## Current capabilities

- `POST /v1/score`: score a transaction from pre-decision features and return a 0–100 risk score, risk band and reason codes.
- `GET /health`: lightweight liveness endpoint.
- Strong request validation, typed responses, and OpenAPI docs at `/docs`.
- Automated unit and HTTP tests, Ruff linting, GitHub Actions CI and a non-root Docker image.
- Architecture, PaySim dataset attribution, leakage-prevention and evaluation documentation.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs, or score an event:

```bash
curl -s http://localhost:8000/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id":"txn-demo-001","transaction_type":"TRANSFER","amount":250000,"origin_balance":100000}'
```

Example response (scores are **heuristic**, not probabilities):

```json
{
  "transaction_id": "txn-demo-001",
  "risk_score": 75,
  "risk_band": "high",
  "reasons": [
    {"code": "HIGH_RISK_TYPE", "points": 25, "description": "Transfer or cash-out transaction"},
    {"code": "VERY_LARGE_AMOUNT", "points": 25, "description": "Transaction amount is at least 100000"},
    {"code": "EXCEEDS_ORIGIN_BALANCE", "points": 25, "description": "Amount exceeds known origin balance"}
  ],
  "model_version": "heuristic-v1"
}
```

The API uses pre-decision fields only. It intentionally excludes `newbalanceOrig`, `newbalanceDest`, `isFraud` and `isFlaggedFraud` from scoring.

## Run tests and checks

```bash
ruff check .
pytest -q
docker build -t realtime-fraud-risk-engine .
docker run --rm -p 8000:8000 realtime-fraud-risk-engine
```

Or `docker compose up --build`.

## Repository layout

```text
app/                     FastAPI app, request schemas and scoring rules
tests/                   Unit and API contract tests
docs/architecture.md     Architecture and diagrams
docs/paysim.md           Dataset schema, source and attribution
docs/ml-plan.md          Temporal validation and leakage-safe feature plan
docs/security.md         Security boundaries and deployment considerations
.github/workflows/ci.yml CI checks and container build
Dockerfile               Non-root container
docker-compose.yml       Local development
pyproject.toml           Package and tool configuration
```

## Dataset and ML roadmap

PaySim is a synthetic mobile-money transaction dataset associated with research by López-Rojas, Elmir and Axelsson (2016). Obtain it independently from [Kaggle — PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1); no dataset, personal identifiers, or model weights are included in this repository. Review the dataset's terms before use. See [dataset documentation](docs/paysim.md) and the [leakage-safe ML plan](docs/ml-plan.md).

Roadmap: (1) deterministic rules and API contract (implemented), (2) reproducible PaySim ingestion with temporal splits, (3) calibrated supervised baseline and PR-AUC/recall-at-review-capacity, (4) model registry and drift monitoring, (5) event-driven Kafka ingestion with idempotency, (6) cloud deployment. **No performance metrics or production readiness are claimed until measured.**

## Architecture

See [architecture and sequence diagrams](docs/architecture.md). This is currently a stateless synchronous scoring API, not a streaming pipeline. The term “realtime” describes low-complexity request-time scoring; latency SLOs have not been benchmarked.

## Engineering trade-offs

- **Explainability before complexity:** deterministic rules provide transparent baselines and testable reason codes.
- **No data leakage:** never score on labels or post-transaction balances.
- **No persistence by default:** reduces operational and privacy surface, but does not provide audit storage or deduplication.
- **No auth or rate limiting yet:** local demonstration only; protect behind an authenticated gateway before any deployment.
- **No trained ML model yet:** heuristic scores must not be interpreted as calibrated fraud probabilities.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks. This repository is an educational portfolio demonstration and should not be used to automatically block real payments.

## License

MIT; see [LICENSE](LICENSE).
