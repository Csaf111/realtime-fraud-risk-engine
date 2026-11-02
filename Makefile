.PHONY: help install lint format typecheck test test-fast cov run train train-synthetic docker-build docker-run smoke clean

PY ?= python
IMAGE ?= realtime-fraud-risk-engine:local

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install package with dev + train extras
	$(PY) -m pip install -e ".[dev]"

lint:  ## Ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## Auto-format
	ruff check --fix .
	ruff format .

typecheck:  ## mypy
	mypy

test:  ## All tests with coverage
	pytest --cov --cov-report=term-missing

test-fast:  ## Skip slow end-to-end training tests
	pytest -m "not slow"

run:  ## Run the API with reload
	uvicorn fraud_engine.api:app --reload --app-dir src

train:  ## Train on the real PaySim CSV (see docs/dataset-paysim.md)
	$(PY) -m fraud_engine.train --data data/raw/paysim.csv --out models/baseline.json

train-synthetic:  ## Train on synthetic data (pipeline smoke test only)
	$(PY) -m fraud_engine.train --synthetic 20000 --out models/baseline.json

docker-build:  ## Build the image
	docker build -t $(IMAGE) .

docker-run:  ## Run the image on :8000
	docker run --rm -p 8000:8000 $(IMAGE)

smoke:  ## Smoke-test a running API (BASE_URL=http://localhost:8000)
	./scripts/smoke_test.sh $${BASE_URL:-http://localhost:8000}

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist *.egg-info
