#!/usr/bin/env bash
# Smoke-test a running API. Usage: scripts/smoke_test.sh [BASE_URL]
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
echo "Smoke testing ${BASE_URL}"

retry() {
  for _ in $(seq 1 30); do
    if curl -fsS "$1" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "Timed out waiting for $1" >&2
  return 1
}

retry "${BASE_URL}/health"
curl -fsS "${BASE_URL}/ready"; echo

benign=$(curl -fsS -X POST "${BASE_URL}/v1/score" -H 'content-type: application/json' \
  -d @scripts/sample_benign.json)
echo "benign -> ${benign}"
echo "${benign}" | grep -q '"decision":"APPROVE"' || { echo "Expected APPROVE" >&2; exit 1; }

fraud=$(curl -fsS -X POST "${BASE_URL}/v1/score" -H 'content-type: application/json' \
  -d @scripts/sample_fraud.json)
echo "fraud  -> ${fraud}"
echo "${fraud}" | grep -q '"decision":"BLOCK"' || { echo "Expected BLOCK" >&2; exit 1; }

curl -fsS "${BASE_URL}/metrics" | grep -q fre_decisions_total
echo "Smoke test passed"
