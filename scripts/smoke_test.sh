#!/usr/bin/env bash
# Smoke-test a running API. Usage: [FRE_SMOKE_API_KEY=...] scripts/smoke_test.sh [BASE_URL]
# When FRE_SMOKE_API_KEY is set (the target runs with FRE_API_KEYS), scoring calls send it
# and the script also checks that an unauthenticated score request is refused with 401.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${FRE_SMOKE_API_KEY:-}"
echo "Smoke testing ${BASE_URL}"

# The key goes to curl through a file descriptor, never on its command line.
score() {
  if [ -n "${API_KEY}" ]; then
    curl -fsS -X POST "${BASE_URL}/v1/score" -H 'content-type: application/json' \
      -H @<(printf 'x-api-key: %s\n' "${API_KEY}") -d "@$1"
  else
    curl -fsS -X POST "${BASE_URL}/v1/score" -H 'content-type: application/json' -d "@$1"
  fi
}

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

if [ -n "${API_KEY}" ]; then
  status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/v1/score" \
    -H 'content-type: application/json' -d @scripts/sample_benign.json)
  [ "${status}" = "401" ] || { echo "Expected 401 without an API key, got ${status}" >&2; exit 1; }
  echo "unauthenticated -> 401"
fi

benign=$(score scripts/sample_benign.json)
echo "benign -> ${benign}"
echo "${benign}" | grep -q '"decision":"APPROVE"' || { echo "Expected APPROVE" >&2; exit 1; }

fraud=$(score scripts/sample_fraud.json)
echo "fraud  -> ${fraud}"
echo "${fraud}" | grep -q '"decision":"BLOCK"' || { echo "Expected BLOCK" >&2; exit 1; }

curl -fsS "${BASE_URL}/metrics" | grep -q fre_decisions_total
echo "Smoke test passed"
