# Local API latency benchmark

Run a bounded, synthetic HTTP load probe against a **local** instance of the fraud API:

```bash
make run
# separate shell:
python scripts/latency_benchmark.py --requests 1000 --concurrency 8 --warmup 50
```

The script requires only Python's standard library. It reports throughput, successful response latency (mean, p50, p95, p99, maximum), total requests and failure categories. It exits nonzero on any measured failure. Warmup responses are excluded from measurements; failures during warmup are not reported. The synthetic account identifiers are unique per request, preventing artificial velocity buildup, but they do not exercise realistic repeated-account fraud patterns.

**Safety:** The CLI only accepts loopback HTTP targets; do not direct load at public or production systems. A local deployment may still depend on remote Redis or other services: isolate those dependencies first. No real PaySim records or customer data are transmitted.

**Interpretation:** These are client-observed round-trip times, not server-only inference latencies. Results depend on host CPU, worker count, network stack, request concurrency, loaded model and external state backend. Throughput includes failed requests; latency percentiles include only successes. The script is not a distributed load generator or capacity certification. Compare runs with identical hardware, environment, model, traffic and concurrency. Record the Git commit, Python version and container configuration alongside measurements. No performance claims are made without measured results.

For realistic capacity engineering, add mixed account distributions, burst tests, sustained soak tests, Redis concurrency tests, error budgets, and resource utilisation monitoring. Crucially, v1 uses post-transaction balances and is **not safe for live pre-authorisation decisions**.
