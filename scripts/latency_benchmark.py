"""Bounded HTTP load probe for local or explicitly authorised test environments.

No third-party dependencies. Uses unique synthetic account identifiers per request
to avoid contaminating velocity state across measurements.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def percentile(values: list[float], pct: float) -> float:
    """Linear interpolation between adjacent sorted observations."""
    if not values:
        raise ValueError("at least one observation required")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def payload(index: int) -> bytes:
    """Synthetic transaction with isolated account keys."""
    return json.dumps({
        "transaction_id": f"bench-{index}",
        "step": 10,
        "type": "PAYMENT",
        "amount": 25.0,
        "nameOrig": f"CbenchOrigin{index}",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 975.0,
        "nameDest": f"MbenchMerchant{index}",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 25.0,
    }).encode("utf-8")


def probe(url: str, index: int, timeout: float) -> tuple[float, bool, str]:
    request = urllib.request.Request(
        url, data=payload(index), headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            ok = response.status == 200 and body.get("transaction_id") == f"bench-{index}"
            return (time.perf_counter() - start) * 1000, ok, "" if ok else "invalid_response"
    except urllib.error.HTTPError as exc:
        return (time.perf_counter() - start) * 1000, False, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return (time.perf_counter() - start) * 1000, False, type(exc).__name__


def run(url: str, requests: int, concurrency: int, timeout: float, warmup: int) -> dict:
    if requests < 1 or concurrency < 1 or warmup < 0 or timeout <= 0:
        raise ValueError("requests/concurrency must be positive, warmup nonnegative")
    # Warmup uses disjoint identifiers; measured requests start after warmup completes.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda i: probe(url, i, timeout), range(warmup)))
        start = time.perf_counter()
        futures = [pool.submit(probe, url, warmup + i, timeout) for i in range(requests)]
        results = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - start
    successes = [ms for ms, ok, _ in results if ok]
    failures: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            failures[reason] = failures.get(reason, 0) + 1
    return {
        "requests": requests,
        "concurrency": concurrency,
        "warmup": warmup,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_rps": round(requests / elapsed, 2) if elapsed else None,
        "successes": len(successes),
        "failures": failures,
        "latency_ms": ({
            "mean": round(statistics.mean(successes), 3),
            "p50": round(percentile(successes, 50), 3),
            "p95": round(percentile(successes, 95), 3),
            "p99": round(percentile(successes, 99), 3),
            "max": round(max(successes), 3),
        } if successes else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/score")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not args.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("only local loopback targets are allowed")
    try:
        result = run(args.url, args.requests, args.concurrency, args.timeout, args.warmup)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
