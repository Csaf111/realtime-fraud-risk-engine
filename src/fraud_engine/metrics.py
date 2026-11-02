"""Minimal, dependency-free Prometheus-style metrics."""

from __future__ import annotations

import threading
from collections import Counter

_BUCKETS_MS = (1, 2, 5, 10, 25, 50, 100, 250)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.decisions: Counter[str] = Counter()
        self.requests: Counter[str] = Counter()
        self.latency_buckets: Counter[float] = Counter()
        self.latency_sum = 0.0
        self.latency_count = 0

    def observe(self, decision: str, latency_ms: float) -> None:
        with self._lock:
            self.decisions[decision] += 1
            self.latency_sum += latency_ms
            self.latency_count += 1
            for b in _BUCKETS_MS:
                if latency_ms <= b:
                    self.latency_buckets[b] += 1

    def request(self, endpoint: str) -> None:
        with self._lock:
            self.requests[endpoint] += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP fre_requests_total Requests by endpoint.",
                "# TYPE fre_requests_total counter",
            ]
            lines += [
                f'fre_requests_total{{endpoint="{k}"}} {v}'
                for k, v in sorted(self.requests.items())
            ]
            lines += [
                "# HELP fre_decisions_total Scoring decisions by outcome.",
                "# TYPE fre_decisions_total counter",
            ]
            lines += [
                f'fre_decisions_total{{decision="{k}"}} {v}'
                for k, v in sorted(self.decisions.items())
            ]
            lines += [
                "# HELP fre_scoring_latency_ms Engine scoring latency in milliseconds.",
                "# TYPE fre_scoring_latency_ms histogram",
            ]
            lines += [
                f'fre_scoring_latency_ms_bucket{{le="{b}"}} {self.latency_buckets[b]}'
                for b in _BUCKETS_MS
            ]
            lines += [
                f'fre_scoring_latency_ms_bucket{{le="+Inf"}} {self.latency_count}',
                f"fre_scoring_latency_ms_sum {self.latency_sum:.3f}",
                f"fre_scoring_latency_ms_count {self.latency_count}",
            ]
        return "\n".join(lines) + "\n"
