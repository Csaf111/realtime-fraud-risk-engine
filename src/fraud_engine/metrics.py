"""Minimal, dependency-free Prometheus-style metrics.

Besides request counts and latency, the engine exports two monitoring signals that are the
raw material for drift detection:

* ``fre_rule_hits_total{rule="R00x"}``: how often each rule fired. Divided by
  ``fre_scored_total`` it gives a per-rule hit rate; a sudden change in a rule's hit rate is
  usually the first sign of an upstream data change or a new attack pattern.
* ``fre_risk_score``: a histogram of final risk scores, so the score distribution can be
  compared over time (for example a PSI against the training window) in Prometheus/Grafana.

Only rule codes and scores are recorded. Nothing about the transaction itself (accounts,
amounts) is ever exported as a label, which keeps metric cardinality bounded and avoids
leaking customer data into the monitoring stack.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable

_BUCKETS_MS = (1, 2, 5, 10, 25, 50, 100, 250)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.decisions: Counter[str] = Counter()
        self.requests: Counter[str] = Counter()
        self.latency_buckets: Counter[float] = Counter()
        self.latency_sum = 0.0
        self.latency_count = 0
        self.rule_hits: Counter[str] = Counter()
        self.score_buckets: Counter[float] = Counter()
        self.score_sum = 0.0
        self.score_count = 0

    def observe(
        self,
        decision: str,
        latency_ms: float,
        risk_score: float | None = None,
        rule_codes: Iterable[str] = (),
    ) -> None:
        """Record one scored transaction.

        ``risk_score`` and ``rule_codes`` are optional so callers that only track latency
        keep working; when given, they feed the score histogram and per-rule hit counters.
        """
        with self._lock:
            self.decisions[decision] += 1
            for code in set(rule_codes):
                self.rule_hits[code] += 1
            if risk_score is not None:
                self.score_sum += risk_score
                self.score_count += 1
                for sb in SCORE_BUCKETS:
                    if risk_score <= sb:
                        self.score_buckets[sb] += 1
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
                "# HELP fre_scored_total Transactions scored (denominator for rule hit rates).",
                "# TYPE fre_scored_total counter",
                f"fre_scored_total {self.latency_count}",
                "# HELP fre_rule_hits_total Times each rule fired.",
                "# TYPE fre_rule_hits_total counter",
            ]
            lines += [
                f'fre_rule_hits_total{{rule="{k}"}} {v}' for k, v in sorted(self.rule_hits.items())
            ]
            lines += [
                "# HELP fre_risk_score Distribution of final risk scores (for drift monitoring).",
                "# TYPE fre_risk_score histogram",
            ]
            lines += [
                f'fre_risk_score_bucket{{le="{b}"}} {self.score_buckets[b]}' for b in SCORE_BUCKETS
            ]
            lines += [
                f'fre_risk_score_bucket{{le="+Inf"}} {self.score_count}',
                f"fre_risk_score_sum {self.score_sum:.6f}",
                f"fre_risk_score_count {self.score_count}",
            ]
        return "\n".join(lines) + "\n"
