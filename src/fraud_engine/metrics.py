"""Minimal, dependency-free Prometheus-style metrics.

Besides request counts and latency, the engine exports two monitoring signals that are the
raw material for drift detection:

* ``fre_rule_hits_total{rule="R00x"}``: how often each rule fired. Divided by
  ``fre_scored_total`` it gives a per-rule hit rate; a sudden change in a rule's hit rate is
  usually the first sign of an upstream data change or a new attack pattern.
* ``fre_risk_score``: a histogram of final risk scores, so the score distribution can be
  compared over time (for example a PSI against the training window) in Prometheus/Grafana.

When a shadow model is configured (``FRE_SHADOW_MODEL_PATH``), ``fre_shadow_*`` series
compare what the candidate *would* have decided against the live decision, so a new model
can be judged on real traffic before it is promoted.

Only rule codes, decisions and scores are recorded. Nothing about the transaction itself (accounts,
amounts) is ever exported as a label, which keeps metric cardinality bounded and avoids
leaking customer data into the monitoring stack.

When a :class:`~fraud_engine.drift.DriftMonitor` is attached (``FRE_SCORE_REFERENCE_PATH``),
``fre_score_psi`` reports the PSI of the last ``FRE_DRIFT_WINDOW`` scores against the
training-time score reference.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable

from .drift import SCORE_BUCKETS, DriftMonitor

__all__ = ["SCORE_BUCKETS", "Metrics"]

_BUCKETS_MS = (1, 2, 5, 10, 25, 50, 100, 250)


class Metrics:
    def __init__(self, drift: DriftMonitor | None = None) -> None:
        self.drift = drift
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
        self.shadow_scored = 0
        self.shadow_errors = 0
        self.shadow_decisions: Counter[str] = Counter()
        self.shadow_disagreements: Counter[tuple[str, str]] = Counter()
        self.shadow_abs_delta_sum = 0.0
        self.shadow_latency_sum = 0.0

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
                if self.drift is not None:
                    self.drift.observe(risk_score)
            self.latency_sum += latency_ms
            self.latency_count += 1
            for b in _BUCKETS_MS:
                if latency_ms <= b:
                    self.latency_buckets[b] += 1

    def observe_shadow(
        self,
        primary_decision: str,
        primary_score: float,
        shadow_decision: str | None,
        shadow_score: float | None,
        latency_ms: float,
    ) -> None:
        """Record one shadow comparison. ``shadow_decision`` is ``None`` if the shadow failed."""
        with self._lock:
            self.shadow_latency_sum += latency_ms
            if shadow_decision is None or shadow_score is None:
                self.shadow_errors += 1
                return
            self.shadow_scored += 1
            self.shadow_decisions[shadow_decision] += 1
            self.shadow_abs_delta_sum += abs(shadow_score - primary_score)
            if shadow_decision != primary_decision:
                self.shadow_disagreements[(primary_decision, shadow_decision)] += 1

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
            if self.drift is not None:
                lines += self._render_drift(self.drift)
            if self.shadow_scored or self.shadow_errors:
                lines += self._render_shadow()
        return "\n".join(lines) + "\n"

    def _render_shadow(self) -> list[str]:
        lines = [
            "# HELP fre_shadow_scored_total Transactions the shadow model scored successfully.",
            "# TYPE fre_shadow_scored_total counter",
            f"fre_shadow_scored_total {self.shadow_scored}",
            "# HELP fre_shadow_errors_total Shadow scoring failures (primary unaffected).",
            "# TYPE fre_shadow_errors_total counter",
            f"fre_shadow_errors_total {self.shadow_errors}",
            "# HELP fre_shadow_decisions_total Decisions the shadow model would have made.",
            "# TYPE fre_shadow_decisions_total counter",
        ]
        lines += [
            f'fre_shadow_decisions_total{{decision="{k}"}} {v}'
            for k, v in sorted(self.shadow_decisions.items())
        ]
        lines += [
            "# HELP fre_shadow_disagreements_total Live vs shadow decision mismatches.",
            "# TYPE fre_shadow_disagreements_total counter",
        ]
        lines += [
            f'fre_shadow_disagreements_total{{primary="{p}",shadow="{s}"}} {v}'
            for (p, s), v in sorted(self.shadow_disagreements.items())
        ]
        lines += [
            "# HELP fre_shadow_abs_score_delta_sum Sum of |shadow - live| risk score.",
            "# TYPE fre_shadow_abs_score_delta_sum counter",
            f"fre_shadow_abs_score_delta_sum {self.shadow_abs_delta_sum:.6f}",
            "# HELP fre_shadow_latency_ms_sum Total extra latency spent on shadow scoring.",
            "# TYPE fre_shadow_latency_ms_sum counter",
            f"fre_shadow_latency_ms_sum {self.shadow_latency_sum:.3f}",
        ]
        return lines

    @staticmethod
    def _render_drift(drift: DriftMonitor) -> list[str]:
        lines = [
            "# HELP fre_score_drift_window_count Live scores in the drift window.",
            "# TYPE fre_score_drift_window_count gauge",
            f"fre_score_drift_window_count {drift.count}",
        ]
        value = drift.psi()
        if value is not None:  # omitted until min_samples scores have been seen
            lines += [
                "# HELP fre_score_psi PSI of live risk scores vs the training score reference.",
                "# TYPE fre_score_psi gauge",
                f"fre_score_psi {value:.6f}",
            ]
        return lines
