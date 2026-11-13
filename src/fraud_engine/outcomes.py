"""Label-delayed performance monitoring: join late fraud labels to served decisions.

Fraud labels (chargebacks, analyst verdicts) arrive hours or days after the decision. Score
and feature PSI say the *inputs* moved; only labels say whether the engine still catches
fraud. :class:`OutcomeTracker` closes that loop for live traffic:

1. When a transaction with a ``transaction_id`` is scored (and the score is a real,
   state-updating decision), :meth:`OutcomeTracker.record_decision` remembers
   ``(risk_score, decision)`` in a bounded, insertion-ordered buffer. Nothing else about the
   transaction is kept: no accounts, amounts or features.
2. When the label arrives (``POST /v1/labels``), :meth:`OutcomeTracker.record_label` joins
   it to the stored decision, moves the pair out of the pending buffer and into a rolling
   window of the last ``window`` labelled outcomes.
3. ``/metrics`` exports the decision x label confusion counts plus rolling precision and
   recall at the REVIEW (``alert``) and BLOCK levels.

Leakage guarantee: labels flow *only* into this tracker. They never reach
``compute_features``, the rules, the model or the ``VelocityStore``, so posting a label can
never change a later score (``test_labels_do_not_change_later_scores``). A label is also
accepted only for a decision this replica actually served, so the join can't invent outcomes.

State is per replica and in memory. Route labels to the replica that scored the transaction
(or aggregate the counters in Prometheus, which works because they are plain sums); a
label for an unknown or evicted id is counted as ``unknown`` and ignored.

Pure Python and stdlib only, so it adds nothing to the serving image.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from enum import StrEnum

ALERT_DECISIONS = frozenset({"REVIEW", "BLOCK"})


class LabelStatus(StrEnum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class RateSnapshot:
    """Rolling precision / recall at one alert level. ``None`` when undefined (0/0)."""

    precision: float | None
    recall: float | None


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


class OutcomeTracker:
    """Bounded join of served decisions with late-arriving labels.

    Args:
        pending_capacity: decisions awaiting a label. When full, the oldest is evicted
            (counted in :attr:`evicted`), so memory stays bounded however many labels never
            arrive. ``0`` disables tracking entirely.
        window: number of most recent labelled outcomes used for precision/recall.
    """

    def __init__(self, pending_capacity: int = 100_000, window: int = 5000) -> None:
        if pending_capacity < 0:
            raise ValueError("pending_capacity must be >= 0")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.pending_capacity = pending_capacity
        self.window = window
        self._pending: OrderedDict[str, tuple[float, str]] = OrderedDict()
        # Bounded memory of ids already labelled, so a retried label is a duplicate rather
        # than "unknown" and is never double-counted.
        self._labelled_ids: OrderedDict[str, None] = OrderedDict()
        self._window: deque[tuple[str, bool]] = deque()
        self._lock = threading.Lock()
        self.evicted = 0
        self.label_status: Counter[str] = Counter()
        self.outcomes: Counter[tuple[str, bool]] = Counter()

    @property
    def enabled(self) -> bool:
        return self.pending_capacity > 0

    def record_decision(self, transaction_id: str | None, risk_score: float, decision: str) -> None:
        """Remember a served decision until its label arrives. Ids are required to join."""
        if not transaction_id or not self.enabled:
            return
        with self._lock:
            # A re-scored id keeps only its latest decision, which is the one that stands.
            self._pending.pop(transaction_id, None)
            self._labelled_ids.pop(transaction_id, None)
            self._pending[transaction_id] = (risk_score, decision)
            while len(self._pending) > self.pending_capacity:
                self._pending.popitem(last=False)
                self.evicted += 1

    def record_label(self, transaction_id: str, is_fraud: bool) -> LabelStatus:
        """Join one label to its stored decision."""
        with self._lock:
            entry = self._pending.pop(transaction_id, None)
            if entry is None:
                status = (
                    LabelStatus.DUPLICATE
                    if transaction_id in self._labelled_ids
                    else LabelStatus.UNKNOWN
                )
                self.label_status[status.value] += 1
                return status
            _, decision = entry
            self._labelled_ids[transaction_id] = None
            while len(self._labelled_ids) > max(self.pending_capacity, 1):
                self._labelled_ids.popitem(last=False)
            self.outcomes[(decision, is_fraud)] += 1
            if len(self._window) == self.window:
                self._window.popleft()
            self._window.append((decision, is_fraud))
            self.label_status[LabelStatus.MATCHED.value] += 1
            return LabelStatus.MATCHED

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def window_count(self) -> int:
        return len(self._window)

    def rates(self) -> dict[str, RateSnapshot]:
        """Precision/recall over the rolling window at the ``alert`` and ``block`` levels.

        ``alert`` counts REVIEW or BLOCK as a positive call (what an analyst queue sees);
        ``block`` counts only BLOCK.
        """
        with self._lock:
            window = list(self._window)
        frauds = sum(1 for _, y in window if y)
        out: dict[str, RateSnapshot] = {}
        for level, positives in (("alert", ALERT_DECISIONS), ("block", frozenset({"BLOCK"}))):
            called = [y for d, y in window if d in positives]
            tp = sum(called)
            out[level] = RateSnapshot(
                precision=_ratio(tp, len(called)),
                recall=_ratio(tp, frauds),
            )
        return out

    def render(self) -> list[str]:
        """Prometheus exposition lines (labels are fixed enums, never ids)."""
        with self._lock:
            status = dict(self.label_status)
            outcomes = dict(self.outcomes)
            pending, evicted, wcount = len(self._pending), self.evicted, len(self._window)
        lines = [
            "# HELP fre_labels_received_total Delayed labels received, by join status.",
            "# TYPE fre_labels_received_total counter",
        ]
        lines += [
            f'fre_labels_received_total{{status="{s.value}"}} {status.get(s.value, 0)}'
            for s in LabelStatus
        ]
        lines += [
            "# HELP fre_labelled_outcomes_total Served decisions joined to their label.",
            "# TYPE fre_labelled_outcomes_total counter",
        ]
        lines += [
            f'fre_labelled_outcomes_total{{decision="{d}",label="{"fraud" if y else "legit"}"}} {n}'
            for (d, y), n in sorted(outcomes.items())
        ]
        lines += [
            "# HELP fre_label_pending Served decisions still awaiting a label.",
            "# TYPE fre_label_pending gauge",
            f"fre_label_pending {pending}",
            "# HELP fre_label_pending_evicted_total Decisions dropped unlabelled (buffer full).",
            "# TYPE fre_label_pending_evicted_total counter",
            f"fre_label_pending_evicted_total {evicted}",
            "# HELP fre_label_window_count Labelled outcomes in the precision/recall window.",
            "# TYPE fre_label_window_count gauge",
            f"fre_label_window_count {wcount}",
        ]
        rates = self.rates()
        for metric, attr, text in (
            ("fre_label_precision", "precision", "Rolling precision of decisions vs labels."),
            ("fre_label_recall", "recall", "Rolling recall of decisions vs labels."),
        ):
            values = [(lvl, getattr(r, attr)) for lvl, r in rates.items()]
            defined = [(lvl, v) for lvl, v in values if v is not None]
            if defined:  # omitted while undefined (no calls / no frauds in the window)
                lines += [f"# HELP {metric} {text}", f"# TYPE {metric} gauge"]
                lines += [f'{metric}{{level="{lvl}"}} {v:.6f}' for lvl, v in defined]
        return lines
