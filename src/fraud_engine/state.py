"""Point-in-time velocity state.

The same ``VelocityStore`` is used online (inside the API) and offline (when building
training features), so the features a model is trained on are computed exactly the way
they are served. That removes a whole class of train/serve skew.

Leakage guarantee: ``snapshot()`` is always called *before* ``record()`` for the current
event, so a transaction never sees itself or anything that happened after it.

The in-memory implementation is per-process. For multi-replica deployments swap it for a
shared store (e.g. Redis sorted sets keyed by account) behind the same interface.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocitySnapshot:
    orig_txn_count: int = 0
    orig_amount_sum: float = 0.0
    dest_inbound_count: int = 0
    dest_inbound_sum: float = 0.0


class VelocityStore:
    def __init__(self, window_steps: int = 24, max_events_per_key: int = 10_000) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        self.window_steps = window_steps
        self.max_events_per_key = max_events_per_key
        self._orig: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._dest: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _window(self, events: deque[tuple[int, float]], step: int) -> tuple[int, float]:
        lower = step - self.window_steps  # exclusive
        count, total = 0, 0.0
        for ev_step, amount in events:
            if lower < ev_step <= step:
                count += 1
                total += amount
        return count, total

    def _prune(self, events: deque[tuple[int, float]], step: int) -> None:
        lower = step - self.window_steps
        while events and events[0][0] <= lower:
            events.popleft()
        while len(events) > self.max_events_per_key:
            events.popleft()

    def snapshot(self, name_orig: str, name_dest: str, step: int) -> VelocitySnapshot:
        """Return activity strictly before the current event, within the window."""
        with self._lock:
            oc, osum = self._window(self._orig.get(name_orig, deque()), step)
            dc, dsum = self._window(self._dest.get(name_dest, deque()), step)
        return VelocitySnapshot(oc, osum, dc, dsum)

    def record(self, name_orig: str, name_dest: str, step: int, amount: float) -> None:
        with self._lock:
            o = self._orig[name_orig]
            o.append((step, amount))
            self._prune(o, step)
            d = self._dest[name_dest]
            d.append((step, amount))
            self._prune(d, step)

    def reset(self) -> None:
        with self._lock:
            self._orig.clear()
            self._dest.clear()

    def __len__(self) -> int:
        return len(self._orig) + len(self._dest)
