"""Transparent, points-based risk rules.

Each rule is a small pure function with a stable code, a human-readable description and
a points value. The rules score is ``min(sum(points), 100) / 100``. Rules give every
decision an explanation an analyst (or a customer) can read, and they work from day one
without any trained model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .features import dest_balance_error, is_merchant, orig_balance_error
from .state import VelocitySnapshot

RuleFn = Callable[[Mapping[str, Any], VelocitySnapshot, "RuleContext"], bool]


@dataclass(frozen=True)
class RuleContext:
    high_amount_threshold: float = 200_000.0
    dest_velocity_threshold: int = 5
    orig_velocity_threshold: int = 3
    balance_tolerance: float = 1.0


@dataclass(frozen=True)
class Rule:
    code: str
    description: str
    points: float
    fn: RuleFn


def _type(txn: Mapping[str, Any]) -> str:
    t = txn["type"]
    return str(getattr(t, "value", t))


def _high_risk_type(txn, v, ctx) -> bool:
    return _type(txn) in {"TRANSFER", "CASH_OUT"}


def _account_drained(txn, v, ctx) -> bool:
    old = txn["oldbalanceOrg"]
    return (
        _type(txn) in {"TRANSFER", "CASH_OUT"}
        and old > 0
        and txn["newbalanceOrig"] == 0
        and txn["amount"] >= 0.95 * old
    )


def _high_amount(txn, v, ctx) -> bool:
    return txn["amount"] >= ctx.high_amount_threshold


def _orig_inconsistent(txn, v, ctx) -> bool:
    return abs(orig_balance_error(txn)) > ctx.balance_tolerance


def _dest_unchanged(txn, v, ctx) -> bool:
    return (
        not is_merchant(str(txn["nameDest"]))
        and txn["amount"] > 0
        and abs(dest_balance_error(txn)) > ctx.balance_tolerance
        and txn["oldbalanceDest"] == txn["newbalanceDest"]
    )


def _exceeds_balance(txn, v, ctx) -> bool:
    return _type(txn) != "CASH_IN" and txn["amount"] > txn["oldbalanceOrg"] + ctx.balance_tolerance


def _dest_velocity(txn, v, ctx) -> bool:
    return v.dest_inbound_count >= ctx.dest_velocity_threshold


def _orig_velocity(txn, v, ctx) -> bool:
    return v.orig_txn_count >= ctx.orig_velocity_threshold


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("R001", "High-risk transaction type (TRANSFER / CASH_OUT)", 15, _high_risk_type),
    Rule("R002", "Origin account emptied by this transaction", 30, _account_drained),
    Rule("R003", "Amount at or above high-value threshold", 15, _high_amount),
    Rule("R004", "Origin balances inconsistent with amount", 10, _orig_inconsistent),
    Rule("R005", "Destination balance unchanged after inbound funds", 20, _dest_unchanged),
    Rule("R006", "Amount exceeds origin's available balance", 10, _exceeds_balance),
    Rule("R007", "Destination received many payments in the window", 15, _dest_velocity),
    Rule("R008", "Origin sent many payments in the window", 10, _orig_velocity),
)


def evaluate_rules(
    txn: Mapping[str, Any],
    velocity: VelocitySnapshot,
    ctx: RuleContext | None = None,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
) -> tuple[float, list[Rule]]:
    """Return (rules_score in [0, 1], list of rules that fired, highest points first)."""
    ctx = ctx or RuleContext()
    hits = [r for r in rules if r.fn(txn, velocity, ctx)]
    hits.sort(key=lambda r: (-r.points, r.code))
    score = min(sum(r.points for r in hits), 100.0) / 100.0
    return score, hits
