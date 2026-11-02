"""Feature engineering shared by the online API and the offline training pipeline.

Every feature here is computable at authorisation time from:
  * the transaction payload itself, and
  * a ``VelocitySnapshot`` of *prior* activity.

No feature uses labels (``isFraud``), the legacy rule flag (``isFlaggedFraud``), future
events, or statistics computed over the whole dataset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .state import VelocitySnapshot

FEATURE_NAMES: tuple[str, ...] = (
    "log_amount",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_DEBIT",
    "orig_balance_error",
    "dest_balance_error",
    "orig_drained",
    "amount_to_orig_balance",
    "dest_is_merchant",
    "dest_zero_balances",
    "hour_sin",
    "hour_cos",
    "orig_txn_count_window",
    "log_orig_amount_window",
    "dest_inbound_count_window",
    "log_dest_inbound_window",
)

# Types where money LEAVES the origin account. CASH_IN credits the origin.
_DEBIT_TYPES = {"CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"}
_RATIO_CAP = 10.0


def signed_log1p(x: float) -> float:
    """Symmetric log transform that keeps sign and tames heavy tails."""
    return math.copysign(math.log1p(abs(x)), x)


def _type(txn: Mapping[str, Any]) -> str:
    t = txn["type"]
    return str(getattr(t, "value", t))


def orig_balance_error(txn: Mapping[str, Any]) -> float:
    """How far the origin's reported balances deviate from the amount moved."""
    old, new, amount = txn["oldbalanceOrg"], txn["newbalanceOrig"], txn["amount"]
    if _type(txn) in _DEBIT_TYPES:
        return float(old - amount - new)
    return float(old + amount - new)


def dest_balance_error(txn: Mapping[str, Any]) -> float:
    """How far the destination's reported balances deviate from the amount received."""
    return float(txn["oldbalanceDest"] + txn["amount"] - txn["newbalanceDest"])


def is_merchant(name: str) -> bool:
    # PaySim encodes merchants with an 'M' prefix and customers with 'C'.
    return name.startswith("M")


def compute_features(
    txn: Mapping[str, Any], velocity: VelocitySnapshot | None = None
) -> dict[str, float]:
    """Compute the model feature vector for one transaction."""
    v = velocity or VelocitySnapshot()
    t = _type(txn)
    amount = float(txn["amount"])
    old_orig = float(txn["oldbalanceOrg"])
    new_orig = float(txn["newbalanceOrig"])
    dest = str(txn["nameDest"])
    dest_merchant = is_merchant(dest)

    hour = int(txn["step"]) % 24
    angle = 2.0 * math.pi * hour / 24.0

    feats = {
        "log_amount": math.log1p(amount),
        "type_TRANSFER": float(t == "TRANSFER"),
        "type_CASH_OUT": float(t == "CASH_OUT"),
        "type_PAYMENT": float(t == "PAYMENT"),
        "type_DEBIT": float(t == "DEBIT"),
        "orig_balance_error": signed_log1p(orig_balance_error(txn)),
        "dest_balance_error": signed_log1p(dest_balance_error(txn)),
        "orig_drained": float(t in _DEBIT_TYPES and old_orig > 0 and new_orig == 0),
        "amount_to_orig_balance": min(amount / (old_orig + 1.0), _RATIO_CAP),
        "dest_is_merchant": float(dest_merchant),
        "dest_zero_balances": float(
            not dest_merchant
            and float(txn["oldbalanceDest"]) == 0
            and float(txn["newbalanceDest"]) == 0
            and amount > 0
        ),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "orig_txn_count_window": float(v.orig_txn_count),
        "log_orig_amount_window": math.log1p(v.orig_amount_sum),
        "dest_inbound_count_window": float(v.dest_inbound_count),
        "log_dest_inbound_window": math.log1p(v.dest_inbound_sum),
    }
    assert tuple(feats) == FEATURE_NAMES  # keeps the ordering contract explicit
    return feats


def to_vector(features: Mapping[str, float]) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]
