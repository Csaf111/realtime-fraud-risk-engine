"""Explicit pre-authorisation feature contract.

This module is intentionally independent of the legacy PaySim feature vector.
A new model must be trained on this contract before it can be served; existing
models must never silently receive a reordered or shortened feature vector.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .features import is_merchant
from .state import VelocitySnapshot

PREAUTH_FEATURE_NAMES: tuple[str, ...] = (
    "log_amount",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_DEBIT",
    "amount_to_available_balance",
    "dest_is_merchant",
    "hour_sin",
    "hour_cos",
    "orig_txn_count_window",
    "log_orig_amount_window",
    "dest_inbound_count_window",
    "log_dest_inbound_window",
)

# Data available before an authorisation decision. In a real integration,
# available balance must be fetched from an authoritative pre-debit ledger.
PREAUTH_REQUIRED_FIELDS = frozenset(
    {"step", "type", "amount", "nameDest", "available_balance"}
)
FORBIDDEN_POST_EVENT_FIELDS = frozenset(
    {"newbalanceOrig", "newbalanceDest", "isFraud", "isFlaggedFraud"}
)


def compute_preauth_features(
    txn: Mapping[str, Any], velocity: VelocitySnapshot | None = None
) -> dict[str, float]:
    """Return features without reading post-event balances or labels.

    Reject non-finite/negative money even when called outside the HTTP schema.
    A pre-event velocity snapshot must be supplied by the caller; absent state
    defaults to an empty snapshot for offline experiments only.
    """
    missing = PREAUTH_REQUIRED_FIELDS.difference(txn)
    if missing:
        raise ValueError(f"Missing pre-authorisation fields: {sorted(missing)}")
    amount = float(txn["amount"])
    balance = float(txn["available_balance"])
    if not (math.isfinite(amount) and math.isfinite(balance)):
        raise ValueError("Amount and available balance must be finite")
    if amount < 0 or balance < 0:
        raise ValueError("Amount and available balance must be non-negative")
    step = int(txn["step"])
    if step < 0 or step != txn["step"]:
        raise ValueError("step must be a non-negative integer")
    kind = str(getattr(txn["type"], "value", txn["type"]))
    if kind not in {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}:
        raise ValueError("Unknown transaction type")
    v = velocity if velocity is not None else VelocitySnapshot()
    angle = 2 * math.pi * (step % 24) / 24
    features = {
        "log_amount": math.log1p(amount),
        "type_TRANSFER": float(kind == "TRANSFER"),
        "type_CASH_OUT": float(kind == "CASH_OUT"),
        "type_PAYMENT": float(kind == "PAYMENT"),
        "type_DEBIT": float(kind == "DEBIT"),
        "amount_to_available_balance": min(amount / (balance + 1), 10.0),
        "dest_is_merchant": float(is_merchant(str(txn["nameDest"]))),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "orig_txn_count_window": float(v.orig_txn_count),
        "log_orig_amount_window": math.log1p(v.orig_amount_sum),
        "dest_inbound_count_window": float(v.dest_inbound_count),
        "log_dest_inbound_window": math.log1p(v.dest_inbound_sum),
    }
    return features
