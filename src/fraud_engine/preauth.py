"""Point-in-time-safe rules-only scoring for the v2 authorisation contract.

This is deliberately separate from the v1 model and rules: v1 uses post-event
balances and must never be silently reused for an authorisation decision.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Decision, RiskResult, RuleHit, TxnType
from .state import VelocitySnapshot


class PreAuthTransaction(BaseModel):
    """Only fields known before funds move; extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    transaction_id: str | None = Field(default=None, max_length=128)
    step: int = Field(ge=0)
    type: TxnType
    amount: float = Field(ge=0, allow_inf_nan=False)
    nameOrig: str = Field(min_length=1, max_length=64)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    nameDest: str = Field(min_length=1, max_length=64)


def score_preauth(
    txn: PreAuthTransaction,
    velocity: VelocitySnapshot,
    *,
    review_threshold: float = 0.35,
    block_threshold: float = 0.7,
    high_amount_threshold: float = 200_000,
) -> RiskResult:
    """Transparent heuristic baseline; NOT a calibrated fraud probability."""
    started = time.perf_counter()
    hits: list[RuleHit] = []

    def hit(code: str, description: str, points: float) -> None:
        hits.append(RuleHit(code=code, description=description, points=points))

    if txn.type in (TxnType.TRANSFER, TxnType.CASH_OUT):
        hit("P001", "High-risk transfer or cash-out category", 15)
    if txn.amount >= high_amount_threshold:
        hit("P002", "High-value transaction", 15)
    if txn.type != TxnType.CASH_IN and txn.amount > txn.oldbalanceOrg:
        hit("P003", "Requested debit exceeds available origin balance", 30)
    if (
        txn.type != TxnType.CASH_IN
        and txn.oldbalanceOrg > 0
        and txn.amount / txn.oldbalanceOrg >= 0.95
    ):
        hit("P004", "Requested debit consumes at least 95% of available balance", 20)
    if velocity.orig_txn_count >= 3:
        hit("P005", "Origin has at least three recent prior transactions", 10)
    if velocity.dest_inbound_count >= 5:
        hit("P006", "Destination has at least five recent prior inbound transactions", 15)
    if (
        not txn.nameDest.startswith("M")
        and txn.amount > 0
        and txn.oldbalanceOrg == 0
    ):
        hit("P007", "Positive payment from an origin with zero available balance", 10)

    risk = round(min(sum(h.points for h in hits) / 100, 1.0), 6)
    decision = (
        Decision.BLOCK if risk >= block_threshold
        else Decision.REVIEW if risk >= review_threshold
        else Decision.APPROVE
    )
    return RiskResult(
        transaction_id=txn.transaction_id,
        risk_score=risk,
        decision=decision,
        rules_score=risk,
        model_score=None,
        reasons=[f"{h.code}: {h.description}" for h in hits],
        rule_hits=hits,
        top_features=[],
        model_version="preauth-rules-v2.0",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )
