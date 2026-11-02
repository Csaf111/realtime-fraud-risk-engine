"""Transparent, deterministic heuristic scoring rules.

This educational baseline is not calibrated and must not be used for real
payment authorization or as a fraud probability.
"""

from app.schemas import RiskReason, ScoreRequest, ScoreResponse, TransactionType

MODEL_VERSION = "heuristic-v1"
HIGH_RISK_TYPES = frozenset({TransactionType.TRANSFER, TransactionType.CASH_OUT})


def score_transaction(transaction: ScoreRequest) -> ScoreResponse:
    """Score only pre-decision features, returning an auditable explanation."""
    reasons: list[RiskReason] = []

    if transaction.transaction_type in HIGH_RISK_TYPES:
        reasons.append(
            RiskReason(
                code="HIGH_RISK_TYPE",
                points=25,
                description="Transfer or cash-out transaction",
            )
        )

    if transaction.amount >= 100_000:
        reasons.append(
            RiskReason(
                code="VERY_LARGE_AMOUNT",
                points=25,
                description="Transaction amount is at least 100000",
            )
        )
    elif transaction.amount >= 10_000:
        reasons.append(
            RiskReason(
                code="LARGE_AMOUNT",
                points=15,
                description="Transaction amount is at least 10000",
            )
        )

    if (
        transaction.origin_balance is not None
        and transaction.amount > transaction.origin_balance
    ):
        reasons.append(
            RiskReason(
                code="EXCEEDS_ORIGIN_BALANCE",
                points=25,
                description="Amount exceeds known origin balance",
            )
        )

    risk_score = min(100, sum(reason.points for reason in reasons))
    if risk_score >= 60:
        band = "high"
    elif risk_score >= 30:
        band = "medium"
    else:
        band = "low"

    return ScoreResponse(
        transaction_id=transaction.transaction_id,
        risk_score=risk_score,
        risk_band=band,
        reasons=reasons,
        model_version=MODEL_VERSION,
    )
