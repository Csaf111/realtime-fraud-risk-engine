"""Request/response schemas.

Field names on ``Transaction`` deliberately mirror the PaySim CSV columns so that a
row from the dataset can be posted to the API unchanged (minus the label columns).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TxnType(StrEnum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class Transaction(BaseModel):
    """A single transaction as it would arrive at authorisation time.

    ``isFraud`` and ``isFlaggedFraud`` are intentionally NOT accepted: they are labels /
    post-hoc system outputs and would leak the answer (see docs/leakage-safe-preprocessing.md).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    transaction_id: str | None = Field(default=None, max_length=128)
    step: int = Field(..., ge=0, description="Simulated hour since start (PaySim 'step').")
    type: TxnType
    amount: float = Field(..., ge=0)
    nameOrig: str = Field(..., min_length=1, max_length=64)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str = Field(..., min_length=1, max_length=64)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


class RuleHit(BaseModel):
    code: str
    description: str
    points: float


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(
        ..., description="Log-odds contribution relative to the training mean."
    )


class RiskResult(BaseModel):
    transaction_id: str | None
    risk_score: float = Field(..., ge=0, le=1)
    decision: Decision
    rules_score: float
    model_score: float | None
    reasons: list[str]
    rule_hits: list[RuleHit]
    top_features: list[FeatureContribution]
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(..., min_length=1)
    update_state: bool = True


class BatchResponse(BaseModel):
    results: list[RiskResult]
    count: int
