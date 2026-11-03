"""API contracts for transaction risk scoring."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TransactionType(StrEnum):
    """Supported PaySim transaction categories."""

    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class ScoreRequest(BaseModel):
    """Fields available at decision time; labels and post-event balances are excluded."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    transaction_type: TransactionType
    amount: float = Field(gt=0, allow_inf_nan=False)
    origin_balance: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RiskReason(BaseModel):
    """One transparent contribution to the risk score."""

    code: str
    points: int = Field(ge=0)
    description: str


class ScoreResponse(BaseModel):
    """Heuristic score: not a fraud probability."""

    transaction_id: str
    risk_score: int = Field(ge=0, le=100)
    risk_band: str
    reasons: list[RiskReason]
    model_version: str
