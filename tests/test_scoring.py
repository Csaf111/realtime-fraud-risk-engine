"""Deterministic risk rule and threshold tests."""

import pytest
from pydantic import ValidationError

from app.schemas import ScoreRequest, TransactionType
from app.scoring import score_transaction


def request(
    transaction_type: TransactionType = TransactionType.PAYMENT,
    amount: float = 100,
    origin_balance: float | None = None,
) -> ScoreRequest:
    return ScoreRequest(
        transaction_id="txn-1",
        transaction_type=transaction_type,
        amount=amount,
        origin_balance=origin_balance,
    )


def test_small_payment_is_low_risk() -> None:
    result = score_transaction(request())
    assert result.risk_score == 0
    assert result.risk_band == "low"
    assert result.reasons == []


def test_transfer_raises_type_reason() -> None:
    result = score_transaction(request(TransactionType.TRANSFER))
    assert result.risk_score == 25
    assert [reason.code for reason in result.reasons] == ["HIGH_RISK_TYPE"]


@pytest.mark.parametrize(
    ("amount", "expected_score", "expected_code"),
    [
        (9_999.99, 0, None),
        (10_000, 15, "LARGE_AMOUNT"),
        (99_999.99, 15, "LARGE_AMOUNT"),
        (100_000, 25, "VERY_LARGE_AMOUNT"),
    ],
)
def test_amount_thresholds(
    amount: float, expected_score: int, expected_code: str | None
) -> None:
    result = score_transaction(request(amount=amount))
    assert result.risk_score == expected_score
    assert [reason.code for reason in result.reasons] == (
        [expected_code] if expected_code else []
    )


def test_combined_reasons_and_high_band() -> None:
    result = score_transaction(
        request(TransactionType.TRANSFER, amount=250_000, origin_balance=100_000)
    )
    assert result.risk_score == 75
    assert result.risk_band == "high"
    assert [reason.code for reason in result.reasons] == [
        "HIGH_RISK_TYPE",
        "VERY_LARGE_AMOUNT",
        "EXCEEDS_ORIGIN_BALANCE",
    ]
    assert sum(reason.points for reason in result.reasons) == result.risk_score


def test_missing_balance_is_not_assumed_zero() -> None:
    result = score_transaction(request(amount=500, origin_balance=None))
    assert result.risk_score == 0


def test_known_zero_balance_triggers_reason() -> None:
    result = score_transaction(request(amount=500, origin_balance=0))
    assert result.risk_score == 25
    assert result.reasons[0].code == "EXCEEDS_ORIGIN_BALANCE"


def test_equal_balance_does_not_trigger_reason() -> None:
    result = score_transaction(request(amount=500, origin_balance=500))
    assert result.risk_score == 0


@pytest.mark.parametrize("amount", [0, -1, float("inf"), float("nan")])
def test_invalid_amount_rejected(amount: float) -> None:
    with pytest.raises(ValidationError):
        request(amount=amount)


def test_unknown_transaction_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest(transaction_id="txn-1", transaction_type="WIRE", amount=10)
