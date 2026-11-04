import pandas as pd
import pytest

from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.preprocessing import (
    LABEL,
    LeakageError,
    build_feature_frame,
    fit_standardiser,
    leakage_checks,
    load_paysim,
    temporal_split,
    validate_schema,
)
from fraud_engine.state import VelocityStore
from fraud_engine.synthetic import write_csv


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> pd.DataFrame:
    path = write_csv(tmp_path_factory.mktemp("d") / "s.csv", n=3000, fraud_rate=0.03)
    return load_paysim(path)


@pytest.fixture(scope="module")
def frame(raw) -> pd.DataFrame:
    return build_feature_frame(raw)


def test_schema_validation_rejects_missing_columns(raw):
    with pytest.raises(ValueError, match="Missing"):
        validate_schema(raw.drop(columns=["amount"]))


def test_schema_validation_rejects_bad_type(raw):
    bad = raw.copy()
    bad["type"] = bad["type"].astype(str)
    bad.loc[0, "type"] = "WIRE"
    with pytest.raises(ValueError, match="Unknown"):
        validate_schema(bad)


def test_schema_validation_rejects_negative_money(raw):
    bad = raw.copy()
    bad.loc[0, "amount"] = -1
    with pytest.raises(ValueError, match="Negative"):
        validate_schema(bad)


def test_feature_frame_excludes_leaky_columns(frame):
    assert "isFlaggedFraud" not in frame.columns
    assert list(frame.columns[: len(FEATURE_NAMES)]) == list(FEATURE_NAMES)


def test_offline_features_match_online(raw, frame):
    """Train/serve parity: replaying rows online yields identical features."""
    df = (
        raw.drop(columns=["isFlaggedFraud"])
        .sort_values("step", kind="mergesort")
        .reset_index(drop=True)
    )
    store = VelocityStore(24)
    for i, rec in enumerate(df.head(300).to_dict("records")):
        rec["type"] = str(rec["type"])
        snap = store.snapshot(rec["nameOrig"], rec["nameDest"], int(rec["step"]))
        online = compute_features(rec, snap)
        store.record(rec["nameOrig"], rec["nameDest"], int(rec["step"]), float(rec["amount"]))
        offline = frame.iloc[i][list(FEATURE_NAMES)].to_dict()
        assert online == pytest.approx(offline)


def test_features_do_not_depend_on_labels(raw):
    """Flipping every label must not change a single feature value."""
    flipped = raw.copy()
    flipped[LABEL] = 1 - flipped[LABEL]
    a = build_feature_frame(raw)[list(FEATURE_NAMES)]
    b = build_feature_frame(flipped)[list(FEATURE_NAMES)]
    pd.testing.assert_frame_equal(a, b)


def test_future_rows_do_not_change_past_features(raw):
    """Point-in-time: truncating the future leaves earlier features untouched."""
    full = build_feature_frame(raw)
    cut_step = int(raw["step"].median())
    past = build_feature_frame(raw[raw["step"] <= cut_step])
    pd.testing.assert_frame_equal(
        full.iloc[: len(past)][list(FEATURE_NAMES)].reset_index(drop=True),
        past[list(FEATURE_NAMES)].reset_index(drop=True),
    )


def test_temporal_split_has_gap(frame):
    train, test, info = temporal_split(frame, test_frac=0.25, gap_steps=10)
    assert train["step"].max() + 10 < test["step"].min()
    assert info.train_rows + info.gap_rows + info.test_rows == len(frame)


def test_temporal_split_validates_args(frame):
    with pytest.raises(ValueError):
        temporal_split(frame, test_frac=1.5)
    with pytest.raises(ValueError, match="empty"):
        temporal_split(frame, test_frac=0.2, gap_steps=10_000)


def test_standardiser_uses_train_only(frame):
    train, _test, _ = temporal_split(frame, test_frac=0.3, gap_steps=5)
    mean, scale = fit_standardiser(train[list(FEATURE_NAMES)])
    assert mean == pytest.approx(train[list(FEATURE_NAMES)].mean().to_numpy())
    assert (scale > 0).all()


def test_leakage_checks_pass(frame):
    train, test, _ = temporal_split(frame, test_frac=0.25, gap_steps=5)
    report = leakage_checks(train, test, list(FEATURE_NAMES), 5)
    assert 0 < report["train_fraud_rate"] < 1


@pytest.mark.parametrize("col", ["isFraud", "isFlaggedFraud", "nameOrig", "step"])
def test_leakage_checks_reject_forbidden_features(frame, col):
    train, test, _ = temporal_split(frame, test_frac=0.25, gap_steps=5)
    with pytest.raises(LeakageError, match="Forbidden"):
        leakage_checks(train, test, [*FEATURE_NAMES, col], 5)


def test_leakage_checks_reject_overlap(frame):
    with pytest.raises(LeakageError, match="overlaps"):
        leakage_checks(frame, frame, list(FEATURE_NAMES), 0)


def test_leakage_checks_reject_single_class(frame):
    train, test, _ = temporal_split(frame, test_frac=0.25, gap_steps=5)
    test = test.assign(**{LABEL: 0})
    with pytest.raises(LeakageError, match="single class"):
        leakage_checks(train, test, list(FEATURE_NAMES), 5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
def test_schema_rejects_nonfinite_money(raw, column, value):
    bad = raw.copy()
    bad.loc[0, column] = value
    with pytest.raises(ValueError, match="Non-finite"):
        validate_schema(bad)


@pytest.mark.parametrize("value", [-1, None])
def test_schema_rejects_invalid_step(raw, value):
    bad = raw.copy()
    bad["step"] = bad["step"].astype("Int64")
    bad.loc[0, "step"] = value
    with pytest.raises(ValueError, match="Invalid transaction step"):
        validate_schema(bad)


def test_schema_rejects_nonbinary_label(raw):
    bad = raw.copy()
    bad["isFraud"] = bad["isFraud"].astype("int64")
    bad.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_schema(bad)


@pytest.mark.parametrize("column", ["nameOrig", "nameDest"])
def test_schema_rejects_missing_account_identifier(raw, column):
    bad = raw.copy()
    bad.loc[0, column] = None
    with pytest.raises(ValueError, match="Missing account"):
        validate_schema(bad)
