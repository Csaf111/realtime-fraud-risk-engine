from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.feature_drift import (
    FEATURE_REFERENCE_FORMAT,
    FeatureBins,
    FeatureDriftMonitor,
    FeatureReference,
    quantile_edges,
    value_bin,
)
from fraud_engine.features import FEATURE_NAMES, compute_features
from fraud_engine.metrics import Metrics
from fraud_engine.model import LogisticBaseline
from fraud_engine.preprocessing import LABEL, build_feature_frame, load_paysim, temporal_split
from fraud_engine.scoring import RiskEngine
from fraud_engine.synthetic import write_csv
from fraud_engine.train import _feature_reference, feature_reference_path_for, main, train

from .conftest import fraud_txn, make_txn


def _feature_psi(text: str) -> dict[str, float]:
    out = {}
    for ln in text.splitlines():
        if ln.startswith("fre_feature_psi{"):
            name = ln.split('"')[1]
            out[name] = float(ln.split()[1])
    return out


def _benign_reference(n: int = 200) -> FeatureReference:
    """Reference built from many varied benign transactions (rules-free of fraud)."""
    rows = [
        compute_features(make_txn(amount=100.0 + 37.0 * i, step=i % 24, oldbalanceOrg=5e4))
        for i in range(n)
    ]
    return FeatureReference.from_rows(rows, steps=[0, 23], model_version="t")


# --- binning -------------------------------------------------------------------------


def test_quantile_edges_deciles():
    assert quantile_edges(range(1, 101)) == [10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_quantile_edges_dedupe_binary_and_constant():
    assert quantile_edges([0] * 90 + [1] * 10) == [0]
    assert quantile_edges([0] * 50 + [1] * 50) == [0, 1]
    assert quantile_edges([3.5] * 7) == [3.5]


@pytest.mark.parametrize(
    ("values", "n_bins"), [([], 10), ([1.0, math.nan], 10), ([1.0], 1), ([1.0, math.inf], 4)]
)
def test_quantile_edges_reject_bad_input(values, n_bins):
    with pytest.raises(ValueError):
        quantile_edges(values, n_bins)


@pytest.mark.parametrize(
    ("x", "expected"), [(-5.0, 0), (0.0, 0), (0.5, 1), (1.0, 1), (1.5, 2), (2.0, 2), (99.0, 3)]
)
def test_value_bin_upper_edge_inclusive_with_overflow(x, expected):
    assert value_bin(x, [0.0, 1.0, 2.0]) == expected


def test_value_bin_rejects_non_finite():
    with pytest.raises(ValueError):
        value_bin(math.nan, [0.0])


def test_feature_bins_from_values():
    b = FeatureBins.from_values([0] * 90 + [1] * 10)
    assert b.edges == [0.0]
    assert b.proportions == pytest.approx([0.9, 0.1])
    assert b.n_bins == 2


@pytest.mark.parametrize(
    ("edges", "props"),
    [
        ([], [1.0]),
        ([1.0, 1.0], [0.3, 0.3, 0.4]),
        ([2.0, 1.0], [0.3, 0.3, 0.4]),
        ([math.nan], [0.5, 0.5]),
        ([1.0], [1.0]),
        ([1.0], [0.7, 0.7]),
        ([1.0], [-0.5, 1.5]),
    ],
)
def test_feature_bins_validation(edges, props):
    with pytest.raises(ValueError):
        FeatureBins(edges=edges, proportions=props)


# --- reference -----------------------------------------------------------------------


def test_reference_covers_exactly_feature_names_and_round_trips(tmp_path):
    ref = _benign_reference()
    assert tuple(ref.features) == FEATURE_NAMES
    assert ref.n == 200
    path = tmp_path / "f.json"
    ref.save(path)
    assert json.loads(path.read_text())["format"] == FEATURE_REFERENCE_FORMAT
    assert FeatureReference.load(path) == ref


def test_reference_rejects_feature_mismatch_and_empty():
    good = _benign_reference().features
    missing = dict(list(good.items())[:-1])
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        FeatureReference(features=missing, n=1)
    reordered = dict(reversed(list(good.items())))
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        FeatureReference(features=reordered, n=1)
    with pytest.raises(ValueError):
        FeatureReference(features=good, n=0)
    with pytest.raises(ValueError):
        FeatureReference.from_rows([])


def test_reference_load_rejects_bad_files(tmp_path):
    good = _benign_reference().as_dict()
    path = tmp_path / "r.json"
    path.write_text(json.dumps({**good, "format": "other"}))
    with pytest.raises(ValueError, match="format"):
        FeatureReference.load(path)
    path.write_text(json.dumps({**good, "features": []}))
    with pytest.raises(ValueError, match="no features"):
        FeatureReference.load(path)
    extra = {**good["features"], "isFraud": {"edges": [0.0], "proportions": [0.5, 0.5]}}
    path.write_text(json.dumps({**good, "features": extra}))
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        FeatureReference.load(path)


# --- monitor -------------------------------------------------------------------------


def test_monitor_waits_for_min_samples_and_is_stable_on_reference_traffic():
    ref = _benign_reference()
    mon = FeatureDriftMonitor(ref, window=500, min_samples=200)
    for i in range(199):
        mon.observe(
            compute_features(make_txn(amount=100.0 + 37.0 * i, step=i % 24, oldbalanceOrg=5e4))
        )
    assert mon.psi() is None
    mon.observe(compute_features(make_txn(amount=100.0 + 37.0 * 199, step=7, oldbalanceOrg=5e4)))
    values = mon.psi()
    assert values is not None
    assert set(values) == set(FEATURE_NAMES)
    assert max(values.values()) < 0.1


def test_monitor_pinpoints_the_feature_that_moved():
    ref = _benign_reference()
    mon = FeatureDriftMonitor(ref, window=100, min_samples=50)
    for i in range(100):  # same traffic, but every transaction is 1000x larger
        mon.observe(
            compute_features(make_txn(amount=1e5 + 37.0 * i, step=i % 24, oldbalanceOrg=5e4))
        )
    values = mon.psi()
    assert values is not None
    assert values["log_amount"] > 0.25
    assert values["hour_sin"] < 0.1
    assert values["type_TRANSFER"] < 0.1


def test_monitor_window_forgets_and_skips_non_finite():
    ref = _benign_reference()
    mon = FeatureDriftMonitor(ref, window=5, min_samples=5)
    base = compute_features(make_txn())
    for _ in range(8):
        mon.observe(base)
    assert mon.count == 5
    assert all(sum(c) == 5 for c in mon.live_counts().values())
    mon.observe({**base, "log_amount": math.nan})
    assert mon.skipped == 1
    assert mon.count == 5


@pytest.mark.parametrize(("window", "min_samples"), [(0, 1), (10, 0), (10, 11)])
def test_monitor_validation(window, min_samples):
    with pytest.raises(ValueError):
        FeatureDriftMonitor(_benign_reference(10), window=window, min_samples=min_samples)


# --- engine hook ---------------------------------------------------------------------


def test_engine_feeds_point_in_time_features_to_observer():
    seen = []
    engine = RiskEngine(feature_observer=seen.append)
    first = make_txn(nameOrig="C1", step=1)
    engine.score(first)
    engine.score(make_txn(nameOrig="C1", step=2, transaction_id="t2"))
    assert seen[0] == compute_features(first)  # empty velocity: no self-leakage
    assert seen[1]["orig_txn_count_window"] == 1.0


def test_engine_survives_observer_failure():
    def boom(_features):
        raise RuntimeError("monitoring is down")

    result = RiskEngine(feature_observer=boom).score(make_txn())
    assert 0.0 <= result.risk_score <= 1.0


# --- metrics + API -------------------------------------------------------------------


def test_metrics_without_feature_drift():
    m = Metrics()
    m.observe("APPROVE", 1.0, risk_score=0.1)
    assert "fre_feature_psi" not in m.render()


def test_metrics_exports_feature_psi_once_warm():
    fd = FeatureDriftMonitor(_benign_reference(), window=10, min_samples=2)
    m = Metrics(feature_drift=fd)
    fd.observe(compute_features(make_txn()))
    text = m.render()
    assert "fre_feature_drift_window_count 1" in text
    assert "fre_feature_drift_skipped_total 0" in text
    assert "fre_feature_psi{" not in text
    fd.observe(compute_features(make_txn()))
    values = _feature_psi(m.render())
    assert set(values) == set(FEATURE_NAMES)


@pytest.fixture
def feature_drift_client(tmp_path):
    path = tmp_path / "feat.json"
    _benign_reference().save(path)
    settings = Settings(feature_reference_path=str(path), drift_window=50, drift_min_samples=3)
    with TestClient(create_app(settings)) as c:
        yield c


def test_api_reports_feature_drift(feature_drift_client):
    ready = feature_drift_client.get("/ready").json()
    assert ready["feature_drift_reference"] == "loaded"
    for i in range(3):
        feature_drift_client.post("/v1/score", json=fraud_txn(transaction_id=f"f{i}"))
    text = feature_drift_client.get("/metrics").text
    assert "fre_feature_drift_window_count 3" in text
    values = _feature_psi(text)
    assert values["type_TRANSFER"] > 0.25  # reference traffic had no transfers
    # Labels are only the fixed feature names: no IDs or amounts leak into metrics.
    assert "C999999999" not in text and "C100000001" not in text


def test_api_without_feature_reference(client):
    assert client.get("/ready").json()["feature_drift_reference"] == "none"
    client.post("/v1/score", json=make_txn())
    assert "fre_feature_psi" not in client.get("/metrics").text


def test_api_fails_fast_on_bad_feature_reference(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with (
        pytest.raises(ValueError),
        TestClient(create_app(Settings(feature_reference_path=str(path)))),
    ):
        pass


def test_settings_feature_reference_from_env(monkeypatch):
    monkeypatch.setenv("FRE_FEATURE_REFERENCE_PATH", "/x/feat.json")
    assert Settings.from_env().feature_reference_path == "/x/feat.json"


# --- training ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("feature-drift")
    data = write_csv(tmp / "s.csv", n=3000, fraud_rate=0.03)
    out = tmp / "model.json"
    metrics = train(data, out, gap_steps=5, version="fd-test")
    return data, out, metrics


def test_train_writes_feature_reference_from_training_window_only(trained):
    _, out, metrics = trained
    path = feature_reference_path_for(out)
    assert metrics["feature_reference"]["path"] == str(path)
    ref = FeatureReference.load(path)
    assert ref.model_version == "fd-test"
    assert tuple(ref.features) == FEATURE_NAMES
    assert ref.steps[1] <= metrics["split"]["train_max_step"]
    assert ref.steps[1] < metrics["split"]["test_min_step"]


def test_feature_reference_is_label_independent(trained):
    data, out, _ = trained
    model = LogisticBaseline.load(out)
    frame = build_feature_frame(load_paysim(data), window_steps=24)
    train_df, _, _ = temporal_split(frame, test_frac=0.2, gap_steps=5)
    flipped = train_df.copy()
    flipped[LABEL] = 1 - flipped[LABEL]
    if "isFlaggedFraud" in flipped:
        flipped["isFlaggedFraud"] = 1 - flipped["isFlaggedFraud"]
    assert _feature_reference(model, train_df, 0.2) == _feature_reference(model, flipped, 0.2)


def test_trained_feature_reference_serves(trained):
    _, out, _ = trained
    settings = Settings(
        model_path=str(out),
        feature_reference_path=str(feature_reference_path_for(out)),
        drift_window=10,
        drift_min_samples=1,
    )
    with TestClient(create_app(settings)) as c:
        c.post("/v1/score", json=make_txn())
        values = _feature_psi(c.get("/metrics").text)
    assert set(values) == set(FEATURE_NAMES)
    assert all(v >= 0 for v in values.values())


@pytest.mark.slow
def test_cli_prints_feature_reference_path(tmp_path, capsys):
    out = tmp_path / "m.json"
    main(["--synthetic", "2000", "--out", str(out), "--gap-steps", "5"])
    assert "Saved feature drift reference" in capsys.readouterr().out
    assert feature_reference_path_for(out).exists()
