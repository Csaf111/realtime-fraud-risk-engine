from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from fraud_engine.api import create_app
from fraud_engine.config import Settings
from fraud_engine.drift import (
    N_BINS,
    SCORE_BUCKETS,
    DriftMonitor,
    ScoreReference,
    bin_counts,
    psi,
    score_bin,
)
from fraud_engine.metrics import SCORE_BUCKETS as METRIC_BUCKETS
from fraud_engine.metrics import Metrics
from fraud_engine.model import LogisticBaseline
from fraud_engine.preprocessing import LABEL, build_feature_frame, load_paysim, temporal_split
from fraud_engine.synthetic import write_csv
from fraud_engine.train import _score_reference, main, reference_path_for, train

from .conftest import fraud_txn, make_txn

UNIFORM = [1 / N_BINS] * N_BINS


def _psi_value(text: str) -> float:
    (line,) = [ln for ln in text.splitlines() if ln.startswith("fre_score_psi ")]
    return float(line.split()[1])


def _reference(props: list[float] | None = None) -> ScoreReference:
    return ScoreReference(proportions=props or UNIFORM, n=1000)


# --- binning -------------------------------------------------------------------------


def test_bins_match_served_histogram():
    assert SCORE_BUCKETS == METRIC_BUCKETS


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, 0), (0.1, 0), (0.1000001, 1), (0.5, 4), (0.95, 9), (1.0, 9), (-1.0, 0), (2.0, 9)],
)
def test_score_bin_uses_inclusive_upper_edges(score, expected):
    assert score_bin(score) == expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_score_bin_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        score_bin(bad)


def test_bin_counts():
    assert bin_counts([0.05, 0.1, 0.15, 1.0]) == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]


# --- PSI -----------------------------------------------------------------------------


def test_psi_zero_for_identical_distributions():
    assert psi(UNIFORM, [5] * N_BINS) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_hand_computation():
    ref = [0.5, 0.5]
    live = [0.25, 0.75]
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    assert psi(ref, live) == pytest.approx(expected)


def test_psi_is_finite_with_empty_bins_and_grows_with_shift():
    ref = [1, 0, 0, 0]
    small = psi(ref, [9, 1, 0, 0])
    large = psi(ref, [0, 0, 0, 1])
    assert math.isfinite(large)
    assert 0 < small < large
    assert large > 0.25


@pytest.mark.parametrize(
    ("ref", "live", "eps"),
    [([1, 1], [1], 1e-4), ([0, 0], [1, 1], 1e-4), ([1, -1], [1, 1], 1e-4), ([1], [1], 0.0)],
)
def test_psi_rejects_bad_inputs(ref, live, eps):
    with pytest.raises(ValueError):
        psi(ref, live, epsilon=eps)


# --- reference -----------------------------------------------------------------------


def test_reference_round_trip(tmp_path):
    ref = ScoreReference.from_scores(
        [0.05, 0.05, 0.95, 0.55], model_weight=0.6, steps=[3, 9], model_version="v"
    )
    assert ref.n == 4
    assert ref.proportions[0] == 0.5
    path = tmp_path / "ref.json"
    ref.save(path)
    assert json.loads(path.read_text())["bins"] == list(SCORE_BUCKETS)
    assert ScoreReference.load(path) == ref


@pytest.mark.parametrize(
    "kwargs",
    [
        {"proportions": [1.0], "n": 1},
        {"proportions": [0.5] * N_BINS, "n": 1},
        {"proportions": [math.nan] + [0.0] * (N_BINS - 1), "n": 1},
        {"proportions": UNIFORM, "n": 0},
    ],
)
def test_reference_validation(kwargs):
    with pytest.raises(ValueError):
        ScoreReference(**kwargs)


def test_reference_load_rejects_wrong_format_or_bins(tmp_path):
    good = _reference().as_dict()
    path = tmp_path / "r.json"
    path.write_text(json.dumps({**good, "format": "other"}))
    with pytest.raises(ValueError, match="format"):
        ScoreReference.load(path)
    path.write_text(json.dumps({**good, "bins": [0.5, 1.0]}))
    with pytest.raises(ValueError, match="bins"):
        ScoreReference.load(path)


# --- monitor -------------------------------------------------------------------------


def test_monitor_waits_for_min_samples_then_reports():
    mon = DriftMonitor(_reference(), window=100, min_samples=10)
    for i in range(9):
        mon.observe((i + 0.5) / 10)
    assert mon.psi() is None
    mon.observe(0.95)
    assert mon.count == 10
    assert mon.psi() == pytest.approx(0.0, abs=1e-12)


def test_monitor_window_forgets_old_scores():
    ref = _reference([1.0] + [0.0] * (N_BINS - 1))
    mon = DriftMonitor(ref, window=5, min_samples=5)
    for _ in range(5):
        mon.observe(0.99)
    assert mon.psi() > 1.0
    for _ in range(5):
        mon.observe(0.01)
    assert mon.count == 5
    assert mon.live_counts() == [5] + [0] * (N_BINS - 1)
    assert mon.psi() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("window", "min_samples"), [(0, 1), (10, 0), (10, 11)])
def test_monitor_validation(window, min_samples):
    with pytest.raises(ValueError):
        DriftMonitor(_reference(), window=window, min_samples=min_samples)


# --- metrics + API -------------------------------------------------------------------


def test_metrics_without_drift_has_no_psi():
    m = Metrics()
    m.observe("APPROVE", 1.0, risk_score=0.5)
    assert "fre_score_psi" not in m.render()


def test_metrics_exports_psi_once_warm():
    m = Metrics(drift=DriftMonitor(_reference(), window=10, min_samples=2))
    m.observe("APPROVE", 1.0, risk_score=0.05)
    text = m.render()
    assert "fre_score_drift_window_count 1" in text
    assert "\nfre_score_psi " not in text
    m.observe("APPROVE", 1.0)  # no score: not fed to the drift window
    m.observe("APPROVE", 1.0, risk_score=0.05)
    text = m.render()
    assert "fre_score_drift_window_count 2" in text
    value = _psi_value(text)
    assert value > 0.25  # everything in one bin vs a uniform reference


@pytest.fixture
def drift_client(tmp_path):
    path = tmp_path / "ref.json"
    _reference([1.0] + [0.0] * (N_BINS - 1)).save(path)
    settings = Settings(score_reference_path=str(path), drift_window=50, drift_min_samples=3)
    with TestClient(create_app(settings)) as c:
        yield c


def test_api_reports_drift(drift_client):
    assert drift_client.get("/ready").json()["drift_reference"] == "loaded"
    for i in range(3):
        drift_client.post("/v1/score", json=fraud_txn(transaction_id=f"f{i}"))
    text = drift_client.get("/metrics").text
    assert "fre_score_drift_window_count 3" in text
    assert _psi_value(text) > 0.25


def test_api_without_reference(client):
    assert client.get("/ready").json()["drift_reference"] == "none"
    client.post("/v1/score", json=make_txn())
    assert "fre_score_psi" not in client.get("/metrics").text


def test_api_fails_fast_on_bad_reference(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with (
        pytest.raises(ValueError),
        TestClient(create_app(Settings(score_reference_path=str(path)))),
    ):
        pass


def test_settings_drift_validation(monkeypatch):
    with pytest.raises(ValueError):
        Settings(drift_window=0)
    with pytest.raises(ValueError):
        Settings(drift_window=10, drift_min_samples=11)
    monkeypatch.setenv("FRE_SCORE_REFERENCE_PATH", "/x/ref.json")
    monkeypatch.setenv("FRE_DRIFT_WINDOW", "200")
    monkeypatch.setenv("FRE_DRIFT_MIN_SAMPLES", "20")
    s = Settings.from_env()
    assert (s.score_reference_path, s.drift_window, s.drift_min_samples) == (
        "/x/ref.json",
        200,
        20,
    )


# --- training ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drift")
    data = write_csv(tmp / "s.csv", n=3000, fraud_rate=0.03)
    out = tmp / "model.json"
    metrics = train(data, out, gap_steps=5, version="drift-test", model_weight=0.5)
    return data, out, metrics


def test_train_writes_reference_from_training_window_only(trained):
    _, out, metrics = trained
    path = reference_path_for(out)
    assert metrics["score_reference"]["path"] == str(path)
    ref = ScoreReference.load(path)
    assert ref.model_version == "drift-test"
    assert ref.model_weight == 0.5
    # Built from the most recent slice of the training window, never the test window.
    assert ref.steps[1] <= metrics["split"]["train_max_step"]
    assert ref.steps[1] < metrics["split"]["test_min_step"]


def test_reference_is_label_independent(trained):
    data, out, _ = trained
    model = LogisticBaseline.load(out)
    frame = build_feature_frame(load_paysim(data), window_steps=24)
    train_df, _, _ = temporal_split(frame, test_frac=0.2, gap_steps=5)
    flipped = train_df.copy()
    flipped[LABEL] = 1 - flipped[LABEL]
    if "isFlaggedFraud" in flipped:
        flipped["isFlaggedFraud"] = 1 - flipped["isFlaggedFraud"]
    assert _score_reference(model, train_df, 0.2, 0.5) == _score_reference(model, flipped, 0.2, 0.5)


def test_trained_reference_serves(trained):
    _, out, _ = trained
    settings = Settings(
        model_path=str(out),
        score_reference_path=str(reference_path_for(out)),
        drift_window=10,
        drift_min_samples=1,
    )
    with TestClient(create_app(settings)) as c:
        c.post("/v1/score", json=make_txn())
        assert _psi_value(c.get("/metrics").text) >= 0


@pytest.mark.slow
def test_cli_prints_reference_path(tmp_path, capsys):
    out = tmp_path / "m.json"
    main(["--synthetic", "2000", "--out", str(out), "--gap-steps", "5"])
    assert "Saved drift score reference" in capsys.readouterr().out
    assert reference_path_for(out).exists()
