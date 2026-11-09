"""Contract tests for the standalone load probe (no server required)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "latency_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "latency_benchmark.py"
)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_percentile_interpolation():
    assert benchmark.percentile([10, 20, 30, 40], 50) == 25
    assert benchmark.percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)


def test_percentile_empty():
    with pytest.raises(ValueError):
        benchmark.percentile([], 99)


def test_payload_is_unique_and_contains_no_labels():
    import json

    first = json.loads(benchmark.payload(1))
    second = json.loads(benchmark.payload(2))
    assert first["nameOrig"] != second["nameOrig"]
    assert first["transaction_id"] != second["transaction_id"]
    assert "isFraud" not in first
    assert "isFlaggedFraud" not in first


def test_run_summary_with_mock_probe(monkeypatch):
    monkeypatch.setattr(
        benchmark, "probe",
        lambda _url, i, _timeout: (float(i + 1), i % 2 == 0, "" if i % 2 == 0 else "http_503"),
    )
    result = benchmark.run("http://localhost:8000/v1/score", 4, 2, 1.0, 0)
    assert result["successes"] == 2
    assert result["failures"] == {"http_503": 2}
    assert result["latency_ms"]["p50"] == 2
    assert result["throughput_rps"] > 0


@pytest.mark.parametrize("requests,concurrency,timeout,warmup", [
    (0, 1, 1, 0), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, -1),
])
def test_invalid_configuration(requests, concurrency, timeout, warmup):
    with pytest.raises(ValueError):
        benchmark.run("http://localhost:8000/v1/score", requests, concurrency, timeout, warmup)
