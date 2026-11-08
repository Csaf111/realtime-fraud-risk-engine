from __future__ import annotations

import json
from typing import Any

import pytest

from fraud_engine.config import Settings
from fraud_engine.schemas import Decision, RiskResult
from fraud_engine.scoring import RiskEngine
from fraud_engine.stream import (
    DeadLetter,
    StreamProcessor,
    main,
    run_kafka,
)

from .conftest import fraud_txn, make_txn


def _collect(**kwargs: Any) -> tuple[StreamProcessor, list[RiskResult], list[DeadLetter]]:
    results: list[RiskResult] = []
    dead: list[DeadLetter] = []
    proc = StreamProcessor(RiskEngine(), results.append, dead.append, **kwargs)
    return proc, results, dead


def _msg(**overrides: Any) -> bytes:
    return json.dumps(make_txn(**overrides)).encode()


def test_scores_valid_messages_in_order():
    proc, results, dead = _collect()
    stats = proc.run(
        [_msg(transaction_id="a"), json.dumps(fraud_txn(transaction_id="b")), _msg(step=11)]
    )
    assert [r.transaction_id for r in results] == ["a", "b", None]
    assert results[1].decision == Decision.BLOCK
    assert dead == []
    assert stats.received == stats.scored == 3
    assert stats.decisions == {"APPROVE": 2, "BLOCK": 1}
    assert proc.watermark == 11


def test_stream_matches_api_engine_exactly():
    """Online/offline parity: the stream path is the RiskEngine path, nothing else."""
    txns = [
        make_txn(nameOrig=f"C{i}", nameDest="CMULE", type="TRANSFER", newbalanceDest=100.0)
        for i in range(6)
    ]
    direct = RiskEngine()
    expected = [direct.score(t) for t in txns]
    proc, results, _ = _collect()
    proc.run(json.dumps(t) for t in txns)
    strip = {"latency_ms"}
    assert [r.model_dump(exclude=strip) for r in results] == [
        e.model_dump(exclude=strip) for e in expected
    ]
    assert any(h.code == "R007" for h in results[-1].rule_hits)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"\xff\xfe not utf8", "decode"),
        (b"{not json", "json"),
        (json.dumps({**make_txn(), "isFraud": 1}).encode(), "schema"),
        (json.dumps({**make_txn(), "isFlaggedFraud": 0}).encode(), "schema"),
        (json.dumps({**make_txn(), "amount": -5}).encode(), "schema"),
        (b'{"step": 1, "amount": "NaN"}', "schema"),
        (b"[1, 2, 3]", "schema"),
    ],
)
def test_bad_messages_are_dead_lettered_not_raised(raw: bytes, reason: str):
    proc, results, dead = _collect()
    assert proc.process(raw) is None
    assert results == []
    assert len(dead) == 1 and dead[0].reason == reason
    assert proc.stats.dead_letter_reasons == {reason: 1}
    # The consumer keeps going after a poison message.
    assert proc.process(_msg()) is not None


def test_labels_never_reach_the_engine_or_state():
    proc, _, dead = _collect()
    proc.process(json.dumps({**fraud_txn(), "isFraud": 1}))
    assert dead[0].reason == "schema" and "isFraud" in dead[0].error
    assert len(proc.engine.store) == 0


def test_oversized_message_rejected_without_parsing():
    proc, _, dead = _collect(max_message_bytes=50)
    proc.process(_msg())
    assert dead[0].reason == "too_large"
    assert dead[0].payload == ""


def test_dead_letter_payload_is_truncated():
    proc, _, dead = _collect()
    proc.process("x" * 5000)
    assert len(dead[0].payload) < 1100
    assert json.loads(dead[0].to_json())["reason"] == "json"


def test_late_events_counted_but_scored_by_default():
    proc, results, dead = _collect()
    proc.run([_msg(step=20), _msg(step=5)])
    assert len(results) == 2 and dead == []
    assert proc.stats.late == 1
    assert proc.watermark == 20


def test_late_events_beyond_tolerance_rejected_and_not_recorded():
    proc, results, dead = _collect(allowed_lateness_steps=2)
    proc.process(_msg(step=20, nameOrig="CA"))
    proc.process(_msg(step=19, nameOrig="CB"))  # within tolerance
    proc.process(_msg(step=10, nameOrig="CLATE"))  # too late
    assert [r.transaction_id for r in results] == [None, None]
    assert dead[0].reason == "late" and "watermark 20" in dead[0].error
    snap = proc.engine.store.snapshot("CLATE", "nobody", 10)
    assert snap.orig_txn_count == 0


def test_same_step_is_not_late():
    proc, _, _ = _collect(allowed_lateness_steps=0)
    proc.run([_msg(step=7), _msg(step=7)])
    assert proc.stats.late == 0 and proc.stats.scored == 2


def test_update_state_false_leaves_store_untouched():
    proc, _, _ = _collect(update_state=False)
    proc.run([_msg(), _msg()])
    assert len(proc.engine.store) == 0


def test_default_dead_letter_sink_is_noop():
    proc = StreamProcessor(RiskEngine(), lambda r: None)
    assert proc.process(b"{") is None
    assert proc.stats.dead_lettered == 1


@pytest.mark.parametrize("kwargs", [{"allowed_lateness_steps": -1}, {"max_message_bytes": 0}])
def test_invalid_config_rejected(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        StreamProcessor(RiskEngine(), lambda r: None, **kwargs)


def test_engine_settings_are_honoured():
    results: list[RiskResult] = []
    proc = StreamProcessor(RiskEngine(Settings(review_threshold=0.0)), results.append)
    proc.process(_msg())
    assert results[0].decision == Decision.REVIEW


# --- Kafka adapter -------------------------------------------------------------------


class FakeMsg:
    def __init__(self, value: bytes | None, offset: int, err: str | None = None) -> None:
        self._value, self._offset, self._err = value, offset, err

    def value(self) -> bytes | None:
        return self._value

    def offset(self) -> int:
        return self._offset

    def error(self) -> str | None:
        return self._err


class FakeConsumer:
    def __init__(self, items: list[FakeMsg | None]) -> None:
        self.items = list(items)
        self.committed: list[int] = []

    def poll(self, timeout: float) -> FakeMsg | None:
        return self.items.pop(0) if self.items else None

    def commit(self, message: FakeMsg, asynchronous: bool = True) -> None:
        assert asynchronous is False
        self.committed.append(message.offset())


def test_run_kafka_commits_after_processing_and_skips_broker_errors():
    consumer = FakeConsumer(
        [
            FakeMsg(_msg(transaction_id="k1"), 100),
            None,
            FakeMsg(None, 101, err="partition EOF"),
            FakeMsg(b"garbage", 102),
            FakeMsg(_msg(transaction_id="k2"), 103),
        ]
    )
    proc, results, dead = _collect()
    stats = run_kafka(proc, consumer, poll_timeout=0, max_idle_polls=3)
    assert [r.transaction_id for r in results] == ["k1", "k2"]
    assert dead[0].offset == 102
    # Broker errors are not committed; poison messages are (they went to dead letter).
    assert consumer.committed == [100, 102, 103]
    assert stats.received == 3


def test_run_kafka_stops_on_max_messages_and_should_stop():
    consumer = FakeConsumer([FakeMsg(_msg(), i) for i in range(5)])
    proc, results, _ = _collect()
    run_kafka(proc, consumer, max_messages=2)
    assert len(results) == 2
    run_kafka(proc, consumer, should_stop=lambda: True)
    assert len(results) == 2


def test_sink_failure_prevents_commit():
    def boom(_: RiskResult) -> None:
        raise RuntimeError("downstream unavailable")

    consumer = FakeConsumer([FakeMsg(_msg(), 7)])
    proc = StreamProcessor(RiskEngine(), boom)
    with pytest.raises(RuntimeError):
        run_kafka(proc, consumer, max_idle_polls=1)
    assert consumer.committed == []  # message will be redelivered


# --- CLI -----------------------------------------------------------------------------


def test_cli_jsonl_file(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join([json.dumps(make_txn()), "", "{bad", json.dumps(fraud_txn())]) + "\n")
    out = tmp_path / "out.jsonl"
    dl = tmp_path / "dl.jsonl"
    assert main(["--input", str(src), "--output", str(out), "--dead-letter", str(dl)]) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["decision"] for x in lines] == ["APPROVE", "BLOCK"]
    assert json.loads(dl.read_text())["reason"] == "json"
    stats = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert stats["scored"] == 2 and stats["dead_lettered"] == 1


def test_cli_stdin_to_stdout(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(make_txn(step=3)) + "\n"))
    assert main(["--input", "-", "--no-update-state", "--allowed-lateness", "0"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["decision"] == "APPROVE"
