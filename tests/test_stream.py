from __future__ import annotations

import json
from typing import Any

import pytest

from fraud_engine.config import Settings
from fraud_engine.schemas import Decision, RiskResult
from fraud_engine.scoring import RiskEngine
from fraud_engine.stream import (
    DeadLetter,
    KafkaDeliveryError,
    KafkaTopicSink,
    MetricsTextfileWriter,
    StreamProcessor,
    StreamStats,
    main,
    run_kafka,
    write_metrics_textfile,
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
    assert stats.committed == 3 and stats.broker_errors == 1


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


# --- Kafka producer sink --------------------------------------------------------------


class FakeProducer:
    """confluent-kafka style producer: delivery callbacks fire inside flush()."""

    def __init__(self, fail: str | None = None, drop: bool = False, raise_on_produce=None):
        self.fail, self.drop, self.raise_on_produce = fail, drop, raise_on_produce
        self.pending: list[Any] = []
        self.records: list[tuple[str, bytes | None, bytes | None]] = []
        self.log: list[str] = []

    def produce(self, topic, value=None, key=None, on_delivery=None):
        if self.raise_on_produce is not None:
            raise self.raise_on_produce
        self.log.append(f"produce:{topic}")
        self.pending.append((topic, value, key, on_delivery))

    def flush(self, timeout: float = -1) -> int:
        self.log.append("flush")
        if self.drop:
            return len(self.pending)
        for topic, value, key, cb in self.pending:
            if self.fail is None:
                self.records.append((topic, key, value))
            if cb is not None:
                cb(self.fail, None)
        self.pending = []
        return 0


class LoggingConsumer(FakeConsumer):
    def __init__(self, items, log: list[str]) -> None:
        super().__init__(items)
        self.log = log

    def commit(self, message: FakeMsg, asynchronous: bool = True) -> None:
        super().commit(message, asynchronous)
        self.log.append(f"commit:{message.offset()}")


def test_kafka_sink_keys_results_and_confirms_before_commit():
    producer = FakeProducer()
    out = KafkaTopicSink(producer, "risk-results")
    dlq = KafkaTopicSink(producer, "risk-dlq")
    proc = StreamProcessor(RiskEngine(), out, dlq)
    consumer = LoggingConsumer(
        [
            FakeMsg(_msg(transaction_id="t1"), 5),
            FakeMsg(b"{bad", 6),
            FakeMsg(json.dumps(fraud_txn()).encode(), 7),
        ],
        producer.log,
    )
    run_kafka(proc, consumer, max_idle_polls=1)

    topics = [t for t, _, _ in producer.records]
    assert topics == ["risk-results", "risk-dlq", "risk-results"]
    first = json.loads(producer.records[0][2] or b"")
    assert producer.records[0][1] == b"t1" and first["transaction_id"] == "t1"
    assert producer.records[1][1] is None
    assert json.loads(producer.records[1][2] or b"")["reason"] == "json"
    assert producer.records[2][1] is None  # no transaction_id -> unkeyed
    assert json.loads(producer.records[2][2] or b"")["decision"] == "BLOCK"
    # Each input offset is committed only after its output was flushed and acknowledged.
    assert producer.log == [
        "produce:risk-results", "flush", "commit:5",
        "produce:risk-dlq", "flush", "commit:6",
        "produce:risk-results", "flush", "commit:7",
    ]  # fmt: skip
    assert out.delivered == 2 and dlq.delivered == 1


@pytest.mark.parametrize(
    ("producer", "match"),
    [
        (FakeProducer(fail="broker down"), "delivery to 'out' failed: broker down"),
        (FakeProducer(drop=True), "no delivery report"),
        (FakeProducer(raise_on_produce=BufferError("queue full")), "produce to 'out' failed"),
    ],
)
def test_kafka_sink_failures_block_commit(producer: FakeProducer, match: str):
    sink = KafkaTopicSink(producer, "out", delivery_timeout=0.01)
    consumer = FakeConsumer([FakeMsg(_msg(transaction_id="x"), 9)])
    proc = StreamProcessor(RiskEngine(), sink)
    with pytest.raises(KafkaDeliveryError, match=match):
        run_kafka(proc, consumer, max_idle_polls=1)
    assert consumer.committed == []  # redelivered on restart
    assert sink.delivered == 0


@pytest.mark.parametrize("kwargs", [{"topic": ""}, {"topic": "t", "delivery_timeout": 0}])
def test_kafka_sink_invalid_config(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        KafkaTopicSink(FakeProducer(), **kwargs)


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


def test_cli_writes_results_and_dead_letters_to_kafka_topics(tmp_path, monkeypatch, capsys):
    producer = FakeProducer()
    seen: list[str] = []

    def fake_make(bootstrap: str) -> FakeProducer:
        seen.append(bootstrap)
        return producer

    monkeypatch.setattr("fraud_engine.stream.make_kafka_producer", fake_make)
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(make_txn(transaction_id="z")) + "\n{bad\n")
    args = ["--input", str(src), "--kafka-bootstrap", "b:9092"]
    assert main([*args, "--output-topic", "res", "--dead-letter-topic", "dlq"]) == 0
    assert seen == ["b:9092"]
    assert [(t, k) for t, k, _ in producer.records] == [("res", b"z"), ("dlq", None)]
    assert capsys.readouterr().out == ""  # nothing went to stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--output-topic", "res", "--output", "x.jsonl"],
        ["--dead-letter-topic", "dlq", "--dead-letter", "x.jsonl"],
    ],
)
def test_cli_rejects_file_and_topic_for_same_output(tmp_path, extra: list[str]):
    src = tmp_path / "in.jsonl"
    src.write_text("")
    with pytest.raises(SystemExit):
        main(["--input", str(src), *extra])


# --- Prometheus metrics export ---------------------------------------------------------


def _parse_prom(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            samples[name] = float(value)
    return samples


def test_render_prometheus_reports_stream_counters():
    proc, _, _ = _collect(allowed_lateness_steps=0)
    proc.run(
        [
            _msg(step=5, transaction_id="secret-id", nameOrig="C999"),
            json.dumps(fraud_txn(step=6)),
            b"{bad",
            _msg(step=1),  # late, beyond tolerance
        ]
    )
    text = proc.stats.render_prometheus()
    samples = _parse_prom(text)
    assert samples["fre_stream_received_total"] == 4
    assert samples["fre_stream_scored_total"] == 2
    assert samples["fre_stream_dead_lettered_total"] == 2
    assert samples['fre_stream_dead_letters_total{reason="json"}'] == 1
    assert samples['fre_stream_dead_letters_total{reason="late"}'] == 1
    assert samples["fre_stream_late_total"] == 1
    assert samples['fre_stream_decisions_total{decision="APPROVE"}'] == 1
    assert samples['fre_stream_decisions_total{decision="BLOCK"}'] == 1
    assert samples["fre_stream_watermark_step"] == 6
    assert any(k.startswith('fre_stream_rule_hits_total{rule="R') for k in samples)
    assert sum(v for k, v in samples.items() if k.startswith("fre_stream_rule_hits_total")) == sum(
        proc.stats.rule_hits.values()
    )
    # Every exported series has HELP/TYPE metadata, and no message content leaks into labels.
    for name in {k.split("{")[0] for k in samples}:
        assert f"# TYPE {name} " in text and f"# HELP {name} " in text
    assert "secret-id" not in text and "C999" not in text


def test_render_prometheus_empty_stats_omits_watermark():
    samples = _parse_prom(StreamStats().render_prometheus())
    assert samples["fre_stream_received_total"] == 0
    assert "fre_stream_watermark_step" not in samples


def test_rule_hits_count_each_rule_once_per_event():
    proc, results, _ = _collect()
    proc.run([json.dumps(fraud_txn())])
    codes = {h.code for h in results[0].rule_hits}
    assert codes and proc.stats.rule_hits == dict.fromkeys(codes, 1)


def test_write_metrics_textfile_is_atomic_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "stream.prom"
    stats = StreamStats(received=3)
    write_metrics_textfile(stats, target)
    assert "fre_stream_received_total 3" in target.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["stream.prom"]

    def boom(*_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("fraud_engine.stream.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_metrics_textfile(StreamStats(received=9), target)
    # The previous file is intact and no temp file is left behind.
    assert "fre_stream_received_total 3" in target.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["stream.prom"]


def test_metrics_writer_rewrites_every_n_messages(tmp_path):
    target = tmp_path / "m.prom"
    writer = MetricsTextfileWriter(target, every=2)
    proc, _, _ = _collect(after_message=writer)
    proc.process(_msg())
    assert writer.writes == 0 and not target.exists()
    proc.process(b"{bad")  # dead letters count as messages too
    assert writer.writes == 1
    assert _parse_prom(target.read_text())["fre_stream_received_total"] == 2
    proc.process(_msg())
    assert writer.writes == 1
    writer.flush(proc.stats)
    assert _parse_prom(target.read_text())["fre_stream_received_total"] == 3


def test_metrics_writer_rejects_bad_interval(tmp_path):
    with pytest.raises(ValueError):
        MetricsTextfileWriter(tmp_path / "m.prom", every=0)


def test_cli_writes_metrics_file(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join([json.dumps(make_txn()), "{bad", json.dumps(fraud_txn())]) + "\n")
    prom = tmp_path / "stream.prom"
    args = ["--input", str(src), "--metrics-file", str(prom), "--metrics-every", "2"]
    assert main([*args, "--output", str(tmp_path / "out.jsonl")]) == 0
    samples = _parse_prom(prom.read_text())
    assert samples["fre_stream_received_total"] == 3
    assert samples["fre_stream_scored_total"] == 2
    stats = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert stats["received"] == 3 and stats["watermark"] == 10


def test_cli_flushes_metrics_even_when_sink_fails(tmp_path, monkeypatch):
    class Broken(FakeProducer):
        def produce(self, *a: Any, **k: Any) -> None:
            raise BufferError("queue full")

    monkeypatch.setattr("fraud_engine.stream.make_kafka_producer", lambda _b: Broken())
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(make_txn()) + "\n")
    prom = tmp_path / "stream.prom"
    with pytest.raises(KafkaDeliveryError):
        main(["--input", str(src), "--output-topic", "res", "--metrics-file", str(prom)])
    assert _parse_prom(prom.read_text())["fre_stream_received_total"] == 1


def test_cli_rejects_bad_metrics_interval(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("")
    with pytest.raises(SystemExit):
        main(["--input", str(src), "--metrics-every", "0"])
