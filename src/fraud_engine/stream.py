"""Streaming consumer mode (skeleton for Kafka ingestion).

The API scores one request at a time. Payment platforms often publish transactions to a
log (Kafka, Kinesis, Pub/Sub) instead, and want risk results written back to another
topic. This module is the transport-agnostic core of that mode:

* ``StreamProcessor`` takes raw message payloads (bytes / str JSON), validates them with
  the **same** ``Transaction`` schema as the API (so label columns are still rejected),
  scores them in arrival order with the **same** ``RiskEngine`` and hands each
  ``RiskResult`` to a sink. Malformed or rejected messages go to a dead-letter sink
  instead of crashing the consumer.
* ``run_kafka`` is a thin at-least-once loop over a confluent-kafka style consumer
  (``poll()`` / ``commit()``). The client library is an optional dependency and is only
  imported when Kafka mode is actually used, so the serving image stays slim.
* ``KafkaTopicSink`` writes results (keyed by ``transaction_id``) and dead letters back
  to Kafka topics, confirming delivery before returning so the consumer only commits
  input offsets whose output is durably on the broker.
* ``StreamStats.render_prometheus`` / ``write_metrics_textfile`` export the running
  counters as ``fre_stream_*`` Prometheus metrics (``--metrics-file``), written atomically in
  the node-exporter textfile-collector format so a batch replay or a consumer without an
  HTTP port can still be scraped.
* ``python -m fraud_engine.stream --input file.jsonl`` replays JSON lines through the
  processor and writes one result per line, handy for backfills and demos.

Point-in-time guarantee: velocity is read before and recorded after each event, exactly
as in the API and the offline replay. Offline training replays rows sorted by ``step``;
a stream only matches that if each account's events arrive in order. Late events (a
``step`` below the highest step already seen) are counted, and can be rejected with
``allowed_lateness_steps`` so they never perturb the state other events read.
See docs/streaming.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import IO, Any, Protocol

from pydantic import ValidationError

from .config import Settings
from .schemas import RiskResult, Transaction
from .scoring import RiskEngine

log = logging.getLogger(__name__)

#: Largest message payload accepted, in bytes. Bigger payloads are dead-lettered
#: without being parsed (a transaction is a few hundred bytes).
MAX_MESSAGE_BYTES = 64 * 1024


@dataclass(frozen=True)
class DeadLetter:
    """A message that could not be scored, with a short machine-readable reason."""

    reason: str  # "too_large" | "decode" | "json" | "schema" | "late"
    error: str
    payload: str
    offset: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "reason": self.reason,
                "error": self.error,
                "payload": self.payload,
                "offset": self.offset,
            }
        )


@dataclass
class StreamStats:
    received: int = 0
    scored: int = 0
    dead_lettered: int = 0
    late: int = 0
    committed: int = 0
    broker_errors: int = 0
    watermark: int | None = None
    decisions: dict[str, int] = field(default_factory=dict)
    dead_letter_reasons: dict[str, int] = field(default_factory=dict)
    rule_hits: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "scored": self.scored,
            "dead_lettered": self.dead_lettered,
            "late": self.late,
            "committed": self.committed,
            "broker_errors": self.broker_errors,
            "watermark": self.watermark,
            "decisions": dict(self.decisions),
            "dead_letter_reasons": dict(self.dead_letter_reasons),
            "rule_hits": dict(self.rule_hits),
        }

    def render_prometheus(self) -> str:
        """Render the counters as Prometheus text exposition (``fre_stream_*``).

        Label values are only decisions, dead-letter reason codes and rule codes, all
        bounded sets defined by this code base. Nothing from the message itself (accounts,
        amounts, transaction IDs) is ever exported, so cardinality stays fixed and no
        customer data reaches the monitoring stack.
        """
        lines: list[str] = []

        def counter(name: str, help_: str, value: int) -> None:
            lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} counter", f"{name} {value}"])

        def labelled(name: str, help_: str, label: str, values: dict[str, int]) -> None:
            lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} counter"])
            lines.extend(f'{name}{{{label}="{k}"}} {v}' for k, v in sorted(values.items()))

        counter("fre_stream_received_total", "Messages received from the stream.", self.received)
        counter("fre_stream_scored_total", "Messages validated and scored.", self.scored)
        counter(
            "fre_stream_dead_lettered_total", "Messages sent to dead letter.", self.dead_lettered
        )
        labelled(
            "fre_stream_dead_letters_total",
            "Dead letters by reason code.",
            "reason",
            self.dead_letter_reasons,
        )
        counter(
            "fre_stream_late_total", "Events behind the step watermark (scored or not).", self.late
        )
        counter(
            "fre_stream_committed_total", "Input offsets committed after handling.", self.committed
        )
        counter(
            "fre_stream_broker_errors_total",
            "Broker-level errors skipped without commit.",
            self.broker_errors,
        )
        labelled(
            "fre_stream_decisions_total",
            "Scoring decisions by outcome.",
            "decision",
            self.decisions,
        )
        labelled("fre_stream_rule_hits_total", "Times each rule fired.", "rule", self.rule_hits)
        if self.watermark is not None:
            lines.extend(
                [
                    "# HELP fre_stream_watermark_step Highest PaySim step scored so far.",
                    "# TYPE fre_stream_watermark_step gauge",
                    f"fre_stream_watermark_step {self.watermark}",
                ]
            )
        return "\n".join(lines) + "\n"


def write_metrics_textfile(stats: StreamStats, path: str | os.PathLike[str]) -> None:
    """Atomically write ``stats`` as a Prometheus textfile (node-exporter collector format).

    The text is written to a temporary file in the same directory and renamed over
    ``path``, so a scraper never reads a half-written file.
    """
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    fd, tmp = tempfile.mkstemp(prefix=".fre_stream_", suffix=".prom.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(stats.render_prometheus())
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class MetricsTextfileWriter:
    """``after_message`` hook that rewrites a metrics textfile every ``every`` messages.

    Call :meth:`flush` once at the end (the CLI does, even on failure) so the final
    counters are always on disk.
    """

    def __init__(self, path: str | os.PathLike[str], *, every: int = 1000) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        self.path = path
        self.every = every
        self.writes = 0
        self._since = 0

    def __call__(self, stats: StreamStats) -> None:
        self._since += 1
        if self._since >= self.every:
            self.flush(stats)

    def flush(self, stats: StreamStats) -> None:
        write_metrics_textfile(stats, self.path)
        self.writes += 1
        self._since = 0


ResultSink = Callable[[RiskResult], None]
DeadLetterSink = Callable[[DeadLetter], None]


def _truncate(text: str, limit: int = 1024) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


class StreamProcessor:
    """Validate, score and route a stream of raw transaction messages, in order."""

    def __init__(
        self,
        engine: RiskEngine,
        sink: ResultSink,
        dead_letter: DeadLetterSink | None = None,
        *,
        update_state: bool = True,
        allowed_lateness_steps: int | None = None,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        after_message: Callable[[StreamStats], None] | None = None,
    ) -> None:
        if allowed_lateness_steps is not None and allowed_lateness_steps < 0:
            raise ValueError("allowed_lateness_steps must be >= 0")
        if max_message_bytes < 1:
            raise ValueError("max_message_bytes must be >= 1")
        self.engine = engine
        self.sink = sink
        self.dead_letter = dead_letter or (lambda dl: None)
        self.update_state = update_state
        self.allowed_lateness_steps = allowed_lateness_steps
        self.max_message_bytes = max_message_bytes
        self.after_message = after_message
        self.stats = StreamStats()

    @property
    def watermark(self) -> int | None:
        """Highest step scored so far (``None`` before the first scored event)."""
        return self.stats.watermark

    def _reject(self, reason: str, error: str, payload: str, offset: int) -> None:
        self.stats.dead_lettered += 1
        self.stats.dead_letter_reasons[reason] = self.stats.dead_letter_reasons.get(reason, 0) + 1
        self.dead_letter(DeadLetter(reason, _truncate(error), _truncate(payload), offset))

    def process(self, raw: bytes | str, offset: int | None = None) -> RiskResult | None:
        """Score one message. Returns the result, or ``None`` if it was dead-lettered."""
        result = self._process(raw, offset)
        if self.after_message is not None:
            self.after_message(self.stats)
        return result

    def _process(self, raw: bytes | str, offset: int | None) -> RiskResult | None:
        self.stats.received += 1
        pos = self.stats.received - 1 if offset is None else offset

        size = len(raw) if isinstance(raw, bytes) else len(raw.encode("utf-8"))
        if size > self.max_message_bytes:
            self._reject("too_large", f"{size} bytes > {self.max_message_bytes}", "", pos)
            return None
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                self._reject("decode", str(exc), raw.decode("utf-8", "replace"), pos)
                return None
        else:
            text = raw
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self._reject("json", str(exc), text, pos)
            return None
        try:
            txn = Transaction.model_validate(payload)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            self._reject("schema", errors, text, pos)
            return None

        if self.watermark is not None and txn.step < self.watermark:
            self.stats.late += 1
            behind = self.watermark - txn.step
            if self.allowed_lateness_steps is not None and behind > self.allowed_lateness_steps:
                self._reject(
                    "late",
                    f"step {txn.step} is {behind} steps behind watermark {self.watermark}",
                    text,
                    pos,
                )
                return None

        result = self.engine.score(txn, update_state=self.update_state)
        wm = self.stats.watermark
        self.stats.watermark = txn.step if wm is None else max(wm, txn.step)
        self.stats.scored += 1
        key = result.decision.value
        self.stats.decisions[key] = self.stats.decisions.get(key, 0) + 1
        for code in {h.code for h in result.rule_hits}:
            self.stats.rule_hits[code] = self.stats.rule_hits.get(code, 0) + 1
        self.sink(result)
        return result

    def run(self, messages: Iterable[bytes | str]) -> StreamStats:
        """Process every message in order and return the running statistics."""
        for i, raw in enumerate(messages):
            self.process(raw, offset=i)
        return self.stats


# --------------------------------------------------------------------------------------
# Kafka adapter (optional dependency, confluent-kafka style client)
# --------------------------------------------------------------------------------------


class KafkaMessage(Protocol):
    def value(self) -> bytes | None: ...
    def error(self) -> Any: ...
    def offset(self) -> int | None: ...


class KafkaConsumerLike(Protocol):
    def poll(self, timeout: float) -> KafkaMessage | None: ...
    def commit(self, message: KafkaMessage, asynchronous: bool = ...) -> Any: ...


def run_kafka(
    processor: StreamProcessor,
    consumer: KafkaConsumerLike,
    *,
    poll_timeout: float = 1.0,
    max_messages: int | None = None,
    max_idle_polls: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> StreamStats:
    """At-least-once loop: poll, process, then commit the message's offset.

    The offset is committed only after the result (or dead letter) has been handed to
    its sink, so a crash replays the message instead of losing it. Sinks should therefore
    be idempotent on ``transaction_id``. Broker-level errors are logged and skipped
    without committing.
    """
    handled = 0
    idle = 0
    while not should_stop():
        if max_messages is not None and handled >= max_messages:
            break
        msg = consumer.poll(poll_timeout)
        if msg is None:
            idle += 1
            if max_idle_polls is not None and idle >= max_idle_polls:
                break
            continue
        idle = 0
        if msg.error():
            log.warning("Kafka error: %s", msg.error())
            processor.stats.broker_errors += 1
            continue
        processor.process(msg.value() or b"", offset=msg.offset())
        consumer.commit(message=msg, asynchronous=False)
        processor.stats.committed += 1
        handled += 1
    return processor.stats


class KafkaDeliveryError(RuntimeError):
    """A record could not be confirmed as delivered to its output topic."""


class KafkaProducerLike(Protocol):
    def produce(
        self,
        topic: str,
        value: bytes | None = ...,
        key: bytes | None = ...,
        on_delivery: Callable[[Any, Any], None] | None = ...,
    ) -> Any: ...
    def flush(self, timeout: float = ...) -> int: ...


class KafkaTopicSink:
    """Synchronous sink that writes results or dead letters to a Kafka topic.

    Each call produces one record and blocks until the broker acknowledges it (or
    ``delivery_timeout`` seconds pass). Any delivery error or timeout raises
    ``KafkaDeliveryError``; under ``run_kafka`` that means the input offset is *not*
    committed and the message is redelivered, which keeps the pipeline at-least-once
    end to end. Results are keyed by ``transaction_id`` so downstream consumers (or a
    compacted topic) can de-duplicate redeliveries; dead letters are unkeyed.

    Flushing per record trades throughput for a simple, provably safe commit order. It is
    the right default for a skeleton; batching would need commit-after-batch-flush.
    """

    def __init__(
        self, producer: KafkaProducerLike, topic: str, *, delivery_timeout: float = 10.0
    ) -> None:
        if not topic:
            raise ValueError("topic must be non-empty")
        if delivery_timeout <= 0:
            raise ValueError("delivery_timeout must be > 0")
        self.producer = producer
        self.topic = topic
        self.delivery_timeout = delivery_timeout
        self.delivered = 0

    def __call__(self, record: RiskResult | DeadLetter) -> None:
        if isinstance(record, RiskResult):
            key = record.transaction_id.encode("utf-8") if record.transaction_id else None
            value = record.model_dump_json().encode("utf-8")
        else:
            key, value = None, record.to_json().encode("utf-8")

        outcome: list[Any] = []

        def on_delivery(err: Any, _msg: Any) -> None:
            outcome.append(err)

        try:
            self.producer.produce(self.topic, value=value, key=key, on_delivery=on_delivery)
        except Exception as exc:  # BufferError, KafkaException, ...
            raise KafkaDeliveryError(f"produce to {self.topic!r} failed: {exc}") from exc
        remaining = self.producer.flush(self.delivery_timeout)
        if not outcome:
            raise KafkaDeliveryError(
                f"no delivery report from {self.topic!r} within {self.delivery_timeout}s "
                f"({remaining} still queued)"
            )
        if outcome[0] is not None:
            raise KafkaDeliveryError(f"delivery to {self.topic!r} failed: {outcome[0]}")
        self.delivered += 1


def make_kafka_producer(bootstrap: str) -> KafkaProducerLike:  # pragma: no cover - needs a broker
    """Build an idempotent confluent-kafka producer (``acks=all``)."""
    try:
        from confluent_kafka import Producer
    except ImportError as exc:
        raise RuntimeError(
            "Kafka mode needs the optional client: pip install confluent-kafka"
        ) from exc
    producer: KafkaProducerLike = Producer(
        {"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all"}
    )
    return producer


def make_kafka_consumer(  # pragma: no cover - needs the optional client and a broker
    bootstrap: str, topic: str, group_id: str
) -> KafkaConsumerLike:
    """Build a confluent-kafka consumer. Requires ``pip install confluent-kafka``."""
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:
        raise RuntimeError(
            "Kafka mode needs the optional client: pip install confluent-kafka"
        ) from exc
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    return consumer


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _iter_lines(fh: IO[str]) -> Iterator[str]:
    for line in fh:
        line = line.strip()
        if line:
            yield line


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m fraud_engine.stream",
        description="Score a stream of JSON transactions in order (JSONL file/stdin or Kafka).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSON-lines file of transactions ('-' for stdin)")
    src.add_argument("--kafka-topic", help="Kafka topic to consume (needs confluent-kafka)")
    p.add_argument("--kafka-bootstrap", default="localhost:9092")
    p.add_argument("--kafka-group", default="fraud-engine")
    p.add_argument("--output", help="Where to write result JSON lines (default: stdout)")
    p.add_argument("--dead-letter", help="Where to write dead-letter JSON lines")
    p.add_argument(
        "--output-topic", help="Kafka topic for results (keyed by transaction_id) instead of a file"
    )
    p.add_argument("--dead-letter-topic", help="Kafka topic for dead letters instead of a file")
    p.add_argument("--delivery-timeout", type=float, default=10.0, metavar="SECONDS")
    p.add_argument("--allowed-lateness", type=int, default=None, metavar="STEPS")
    p.add_argument("--no-update-state", action="store_true", help="Score without recording")
    p.add_argument(
        "--metrics-file",
        help="Write fre_stream_* Prometheus metrics here (textfile-collector format)",
    )
    p.add_argument(
        "--metrics-every",
        type=int,
        default=1000,
        metavar="N",
        help="Rewrite --metrics-file every N messages (default 1000; always at exit)",
    )
    args = p.parse_args(argv)
    if args.metrics_every < 1:
        p.error("--metrics-every must be >= 1")
    if args.output_topic and args.output:
        p.error("--output and --output-topic are mutually exclusive")
    if args.dead_letter_topic and args.dead_letter:
        p.error("--dead-letter and --dead-letter-topic are mutually exclusive")

    engine = RiskEngine(Settings.from_env())
    producer = (
        make_kafka_producer(args.kafka_bootstrap)
        if args.output_topic or args.dead_letter_topic
        else None
    )
    out: IO[str] = (
        sys.stdout if args.output in (None, "-") else open(args.output, "w")  # noqa: SIM115
    )
    dl_fh: IO[str] | None = open(args.dead_letter, "w") if args.dead_letter else None  # noqa: SIM115
    sink: ResultSink
    dead_letter: DeadLetterSink | None
    if producer is not None and args.output_topic:
        sink = KafkaTopicSink(producer, args.output_topic, delivery_timeout=args.delivery_timeout)
    else:
        sink = lambda r: print(r.model_dump_json(), file=out)  # noqa: E731
    if producer is not None and args.dead_letter_topic:
        dead_letter = KafkaTopicSink(
            producer, args.dead_letter_topic, delivery_timeout=args.delivery_timeout
        )
    elif dl_fh is not None:
        dead_letter = lambda d: print(d.to_json(), file=dl_fh)  # noqa: E731
    else:
        dead_letter = None
    metrics = (
        MetricsTextfileWriter(args.metrics_file, every=args.metrics_every)
        if args.metrics_file
        else None
    )
    processor = StreamProcessor(
        engine,
        sink=sink,
        dead_letter=dead_letter,
        update_state=not args.no_update_state,
        allowed_lateness_steps=args.allowed_lateness,
        after_message=metrics,
    )
    try:
        if args.input is not None:
            if args.input == "-":
                stats = processor.run(_iter_lines(sys.stdin))
            else:
                with open(args.input) as fh:
                    stats = processor.run(_iter_lines(fh))
        else:  # pragma: no cover - needs a broker
            consumer = make_kafka_consumer(args.kafka_bootstrap, args.kafka_topic, args.kafka_group)
            stats = run_kafka(processor, consumer)
    finally:
        if metrics is not None:
            metrics.flush(processor.stats)
        if out is not sys.stdout:
            out.close()
        if dl_fh is not None:
            dl_fh.close()
    print(json.dumps(stats.as_dict()), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
