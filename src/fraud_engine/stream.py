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
import sys
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
    decisions: dict[str, int] = field(default_factory=dict)
    dead_letter_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "scored": self.scored,
            "dead_lettered": self.dead_lettered,
            "late": self.late,
            "decisions": dict(self.decisions),
            "dead_letter_reasons": dict(self.dead_letter_reasons),
        }


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
        self.stats = StreamStats()
        self.watermark: int | None = None  # highest step scored so far

    def _reject(self, reason: str, error: str, payload: str, offset: int) -> None:
        self.stats.dead_lettered += 1
        self.stats.dead_letter_reasons[reason] = self.stats.dead_letter_reasons.get(reason, 0) + 1
        self.dead_letter(DeadLetter(reason, _truncate(error), _truncate(payload), offset))

    def process(self, raw: bytes | str, offset: int | None = None) -> RiskResult | None:
        """Score one message. Returns the result, or ``None`` if it was dead-lettered."""
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
        self.watermark = txn.step if self.watermark is None else max(self.watermark, txn.step)
        self.stats.scored += 1
        key = result.decision.value
        self.stats.decisions[key] = self.stats.decisions.get(key, 0) + 1
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
            continue
        processor.process(msg.value() or b"", offset=msg.offset())
        consumer.commit(message=msg, asynchronous=False)
        handled += 1
    return processor.stats


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
    p.add_argument("--output", default="-", help="Where to write result JSON lines")
    p.add_argument("--dead-letter", help="Where to write dead-letter JSON lines")
    p.add_argument("--allowed-lateness", type=int, default=None, metavar="STEPS")
    p.add_argument("--no-update-state", action="store_true", help="Score without recording")
    args = p.parse_args(argv)

    engine = RiskEngine(Settings.from_env())
    out: IO[str] = sys.stdout if args.output == "-" else open(args.output, "w")  # noqa: SIM115
    dl_fh: IO[str] | None = open(args.dead_letter, "w") if args.dead_letter else None  # noqa: SIM115
    try:
        processor = StreamProcessor(
            engine,
            sink=lambda r: print(r.model_dump_json(), file=out),
            dead_letter=(lambda d: print(d.to_json(), file=dl_fh)) if dl_fh else None,
            update_state=not args.no_update_state,
            allowed_lateness_steps=args.allowed_lateness,
        )
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
        if out is not sys.stdout:
            out.close()
        if dl_fh is not None:
            dl_fh.close()
    print(json.dumps(stats.as_dict()), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
