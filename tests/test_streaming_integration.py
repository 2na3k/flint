"""Integration tests for Flint streaming — require a running Kafka broker.

Run with:
    docker compose up kafka -d
    uv run pytest -m integration tests/test_streaming_integration.py -v
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import List

import pyarrow as pa
import pytest

from flint.session import Session
from flint.streaming.sinks import Sink
from flint.streaming.sources import StreamingSource

# Consumer rebalance on Kafka 4 takes ~3s on first connection.
# Tests wait long enough after loop start to guarantee messages are received.
_LOOP_WAIT = 6.0  # seconds to run the loop
_BATCH_INTERVAL = 0.2  # fast polling so we don't miss the rebalance window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CollectingSink(Sink):
    """Appends each written pa.Table to a list for assertions."""

    def __init__(self) -> None:
        self.batches: List[pa.Table] = []

    def write(self, batch: pa.Table) -> None:
        self.batches.append(batch)

    def close(self) -> None:
        pass

    @property
    def total_rows(self) -> int:
        return sum(len(b) for b in self.batches)


def produce_messages(bootstrap: str, topic: str, messages: list[dict]) -> None:
    """Synchronously produce JSON messages to a Kafka topic."""
    from confluent_kafka import Producer

    p = Producer({"bootstrap.servers": bootstrap})
    for msg in messages:
        p.produce(topic, value=json.dumps(msg).encode())
    p.flush()


def consume_all(bootstrap: str, topic: str, timeout: float = 8.0) -> list[dict]:
    """Consume all available messages from a topic (waits up to *timeout* seconds)."""
    from confluent_kafka import Consumer

    c = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"test-verifier-{uuid.uuid4().hex[:6]}",
            "auto.offset.reset": "earliest",
        }
    )
    c.subscribe([topic])
    messages = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = c.poll(0.5)
        if msg and not msg.error():
            messages.append(json.loads(msg.value()))
    c.close()
    return messages


@contextmanager
def temp_topic(bootstrap: str, topic: str):
    """Create topic before test, delete after."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    fs = admin.create_topics([NewTopic(topic, num_partitions=1)])
    for t, f in fs.items():
        try:
            f.result()
        except Exception:
            pass
    try:
        yield topic
    finally:
        admin.delete_topics([topic])


def _unique_topic(prefix: str = "flint-test") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _run_loop(sources, pipeline, sinks, tmp_path, wait=_LOOP_WAIT):
    """Helper: start loop in background, wait, stop, join."""
    from flint.streaming.loop import MicroBatchLoop

    loop = MicroBatchLoop(
        sources=sources,
        pipeline=pipeline,
        sinks=sinks,
        batch_size=100,
        temp_dir=str(tmp_path),
    )
    t = loop.start_background(batch_interval=_BATCH_INTERVAL)
    time.sleep(wait)
    loop.stop()
    t.join(timeout=5.0)
    return loop


# ---------------------------------------------------------------------------
# Kafka round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestKafkaRoundTrip:
    def test_kafka_source_reads_messages(self, kafka_bootstrap, tmp_path):
        topic = _unique_topic()
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("val", pa.float64())])
        messages = [{"id": i, "val": float(i) * 1.5} for i in range(10)]

        with temp_topic(kafka_bootstrap, topic):
            produce_messages(kafka_bootstrap, topic, messages)

            from flint.streaming.sources import KafkaSource

            source = KafkaSource(
                topic,
                kafka_bootstrap,
                schema,
                consumer_config={"auto.offset.reset": "earliest"},
            )
            sink = CollectingSink()
            _run_loop([source], [], [sink], tmp_path)
            source.close()

        assert sink.total_rows == 10

    def test_filter_then_kafka_sink(self, kafka_bootstrap, tmp_path):
        in_topic = _unique_topic("flint-in")
        out_topic = _unique_topic("flint-out")
        schema = pa.schema([pa.field("value", pa.int64())])
        messages = [
            {"value": i - 9} for i in range(20)
        ]  # -9..10; exactly 10 positive (1..10)

        with (
            temp_topic(kafka_bootstrap, in_topic),
            temp_topic(kafka_bootstrap, out_topic),
        ):
            produce_messages(kafka_bootstrap, in_topic, messages)

            from flint.planner.node import FilterNode
            from flint.streaming.sinks import KafkaSink
            from flint.streaming.sources import KafkaSource

            source = KafkaSource(
                in_topic,
                kafka_bootstrap,
                schema,
                consumer_config={"auto.offset.reset": "earliest"},
            )
            pipeline = [FilterNode(predicate="value > 0", is_sql=True)]
            sink = KafkaSink(topic=out_topic, bootstrap_servers=kafka_bootstrap)
            _run_loop([source], pipeline, [sink], tmp_path)
            source.close()
            sink.close()

            received = consume_all(kafka_bootstrap, out_topic, timeout=8.0)

        assert len(received) == 10
        assert all(r["value"] > 0 for r in received)

    def test_stdio_sink_prints_output(self, kafka_bootstrap, tmp_path, capsys):
        topic = _unique_topic()
        schema = pa.schema([pa.field("msg", pa.string())])
        messages = [{"msg": f"hello-{i}"} for i in range(5)]

        with temp_topic(kafka_bootstrap, topic):
            produce_messages(kafka_bootstrap, topic, messages)

            from flint.streaming.sinks import StdioSink
            from flint.streaming.sources import KafkaSource

            source = KafkaSource(
                topic,
                kafka_bootstrap,
                schema,
                consumer_config={"auto.offset.reset": "earliest"},
            )
            sink = StdioSink(label="integration-test")
            _run_loop([source], [], [sink], tmp_path)
            source.close()

        out = capsys.readouterr().out
        assert "integration-test" in out

    def test_error_handler_skips_malformed(self, kafka_bootstrap, tmp_path):
        from confluent_kafka import Producer

        topic = _unique_topic()
        schema = pa.schema([pa.field("value", pa.int64())])

        with temp_topic(kafka_bootstrap, topic):
            p = Producer({"bootstrap.servers": kafka_bootstrap})
            p.produce(topic, value=json.dumps({"value": 1}).encode())
            p.produce(topic, value=b"\xff\xfe invalid bytes")
            p.produce(topic, value=json.dumps({"value": 2}).encode())
            p.flush()

            from flint.streaming.loop import MicroBatchLoop
            from flint.streaming.sources import KafkaSource

            source = KafkaSource(
                topic,
                kafka_bootstrap,
                schema,
                consumer_config={"auto.offset.reset": "earliest"},
            )
            sink = CollectingSink()
            errors: list = []

            loop = MicroBatchLoop(
                sources=[source],
                pipeline=[],
                sinks=[sink],
                batch_size=100,
                temp_dir=str(tmp_path),
                error_handler=errors.append,
            )
            t = loop.start_background(batch_interval=_BATCH_INTERVAL)
            time.sleep(_LOOP_WAIT)
            loop.stop()
            t.join(timeout=5.0)
            source.close()

        # 2 valid messages; malformed silently skipped by KafkaSource
        assert sink.total_rows == 2
        assert errors == []


# ---------------------------------------------------------------------------
# WebSocket integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWebSocketIntegration:
    def test_websocket_source_reads_messages(self, tmp_path):
        import asyncio
        import threading

        received_port: list = []

        async def _server(websocket):
            for i in range(5):
                await websocket.send(json.dumps({"v": i}))
            await websocket.wait_closed()

        async def _run_server():
            import websockets

            async with websockets.serve(_server, "localhost", 0) as server:
                port = server.sockets[0].getsockname()[1]
                received_port.append(port)
                await asyncio.sleep(15)

        server_thread = threading.Thread(
            target=lambda: asyncio.run(_run_server()), daemon=True
        )
        server_thread.start()
        # wait for server to be ready
        deadline = time.monotonic() + 3.0
        while not received_port and time.monotonic() < deadline:
            time.sleep(0.05)

        if not received_port:
            pytest.skip("Could not start local WebSocket server")

        port = received_port[0]
        schema = pa.schema([pa.field("v", pa.int64())])

        from flint.streaming.loop import MicroBatchLoop
        from flint.streaming.sources import WebSocketSource

        source = WebSocketSource(f"ws://localhost:{port}", schema, reconnect_delay=0.5)
        source.start()
        time.sleep(1.0)  # let messages queue up

        sink = CollectingSink()
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[sink],
            batch_size=100,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()
        source.close()

        assert sink.total_rows == 5
