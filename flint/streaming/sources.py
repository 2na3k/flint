"""Streaming source implementations for Flint."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import pyarrow as pa


class StreamingSource(ABC):
    """Abstract base class for streaming data sources."""

    @abstractmethod
    def poll(self, batch_size: int) -> Optional[pa.RecordBatch]:
        """Poll for a batch of records.

        Returns a RecordBatch if data is available, or None if empty.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the source."""
        ...

    def _build_record_batch(self, rows: list[dict]) -> pa.RecordBatch:
        """Build a RecordBatch from a list of dicts using this source's schema."""
        arrays = [
            pa.array([r.get(f.name) for r in rows], type=f.type) for f in self._schema
        ]
        return pa.record_batch(arrays, schema=self._schema)


class KafkaSource(StreamingSource):
    """Streaming source that reads from a Kafka topic via confluent-kafka."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str,
        schema: pa.Schema,
        group_id: Optional[str] = None,
        consumer_config: Optional[dict] = None,
    ) -> None:
        from confluent_kafka import Consumer

        self._schema = schema
        self._topic = topic

        config: dict = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id or f"flint-{topic}",
            "auto.offset.reset": "latest",
        }
        if consumer_config:
            config.update(consumer_config)

        self._consumer = Consumer(config)
        self._consumer.subscribe([topic])

    def poll(self, batch_size: int) -> Optional[pa.RecordBatch]:
        messages = self._consumer.consume(num_messages=batch_size, timeout=0.5)
        rows: list[dict] = []
        for msg in messages:
            if msg.error() is not None:
                continue
            try:
                row = json.loads(msg.value().decode("utf-8"))
                rows.append(row)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        if not rows:
            return None
        return self._build_record_batch(rows)

    def close(self) -> None:
        self._consumer.close()


class WebSocketSource(StreamingSource):
    """Streaming source that reads from a WebSocket endpoint."""

    def __init__(
        self,
        uri: str,
        schema: pa.Schema,
        reconnect_delay: float = 1.0,
    ) -> None:
        self._uri = uri
        self._schema = schema
        self._reconnect_delay = reconnect_delay
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch background asyncio thread that feeds messages into the queue."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.run(self._connect_loop())

    async def _connect_loop(self) -> None:
        import websockets

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._uri) as ws:
                    async for message in ws:
                        if self._stop_event.is_set():
                            return
                        try:
                            data = json.loads(message)
                            self._queue.put_nowait(data)
                        except (json.JSONDecodeError, queue.Full):
                            continue
            except Exception:
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(self._reconnect_delay)

    def poll(self, batch_size: int) -> Optional[pa.RecordBatch]:
        rows: list[dict] = []
        for _ in range(batch_size):
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not rows:
            return None
        return self._build_record_batch(rows)

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
