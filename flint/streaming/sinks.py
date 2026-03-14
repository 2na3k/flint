"""Streaming sink implementations for Flint."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

import pyarrow as pa


class Sink(ABC):
    """Abstract base class for streaming sinks."""

    @abstractmethod
    def write(self, batch: pa.Table) -> None:
        """Write a batch of records to the sink."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the sink."""
        ...


class StdioSink(Sink):
    """Sink that prints batches to stdout — useful for debugging."""

    def __init__(self, label: str = "", max_rows: int = 20) -> None:
        self._label = label
        self._max_rows = max_rows

    def write(self, batch: pa.Table) -> None:
        if len(batch) == 0:
            return
        header = f"--- {self._label} ---" if self._label else "---"
        print(header)
        print(batch.slice(0, self._max_rows).to_pandas().to_string(index=False))

    def close(self) -> None:
        pass


class KafkaSink(Sink):
    """Sink that produces records to a Kafka topic as JSON."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str,
        producer_config: Optional[dict] = None,
        key_column: Optional[str] = None,
    ) -> None:
        from confluent_kafka import Producer

        self._topic = topic
        self._key_column = key_column

        config: dict = {"bootstrap.servers": bootstrap_servers}
        if producer_config:
            config.update(producer_config)

        self._producer = Producer(config)

    def write(self, batch: pa.Table) -> None:
        if len(batch) == 0:
            return

        col_names = batch.schema.names
        columns = {name: batch.column(name).to_pylist() for name in col_names}
        n = len(batch)

        for i in range(n):
            row = {k: columns[k][i] for k in col_names}
            value = json.dumps(row, default=str).encode()

            key: Optional[bytes] = None
            if self._key_column and self._key_column in row:
                key = str(row[self._key_column]).encode()

            self._producer.produce(self._topic, value=value, key=key)

        self._producer.flush()

    def close(self) -> None:
        self._producer.flush()
