"""Simple Flint streaming demo — mock HTTP request events, no external deps.

Run:
    uv run python examples/streaming.py
"""

from __future__ import annotations

import queue
import random
import threading
import time
from typing import Optional

import pyarrow as pa

import flint
from flint.streaming.sources import StreamingSource


# ─────────────────────────────────────────────────────────────────────────────
# Mock source — emits fake HTTP request events into a queue
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = pa.schema(
    [
        pa.field("method", pa.string()),
        pa.field("path", pa.string()),
        pa.field("status_code", pa.int64()),
        pa.field("latency_ms", pa.float64()),
    ]
)

METHODS = ["GET", "POST", "PUT", "DELETE"]
PATHS = ["/api/users", "/api/orders", "/api/products", "/health", "/api/auth"]
STATUSES = [200, 200, 200, 201, 400, 404, 500]


class MockRequestSource(StreamingSource):
    """Generates fake HTTP request events at a fixed rate."""

    def __init__(self, rate: float = 5.0) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._rate = rate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        rng = random.Random(42)
        while not self._stop.is_set():
            self._queue.put(
                {
                    "method": rng.choice(METHODS),
                    "path": rng.choice(PATHS),
                    "status_code": rng.choice(STATUSES),
                    "latency_ms": round(rng.uniform(1, 500), 1),
                }
            )
            time.sleep(1.0 / self._rate)

    def poll(self, batch_size: int) -> Optional[pa.RecordBatch]:
        rows = []
        while len(rows) < batch_size:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not rows:
            return None
        arrays = {field.name: [r[field.name] for r in rows] for field in SCHEMA}
        return pa.record_batch(arrays, schema=SCHEMA)

    def close(self) -> None:
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    session = flint.Session(local=True)
    source = MockRequestSource(rate=10.0)

    def flag_slow(row: dict) -> dict:
        return {**row, "slow": row["latency_ms"] > 200}

    print("Streaming mock HTTP requests — press Ctrl+C to stop.\n")

    session.read_stream(source, batch_size=20).filter("status_code >= 200").map(
        flag_slow
    ).write_stdio(label="requests", max_rows=20, batch_interval=1.0)

    session.stop()
    print("\nDone.")
