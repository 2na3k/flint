"""Distributed streaming demo — Ray workers process each micro-batch partition.

Demonstrates session.read_stream() with n_partitions and partition_by so that
transformation work fans out across Ray workers rather than running single-threaded.

Requirements:
    uv add ray
    uv run python examples/streaming_distributed.py

Press Ctrl+C to stop.

What to look for
----------------
Each printed batch shows a `worker_pid` column.  With n_partitions > 1 and
Ray mode you should see multiple distinct PIDs — proof that different OS
processes handled different partitions of the same micro-batch.

Scenarios
---------
  local_threaded    — local=True,  n_partitions=4  (ThreadPoolExecutor, same PID)
  local_single      — local=True,  n_partitions=1  (fast path, no distribution)
  ray_distributed   — local=False, n_partitions=4  (Ray workers, distinct PIDs)

Run one scenario:
    uv run python examples/streaming_distributed.py local_threaded
    uv run python examples/streaming_distributed.py ray_distributed
"""

from __future__ import annotations

import os
import queue
import random
import sys
import threading
import time
from typing import Optional

import pyarrow as pa

import flint
from flint.streaming.sources import StreamingSource


# ─────────────────────────────────────────────────────────────────────────────
# Mock source — emits fake HTTP request events at a steady rate
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.int64()),
        pa.field("method", pa.string()),
        pa.field("path", pa.string()),
        pa.field("status", pa.int64()),
        pa.field("latency_ms", pa.float64()),
    ]
)

METHODS = ["GET", "POST", "PUT", "DELETE"]
PATHS = ["/api/users", "/api/orders", "/api/products", "/health", "/api/auth"]
STATUSES = [200, 200, 200, 201, 400, 404, 500]


class MockRequestSource(StreamingSource):
    """Emits fake HTTP request events at *rate* events/second."""

    def __init__(self, rate: float = 20.0) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._counter = 0
        rng = random.Random(42)
        self._rng = rng

        def _produce():
            req_id = 0
            while not self._stop.is_set():
                self._queue.put(
                    {
                        "request_id": req_id,
                        "method": rng.choice(METHODS),
                        "path": rng.choice(PATHS),
                        "status": rng.choice(STATUSES),
                        "latency_ms": round(rng.uniform(1, 500), 1),
                    }
                )
                req_id += 1
                time.sleep(1.0 / rate)

        threading.Thread(target=_produce, daemon=True).start()

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
# Transform — tags each row with the worker's PID so we can see distribution
# ─────────────────────────────────────────────────────────────────────────────


def tag_worker(row: dict) -> dict:
    """Add worker_pid and slow flag to each row."""
    return {
        **row,
        "worker_pid": os.getpid(),
        "slow": row["latency_ms"] > 250,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────


RAY_ADDRESS = "ray://localhost:10001"


def scenario_local_single():
    """n_partitions=1 — fast path, single-threaded, all work on main process."""
    print("\n" + "═" * 64)
    print("  local_single — n_partitions=1  (fast path, no distribution)")
    print("═" * 64)
    print("\n  All rows processed on a single thread.  Press Ctrl+C to stop.\n")

    session = flint.Session(local=True)
    source = MockRequestSource(rate=20.0)
    try:
        session.read_stream(source, batch_size=20, n_partitions=1) \
            .filter("status >= 200") \
            .map(tag_worker) \
            .write_stdio(label="single", max_rows=5, batch_interval=1.5)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    print("\n  Done.")


def scenario_local_threaded():
    """n_partitions=4 with local=True — ThreadPoolExecutor, same process PID."""
    print("\n" + "═" * 64)
    print("  local_threaded — n_partitions=4, partition_by=['method']")
    print("  Scheduler: ThreadPoolExecutor  (all same PID, but parallel)")
    print("═" * 64)
    print("\n  4 threads handle 4 method-partitions per batch.  Press Ctrl+C to stop.\n")

    session = flint.Session(local=True)
    source = MockRequestSource(rate=30.0)
    try:
        session.read_stream(
            source,
            batch_size=40,
            n_partitions=4,
            partition_by=["method"],
        ).filter("status >= 200") \
         .map(tag_worker) \
         .write_stdio(label="threaded", max_rows=10, batch_interval=2.0)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
    print("\n  Done.")


KAFKA_BOOTSTRAP  = "localhost:9094"
OUT_TOPIC        = "test-integration-ray"


def _ensure_topic(topic: str) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    fs = admin.create_topics([NewTopic(topic, num_partitions=4)])
    for t, f in fs.items():
        try:
            f.result()
            print(f"  Created topic: {t}")
        except Exception as e:
            # TopicExistsException is fine
            if "already exists" not in str(e).lower():
                raise


def scenario_ray_distributed():
    """Ray workers process partitions and sink results into Kafka topic."""
    print("\n" + "═" * 64)
    print("  ray_distributed — n_partitions=4, partition_by=['method']")
    print(f"  Sink: Kafka topic '{OUT_TOPIC}' at {KAFKA_BOOTSTRAP}")
    print(f"  Connecting to Ray cluster at {RAY_ADDRESS}")
    print("═" * 64)

    try:
        import ray
    except ImportError:
        print("\n  ✗ Ray not installed.  Run:  uv add ray")
        return

    try:
        from confluent_kafka.admin import AdminClient
        AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP, "socket.timeout.ms": 2000}) \
            .list_topics(timeout=2)
    except Exception:
        print(f"\n  ✗ Kafka not reachable at {KAFKA_BOOTSTRAP}")
        print("    Start it with:  docker compose up kafka -d")
        return

    _ensure_topic(OUT_TOPIC)

    print("\n  Connecting to Ray cluster ...")
    try:
        session = flint.Session(local=False, n_workers=4, ray_address=RAY_ADDRESS)
    except Exception as e:
        print(f"\n  ✗ Could not connect to Ray at {RAY_ADDRESS}: {e}")
        print("    Start the cluster with:  docker compose up ray-head ray-worker -d")
        return

    print("  Connected — dashboard: http://localhost:8265")
    print(f"\n  Streaming into '{OUT_TOPIC}' — each batch fanned across 4 Ray workers.")
    print("  Press Ctrl+C to stop.\n")

    source = MockRequestSource(rate=40.0)
    try:
        session.read_stream(
            source,
            batch_size=40,
            n_partitions=4,
            partition_by=["method"],
        ).filter("status >= 200") \
         .map(tag_worker) \
         .write_kafka(
             topic=OUT_TOPIC,
             bootstrap_servers=KAFKA_BOOTSTRAP,
             key_column="method",
             batch_interval=2.0,
         )
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        session.stop()
    print("\n  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "local_single": scenario_local_single,
    "local_threaded": scenario_local_threaded,
    "ray_distributed": scenario_ray_distributed,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in SCENARIOS:
            print(f"Unknown scenario {name!r}. Choose from: {list(SCENARIOS)}")
            sys.exit(1)
        SCENARIOS[name]()
    else:
        # Default: run the two local scenarios back-to-back for a quick sanity check
        scenario_local_single()
        scenario_local_threaded()
