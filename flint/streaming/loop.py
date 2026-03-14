"""MicroBatchLoop — orchestrates polling, processing, and sink fan-out."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import pyarrow as pa

if TYPE_CHECKING:
    from flint.planner.node import PartitionSpec


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------


def _assign_partitions(table: pa.Table, spec: "PartitionSpec") -> pa.Array:
    """Return an int32 Array of length ``len(table)`` with partition IDs.

    Dispatches on the concrete PartitionSpec subclass:
    - EvenPartitionSpec      → row-position formula
    - HashPartitionSpec      → DuckDB hash % n
    - UserDefinedPartitionSpec → user fn(RecordBatch) -> int32 Array
    """
    from flint.planner.node import EvenPartitionSpec, HashPartitionSpec, UserDefinedPartitionSpec

    n = spec.n_partitions
    total = len(table)

    if isinstance(spec, EvenPartitionSpec):
        ids = [(i * n) // total for i in range(total)]
        return pa.array(ids, type=pa.int32())

    if isinstance(spec, HashPartitionSpec):
        import duckdb
        conn = duckdb.connect()
        conn.register("__input__", table)
        key_expr = ", ".join(f'"{k}"' for k in spec.keys)
        sql = f"SELECT CAST(hash({key_expr}) % {n} AS INTEGER) FROM __input__"
        result = conn.execute(sql).fetchall()
        return pa.array([row[0] for row in result], type=pa.int32())

    if isinstance(spec, UserDefinedPartitionSpec):
        batch = table.to_batches(max_chunksize=total)[0]
        arr = spec.fn(batch)
        return arr.cast(pa.int32())

    raise NotImplementedError(f"Unsupported PartitionSpec: {type(spec).__name__}")


def _clone_spec_with_n(spec: "PartitionSpec", n: int) -> "PartitionSpec":
    """Return a copy of *spec* with ``n_partitions`` replaced by *n*."""
    return replace(spec, n_partitions=n)


class MicroBatchLoop:
    """Runs micro-batch streaming: poll → execute pipeline → fan out to sinks.

    Parameters
    ----------
    sources:
        One or more StreamingSource objects to poll each batch.
    pipeline:
        List of logical plan nodes applied to each batch via the Executor.
    sinks:
        One or more Sink objects that receive processed batches.
    batch_size:
        Maximum number of records to poll per source per batch.
    temp_dir:
        Temporary directory for intermediate Parquet files (Executor output).
    error_handler:
        Optional callable ``(exc: Exception) -> None`` called instead of
        re-raising on per-batch errors.  If None, exceptions propagate.
    """

    def __init__(
        self,
        sources: list,
        pipeline: list,
        sinks: list,
        batch_size: int,
        temp_dir: str,
        error_handler: Optional[Callable[[Exception], None]] = None,
        partition_spec: Optional["PartitionSpec"] = None,
        scheduler: Any = None,
    ) -> None:
        self._sources = sources
        self._pipeline = pipeline
        self._sinks = sinks
        self._batch_size = batch_size
        self._temp_dir = temp_dir
        self._error_handler = error_handler
        self._partition_spec = partition_spec
        self._scheduler = scheduler
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, batch_interval: float = 1.0) -> None:
        """Block and run the micro-batch loop until KeyboardInterrupt or stop()."""
        try:
            self._run_loop(batch_interval)
        except KeyboardInterrupt:
            pass

    def start_background(self, batch_interval: float = 1.0) -> threading.Thread:
        """Run the loop in a background daemon thread; returns the thread."""
        t = threading.Thread(
            target=self._run_loop,
            args=(batch_interval,),
            daemon=True,
        )
        t.start()
        return t

    def stop(self) -> None:
        """Signal the loop to stop after the current batch completes."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self, batch_interval: float) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()
            self._run_one_batch()
            elapsed = time.monotonic() - start
            sleep_time = max(0.0, batch_interval - elapsed)
            if sleep_time > 0 and not self._stop_event.is_set():
                self._stop_event.wait(timeout=sleep_time)

    def _run_one_batch(self) -> None:
        try:
            table = self._poll_sources()
            if table is None:
                return

            spec = self._partition_spec
            if spec is not None and spec.n_partitions > 1:
                result = self._execute_pipeline_distributed(table)
            else:
                result = self._execute_pipeline(table)

            for sink in self._sinks:
                sink.write(result)
        except Exception as exc:
            if self._error_handler is not None:
                self._error_handler(exc)
            else:
                raise

    def _execute_pipeline_distributed(self, table: pa.Table) -> pa.Table:
        """Split batch into N partitions, run pipeline in parallel, coalesce."""
        from flint.dataframe import InMemoryDataset
        from flint.executor.task import Task
        from flint.utils import generate_id

        spec = self._partition_spec
        effective_n = min(spec.n_partitions, len(table))
        if effective_n <= 1 or self._scheduler is None:
            return self._execute_pipeline(table)

        adjusted_spec = _clone_spec_with_n(spec, effective_n)
        partition_ids = _assign_partitions(table, adjusted_spec)
        tasks = self._build_tasks(table, partition_ids, effective_n)

        if not tasks:
            return table

        completed = self._scheduler.submit_batch(tasks)
        parts = [t.output_dataset.to_arrow() for t in completed]
        return pa.concat_tables(parts) if parts else pa.table({})

    def _build_tasks(
        self,
        table: pa.Table,
        partition_ids: pa.Array,
        effective_n: int,
    ) -> list:
        """Slice table by partition ID, wrap each slice as a Task."""
        from flint.dataframe import InMemoryDataset
        from flint.executor.task import Task
        from flint.utils import generate_id

        pid_list = partition_ids.to_pylist()
        tasks = []
        for p in range(effective_n):
            mask = pa.array([pid == p for pid in pid_list], type=pa.bool_())
            slice_ = table.filter(mask)
            if len(slice_) == 0:
                continue
            dataset = InMemoryDataset(slice_, partition_id=p)
            tasks.append(Task(
                task_id=generate_id(),
                stage_id="streaming",
                partition_id=p,
                pipeline=self._pipeline,
                input_datasets=[dataset],
                temp_dir=self._temp_dir,
            ))
        return tasks

    def _poll_sources(self) -> Optional[pa.Table]:
        tables: list[pa.Table] = []
        for source in self._sources:
            batch = source.poll(self._batch_size)
            if batch is not None:
                tables.append(pa.Table.from_batches([batch]))

        if not tables:
            return None
        return pa.concat_tables(tables)

    def _execute_pipeline(self, table: pa.Table) -> pa.Table:
        if not self._pipeline:
            return table

        from flint.dataframe import InMemoryDataset
        from flint.executor.executor import Executor
        from flint.executor.task import Task
        from flint.utils import generate_id

        dataset = InMemoryDataset(table, partition_id=0)
        task = Task(
            task_id=generate_id(),
            stage_id="streaming",
            partition_id=0,
            pipeline=self._pipeline,
            input_datasets=[dataset],
            temp_dir=self._temp_dir,
        )
        result = Executor().run(task)
        return result.to_arrow()
