"""Driver — orchestrates stage execution, shuffle, and join coordination."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from flint.dataframe import Dataset, DuckDBDataset, InMemoryDataset
from flint.executor.scheduler import Scheduler
from flint.executor.task import Task
from flint.planner.node import (
    BroadcastNode,
    EvenPartitionSpec,
    HashPartitionSpec,
    JoinNode,
    ReadNode,
    ShuffleNode,
    UserDefinedPartitionSpec,
)
from flint.planner.planner import ExecutionPlan, ExecutionStage
from flint.utils import generate_id, generate_temp_path

if TYPE_CHECKING:
    from flint.session import Session


class Driver:
    """Coordinates execution of an ``ExecutionPlan``.

    Responsibilities
    ----------------
    - Execute stages in dependency order.
    - Build ``Task`` objects for each partition in a stage.
    - Pass results from one stage as inputs to the next.
    - Perform shuffle data movement (Driver-side, not Executor-side).
    - Handle join coordination (co-locate left/right partitions).
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._scheduler = Scheduler(
            local=getattr(session, "local", True),
            n_workers=getattr(session, "n_workers", 4),
        )
        self._scheduler.start()

    def execute(self, plan: ExecutionPlan) -> List[Dataset]:
        """Run all stages and return the output partition datasets."""
        stage_results: Dict[str, List[Dataset]] = {}

        for stage in plan.stages:
            datasets = self._execute_stage(stage, stage_results)
            stage_results[stage.stage_id] = datasets

        return stage_results[plan.output_stage_id]

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    def _execute_stage(
        self,
        stage: ExecutionStage,
        stage_results: Dict[str, List[Dataset]],
    ) -> List[Dataset]:
        first_node = stage.pipeline[0] if stage.pipeline else None

        if isinstance(first_node, ReadNode):
            return self._execute_source_stage(stage)

        if isinstance(first_node, ShuffleNode):
            input_datasets = stage_results[stage.depends_on[0]]
            return self._execute_shuffle_stage(stage, first_node, input_datasets)

        if isinstance(first_node, BroadcastNode):
            input_datasets = stage_results[stage.depends_on[0]]
            return self._execute_broadcast_stage(stage, input_datasets)

        if isinstance(first_node, JoinNode):
            left_datasets = stage_results[stage.left_stage_id]
            right_datasets = stage_results[stage.right_stage_id]
            return self._execute_join_stage(stage, first_node, left_datasets, right_datasets)

        # Regular transform stage
        input_datasets = stage_results[stage.depends_on[0]]
        return self._execute_transform_stage(stage, input_datasets)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _execute_source_stage(self, stage: ExecutionStage) -> List[Dataset]:
        """Create one task per partition for source + pipeline nodes."""
        tasks = [
            Task(
                task_id=generate_id(),
                stage_id=stage.stage_id,
                partition_id=pid,
                pipeline=stage.pipeline,
                input_datasets=[],
                temp_dir=self.session.temp_dir,
            )
            for pid in range(stage.n_partitions)
        ]
        self._scheduler.submit_batch(tasks)
        return [t.output_dataset for t in tasks]

    def _execute_transform_stage(
        self, stage: ExecutionStage, input_datasets: List[Dataset]
    ) -> List[Dataset]:
        """One task per partition; passes one input dataset per task."""
        # Align input count to stage partition count (may differ after repartition)
        n = stage.n_partitions
        aligned = _align_datasets(input_datasets, n)

        tasks = [
            Task(
                task_id=generate_id(),
                stage_id=stage.stage_id,
                partition_id=pid,
                pipeline=stage.pipeline,
                input_datasets=[aligned[pid]],
                temp_dir=self.session.temp_dir,
            )
            for pid in range(n)
        ]
        self._scheduler.submit_batch(tasks)
        return [t.output_dataset for t in tasks]

    def _execute_shuffle_stage(
        self,
        stage: ExecutionStage,
        shuffle_node: ShuffleNode,
        input_datasets: List[Dataset],
    ) -> List[Dataset]:
        """Perform partition assignment and data movement in the Driver.

        Steps
        -----
        1. For each input partition, compute per-row partition assignments.
        2. Bucket rows into ``n_out`` groups.
        3. Merge all buckets with the same ID into output partition files.
        """
        spec = shuffle_node.partition_spec
        n_out = spec.n_partitions

        # bucket_tables[pid] accumulates Arrow tables for output partition pid
        bucket_tables: List[List[pa.Table]] = [[] for _ in range(n_out)]

        for dataset in input_datasets:
            table = dataset.to_arrow()
            if len(table) == 0:
                continue
            partition_ids = _assign_partitions(table, spec)
            for pid in range(n_out):
                mask = pa.compute.equal(partition_ids, pid)
                bucket = table.filter(mask)
                if len(bucket) > 0:
                    bucket_tables[pid].append(bucket)

        # Merge and write each output partition
        output_datasets: List[Dataset] = []
        for pid in range(n_out):
            if bucket_tables[pid]:
                merged = pa.concat_tables(bucket_tables[pid])
            else:
                # Empty partition — preserve schema from first non-empty input
                schema = input_datasets[0].schema if input_datasets else pa.schema([])
                merged = schema.empty_table()

            out_path = generate_temp_path(
                self.session.temp_dir,
                prefix=f"shuffle_{stage.stage_id}_p{pid}",
            )
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            pq.write_table(merged, out_path)
            output_datasets.append(
                DuckDBDataset(out_path, pid, merged.schema, num_rows=len(merged))
            )

        return output_datasets

    def _execute_broadcast_stage(
        self,
        stage: ExecutionStage,
        input_datasets: List[Dataset],
    ) -> List[Dataset]:
        """Collect all input partitions into a single InMemoryDataset (broadcast)."""
        tables = [ds.to_arrow() for ds in input_datasets]
        combined = pa.concat_tables(tables) if tables else pa.table({})
        # Return one copy per output partition (all partitions share the same data)
        return [InMemoryDataset(combined, pid) for pid in range(stage.n_partitions)]

    def _execute_join_stage(
        self,
        stage: ExecutionStage,
        join_node: JoinNode,
        left_datasets: List[Dataset],
        right_datasets: List[Dataset],
    ) -> List[Dataset]:
        """Execute local joins: partition_i_left ⋈ partition_i_right."""
        n = stage.n_partitions
        left_aligned = _align_datasets(left_datasets, n)
        right_aligned = _align_datasets(right_datasets, n)

        tasks = [
            Task(
                task_id=generate_id(),
                stage_id=stage.stage_id,
                partition_id=pid,
                pipeline=[join_node],
                input_datasets=[left_aligned[pid], right_aligned[pid]],
                temp_dir=self.session.temp_dir,
            )
            for pid in range(n)
        ]
        self._scheduler.submit_batch(tasks)
        return [t.output_dataset for t in tasks]

    def stop(self) -> None:
        self._scheduler.stop()


# ---------------------------------------------------------------------------
# Partition assignment helpers
# ---------------------------------------------------------------------------


def _assign_partitions(table: pa.Table, spec: object) -> pa.Array:
    """Return an int32 array of partition IDs (one per row) for *table*."""

    if isinstance(spec, HashPartitionSpec):
        return _hash_partition(table, spec.keys, spec.n_partitions)

    if isinstance(spec, EvenPartitionSpec):
        n = len(table)
        n_out = spec.n_partitions
        pids = [int(i * n_out // n) for i in range(n)]
        return pa.array(pids, type=pa.int32())

    if isinstance(spec, UserDefinedPartitionSpec):
        batches = table.to_batches()
        if not batches:
            return pa.array([], type=pa.int32())
        batch = batches[0] if len(batches) == 1 else pa.concat_tables([
            pa.Table.from_batches([b]) for b in batches
        ]).to_batches()[0]
        return spec.fn(batch).cast(pa.int32())

    raise NotImplementedError(f"Unknown PartitionSpec type: {type(spec).__name__}")


def _hash_partition(table: pa.Table, keys: List[str], n: int) -> pa.Array:
    """Assign rows to partitions using DuckDB's stable hash function."""
    conn = __import__("duckdb").connect()
    conn.register("__t__", table)

    if len(keys) == 1:
        hash_expr = f'hash(CAST("{keys[0]}" AS VARCHAR))'
    else:
        concat_expr = " || '|' || ".join(f'CAST("{k}" AS VARCHAR)' for k in keys)
        hash_expr = f"hash({concat_expr})"

    result = conn.execute(f"SELECT CAST({hash_expr} % {n} AS INTEGER) FROM __t__").fetchall()
    return pa.array([r[0] for r in result], type=pa.int32())


# ---------------------------------------------------------------------------
# Dataset alignment helper
# ---------------------------------------------------------------------------


def _align_datasets(datasets: List[Dataset], n: int) -> List[Dataset]:
    """Ensure *datasets* has exactly *n* entries.

    - If len == n: return as-is.
    - If len > n: combine extras into the last partition (simple approach).
    - If len < n: pad with empty datasets.
    """
    if len(datasets) == n:
        return datasets

    if not datasets:
        empty = InMemoryDataset(pa.table({}), 0)
        return [empty] * n

    schema = datasets[0].schema

    if len(datasets) > n:
        # Merge all into one and return n copies... actually just redistribute
        tables = [ds.to_arrow() for ds in datasets]
        combined = pa.concat_tables(tables)
        total = len(combined)
        result = []
        for pid in range(n):
            start = (pid * total) // n
            end = ((pid + 1) * total) // n
            result.append(InMemoryDataset(combined.slice(start, end - start), pid))
        return result

    # Pad with empty partitions
    empty = InMemoryDataset(schema.empty_table(), 0)
    return list(datasets) + [
        InMemoryDataset(schema.empty_table(), pid) for pid in range(len(datasets), n)
    ]
