"""Executor — runs a single Task on one partition using DuckDB + PyArrow."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from flint.dataframe import (
    Dataset,
    DuckDBDataset,
)
from flint.executor.task import Task, TaskStatus
from flint.planner.node import (
    BroadcastNode,
    FilterNode,
    FlatMapNode,
    GroupByAggNode,
    JoinNode,
    LimitNode,
    MapBatchesNode,
    MapNode,
    ReadArrow,
    ReadCsv,
    ReadNode,
    ReadPandas,
    ReadParquet,
    SelectNode,
    ShuffleNode,
    SqlNode,
    WriteCsv,
    WriteParquet,
)
from flint.utils import generate_temp_path


class Executor:
    """Executes a single Task: applies a pipeline of nodes to one partition.

    Execution strategy
    ------------------
    1. Load input data as a ``pyarrow.Table`` (from source node or prior stage).
    2. Walk the pipeline, accumulating SQL-compatible nodes into a pending query.
    3. Before each Python UDF node, flush pending SQL through DuckDB.
    4. For Python UDF nodes (Map/FlatMap/MapBatches/Python Filter), apply in Python.
    5. Write final result to a temp Parquet file; return a ``DuckDBDataset``.
    """

    def run(self, task: Task) -> Dataset:
        """Execute *task* and return the output ``Dataset``."""
        task.status = TaskStatus.RUNNING
        try:
            result = self._execute(task)
            task.output_dataset = result
            task.status = TaskStatus.DONE
            return result
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = exc
            raise

    # ------------------------------------------------------------------
    # Internal execution logic
    # ------------------------------------------------------------------

    def _execute(self, task: Task) -> Dataset:
        pipeline = task.pipeline

        if not pipeline:
            raise ValueError(f"Task {task.task_id} has an empty pipeline")

        # Step 1 — load input table
        if isinstance(pipeline[0], ReadNode):
            table = _read_source(pipeline[0], task.partition_id)
            remaining = pipeline[1:]
        elif isinstance(pipeline[0], JoinNode):
            # Join stage: input_datasets[0]=left, input_datasets[1]=right
            return self._execute_join(pipeline[0], task)
        elif isinstance(pipeline[0], BroadcastNode):
            # Broadcast stage: collect all partitions into one table
            tables = [ds.to_arrow() for ds in task.input_datasets]
            table = pa.concat_tables(tables) if tables else pa.table({})
            remaining = pipeline[1:]
        else:
            if not task.input_datasets:
                raise ValueError(f"Task {task.task_id} has no input datasets")
            table = task.input_datasets[0].to_arrow()
            remaining = pipeline

        # Step 2 — apply the rest of the pipeline
        table = self._apply_pipeline(table, remaining, task)

        # Step 3 — handle write sinks
        if remaining and isinstance(remaining[-1], (WriteParquet, WriteCsv)):
            sink = remaining[-1]
            if isinstance(sink, WriteParquet) and sink.partition_cols:
                return _write_hive_parquet(table, sink, task)
            # Table has already been processed; now write
            part_path = _partition_write_path(sink.path, task.partition_id)
            if isinstance(sink, WriteParquet):
                os.makedirs(os.path.dirname(os.path.abspath(part_path)), exist_ok=True)
                pq.write_table(table, part_path, compression=sink.compression)
            else:
                from flint.io.arrow import write_csv
                write_csv(table, part_path, sink.delimiter, sink.include_header)
            return DuckDBDataset(part_path, task.partition_id, table.schema, num_rows=len(table))

        # Step 4 — write to temp Parquet
        out_path = generate_temp_path(task.temp_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        pq.write_table(table, out_path)
        return DuckDBDataset(out_path, task.partition_id, table.schema, num_rows=len(table))

    def _apply_pipeline(self, table: pa.Table, pipeline: list, task: Task) -> pa.Table:
        """Apply a sequence of nodes to *table*, fusing SQL ops where possible."""
        # SQL accumulator
        select_expr = "*"
        where_clauses: List[str] = []
        limit_val: Optional[int] = None

        def flush_sql(tbl: pa.Table) -> pa.Table:
            nonlocal select_expr, where_clauses, limit_val
            if select_expr == "*" and not where_clauses and limit_val is None:
                return tbl
            result = _exec_sql(tbl, select_expr, where_clauses, limit_val)
            select_expr = "*"
            where_clauses = []
            limit_val = None
            return result

        for node in pipeline:
            if isinstance(node, FilterNode) and node.is_sql:
                where_clauses.append(str(node.predicate))

            elif isinstance(node, SelectNode):
                # Flush pending WHERE before changing SELECT to avoid column aliasing issues
                if where_clauses:
                    table = flush_sql(table)
                select_expr = ", ".join(node.columns)

            elif isinstance(node, LimitNode):
                limit_val = node.limit

            elif isinstance(node, GroupByAggNode):
                table = flush_sql(table)
                table = _exec_groupby(table, node)

            elif isinstance(node, SqlNode):
                table = flush_sql(table)
                table = _exec_raw_sql(table, node.sql)

            elif isinstance(node, FilterNode):
                # Python callable filter
                table = flush_sql(table)
                table = _apply_python_filter(table, node.predicate)

            elif isinstance(node, MapNode):
                table = flush_sql(table)
                table = _apply_map(table, node.fn)

            elif isinstance(node, FlatMapNode):
                table = flush_sql(table)
                table = _apply_flatmap(table, node.fn)

            elif isinstance(node, MapBatchesNode):
                table = flush_sql(table)
                table = _apply_map_batches(table, node.fn, node.batch_size)

            elif isinstance(node, (WriteParquet, WriteCsv)):
                # Handled by caller after pipeline returns
                table = flush_sql(table)
                break

            elif isinstance(node, (ShuffleNode, BroadcastNode)):
                # These are stage boundaries — should not appear inside a transform pipeline
                table = flush_sql(table)
                break

        table = flush_sql(table)
        return table

    def _execute_join(self, node: JoinNode, task: Task) -> Dataset:
        """Execute a local (post-shuffle) join on co-located partitions."""
        if len(task.input_datasets) < 2:
            raise ValueError("JoinNode task requires two input datasets (left and right)")

        left_table = task.input_datasets[0].to_arrow()
        right_table = task.input_datasets[1].to_arrow()

        result = _exec_join(left_table, right_table, node)
        out_path = generate_temp_path(task.temp_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        pq.write_table(result, out_path)
        return DuckDBDataset(out_path, task.partition_id, result.schema, num_rows=len(result))


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------


def _read_source(node: ReadNode, partition_id: int) -> pa.Table:
    """Read one partition from a source node."""
    if isinstance(node, ReadArrow):
        return _slice_table(node.table, partition_id, node.n_partitions or 1)

    if isinstance(node, ReadPandas):
        return _slice_table(node.table, partition_id, node.n_partitions or 1)

    if isinstance(node, ReadParquet):
        if not node.paths:
            return pa.table({})
        idx = partition_id % len(node.paths)
        path = node.paths[idx]
        table = pq.read_table(path)
        # Inject Hive partition columns
        if node.hive_partitioning and node.partition_values:
            table = _add_partition_columns(table, node.partition_values[idx])
        if node.row_group_filter:
            table = _exec_sql(table, "*", [node.row_group_filter], None)
        return table

    if isinstance(node, ReadCsv):
        if not node.paths:
            return pa.table({})
        import pyarrow.csv as pcsv

        idx = partition_id % len(node.paths)
        path = node.paths[idx]
        table = pcsv.read_csv(path)
        if node.hive_partitioning and node.partition_values:
            table = _add_partition_columns(table, node.partition_values[idx])
        return table

    raise NotImplementedError(f"Unsupported source node type: {type(node).__name__}")


def _add_partition_columns(table: pa.Table, part_vals: Dict[str, Any]) -> pa.Table:
    """Append Hive partition key-value pairs as constant columns to *table*."""
    for col, val in part_vals.items():
        if col not in table.schema.names:
            arr = pa.array([val] * len(table))
            table = table.append_column(col, arr)
    return table


def _slice_table(table: pa.Table, partition_id: int, n_partitions: int) -> pa.Table:
    """Slice an in-memory table into a partition."""
    if table is None:
        return pa.table({})
    total = len(table)
    if n_partitions <= 1:
        return table
    start = (partition_id * total) // n_partitions
    end = ((partition_id + 1) * total) // n_partitions
    return table.slice(start, end - start)


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------


def _duckdb_to_table(result: Any) -> pa.Table:
    """Normalise a DuckDB query result to a ``pa.Table``.

    DuckDB may return either a ``pa.Table`` or a ``pa.RecordBatchReader``
    depending on the version and method called.
    """
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatchReader):
        return result.read_all()
    # Fallback: try converting via pandas (slow but safe)
    return pa.Table.from_pandas(result.df())


def _exec_sql(
    table: pa.Table,
    select_expr: str,
    where_clauses: List[str],
    limit_val: Optional[int],
) -> pa.Table:
    """Compose and run a single DuckDB query against *table*."""
    if table.num_columns == 0:
        return table
    conn = duckdb.connect()
    conn.register("__input__", table)
    sql = f"SELECT {select_expr} FROM __input__"
    if where_clauses:
        sql += " WHERE " + " AND ".join(f"({c})" for c in where_clauses)
    if limit_val is not None:
        sql += f" LIMIT {limit_val}"
    return _duckdb_to_table(conn.execute(sql).arrow())


def _exec_groupby(table: pa.Table, node: GroupByAggNode) -> pa.Table:
    """Execute a GROUP BY aggregation via DuckDB."""
    conn = duckdb.connect()
    conn.register("__input__", table)

    agg_exprs = []
    for out_col, agg_fn, in_col in node.aggregations:
        if in_col == "*":
            agg_exprs.append(f"{agg_fn}(*) AS \"{out_col}\"")
        else:
            agg_exprs.append(f"{agg_fn}(\"{in_col}\") AS \"{out_col}\"")

    group_cols = ", ".join(f'"{k}"' for k in node.group_keys)
    select_cols = group_cols + (", " + ", ".join(agg_exprs) if agg_exprs else "")
    sql = f'SELECT {select_cols} FROM __input__'
    if node.group_keys:
        sql += f" GROUP BY {group_cols}"
    return _duckdb_to_table(conn.execute(sql).arrow())


def _exec_raw_sql(table: pa.Table, sql: str) -> pa.Table:
    """Execute raw DuckDB SQL; replaces 'this' with __input__."""
    conn = duckdb.connect()
    conn.register("__input__", table)
    rewritten = sql.replace("this", "__input__")
    return _duckdb_to_table(conn.execute(rewritten).arrow())


def _exec_join(
    left: pa.Table,
    right: pa.Table,
    node: JoinNode,
) -> pa.Table:
    """Execute a local join via DuckDB after shuffle co-location."""
    conn = duckdb.connect()
    conn.register("__left__", left)
    conn.register("__right__", right)

    how_map = {
        "inner": "INNER",
        "left": "LEFT",
        "right": "RIGHT",
        "outer": "FULL OUTER",
        "semi": "SEMI",
        "anti": "ANTI",
    }
    join_type = how_map.get(node.how.lower(), "INNER")

    conditions = " AND ".join(
        f'__left__."{lk}" = __right__."{rk}"'
        for lk, rk in zip(node.left_keys, node.right_keys)
    )

    # Handle column name collisions with suffixes
    left_cols = left.schema.names
    right_cols = right.schema.names
    key_set = set(node.left_keys)

    left_selects = [f'__left__."{c}"' for c in left_cols]
    right_selects = []
    for c in right_cols:
        if c in key_set:
            continue  # skip duplicate join key columns from right
        alias = f"{c}{node.right_suffix}" if c in left_cols else c
        right_selects.append(f'__right__."{c}" AS "{alias}"')

    select_expr = ", ".join(left_selects + right_selects)
    sql = f"SELECT {select_expr} FROM __left__ {join_type} JOIN __right__ ON {conditions}"
    return _duckdb_to_table(conn.execute(sql).arrow())


# ---------------------------------------------------------------------------
# Python UDF helpers
# ---------------------------------------------------------------------------


def _apply_python_filter(table: pa.Table, fn: Any) -> pa.Table:
    """Apply a Python callable ``(row: dict) -> bool`` as a filter."""
    col_names = table.schema.names
    columns = {name: table.column(name).to_pylist() for name in col_names}
    n = len(table)
    mask = pa.array(
        [fn({k: columns[k][i] for k in col_names}) for i in range(n)],
        type=pa.bool_(),
    )
    return table.filter(mask)


def _apply_map(table: pa.Table, fn: Any) -> pa.Table:
    """Apply a Python callable ``(row: dict) -> dict`` to every row."""
    col_names = table.schema.names
    columns = {name: table.column(name).to_pylist() for name in col_names}
    n = len(table)
    result_rows = [fn({k: columns[k][i] for k in col_names}) for i in range(n)]
    if not result_rows:
        return pa.table({})
    keys = list(result_rows[0].keys())
    return pa.table({k: [r[k] for r in result_rows] for k in keys})


def _apply_flatmap(table: pa.Table, fn: Any) -> pa.Table:
    """Apply a Python callable ``(row: dict) -> list[dict]`` to every row."""
    col_names = table.schema.names
    columns = {name: table.column(name).to_pylist() for name in col_names}
    n = len(table)
    result_rows: List[Dict[str, Any]] = []
    for i in range(n):
        row = {k: columns[k][i] for k in col_names}
        result_rows.extend(fn(row))
    if not result_rows:
        return pa.table({})
    keys = list(result_rows[0].keys())
    return pa.table({k: [r[k] for r in result_rows] for k in keys})


def _apply_map_batches(table: pa.Table, fn: Any, batch_size: int) -> pa.Table:
    """Apply a function on ``pa.RecordBatch`` chunks."""
    batches = table.to_batches(max_chunksize=batch_size)
    result_batches = [fn(batch) for batch in batches]
    if not result_batches:
        return pa.table({})
    return pa.Table.from_batches(result_batches)


# ---------------------------------------------------------------------------
# Write path helpers
# ---------------------------------------------------------------------------


def _partition_write_path(base_path: str, partition_id: int) -> str:
    """Derive a per-partition output file path."""
    base, ext = os.path.splitext(base_path)
    if not ext:
        # Treat as directory
        return os.path.join(base_path, f"part-{partition_id:05d}.parquet")
    return f"{base}-part-{partition_id:05d}{ext}"


def _write_hive_parquet(table: pa.Table, sink: "WriteParquet", task: "Task") -> "InMemoryDataset":
    """Write *table* into Hive-partitioned Parquet files.

    After the hash-shuffle stage every partition already contains only rows
    for a specific set of partition-column values, but we still split the
    table by every distinct combination so that we write exactly one file per
    ``col=val/…/part-{id}.parquet`` path.

    Returns an ``InMemoryDataset`` sentinel — data is already on disk, and the
    caller only needs this for bookkeeping (especially over Ray where the return
    value is serialized through the object store, not via file paths).
    """
    from flint.dataframe import InMemoryDataset

    partition_cols = sink.partition_cols  # e.g. ["year", "month"]

    if table.num_columns == 0 or len(table) == 0:
        return InMemoryDataset(pa.table({}), task.partition_id)

    # Group rows by the distinct partition-key combinations using DuckDB
    conn = duckdb.connect()
    conn.register("__input__", table)
    group_cols = ", ".join(f'"{c}"' for c in partition_cols)
    groups = conn.execute(f"SELECT DISTINCT {group_cols} FROM __input__").fetchall()

    for group_vals in groups:
        # Build WHERE clause to extract this group's rows
        conditions = " AND ".join(
            f'"{col}" = {_sql_literal(val)}'
            for col, val in zip(partition_cols, group_vals)
        )
        group_table = _duckdb_to_table(
            conn.execute(f"SELECT * FROM __input__ WHERE {conditions}").arrow()
        )

        # Build the Hive directory path:  base/year=2024/month=1/
        hive_dir = sink.path
        for col, val in zip(partition_cols, group_vals):
            hive_dir = os.path.join(hive_dir, f"{col}={val}")
        os.makedirs(hive_dir, exist_ok=True)

        # Drop the partition columns from the file itself (Hive convention)
        data_cols = [c for c in group_table.schema.names if c not in partition_cols]
        group_table = group_table.select(data_cols) if data_cols else group_table

        out_path = os.path.join(hive_dir, f"part-{task.partition_id:05d}.parquet")
        pq.write_table(group_table, out_path, compression=sink.compression)

    return InMemoryDataset(pa.table({}), task.partition_id)


def _sql_literal(val: Any) -> str:
    """Format a Python value as a SQL literal for DuckDB."""
    if val is None:
        return "NULL"
    if isinstance(val, str):
        escaped = val.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    return str(val)
