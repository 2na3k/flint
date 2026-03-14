# AGENTS.md — Guidance for AI Agents Working on Flint

## What Is Flint?

Flint is a lightweight distributed data pipeline framework (like Apache Spark or smallpond) built on top of **DuckDB**, **PyArrow**, and **Ray**. The core design principle is **lazy evaluation**: transformations build a logical plan (DAG), and computation only happens when `.compute()` is called.

## Key Architectural Principles

1. **Lazy by design** — never trigger computation inside transformation methods (`filter`, `map`, `repartition`, etc.). Only `.compute()` and action methods (`count`, `to_pandas`, `to_arrow`) should trigger execution.
2. **Immutable DataFrames** — every transformation returns a new `DataFrame`; never mutate in place.
3. **DuckDB for compute** — use DuckDB SQL for partition-level data transformations (filter, select, groupby, join). Fuse consecutive SQL ops into a single query.
4. **PyArrow for data representation** — internal data format is `pyarrow.Table`; minimize pandas usage in hot paths.
5. **Ray for distribution** — task execution is distributed via Ray workers. `Session(local=True)` skips Ray for testing.
6. **Shuffle via Driver** — data movement across partitions is done by the Driver (not Executor). Executor stays stateless per-partition.

## Package Manager

Always use `uv`, not `pip` or `poetry`.

```bash
uv add <package>      # add dependency
uv sync               # install all deps
uv run <command>      # run in project env
```

## Linting / Formatting

Use `ruff` for both linting and formatting. Run before finishing a task:

```bash
uv run ruff check .
uv run ruff format .
```

Do not introduce linting violations. Fix any `ruff` errors before finishing a task.

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `flint/session.py` | User-facing entry point. `Session` reads data into `DataFrame`. `GlobalContext` holds global state (Ray cluster). |
| `flint/dataframe.py` | `DataFrame` (lazy API), `Dataset` hierarchy (`InMemoryDataset`, `ParquetDataset`, `CsvDataset`, `DuckDBDataset`), `GroupedDataFrame` proxy. No computation in transform methods. |
| `flint/utils.py` | `generate_id()`, `generate_temp_path()`, `make_temp_dir()`. |
| `flint/io/arrow.py` | `load_dataset(ds) -> pa.Table` dispatch per Dataset subtype. Write helpers. |
| `flint/io/fs.py` | `resolve_filesystem(uri) -> (fsspec.AbstractFileSystem, str)` for s3/gcs/local URIs. |
| `flint/planner/node.py` | All logical plan node `@dataclass`es + `PartitionSpec` hierarchy (`HashPartitionSpec`, `EvenPartitionSpec`, `UserDefinedPartitionSpec`). |
| `flint/planner/planner.py` | `Planner.build(root) -> ExecutionPlan`. Detects stage boundaries at `ShuffleNode`/`JoinNode`. `ExecutionPlan` + `ExecutionStage` dataclasses. |
| `flint/planner/optimizer.py` | `Optimizer` + rules: `FilterFusion`, `PredicatePushdown`, `ProjectionPushdown`, `LimitPushdown`, `SelectFusion`. |
| `flint/executor/task.py` | `Task` dataclass + `TaskStatus` enum. |
| `flint/executor/queue.py` | Thread-safe `TaskQueue`. |
| `flint/executor/scheduler.py` | `Scheduler`: submits Tasks locally (in-process) or via `ray.remote`. |
| `flint/executor/executor.py` | `Executor.run(task) -> Dataset`. DuckDB SQL fusion for SQL nodes; Python UDF execution for map/filter callables. |
| `flint/executor/driver.py` | `Driver.execute(plan) -> List[Dataset]`. Coordinates stage execution, shuffle data movement, join coordination. |
| `flint/telemetry/telemetry.py` | Metrics/observability hooks. |

## Node Types

Source: `ReadParquet`, `ReadCsv`, `ReadArrow`, `ReadPandas`, `ReadDelta`, `ReadKafka`

Transforms: `FilterNode`, `SelectNode`, `MapNode`, `FlatMapNode`, `MapBatchesNode`, `LimitNode`, `GroupByAggNode`, `SqlNode`

Shuffle/Join (stage boundaries): `ShuffleNode`, `RepartitionNode` (user-facing, lowered to `ShuffleNode`), `JoinNode`, `BroadcastNode`

Write: `WriteParquet`, `WriteCsv`

## Partition Specs

`PartitionSpec` is the base. Three implementations:

- `HashPartitionSpec(keys, n_partitions, seed)` — hash(row[keys]) % n. Uses DuckDB for stable cross-process hashing.
- `EvenPartitionSpec(n_partitions)` — distribute rows evenly (two-pass).
- `UserDefinedPartitionSpec(fn, n_partitions)` — `fn(RecordBatch) -> int32 Array` of partition IDs.

`RepartitionNode` stores a `PartitionSpec` and is immediately lowered to `ShuffleNode` by the Planner.

## Join Architecture

```
df1.join(df2, on="user_id")

Planner output:
  JoinNode
    ├── ShuffleNode(HashPartitionSpec(["user_id"], N))  ← wraps df1 pipeline
    └── ShuffleNode(HashPartitionSpec(["user_id"], N))  ← wraps df2 pipeline

Both sides MUST use the same N and same hash spec — enforced by planner.
After shuffle: partition_i of left joins partition_i of right (local join).
For broadcast=True: skip shuffle on small side.
```

## Executor SQL Fusion Pattern

Within a stage, the executor fuses consecutive SQL-compatible nodes into one DuckDB query before flushing for Python UDFs:

```sql
SELECT {select_expr}
FROM   __input__
WHERE  (pred_1) AND (pred_2) AND ...
LIMIT  {limit}
```

Python UDF nodes (`MapNode`, `FlatMapNode`, `MapBatchesNode`, `FilterNode(is_sql=False)`) force a SQL flush before execution.

## Adding New DataFrame Operations

1. Add the method stub to `DataFrame` in `flint/dataframe.py` — it must return `DataFrame`.
2. Create the corresponding logical plan node in `flint/planner/node.py`.
3. Handle the node in `flint/executor/executor.py` (DuckDB or PyArrow based).
4. If it creates a stage boundary (shuffle, join), handle in `flint/executor/driver.py`.
5. If optimizable, add a rule in `flint/planner/optimizer.py`.
6. Write tests in `tests/`.

## Testing

- Write tests alongside new features. Test files go in `tests/`.
- Run tests with: `uv run pytest`
- Unit-test logical plan construction separately from execution (mock the Driver).
- Use `Session(local=True)` for integration tests — no Ray required.

## What NOT To Do

- Do not call `.compute()` or trigger Ray tasks inside transformation methods.
- Do not use `pandas` for internal data representation (only at the `.to_pandas()` boundary).
- Do not add unnecessary abstractions — keep the codebase minimal.
- Do not bypass `ruff` checks.
- Do not use `pip install` — always use `uv add`.
- Do not mutate `Node` objects after creation — the plan DAG is immutable.
- Do not store large data (pa.Table) in nodes except for `ReadArrow`/`ReadPandas` sources.
- **Do not use `MicroBatchLoop` directly in user-facing code or examples.** It is an internal implementation detail of `StreamingDataFrame`. The public API is `write_stdio()` and `write_kafka()` — they register the sink and start the blocking loop. Using `MicroBatchLoop` directly bypasses the clean API and leaks internals.
