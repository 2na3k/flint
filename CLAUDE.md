# Flint

A lightweight distributed data pipeline framework inspired by [smallpond](https://github.com/deepseek-ai/smallpond) and Apache Spark.

## Project Overview

Flint uses **DuckDB** and **PyArrow** as the compute backend, with **Ray** for distributed execution. The API is lazy-evaluated (like Spark, not pandas) — transformations build a logical plan, and execution is triggered explicitly via `.compute()`.

## Tech Stack

- **Python**: 3.12+
- **Package manager**: `uv` (use `uv` for all dependency management, not pip)
- **Linter/Formatter**: `ruff`
- **Core dependencies**: `duckdb`, `pyarrow`, `pandas`, `ray`, `fsspec`, `psutil`

## Coding loop
```text
1. Writing a new feature
2. Writing an unit test then run it, and cover the case if anything fail (ALWAYS)
3. Running integration test (if needed)
4. Lint and format the codebase
```

## Project Structure

```
flint/
├── __init__.py
├── session.py          # Session (entry point) and GlobalContext
├── dataframe.py        # DataFrame (lazy API), Dataset hierarchy, GroupedDataFrame
├── utils.py            # generate_id(), generate_temp_path(), make_temp_dir()
├── io/
│   ├── arrow.py        # Dataset → pa.Table dispatch; write helpers
│   ├── fs.py           # URI → fsspec filesystem resolution
│   └── kafka.py        # Kafka source (future)
├── planner/
│   ├── node.py         # All logical plan node @dataclasses + PartitionSpec hierarchy
│   ├── planner.py      # Planner.build() → ExecutionPlan/ExecutionStage
│   └── optimizer.py    # Optimizer + optimization rules
├── executor/
│   ├── task.py         # Task dataclass + TaskStatus enum
│   ├── queue.py        # Thread-safe TaskQueue
│   ├── scheduler.py    # Scheduler (local + Ray modes)
│   ├── executor.py     # Executor.run(task) → Dataset (DuckDB + PyArrow)
│   └── driver.py       # Driver.execute(plan) → List[Dataset]
└── telemetry/
    └── telemetry.py    # Metrics and observability

tests/
├── conftest.py         # shared fixtures
├── test_dataframe.py   # plan construction unit tests (no execution)
├── test_planner.py     # stage splitting + optimizer rules
├── test_executor.py    # single-task execution with in-process DuckDB
└── test_session.py     # integration tests for read methods

docs/
└── implementation_plan.md   # Full design plan with data structures and algorithms
```

## Architecture

```
User API (DataFrame)
    → Planner (builds DAG of logical nodes, detects stage boundaries)
    → Optimizer (predicate pushdown, filter fusion, projection pushdown)
    → Driver (orchestrates stage execution, shuffle, joins)
    → Scheduler (dispatches Tasks to workers — local or Ray)
    → Executor (runs tasks via DuckDB/PyArrow on one partition)
```

- **DataFrame**: Lazy, immutable; each operation returns a new DataFrame
- **Dataset**: Physical partition unit — `InMemoryDataset`, `ParquetDataset`, `CsvDataset`, `DuckDBDataset`
- **Session**: Entry point (`read_parquet`, `read_csv`, `from_arrow`, `from_pandas`, `sql`)
- **GlobalContext**: Singleton holding cluster state

## Dataset Hierarchy

```
Dataset (base)
├── InMemoryDataset   — pa.Table held in-process (test fixtures, compute() results)
├── ParquetDataset    — path-backed, read via pyarrow.parquet
├── CsvDataset        — path-backed, read via pyarrow.csv
└── DuckDBDataset     — temp parquet file written by Executor
```

## Partition & Shuffle

Three partition strategies, all subclass `PartitionSpec`:

- `HashPartitionSpec(keys, n_partitions)` — stable hash on key columns (uses DuckDB hash)
- `EvenPartitionSpec(n_partitions)` — split rows as evenly as possible
- `UserDefinedPartitionSpec(fn, n_partitions)` — `fn(RecordBatch) -> int32 Array`

`RepartitionNode` (user-facing) is lowered to `ShuffleNode` by the planner.
`JoinNode` inserts `ShuffleNode` on both sides automatically (unless `broadcast=True`).

## Development Commands

```bash
# Install dependencies
uv sync

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Run tests
uv run pytest

# Run with uv
uv run python -c "..."
```

## Code Conventions

- All transformations on `DataFrame` must return a new `DataFrame` (immutable)
- Execution is lazy: no computation happens until `.compute()` is called
- Use `pyarrow` types for schema/data representation internally
- DuckDB is used for SQL-based computation on partitions (filter, select, groupby, join)
- Ray is used for distributed task execution across workers
- `Session(local=True)` runs without Ray for testing and local development
- Follow Spark's naming conventions where applicable (`repartition`, `filter`, `map`, `flatmap`)
- Type hints are required for all public methods
- All code must be unit-tested; test files go in `tests/`

## Streaming Anti-Patterns

**Never use `MicroBatchLoop` directly in examples or user-facing code.** It is an internal class.
The correct streaming API is terminal sink methods on `StreamingDataFrame`:

```python
# CORRECT — write_stdio starts the blocking loop, Ctrl+C stops it
session.read_kafka_stream(...).filter("qty > 50").write_stdio(label="hot-trades")

# WRONG — never do this outside of StreamingDataFrame internals
from flint.streaming.loop import MicroBatchLoop
loop = MicroBatchLoop(sources=[sdf._source], ...)
loop.start(...)
```
