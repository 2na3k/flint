"""DataFrame (lazy API), Dataset hierarchy, and GroupedDataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from flint.planner.node import Node, PartitionSpec
    from flint.session import Session


# ---------------------------------------------------------------------------
# Dataset hierarchy — physical partition units
# ---------------------------------------------------------------------------


class Dataset:
    """Base class for a single physical partition.

    All subclasses must be picklable (Ray uses cloudpickle).
    """

    def __init__(
        self,
        partition_id: int,
        schema: pa.Schema,
        num_rows: Optional[int] = None,
    ) -> None:
        self.partition_id = partition_id
        self.schema = schema
        self.num_rows = num_rows

    def to_arrow(self) -> pa.Table:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(partition_id={self.partition_id}, num_rows={self.num_rows})"


class InMemoryDataset(Dataset):
    """A partition backed by a ``pyarrow.Table`` held in-process.

    Used for small datasets, test fixtures, and materialised results.
    """

    def __init__(self, table: pa.Table, partition_id: int = 0) -> None:
        super().__init__(partition_id, table.schema, len(table))
        self.table = table

    def to_arrow(self) -> pa.Table:
        return self.table


class ParquetDataset(Dataset):
    """A partition backed by a Parquet file on local or cloud FS."""

    def __init__(
        self,
        path: str,
        partition_id: int,
        schema: pa.Schema,
        filesystem: Any = None,
        row_group_filter: Optional[str] = None,
        num_rows: Optional[int] = None,
    ) -> None:
        super().__init__(partition_id, schema, num_rows)
        self.path = path
        self.filesystem = filesystem
        self.row_group_filter = row_group_filter

    def to_arrow(self) -> pa.Table:
        return pq.read_table(self.path, filesystem=self.filesystem)


class CsvDataset(Dataset):
    """A partition backed by a CSV file."""

    def __init__(
        self,
        path: str,
        partition_id: int,
        schema: pa.Schema,
        read_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(partition_id, schema)
        self.path = path
        self.read_options = read_options or {}

    def to_arrow(self) -> pa.Table:
        import pyarrow.csv as pcsv

        return pcsv.read_csv(self.path)


class DuckDBDataset(Dataset):
    """A partition written to a temp Parquet file by the Executor.

    This is the primary output type for any transformation stage.
    """

    def __init__(
        self,
        path: str,
        partition_id: int,
        schema: pa.Schema,
        query_hash: str = "",
        num_rows: Optional[int] = None,
    ) -> None:
        super().__init__(partition_id, schema, num_rows)
        self.path = path
        self.query_hash = query_hash

    def to_arrow(self) -> pa.Table:
        return pq.read_table(self.path)


# ---------------------------------------------------------------------------
# GroupedDataFrame — proxy returned by df.groupby(...)
# ---------------------------------------------------------------------------


class GroupedDataFrame:
    """Proxy object returned by ``DataFrame.groupby()``.

    No plan node is created until ``.agg()`` (or another aggregation) is called.
    """

    def __init__(self, df: DataFrame, keys: List[str]) -> None:
        self._df = df
        self._keys = keys

    def agg(self, aggregations: Dict[str, str]) -> DataFrame:
        """Aggregate columns.

        Parameters
        ----------
        aggregations:
            Mapping of input column → aggregation function name.
            e.g. ``{"price": "sum", "age": "mean"}``.
            Output column names equal input column names.
        """
        from flint.planner.node import GroupByAggNode

        agg_list = [(col, fn, col) for col, fn in aggregations.items()]
        node = GroupByAggNode(
            children=[self._df._node],
            group_keys=self._keys,
            aggregations=agg_list,
        )
        return DataFrame._from_node(node, self._df._session)

    def count(self) -> DataFrame:
        return self.agg({"*": "count"})

    def sum(self, *cols: str) -> DataFrame:
        return self.agg({c: "sum" for c in cols})

    def mean(self, *cols: str) -> DataFrame:
        return self.agg({c: "mean" for c in cols})

    def min(self, *cols: str) -> DataFrame:
        return self.agg({c: "min" for c in cols})

    def max(self, *cols: str) -> DataFrame:
        return self.agg({c: "max" for c in cols})


# ---------------------------------------------------------------------------
# DataFrame — lazy, immutable
# ---------------------------------------------------------------------------


class DataFrame:
    """Lazy, immutable distributed DataFrame.

    Every transformation method returns a new DataFrame with an updated
    logical plan. No computation happens until ``.compute()`` is called.
    """

    def __init__(self, node: Node, session: Session) -> None:
        self._node = node
        self._session = session
        self._cached_datasets: Optional[List[Dataset]] = None

    @classmethod
    def _from_node(cls, node: Node, session: Session) -> DataFrame:
        return cls(node, session)

    # ------------------------------------------------------------------
    # Transformation methods (all lazy — must return DataFrame)
    # ------------------------------------------------------------------

    def filter(self, expr: Union[str, Callable]) -> DataFrame:
        """Filter rows.

        Parameters
        ----------
        expr:
            A DuckDB SQL expression string (e.g. ``"age > 18"``) or a Python
            callable ``(row: dict) -> bool``.  SQL expressions are pushed down
            to DuckDB and can be optimised; callables are applied row-by-row.
        """
        from flint.planner.node import FilterNode

        node = FilterNode(
            children=[self._node],
            predicate=expr,
            is_sql=isinstance(expr, str),
        )
        return DataFrame._from_node(node, self._session)

    def select(self, *columns: str) -> DataFrame:
        """Project to a subset of columns.

        Columns may be SQL expressions understood by DuckDB,
        e.g. ``"cast(price as double)"``.
        """
        from flint.planner.node import SelectNode

        node = SelectNode(children=[self._node], columns=list(columns))
        return DataFrame._from_node(node, self._session)

    def map(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        output_schema: Optional[pa.Schema] = None,
    ) -> DataFrame:
        """Apply ``fn(row: dict) -> dict`` to every row."""
        from flint.planner.node import MapNode

        node = MapNode(children=[self._node], fn=fn, output_schema=output_schema)
        return DataFrame._from_node(node, self._session)

    def flatmap(
        self,
        fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        output_schema: Optional[pa.Schema] = None,
    ) -> DataFrame:
        """Apply ``fn(row: dict) -> list[dict]`` to every row (one-to-many)."""
        from flint.planner.node import FlatMapNode

        node = FlatMapNode(children=[self._node], fn=fn, output_schema=output_schema)
        return DataFrame._from_node(node, self._session)

    def map_batches(
        self,
        fn: Callable[[pa.RecordBatch], pa.RecordBatch],
        batch_size: int = 1024,
        output_schema: Optional[pa.Schema] = None,
    ) -> DataFrame:
        """Apply ``fn(batch: pa.RecordBatch) -> pa.RecordBatch`` for vectorised ops."""
        from flint.planner.node import MapBatchesNode

        node = MapBatchesNode(
            children=[self._node],
            fn=fn,
            batch_size=batch_size,
            output_schema=output_schema,
        )
        return DataFrame._from_node(node, self._session)

    def limit(self, n: int) -> DataFrame:
        """Limit to the first *n* rows (applied per-partition)."""
        from flint.planner.node import LimitNode

        node = LimitNode(children=[self._node], limit=n)
        return DataFrame._from_node(node, self._session)

    def repartition(
        self,
        n_partitions: int,
        partition_by: Union[str, List[str], None] = None,
        by_rows: bool = False,
        partition_spec: Optional[PartitionSpec] = None,
    ) -> DataFrame:
        """Shuffle data into *n_partitions* output partitions.

        Parameters
        ----------
        n_partitions:
            Target number of output partitions.
        partition_by:
            Column(s) to hash-partition on.  Defaults to ``EvenPartitionSpec``.
        by_rows:
            If ``True``, split rows as evenly as possible.
        partition_spec:
            Override with a custom ``PartitionSpec`` instance.
        """
        from flint.planner.node import (
            EvenPartitionSpec,
            HashPartitionSpec,
            RepartitionNode,
        )

        if partition_spec is not None:
            spec = partition_spec
        elif partition_by is not None:
            keys = [partition_by] if isinstance(partition_by, str) else list(partition_by)
            spec = HashPartitionSpec(n_partitions=n_partitions, keys=keys)
        else:
            spec = EvenPartitionSpec(n_partitions=n_partitions)

        node = RepartitionNode(children=[self._node], partition_spec=spec)
        return DataFrame._from_node(node, self._session)

    def groupby(self, *keys: str) -> GroupedDataFrame:
        """Return a ``GroupedDataFrame`` for chained aggregation."""
        return GroupedDataFrame(self, list(keys))

    def join(
        self,
        other: DataFrame,
        on: Union[str, List[str]],
        how: str = "inner",
        partition_spec: Optional[PartitionSpec] = None,
        broadcast: bool = False,
        left_suffix: str = "_left",
        right_suffix: str = "_right",
    ) -> DataFrame:
        """Distributed join.

        Both sides are shuffle-partitioned on *on* columns using
        ``HashPartitionSpec`` unless ``broadcast=True``.

        Parameters
        ----------
        other:
            Right-hand DataFrame.
        on:
            Join key column(s) — used for both sides.
        how:
            Join type: ``"inner"``, ``"left"``, ``"right"``, ``"outer"``,
            ``"semi"``, ``"anti"``.
        partition_spec:
            Override the default ``HashPartitionSpec``.
        broadcast:
            If ``True``, the right side is broadcast to all workers (skip shuffle).
        """
        from flint.planner.node import HashPartitionSpec, JoinNode

        keys = [on] if isinstance(on, str) else list(on)
        spec = partition_spec or HashPartitionSpec(keys=keys, n_partitions=0)

        node = JoinNode(
            children=[self._node, other._node],
            left_keys=keys,
            right_keys=keys,
            how=how,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            broadcast=broadcast,
            partition_spec=spec,
        )
        return DataFrame._from_node(node, self._session)

    def sql(self, query: str) -> DataFrame:
        """Apply a raw DuckDB SQL query.  Reference this DataFrame as ``this``.

        Example::

            df.sql("SELECT *, age * 2 AS double_age FROM this")
        """
        from flint.planner.node import SqlNode

        node = SqlNode(
            children=[self._node],
            sql=query,
            input_names={"this": self._node},
        )
        return DataFrame._from_node(node, self._session)

    def write_parquet(
        self,
        path: str,
        compression: str = "snappy",
        partition_cols: Optional[List[str]] = None,
        n_partitions: int = 200,
    ) -> DataFrame:
        """Add a Parquet write sink to the plan.

        If *partition_cols* are provided the data is first shuffled so that all
        rows sharing the same partition-column values land on the same worker,
        enabling parallel conflict-free Hive-style writes (``col=val/part-*.parquet``).
        """
        from flint.planner.node import WriteParquet

        source = self
        if partition_cols:
            source = self.repartition(n_partitions, partition_by=partition_cols)

        node = WriteParquet(
            children=[source._node],
            path=path,
            compression=compression,
            partition_cols=partition_cols,
        )
        return DataFrame._from_node(node, self._session)

    def write_csv(
        self,
        path: str,
        delimiter: str = ",",
        include_header: bool = True,
    ) -> DataFrame:
        """Add a CSV write sink to the plan."""
        from flint.planner.node import WriteCsv

        node = WriteCsv(
            children=[self._node],
            path=path,
            delimiter=delimiter,
            include_header=include_header,
        )
        return DataFrame._from_node(node, self._session)

    # ------------------------------------------------------------------
    # Action methods (trigger execution)
    # ------------------------------------------------------------------

    def _compute(self) -> List[Dataset]:
        """Internal: run the logical plan and return partition datasets.

        Results are cached — calling twice returns the same list.
        """
        if self._cached_datasets is not None:
            return self._cached_datasets

        from flint.executor.driver import Driver
        from flint.planner.planner import Planner

        plan = Planner().build(self._node)
        driver = Driver(session=self._session)
        datasets = driver.execute(plan)
        self._cached_datasets = datasets
        return datasets

    def compute(self) -> DataFrame:
        """Materialise the DataFrame and return a new cached DataFrame."""
        datasets = self._compute()
        tables = [ds.to_arrow() for ds in datasets]
        combined = pa.concat_tables(tables) if tables else pa.table({})

        from flint.planner.node import ReadArrow

        result = DataFrame._from_node(
            ReadArrow(table=combined, n_partitions=1),
            self._session,
        )
        result._cached_datasets = [InMemoryDataset(combined, 0)]
        return result

    def count(self) -> int:
        """Return the total number of rows across all partitions."""
        datasets = self._compute()
        total = 0
        for ds in datasets:
            if ds.num_rows is not None:
                total += ds.num_rows
            else:
                total += len(ds.to_arrow())
        return total

    def to_pandas(self) -> pd.DataFrame:
        """Collect all partitions and return a single pandas DataFrame."""
        datasets = self._compute()
        tables = [ds.to_arrow() for ds in datasets]
        if not tables:
            return pd.DataFrame()
        return pa.concat_tables(tables).to_pandas()

    def to_arrow(self) -> pa.Table:
        """Collect all partitions and return a single pyarrow Table."""
        datasets = self._compute()
        tables = [ds.to_arrow() for ds in datasets]
        return pa.concat_tables(tables) if tables else pa.table({})

    def show(self, n: int = 20) -> None:
        """Print the first *n* rows to stdout."""
        result = self.limit(n).to_arrow()
        try:
            print(result.to_pandas().to_string(index=False))
        except Exception:
            print(result)

    def explain(self, mode: str = "logical") -> None:
        """Print the query plan.

        Parameters
        ----------
        mode:
            ``"logical"``  — the raw node DAG before optimisation (default).
            ``"optimized"`` — the node DAG after the optimizer has run.
            ``"physical"``  — the execution stages produced by the planner.
        """
        if mode == "logical":
            print("== Logical Plan ==")
            print(_format_node_tree(self._node))
        elif mode == "optimized":
            from flint.planner.optimizer import Optimizer

            optimized = Optimizer().optimize(self._node)
            print("== Optimized Logical Plan ==")
            print(_format_node_tree(optimized))
        elif mode == "physical":
            from flint.planner.planner import Planner

            plan = Planner().build(self._node)
            print("== Physical Plan ==")
            for stage in plan.stages:
                deps = ", ".join(stage.depends_on) if stage.depends_on else "none"
                marker = " ◄ OUTPUT" if stage.stage_id == plan.output_stage_id else ""
                print(f"\nStage {stage.stage_id}  partitions={stage.n_partitions}  depends_on=[{deps}]{marker}")
                for node in stage.pipeline:
                    print(f"  └─ {_format_node_inline(node)}")
        else:
            raise ValueError(f"Unknown mode {mode!r}. Choose 'logical', 'optimized', or 'physical'.")

    def __repr__(self) -> str:
        return f"DataFrame(node={self._node!r})"


# ---------------------------------------------------------------------------
# Plan formatting helpers
# ---------------------------------------------------------------------------


def _format_node_inline(node: Any) -> str:
    """Single-line description of a node for physical plan display."""
    from flint.planner.node import (
        FilterNode,
        FlatMapNode,
        GroupByAggNode,
        HashPartitionSpec,
        JoinNode,
        LimitNode,
        MapBatchesNode,
        MapNode,
        ReadArrow,
        ReadCsv,
        ReadParquet,
        RepartitionNode,
        SelectNode,
        ShuffleNode,
        SqlNode,
        WriteCsv,
        WriteParquet,
    )

    if isinstance(node, ReadParquet):
        base = f"ReadParquet(files={len(node.paths)}, n_partitions={node.n_partitions}"
        if node.hive_partitioning:
            base += f", hive_cols={node.partition_columns}"
        return base + ")"
    if isinstance(node, ReadArrow):
        rows = len(node.table) if node.table is not None else "?"
        return f"ReadArrow(rows={rows}, n_partitions={node.n_partitions})"
    if isinstance(node, ReadCsv):
        return f"ReadCsv(paths={node.paths[:2]}, n_partitions={node.n_partitions})"
    if isinstance(node, FilterNode):
        pred = repr(node.predicate) if not node.is_sql else f'"{node.predicate}"'
        kind = "SQL" if node.is_sql else "Python"
        return f"Filter[{kind}]({pred})"
    if isinstance(node, SelectNode):
        return f"Select({node.columns})"
    if isinstance(node, MapNode):
        return f"Map(fn={node.fn.__name__ if hasattr(node.fn, '__name__') else '...'})"
    if isinstance(node, FlatMapNode):
        return f"FlatMap(fn={node.fn.__name__ if hasattr(node.fn, '__name__') else '...'})"
    if isinstance(node, MapBatchesNode):
        return f"MapBatches(fn=..., batch_size={node.batch_size})"
    if isinstance(node, LimitNode):
        return f"Limit({node.limit})"
    if isinstance(node, GroupByAggNode):
        aggs = [f"{fn}({col})" for _, fn, col in node.aggregations]
        return f"GroupBy({node.group_keys}) Agg({aggs})"
    if isinstance(node, JoinNode):
        return f"Join[{node.how}](left={node.left_keys}, right={node.right_keys}, broadcast={node.broadcast})"
    if isinstance(node, (ShuffleNode, RepartitionNode)):
        spec = node.partition_spec
        if isinstance(spec, HashPartitionSpec):
            return f"Shuffle[Hash](keys={spec.keys}, n={spec.n_partitions})"
        return f"Shuffle[{type(spec).__name__}](n={spec.n_partitions if spec else '?'})"
    if isinstance(node, SqlNode):
        sql_preview = node.sql[:60].replace("\n", " ")
        return f'SQL("{sql_preview}{"…" if len(node.sql) > 60 else ""}")'
    if isinstance(node, WriteParquet):
        return f"WriteParquet(path={node.path!r}, compression={node.compression!r})"
    if isinstance(node, WriteCsv):
        return f"WriteCSV(path={node.path!r})"
    return f"{type(node).__name__}()"


def _format_node_tree(node: Any, prefix: str = "", is_last: bool = True) -> str:
    """Recursive tree formatter for the logical plan."""
    connector = "└─ " if is_last else "├─ "
    line = prefix + connector + _format_node_inline(node)
    child_prefix = prefix + ("   " if is_last else "│  ")
    children = getattr(node, "children", [])
    child_lines = []
    for i, child in enumerate(children):
        child_lines.append(
            _format_node_tree(child, child_prefix, is_last=(i == len(children) - 1))
        )
    return "\n".join([line] + child_lines)
