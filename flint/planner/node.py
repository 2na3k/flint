"""Logical plan nodes — the DAG representation of a DataFrame computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from flint.utils import generate_id


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    node_id: str = field(default_factory=generate_id)
    children: List[Node] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node_id})"


# ---------------------------------------------------------------------------
# Partition specs — describe HOW to assign rows to output partitions
# ---------------------------------------------------------------------------


@dataclass
class PartitionSpec:
    """Base class for partition strategies."""

    n_partitions: int = 0  # 0 means auto-detect from input


@dataclass
class HashPartitionSpec(PartitionSpec):
    """Hash rows by key columns: stable_hash(row[keys]) % n_partitions.

    Uses DuckDB's built-in hash for cross-process consistency.
    """

    keys: List[str] = field(default_factory=list)
    seed: int = 0


@dataclass
class EvenPartitionSpec(PartitionSpec):
    """Distribute rows as evenly as possible across n_partitions.

    Requires a two-pass scan (count rows first, then partition).
    """


@dataclass
class UserDefinedPartitionSpec(PartitionSpec):
    """User-supplied function that assigns each row a partition ID.

    fn: pa.RecordBatch -> pa.Array[int32]  (partition IDs, 0-indexed)
    Must be serializable (cloudpickle) for distributed execution.
    """

    fn: Any = field(
        default=None, compare=False, repr=False
    )  # Callable[[pa.RecordBatch], pa.Array]


# ---------------------------------------------------------------------------
# Source nodes
# ---------------------------------------------------------------------------


@dataclass
class ReadNode(Node):
    """Base for all source nodes."""

    schema: Any = field(default=None, compare=False, repr=False)  # Optional[pa.Schema]
    n_partitions: Optional[int] = None


@dataclass
class ReadParquet(ReadNode):
    paths: List[str] = field(default_factory=list)
    filesystem: Optional[str] = None  # fsspec protocol: "s3", "gcs", None=local
    row_group_filter: Optional[str] = None  # DuckDB SQL predicate for pushdown
    # Hive-style partitioning metadata
    hive_partitioning: bool = False
    partition_columns: List[str] = field(default_factory=list)
    # One dict per path: {"year": 2024, "month": 1}.  Empty list when not partitioned.
    partition_values: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReadCsv(ReadNode):
    paths: List[str] = field(default_factory=list)
    filesystem: Optional[str] = None
    delimiter: str = ","
    has_header: bool = True
    read_options: Dict[str, Any] = field(default_factory=dict)
    # Hive-style partitioning metadata
    hive_partitioning: bool = False
    partition_columns: List[str] = field(default_factory=list)
    partition_values: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReadArrow(ReadNode):
    """In-memory PyArrow Table source. Sliced by partition_id at execution time."""

    table: Any = field(default=None, compare=False, repr=False)  # Optional[pa.Table]


@dataclass
class ReadPandas(ReadNode):
    """Pandas DataFrame converted to Arrow at construction time."""

    table: Any = field(default=None, compare=False, repr=False)  # Optional[pa.Table]


@dataclass
class ReadDelta(ReadNode):
    paths: List[str] = field(default_factory=list)
    filesystem: Optional[str] = None
    version: Optional[int] = None


@dataclass
class ReadKafka(ReadNode):
    topic: str = ""
    bootstrap_servers: str = ""
    consumer_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transformation nodes
# ---------------------------------------------------------------------------


@dataclass
class FilterNode(Node):
    """Filter rows by a SQL predicate string or a Python callable.

    is_sql=True  → predicate is a DuckDB SQL expression string (pushdown candidate)
    is_sql=False → predicate is a Python callable: (row: dict) -> bool
    """

    predicate: Any = field(default=None, compare=False, repr=False)
    is_sql: bool = True


@dataclass
class SelectNode(Node):
    """Project to a subset of columns. Columns may be SQL expressions."""

    columns: List[str] = field(default_factory=list)


@dataclass
class MapNode(Node):
    """Apply fn(row: dict) -> dict to every row."""

    fn: Any = field(default=None, compare=False, repr=False)
    output_schema: Any = field(
        default=None, compare=False, repr=False
    )  # Optional[pa.Schema]


@dataclass
class FlatMapNode(Node):
    """Apply fn(row: dict) -> list[dict] to every row (one-to-many)."""

    fn: Any = field(default=None, compare=False, repr=False)
    output_schema: Any = field(default=None, compare=False, repr=False)


@dataclass
class MapBatchesNode(Node):
    """Apply fn(batch: pa.RecordBatch) -> pa.RecordBatch for vectorized ops."""

    fn: Any = field(default=None, compare=False, repr=False)
    batch_size: int = 1024
    output_schema: Any = field(default=None, compare=False, repr=False)


@dataclass
class LimitNode(Node):
    """Limit to first n rows (applied per-partition unless after a shuffle)."""

    limit: int = 0


@dataclass
class RepartitionNode(Node):
    """User-facing repartition. Lowered to ShuffleNode by the planner."""

    partition_spec: Any = field(default=None)  # PartitionSpec


@dataclass
class ShuffleNode(Node):
    """Physical shuffle — always a stage boundary.

    Executor computes partition assignments; Driver moves the data.
    """

    partition_spec: Any = field(default=None)  # PartitionSpec


@dataclass
class GroupByAggNode(Node):
    """GROUP BY + aggregation.

    aggregations: list of (output_col, agg_fn, input_col)
      e.g. [("total", "sum", "price"), ("cnt", "count", "*")]
    """

    group_keys: List[str] = field(default_factory=list)
    aggregations: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class JoinNode(Node):
    """Distributed join — always a stage boundary.

    children[0] = left, children[1] = right.
    The planner inserts ShuffleNodes on both sides (unless broadcast=True).
    """

    left_keys: List[str] = field(default_factory=list)
    right_keys: List[str] = field(default_factory=list)
    how: str = "inner"  # "inner", "left", "right", "outer", "semi", "anti"
    left_suffix: str = "_left"
    right_suffix: str = "_right"
    broadcast: bool = False
    partition_spec: Any = field(
        default=None
    )  # Optional[PartitionSpec], overrides default HashPartitionSpec


@dataclass
class BroadcastNode(Node):
    """Wraps the small side of a broadcast join.

    The Driver collects all partitions of this side and sends them to every worker.
    """


@dataclass
class SqlNode(Node):
    """Raw DuckDB SQL escape hatch.

    'this' in input_names refers to the primary input DataFrame's node.
    Additional named DataFrames can be passed for multi-input SQL.
    """

    sql: str = ""
    input_names: Dict[str, Any] = field(default_factory=dict)  # alias -> Node


# ---------------------------------------------------------------------------
# Write (sink) nodes
# ---------------------------------------------------------------------------


@dataclass
class WriteNode(Node):
    """Base for all sink nodes."""

    path: str = ""
    filesystem: Optional[str] = None


@dataclass
class WriteParquet(WriteNode):
    compression: str = "snappy"
    partition_cols: Optional[List[str]] = None


@dataclass
class WriteCsv(WriteNode):
    delimiter: str = ","
    include_header: bool = True
