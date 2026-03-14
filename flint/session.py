"""Session — user-facing entry point for Flint."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd
import pyarrow as pa

from flint.dataframe import DataFrame
from flint.utils import make_temp_dir

if TYPE_CHECKING:
    from flint.streaming.dataframe import StreamingDataFrame


class Session:
    """Entry point for Flint.  Analogous to ``SparkSession``.

    Parameters
    ----------
    local:
        If ``True``, run without Ray (in-process threads).  Ideal for testing
        and single-machine workloads.
    n_workers:
        Number of worker threads (local mode) or Ray actors (distributed mode).
    temp_dir:
        Directory for intermediate Parquet files.  Created automatically if
        not provided.

    Example
    -------
    ::

        session = Session()
        df = session.read_parquet("data/*.parquet")
        result = df.filter("age > 18").groupby("country").agg({"score": "sum"}).compute()
        print(result.to_pandas())
        session.stop()
    """

    def __init__(
        self,
        local: bool = True,
        n_workers: int = 4,
        temp_dir: Optional[str] = None,
        ray_address: Optional[str] = None,
        ray_init_kwargs: Optional[dict] = None,
    ) -> None:
        self.local = local
        self.n_workers = n_workers
        self.temp_dir = temp_dir or make_temp_dir()
        self._ray_address = ray_address
        self._ray_init_kwargs = ray_init_kwargs or {}

        from flint.executor.scheduler import Scheduler

        self._scheduler = Scheduler(local=local, n_workers=n_workers)

        if not local:
            self._init_ray()

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def read_parquet(
        self,
        path: str,
        n_partitions: Optional[int] = None,
        filesystem: Optional[str] = None,
    ) -> DataFrame:
        """Read Parquet file(s) or a Hive-partitioned directory into a lazy DataFrame.

        Parameters
        ----------
        path:
            File path, glob pattern (``"data/*.parquet"``), or a directory.
            If *path* is a directory whose immediate children are ``key=value``
            subdirectories, Hive partitioning is detected automatically and the
            partition columns are added to every row.
        n_partitions:
            Override the number of partitions.  For Hive datasets this is
            ignored — one Flint partition is created per file.
        filesystem:
            fsspec protocol string (``"s3"``, ``"gcs"``, etc.).  ``None`` = local.

        Example
        -------
        ::

            # Hive-partitioned dataset:
            # data/year=2024/month=01/part-0.parquet
            # data/year=2024/month=02/part-0.parquet
            df = session.read_parquet("data/")
            df.filter("year = 2024").show()
        """
        from flint.io.fs import discover_dataset
        from flint.planner.node import ReadParquet

        files, part_cols, part_vals = discover_dataset(
            path, format="parquet", filesystem=filesystem
        )
        if not files:
            raise ValueError(f"No Parquet files found at {path!r}")

        # For Hive datasets, always one partition per file (preserves partition semantics)
        n = len(files) if part_cols else (n_partitions or max(1, len(files)))
        node = ReadParquet(
            paths=files,
            n_partitions=n,
            filesystem=filesystem,
            hive_partitioning=bool(part_cols),
            partition_columns=part_cols,
            partition_values=part_vals,
        )
        return DataFrame._from_node(node, self)

    def read_csv(
        self,
        path: str,
        n_partitions: Optional[int] = None,
        filesystem: Optional[str] = None,
        delimiter: str = ",",
        has_header: bool = True,
        **kwargs: Any,
    ) -> DataFrame:
        """Read CSV file(s) or a Hive-partitioned directory into a lazy DataFrame.

        Hive partitioning is detected automatically when *path* is a directory
        whose subdirectories follow the ``key=value`` naming convention.
        """
        from flint.io.fs import discover_dataset
        from flint.planner.node import ReadCsv

        files, part_cols, part_vals = discover_dataset(
            path, format="csv", filesystem=filesystem
        )
        if not files:
            raise ValueError(f"No CSV files found at {path!r}")

        n = len(files) if part_cols else (n_partitions or max(1, len(files)))
        node = ReadCsv(
            paths=files,
            n_partitions=n,
            filesystem=filesystem,
            delimiter=delimiter,
            has_header=has_header,
            read_options=kwargs,
            hive_partitioning=bool(part_cols),
            partition_columns=part_cols,
            partition_values=part_vals,
        )
        return DataFrame._from_node(node, self)

    def from_arrow(self, table: pa.Table, n_partitions: int = 1) -> DataFrame:
        """Create a DataFrame from an existing ``pyarrow.Table``."""
        from flint.planner.node import ReadArrow

        node = ReadArrow(table=table, n_partitions=n_partitions, schema=table.schema)
        return DataFrame._from_node(node, self)

    def from_pandas(self, df: pd.DataFrame, n_partitions: int = 1) -> DataFrame:
        """Create a DataFrame from a pandas DataFrame."""
        from flint.planner.node import ReadPandas

        table = pa.Table.from_pandas(df)
        node = ReadPandas(table=table, n_partitions=n_partitions, schema=table.schema)
        return DataFrame._from_node(node, self)

    def read_kafka_stream(
        self,
        topic: str,
        bootstrap_servers: str,
        schema: pa.Schema,
        batch_size: int = 100,
        group_id: Optional[str] = None,
        consumer_config: Optional[dict] = None,
    ) -> "StreamingDataFrame":
        """Create a StreamingDataFrame from a Kafka topic.

        Parameters
        ----------
        topic:
            Kafka topic name.
        bootstrap_servers:
            Comma-separated list of broker host:port pairs.
        schema:
            PyArrow schema describing the expected message fields.
        batch_size:
            Max records to poll per micro-batch.
        group_id:
            Consumer group ID.  Defaults to ``flint-<topic>``.
        consumer_config:
            Extra confluent-kafka Consumer config overrides.
        """
        from flint.streaming.dataframe import StreamingDataFrame
        from flint.streaming.sources import KafkaSource

        source = KafkaSource(
            topic, bootstrap_servers, schema, group_id, consumer_config
        )
        return StreamingDataFrame(source=source, session=self, batch_size=batch_size)

    def read_websocket_stream(
        self,
        uri: str,
        schema: pa.Schema,
        batch_size: int = 50,
        reconnect_delay: float = 1.0,
    ) -> "StreamingDataFrame":
        """Create a StreamingDataFrame from a WebSocket endpoint.

        Parameters
        ----------
        uri:
            WebSocket URI (e.g. ``ws://localhost:8765``).
        schema:
            PyArrow schema describing the expected JSON message fields.
        batch_size:
            Max records to drain per micro-batch.
        reconnect_delay:
            Seconds to wait before reconnecting after a disconnection.
        """
        from flint.streaming.dataframe import StreamingDataFrame
        from flint.streaming.sources import WebSocketSource

        source = WebSocketSource(uri, schema, reconnect_delay)
        source.start()
        return StreamingDataFrame(source=source, session=self, batch_size=batch_size)

    def read_stream(
        self,
        source: "StreamingSource",
        batch_size: int = 100,
        n_partitions: int = 1,
        partition_by: Optional[List[str]] = None,
        partition_fn: Optional[Any] = None,
    ) -> "StreamingDataFrame":
        """Create a StreamingDataFrame from any custom StreamingSource.

        Parameters
        ----------
        source:
            A ``StreamingSource`` instance (already started/connected).
        batch_size:
            Max records to drain per micro-batch.
        n_partitions:
            Number of parallel workers to distribute each micro-batch across.
            ``1`` (default) runs single-threaded with zero overhead.
        partition_by:
            Column names to hash-partition by. Rows with the same key always
            land on the same worker. Requires ``n_partitions > 1``.
        partition_fn:
            Callable ``(pa.RecordBatch) -> pa.Array[int32]`` assigning each row
            a partition ID. Mutually exclusive with ``partition_by``.
        """
        from flint.streaming.dataframe import StreamingDataFrame

        partition_spec = None
        if n_partitions > 1:
            if partition_by is not None and partition_fn is not None:
                raise ValueError(
                    "Provide at most one of partition_by or partition_fn, not both."
                )
            if partition_by is not None:
                from flint.planner.node import HashPartitionSpec

                partition_spec = HashPartitionSpec(
                    keys=partition_by, n_partitions=n_partitions
                )
            elif partition_fn is not None:
                from flint.planner.node import UserDefinedPartitionSpec

                partition_spec = UserDefinedPartitionSpec(
                    fn=partition_fn, n_partitions=n_partitions
                )
            else:
                from flint.planner.node import EvenPartitionSpec

                partition_spec = EvenPartitionSpec(n_partitions=n_partitions)

        return StreamingDataFrame(
            source=source,
            session=self,
            batch_size=batch_size,
            partition_spec=partition_spec,
        )

    @property
    def scheduler(self):
        """The Scheduler instance used for distributed task execution."""
        return self._scheduler

    def sql(self, query: str, **named_dfs: DataFrame) -> DataFrame:
        """Run a DuckDB SQL query referencing named DataFrames.

        Example
        -------
        ::

            result = session.sql(
                "SELECT * FROM df1 JOIN df2 ON df1.id = df2.id",
                df1=df1,
                df2=df2,
            )
        """
        from flint.planner.node import SqlNode

        if not named_dfs:
            raise ValueError(
                "Provide at least one named DataFrame: session.sql(q, df=df)"
            )

        # Use the first DataFrame as the primary node; others are in input_names
        primary_name, primary_df = next(iter(named_dfs.items()))
        input_names = {name: df._node for name, df in named_dfs.items()}
        children = [df._node for df in named_dfs.values()]

        node = SqlNode(sql=query, children=children, input_names=input_names)
        return DataFrame._from_node(node, primary_df._session)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Release resources (temp directory, scheduler, Ray cluster if started)."""
        import shutil

        try:
            self._scheduler.stop()
        except Exception:
            pass

        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        if self.local:
            mode = "local"
        elif self._ray_address:
            mode = f"ray(address={self._ray_address!r})"
        else:
            mode = f"ray(n_workers={self.n_workers})"
        return f"Session(mode={mode}, temp_dir={self.temp_dir!r})"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_ray(self) -> None:
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "Ray is required for distributed mode: uv add ray"
            ) from exc

        if ray.is_initialized():
            return

        kwargs = dict(self._ray_init_kwargs)

        if self._ray_address:
            # Connect to an existing cluster (head node address or "auto")
            kwargs["address"] = self._ray_address
        else:
            # Start a local Ray cluster (single machine)
            kwargs.setdefault("num_cpus", self.n_workers)

        kwargs.setdefault("ignore_reinit_error", True)
        ray.init(**kwargs)


# ---------------------------------------------------------------------------
# GlobalContext — singleton holding cluster-level state
# ---------------------------------------------------------------------------


class GlobalContext:
    """Singleton that holds global Flint state (Ray cluster, active sessions).

    Analogous to ``SparkContext``.  Users rarely interact with this directly.
    """

    _instance: Optional[GlobalContext] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: List[Session] = []

    @classmethod
    def get(cls) -> GlobalContext:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = GlobalContext()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for testing)."""
        with cls._lock:
            cls._instance = None

    def register_session(self, session: Session) -> None:
        self._sessions.append(session)

    def stop_all(self) -> None:
        for session in self._sessions:
            session.stop()
        self._sessions.clear()
