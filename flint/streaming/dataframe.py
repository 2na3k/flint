"""StreamingDataFrame — lazy, immutable streaming API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

import pyarrow as pa

if TYPE_CHECKING:
    from flint.planner.node import Node
    from flint.session import Session
    from flint.streaming.sinks import Sink
    from flint.streaming.sources import StreamingSource


class StreamingDataFrame:
    """Lazy, immutable streaming DataFrame.

    Transforms build up a pipeline list; execution starts with ``start()``.
    """

    def __init__(
        self,
        source: "StreamingSource",
        session: "Session",
        pipeline: Optional[List["Node"]] = None,
        sinks: Optional[List["Sink"]] = None,
        batch_size: int = 100,
        partition_spec: Optional[Any] = None,
    ) -> None:
        self._source = source
        self._session = session
        self._pipeline: List["Node"] = pipeline if pipeline is not None else []
        self._sinks: List["Sink"] = sinks if sinks is not None else []
        self._batch_size = batch_size
        self._partition_spec = partition_spec

    # ------------------------------------------------------------------
    # Immutable transform methods
    # ------------------------------------------------------------------

    def filter(self, expr) -> "StreamingDataFrame":
        """Filter rows using a SQL string or Python callable."""
        from flint.planner.node import FilterNode

        node = FilterNode(predicate=expr, is_sql=isinstance(expr, str))
        return self._with_node(node)

    def select(self, *columns: str) -> "StreamingDataFrame":
        """Project to a subset of columns."""
        from flint.planner.node import SelectNode

        node = SelectNode(columns=list(columns))
        return self._with_node(node)

    def map(
        self,
        fn: Callable,
        output_schema: Optional[pa.Schema] = None,
    ) -> "StreamingDataFrame":
        """Apply ``fn(row: dict) -> dict`` to every row."""
        from flint.planner.node import MapNode

        node = MapNode(fn=fn, output_schema=output_schema)
        return self._with_node(node)

    def flatmap(
        self,
        fn: Callable,
        output_schema: Optional[pa.Schema] = None,
    ) -> "StreamingDataFrame":
        """Apply ``fn(row: dict) -> list[dict]`` to every row (one-to-many)."""
        from flint.planner.node import FlatMapNode

        node = FlatMapNode(fn=fn, output_schema=output_schema)
        return self._with_node(node)

    def map_batches(
        self,
        fn: Callable,
        batch_size: int = 1024,
        output_schema: Optional[pa.Schema] = None,
    ) -> "StreamingDataFrame":
        """Apply ``fn(batch: pa.RecordBatch) -> pa.RecordBatch`` for vectorised ops."""
        from flint.planner.node import MapBatchesNode

        node = MapBatchesNode(fn=fn, batch_size=batch_size, output_schema=output_schema)
        return self._with_node(node)

    # ------------------------------------------------------------------
    # Terminal sink methods — register sink and start the blocking loop
    # ------------------------------------------------------------------

    def write_stdio(
        self, label: str = "", max_rows: int = 20, batch_interval: float = 1.0
    ) -> None:
        """Print each micro-batch to stdout and start the streaming loop (blocking).

        Press Ctrl+C to stop.
        """
        from flint.streaming.sinks import StdioSink

        self._sinks.append(StdioSink(label=label, max_rows=max_rows))
        self._run(batch_interval=batch_interval)

    def write_kafka(
        self,
        topic: str,
        bootstrap_servers: str,
        producer_config: Optional[dict] = None,
        key_column: Optional[str] = None,
        batch_interval: float = 1.0,
    ) -> None:
        """Write each micro-batch to Kafka and start the streaming loop (blocking).

        Press Ctrl+C to stop.
        """
        from flint.streaming.sinks import KafkaSink

        self._sinks.append(
            KafkaSink(
                topic=topic,
                bootstrap_servers=bootstrap_servers,
                producer_config=producer_config,
                key_column=key_column,
            )
        )
        self._run(batch_interval=batch_interval)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run(self, batch_interval: float) -> None:
        """Build and run the micro-batch loop; close source + sinks on exit."""
        from flint.planner.node import JoinNode, RepartitionNode, ShuffleNode
        from flint.streaming.loop import MicroBatchLoop

        # Guard: stage-boundary nodes are incompatible with per-batch execution
        for node in self._pipeline:
            if isinstance(node, (ShuffleNode, JoinNode)):
                raise ValueError(
                    f"{type(node).__name__} is not supported in streaming pipelines. "
                    "Stage boundaries require cross-partition data movement which is "
                    "incompatible with per-micro-batch execution."
                )

        # Extract RepartitionNode if present — use its spec, strip node from pipeline
        partition_spec = self._partition_spec
        pipeline = self._pipeline
        repartition_idx = next(
            (i for i, n in enumerate(pipeline) if isinstance(n, RepartitionNode)),
            None,
        )
        if repartition_idx is not None:
            partition_spec = pipeline[repartition_idx].partition_spec
            pipeline = pipeline[:repartition_idx] + pipeline[repartition_idx + 1:]

        scheduler = self._session.scheduler if partition_spec is not None else None

        loop = MicroBatchLoop(
            sources=[self._source],
            pipeline=pipeline,
            sinks=self._sinks,
            batch_size=self._batch_size,
            temp_dir=self._session.temp_dir,
            partition_spec=partition_spec,
            scheduler=scheduler,
        )
        try:
            loop.start(batch_interval=batch_interval)
        finally:
            self._source.close()
            for sink in self._sinks:
                sink.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _with_node(self, node: "Node") -> "StreamingDataFrame":
        """Return a new StreamingDataFrame with *node* appended to the pipeline."""
        return StreamingDataFrame(
            source=self._source,
            session=self._session,
            pipeline=self._pipeline + [node],
            sinks=list(self._sinks),
            batch_size=self._batch_size,
            partition_spec=self._partition_spec,
        )

    def __repr__(self) -> str:
        return (
            f"StreamingDataFrame("
            f"source={type(self._source).__name__}, "
            f"pipeline={[type(n).__name__ for n in self._pipeline]}, "
            f"sinks={[type(s).__name__ for s in self._sinks]})"
        )
