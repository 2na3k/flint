"""Unit tests for the Flint streaming layer (no real Kafka or WebSocket needed)."""

from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from flint.session import Session
from flint.streaming.loop import MicroBatchLoop
from flint.streaming.sinks import Sink, StdioSink
from flint.streaming.sources import StreamingSource


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeSource(StreamingSource):
    """Returns a fixed table on first poll(), None thereafter."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table
        self._polled = False
        self._schema = table.schema

    def poll(self, batch_size: int) -> Optional[pa.RecordBatch]:
        if self._polled:
            return None
        self._polled = True
        return self._table.to_batches(max_chunksize=batch_size)[0]

    def close(self) -> None:
        pass


class CollectingSink(Sink):
    """Appends each written pa.Table to a list for assertions."""

    def __init__(self) -> None:
        self.batches: List[pa.Table] = []

    def write(self, batch: pa.Table) -> None:
        self.batches.append(batch)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# KafkaSource tests
# ---------------------------------------------------------------------------


class TestKafkaSource:
    def _make_message(self, value: bytes, error=None):
        msg = MagicMock()
        msg.error.return_value = error
        msg.value.return_value = value
        return msg

    def test_poll_returns_none_on_empty(self, tmp_path):
        schema = pa.schema([pa.field("x", pa.int64())])
        with patch("confluent_kafka.Consumer") as MockConsumer:
            consumer = MockConsumer.return_value
            consumer.consume.return_value = []
            from flint.streaming.sources import KafkaSource

            src = KafkaSource.__new__(KafkaSource)
            src._schema = schema
            src._topic = "test"
            src._consumer = consumer
            result = src.poll(10)
        assert result is None

    def test_poll_returns_record_batch_with_correct_schema(self, tmp_path):
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("val", pa.float64())])
        import json

        with patch("confluent_kafka.Consumer") as MockConsumer:
            consumer = MockConsumer.return_value
            msg1 = self._make_message(json.dumps({"id": 1, "val": 1.5}).encode())
            msg2 = self._make_message(json.dumps({"id": 2, "val": 2.5}).encode())
            consumer.consume.return_value = [msg1, msg2]
            from flint.streaming.sources import KafkaSource

            src = KafkaSource.__new__(KafkaSource)
            src._schema = schema
            src._consumer = consumer
            src._topic = "test"
            result = src.poll(10)

        assert result is not None
        assert result.schema == schema
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]

    def test_poll_skips_errored_messages(self):
        schema = pa.schema([pa.field("x", pa.int64())])
        import json

        with patch("confluent_kafka.Consumer") as MockConsumer:
            consumer = MockConsumer.return_value
            err_msg = self._make_message(b"", error=MagicMock())
            good_msg = self._make_message(json.dumps({"x": 42}).encode())
            consumer.consume.return_value = [err_msg, good_msg]
            from flint.streaming.sources import KafkaSource

            src = KafkaSource.__new__(KafkaSource)
            src._schema = schema
            src._consumer = consumer
            src._topic = "test"
            result = src.poll(10)

        assert result is not None
        assert result.num_rows == 1
        assert result.column("x").to_pylist() == [42]

    def test_poll_skips_malformed_json(self):
        schema = pa.schema([pa.field("x", pa.int64())])
        with patch("confluent_kafka.Consumer") as MockConsumer:
            consumer = MockConsumer.return_value
            bad_msg = self._make_message(b"not-valid-json")
            consumer.consume.return_value = [bad_msg]
            from flint.streaming.sources import KafkaSource

            src = KafkaSource.__new__(KafkaSource)
            src._schema = schema
            src._consumer = consumer
            src._topic = "test"
            result = src.poll(10)

        assert result is None


# ---------------------------------------------------------------------------
# WebSocketSource tests
# ---------------------------------------------------------------------------


class TestWebSocketSource:
    def test_poll_returns_none_when_empty(self):
        schema = pa.schema([pa.field("v", pa.int64())])
        from flint.streaming.sources import WebSocketSource

        src = WebSocketSource.__new__(WebSocketSource)
        src._schema = schema
        src._queue = queue.Queue()
        src._stop_event = threading.Event()
        src._thread = None

        result = src.poll(5)
        assert result is None

    def test_poll_drains_queue_up_to_batch_size(self):
        schema = pa.schema([pa.field("v", pa.int64())])
        from flint.streaming.sources import WebSocketSource

        src = WebSocketSource.__new__(WebSocketSource)
        src._schema = schema
        src._queue = queue.Queue()
        src._stop_event = threading.Event()
        src._thread = None

        for i in range(10):
            src._queue.put_nowait({"v": i})

        result = src.poll(5)
        assert result is not None
        assert result.num_rows == 5
        # 5 remaining in queue
        assert src._queue.qsize() == 5

    def test_poll_does_not_deadlock(self):
        """Ensure poll() completes quickly when queue is empty."""
        schema = pa.schema([pa.field("v", pa.int64())])
        from flint.streaming.sources import WebSocketSource

        src = WebSocketSource.__new__(WebSocketSource)
        src._schema = schema
        src._queue = queue.Queue()
        src._stop_event = threading.Event()
        src._thread = None

        start = time.monotonic()
        src.poll(100)
        assert time.monotonic() - start < 1.0


# ---------------------------------------------------------------------------
# StdioSink tests
# ---------------------------------------------------------------------------


class TestStdioSink:
    def test_write_prints_header_and_data(self, capsys):
        sink = StdioSink(label="test-label", max_rows=10)
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        sink.write(table)
        out = capsys.readouterr().out
        assert "test-label" in out
        assert "1" in out

    def test_write_silent_on_empty(self, capsys):
        sink = StdioSink(label="silent")
        sink.write(pa.table({"a": pa.array([], type=pa.int64())}))
        out = capsys.readouterr().out
        assert out == ""

    def test_close_is_noop(self):
        sink = StdioSink()
        sink.close()  # should not raise


# ---------------------------------------------------------------------------
# KafkaSink tests
# ---------------------------------------------------------------------------


class TestKafkaSink:
    def _make_sink(self, key_column=None):
        with patch("confluent_kafka.Producer") as MockProducer:
            producer = MockProducer.return_value
            from flint.streaming.sinks import KafkaSink

            sink = KafkaSink.__new__(KafkaSink)
            sink._topic = "out"
            sink._key_column = key_column
            sink._producer = producer
            return sink, producer

    def test_write_produces_json_per_row(self):
        import json

        sink, producer = self._make_sink()
        table = pa.table({"id": [1, 2], "val": [10.0, 20.0]})
        sink.write(table)

        assert producer.produce.call_count == 2
        producer.flush.assert_called_once()

        call_args = producer.produce.call_args_list
        row0 = json.loads(call_args[0].kwargs["value"])
        assert row0["id"] == 1
        assert row0["val"] == 10.0

    def test_write_uses_key_column_when_set(self):
        sink, producer = self._make_sink(key_column="id")
        table = pa.table({"id": [7], "val": [3.14]})
        sink.write(table)

        call_kwargs = producer.produce.call_args.kwargs
        assert call_kwargs["key"] == b"7"

    def test_write_flushes_after_all_rows(self):
        sink, producer = self._make_sink()
        table = pa.table({"x": [1, 2, 3]})
        sink.write(table)
        producer.flush.assert_called_once()

    def test_write_empty_table_skips_produce(self):
        sink, producer = self._make_sink()
        sink.write(pa.table({"x": pa.array([], type=pa.int64())}))
        producer.produce.assert_not_called()


# ---------------------------------------------------------------------------
# MicroBatchLoop tests
# ---------------------------------------------------------------------------


class TestMicroBatchLoop:
    def test_empty_poll_skips_executor_and_sinks(self, tmp_path):
        source = FakeSource(pa.table({"v": pa.array([], type=pa.int64())}))
        source._polled = True  # force empty
        sink = CollectingSink()
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()
        assert sink.batches == []

    def test_pipeline_applied_via_executor(self, tmp_path):
        from flint.planner.node import FilterNode

        table = pa.table({"value": [1, -1, 2, -2]})
        source = FakeSource(table)
        sink = CollectingSink()
        pipeline = [FilterNode(predicate="value > 0", is_sql=True)]

        loop = MicroBatchLoop(
            sources=[source],
            pipeline=pipeline,
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()

        assert len(sink.batches) == 1
        result = sink.batches[0]
        assert result.column("value").to_pylist() == [1, 2]

    def test_stop_signal_exits_loop(self, tmp_path):
        # Use a source that never returns data
        class InfiniteEmptySource(StreamingSource):
            def poll(self, batch_size):
                return None

            def close(self):
                pass

        source = InfiniteEmptySource()
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        t = loop.start_background(batch_interval=0.05)
        time.sleep(0.15)
        loop.stop()
        t.join(timeout=2.0)
        assert not t.is_alive()

    def test_error_handler_called_instead_of_propagating(self, tmp_path):
        class ErrorSource(StreamingSource):
            def poll(self, batch_size):
                raise RuntimeError("boom")

            def close(self):
                pass

        errors = []
        loop = MicroBatchLoop(
            sources=[ErrorSource()],
            pipeline=[],
            sinks=[],
            batch_size=10,
            temp_dir=str(tmp_path),
            error_handler=errors.append,
        )
        loop._run_one_batch()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)


# ---------------------------------------------------------------------------
# StreamingDataFrame tests
# ---------------------------------------------------------------------------


class TestStreamingDataFrame:
    def _make_sdf(self, tmp_path):
        session = Session(local=True, temp_dir=str(tmp_path))
        table = pa.table({"value": [1, 2, 3]})
        source = FakeSource(table)
        from flint.streaming.dataframe import StreamingDataFrame

        return StreamingDataFrame(source=source, session=session), session

    def test_filter_is_immutable(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        filtered = sdf.filter("value > 1")
        assert len(sdf._pipeline) == 0
        assert len(filtered._pipeline) == 1

    def test_select_is_immutable(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        selected = sdf.select("value")
        assert len(sdf._pipeline) == 0
        assert len(selected._pipeline) == 1

    def test_map_is_immutable(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        mapped = sdf.map(lambda r: r)
        assert len(sdf._pipeline) == 0
        assert len(mapped._pipeline) == 1

    def test_write_stdio_registers_sink(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        with patch.object(sdf, "_run"):
            sdf.write_stdio(label="x")
        assert len(sdf._sinks) == 1

    def test_write_kafka_registers_sink(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        with patch("confluent_kafka.Producer"), patch.object(sdf, "_run"):
            sdf.write_kafka(topic="t", bootstrap_servers="localhost:9092")
        assert len(sdf._sinks) == 1

    def test_chained_transforms_build_pipeline(self, tmp_path):
        sdf, session = self._make_sdf(tmp_path)
        result = sdf.filter("value > 0").select("value").map(lambda r: r)
        assert len(result._pipeline) == 3
        # Original unchanged
        assert len(sdf._pipeline) == 0


# ---------------------------------------------------------------------------
# Integration-style: filter + select end-to-end (no Kafka)
# ---------------------------------------------------------------------------


class TestStreamingEndToEnd:
    def test_filter_and_select_pipeline(self, tmp_path):
        from flint.planner.node import FilterNode, SelectNode

        table = pa.table({"value": [1, -1, 2, -2], "name": ["a", "b", "c", "d"]})
        source = FakeSource(table)
        sink = CollectingSink()

        pipeline = [
            FilterNode(predicate="value > 0", is_sql=True),
            SelectNode(columns=["value"]),
        ]
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=pipeline,
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()

        assert len(sink.batches) == 1
        result = sink.batches[0]
        assert result.schema.names == ["value"]
        assert sorted(result.column("value").to_pylist()) == [1, 2]

    def test_multiple_sources_concatenated(self, tmp_path):
        src1 = FakeSource(pa.table({"x": [1, 2]}))
        src2 = FakeSource(pa.table({"x": [3, 4]}))
        sink = CollectingSink()

        loop = MicroBatchLoop(
            sources=[src1, src2],
            pipeline=[],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()

        assert len(sink.batches) == 1
        assert sorted(sink.batches[0].column("x").to_pylist()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Session method tests
# ---------------------------------------------------------------------------


class TestSessionStreamingMethods:
    def test_read_kafka_stream_returns_streaming_dataframe(self, tmp_path):
        with patch("confluent_kafka.Consumer") as MockConsumer:
            MockConsumer.return_value.subscribe.return_value = None
            session = Session(local=True, temp_dir=str(tmp_path))
            schema = pa.schema([pa.field("id", pa.int64())])
            sdf = session.read_kafka_stream(
                topic="test", bootstrap_servers="localhost:9092", schema=schema
            )
            from flint.streaming.dataframe import StreamingDataFrame
            from flint.streaming.sources import KafkaSource

            assert isinstance(sdf, StreamingDataFrame)
            assert isinstance(sdf._source, KafkaSource)

    def test_read_websocket_stream_returns_streaming_dataframe(self, tmp_path):
        with patch("flint.streaming.sources.WebSocketSource.start"):
            session = Session(local=True, temp_dir=str(tmp_path))
            schema = pa.schema([pa.field("msg", pa.string())])
            sdf = session.read_websocket_stream(
                uri="ws://localhost:9999", schema=schema
            )
            from flint.streaming.dataframe import StreamingDataFrame
            from flint.streaming.sources import WebSocketSource

            assert isinstance(sdf, StreamingDataFrame)
            assert isinstance(sdf._source, WebSocketSource)


# ---------------------------------------------------------------------------
# Distributed micro-batch tests
# ---------------------------------------------------------------------------


class TestDistributedMicroBatch:
    def _make_session(self, tmp_path):
        return Session(local=True, temp_dir=str(tmp_path))

    def _make_source(self, table):
        return FakeSource(table)

    # ------------------------------------------------------------------
    # _assign_partitions
    # ------------------------------------------------------------------

    def test_assign_partitions_even(self):
        from flint.planner.node import EvenPartitionSpec
        from flint.streaming.loop import _assign_partitions

        table = pa.table({"v": list(range(12))})
        ids = _assign_partitions(table, EvenPartitionSpec(n_partitions=4))
        assert ids.type == pa.int32()
        assert len(ids) == 12
        counts = [ids.to_pylist().count(p) for p in range(4)]
        assert counts == [3, 3, 3, 3]

    def test_assign_partitions_even_uneven(self):
        from flint.planner.node import EvenPartitionSpec
        from flint.streaming.loop import _assign_partitions

        table = pa.table({"v": list(range(10))})
        ids = _assign_partitions(table, EvenPartitionSpec(n_partitions=3))
        assert len(ids) == 10
        counts = [ids.to_pylist().count(p) for p in range(3)]
        assert sum(counts) == 10
        # No partition gets 0 rows — distribution is reasonable
        assert all(c > 0 for c in counts)

    def test_assign_partitions_hash_deterministic(self):
        from flint.planner.node import HashPartitionSpec
        from flint.streaming.loop import _assign_partitions

        table = pa.table(
            {"user_id": [1, 2, 3, 1, 2, 3], "val": [10, 20, 30, 40, 50, 60]}
        )
        spec = HashPartitionSpec(keys=["user_id"], n_partitions=4)
        ids1 = _assign_partitions(table, spec)
        ids2 = _assign_partitions(table, spec)
        assert ids1.to_pylist() == ids2.to_pylist()

    def test_assign_partitions_hash_same_key_same_partition(self):
        from flint.planner.node import HashPartitionSpec
        from flint.streaming.loop import _assign_partitions

        table = pa.table({"user_id": [1, 2, 3, 1, 2, 3]})
        ids = _assign_partitions(
            table, HashPartitionSpec(keys=["user_id"], n_partitions=4)
        )
        pid = ids.to_pylist()
        # rows 0 and 3 have user_id=1, rows 1 and 4 have user_id=2, etc.
        assert pid[0] == pid[3]  # user_id=1
        assert pid[1] == pid[4]  # user_id=2
        assert pid[2] == pid[5]  # user_id=3

    def test_assign_partitions_udf(self):
        from flint.planner.node import UserDefinedPartitionSpec
        from flint.streaming.loop import _assign_partitions

        def fn(batch):
            return pa.array([i % 3 for i in range(len(batch))], type=pa.int32())

        table = pa.table({"v": list(range(9))})
        ids = _assign_partitions(table, UserDefinedPartitionSpec(fn=fn, n_partitions=3))
        assert ids.to_pylist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    # ------------------------------------------------------------------
    # small-batch fallback
    # ------------------------------------------------------------------

    def test_small_batch_falls_back_to_single_threaded(self, tmp_path):
        from unittest.mock import patch

        from flint.planner.node import EvenPartitionSpec

        table = pa.table({"v": [1, 2]})
        source = FakeSource(table)
        sink = CollectingSink()
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
            partition_spec=EvenPartitionSpec(n_partitions=10),
            scheduler=None,
        )
        with patch.object(
            loop,
            "_execute_pipeline_distributed",
            wraps=loop._execute_pipeline_distributed,
        ):
            loop._run_one_batch()
        # effective_n = min(10, 2) = 2, but no scheduler → falls back inside distributed
        assert len(sink.batches) == 1

    # ------------------------------------------------------------------
    # distributed execution
    # ------------------------------------------------------------------

    def test_distributes_across_partitions(self, tmp_path):
        from unittest.mock import MagicMock

        from flint.dataframe import InMemoryDataset
        from flint.planner.node import EvenPartitionSpec

        table = pa.table({"v": list(range(8))})
        source = FakeSource(table)
        sink = CollectingSink()

        mock_scheduler = MagicMock()

        def fake_submit(tasks):
            for t in tasks:
                t.output_dataset = InMemoryDataset(
                    t.input_datasets[0].to_arrow(), t.partition_id
                )
            return tasks

        mock_scheduler.submit_batch.side_effect = fake_submit

        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
            partition_spec=EvenPartitionSpec(n_partitions=4),
            scheduler=mock_scheduler,
        )
        loop._run_one_batch()

        mock_scheduler.submit_batch.assert_called_once()
        submitted_tasks = mock_scheduler.submit_batch.call_args[0][0]
        assert len(submitted_tasks) == 4

    def test_coalesces_results(self, tmp_path):
        from flint.executor.scheduler import Scheduler
        from flint.planner.node import EvenPartitionSpec
        from flint.planner.node import FilterNode

        table = pa.table({"v": list(range(6))})
        source = FakeSource(table)
        sink = CollectingSink()
        scheduler = Scheduler(local=True, n_workers=3)

        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[FilterNode(predicate="v >= 0", is_sql=True)],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
            partition_spec=EvenPartitionSpec(n_partitions=3),
            scheduler=scheduler,
        )
        loop._run_one_batch()
        scheduler.stop()

        assert len(sink.batches) == 1
        assert sorted(sink.batches[0].column("v").to_pylist()) == [0, 1, 2, 3, 4, 5]

    def test_n_partitions_1_skips_distributed_path(self, tmp_path):
        from unittest.mock import patch

        table = pa.table({"v": [1, 2, 3]})
        source = FakeSource(table)
        sink = CollectingSink()
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=[],
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        with patch.object(loop, "_execute_pipeline_distributed") as mock_dist:
            loop._run_one_batch()
        mock_dist.assert_not_called()

    # ------------------------------------------------------------------
    # session.read_stream spec inference
    # ------------------------------------------------------------------

    def test_read_stream_default_is_no_distribution(self, tmp_path):
        session = self._make_session(tmp_path)
        sdf = session.read_stream(FakeSource(pa.table({"v": [1]})))
        assert sdf._partition_spec is None

    def test_read_stream_infers_even(self, tmp_path):
        from flint.planner.node import EvenPartitionSpec

        session = self._make_session(tmp_path)
        sdf = session.read_stream(FakeSource(pa.table({"v": [1]})), n_partitions=4)
        assert isinstance(sdf._partition_spec, EvenPartitionSpec)
        assert sdf._partition_spec.n_partitions == 4

    def test_read_stream_infers_hash(self, tmp_path):
        from flint.planner.node import HashPartitionSpec

        session = self._make_session(tmp_path)
        sdf = session.read_stream(
            FakeSource(pa.table({"v": [1]})),
            n_partitions=4,
            partition_by=["user_id"],
        )
        assert isinstance(sdf._partition_spec, HashPartitionSpec)
        assert sdf._partition_spec.keys == ["user_id"]
        assert sdf._partition_spec.n_partitions == 4

    def test_read_stream_infers_udf(self, tmp_path):
        from flint.planner.node import UserDefinedPartitionSpec

        session = self._make_session(tmp_path)

        def fn(batch):
            return pa.array([0] * len(batch), type=pa.int32())

        sdf = session.read_stream(
            FakeSource(pa.table({"v": [1]})),
            n_partitions=4,
            partition_fn=fn,
        )
        assert isinstance(sdf._partition_spec, UserDefinedPartitionSpec)
        assert sdf._partition_spec.fn is fn

    def test_read_stream_both_kwargs_raises(self, tmp_path):
        session = self._make_session(tmp_path)
        with pytest.raises(ValueError, match="at most one"):
            session.read_stream(
                FakeSource(pa.table({"v": [1]})),
                n_partitions=4,
                partition_by=["x"],
                partition_fn=lambda b: b,
            )

    def test_partition_spec_propagates_through_transforms(self, tmp_path):
        session = self._make_session(tmp_path)
        sdf = session.read_stream(FakeSource(pa.table({"v": [1]})), n_partitions=4)
        filtered = sdf.filter("v > 0").map(lambda r: r)
        assert filtered._partition_spec == sdf._partition_spec

    # ------------------------------------------------------------------
    # RepartitionNode / guard tests
    # ------------------------------------------------------------------

    def test_repartition_node_stripped_from_pipeline(self, tmp_path):
        from unittest.mock import patch

        from flint.planner.node import (
            EvenPartitionSpec,
            FilterNode,
            RepartitionNode,
            SelectNode,
        )
        from flint.streaming.dataframe import StreamingDataFrame

        session = self._make_session(tmp_path)
        source = FakeSource(pa.table({"v": [1, 2]}))
        spec = EvenPartitionSpec(n_partitions=3)
        sdf = StreamingDataFrame(
            source=source,
            session=session,
            pipeline=[
                FilterNode(predicate="v > 0", is_sql=True),
                RepartitionNode(partition_spec=spec),
                SelectNode(columns=["v"]),
            ],
            sinks=[],
        )

        captured = {}
        original_init = MicroBatchLoop.__init__

        def capturing_init(self_loop, **kwargs):
            captured.update(kwargs)
            # prevent actual loop from running
            original_init(
                self_loop,
                **{
                    **kwargs,
                    "sources": [FakeSource(pa.table({"v": pa.array([], pa.int64())}))],
                },
            )

        with (
            patch.object(MicroBatchLoop, "__init__", capturing_init),
            patch.object(MicroBatchLoop, "start"),
        ):
            sdf.write_stdio(label="x")

        assert captured["partition_spec"] == spec
        node_types = [type(n).__name__ for n in captured["pipeline"]]
        assert "RepartitionNode" not in node_types
        assert "FilterNode" in node_types
        assert "SelectNode" in node_types

    def test_shuffle_node_in_pipeline_raises(self, tmp_path):
        from flint.planner.node import ShuffleNode
        from flint.streaming.dataframe import StreamingDataFrame

        session = self._make_session(tmp_path)
        sdf = StreamingDataFrame(
            source=FakeSource(pa.table({"v": [1]})),
            session=session,
            pipeline=[ShuffleNode()],
            sinks=[],
        )
        with pytest.raises(ValueError, match="ShuffleNode"):
            sdf._run(batch_interval=0.1)

    def test_join_node_in_pipeline_raises(self, tmp_path):
        from flint.planner.node import JoinNode
        from flint.streaming.dataframe import StreamingDataFrame

        session = self._make_session(tmp_path)
        sdf = StreamingDataFrame(
            source=FakeSource(pa.table({"v": [1]})),
            session=session,
            pipeline=[JoinNode()],
            sinks=[],
        )
        with pytest.raises(ValueError, match="JoinNode"):
            sdf._run(batch_interval=0.1)


# ---------------------------------------------------------------------------
# Streaming GroupBy tests
# ---------------------------------------------------------------------------


class TestStreamingGroupBy:
    def _make_sdf(self, tmp_path, table=None):
        session = Session(local=True, temp_dir=str(tmp_path))
        if table is None:
            table = pa.table({"group": ["a", "b"], "val": [1, 2]})
        source = FakeSource(table)
        from flint.streaming.dataframe import StreamingDataFrame

        return StreamingDataFrame(source=source, session=session), session

    def test_groupby_api_immutability(self, tmp_path):
        """groupby().agg() appends GroupByAggNode; original SDF is unchanged."""
        from flint.planner.node import GroupByAggNode
        from flint.streaming.dataframe import GroupedStreamingDataFrame

        sdf, _ = self._make_sdf(tmp_path)
        gsdf = sdf.groupby("group")
        assert isinstance(gsdf, GroupedStreamingDataFrame)
        agged = gsdf.agg({"val": "sum"})
        assert len(sdf._pipeline) == 0
        assert len(agged._pipeline) == 1
        assert isinstance(agged._pipeline[0], GroupByAggNode)

    def test_groupby_shorthand_methods(self, tmp_path):
        """sum/mean/min/max/count all produce GroupByAggNode."""
        from flint.planner.node import GroupByAggNode

        table = pa.table({"group": ["a"], "val": [1]})
        sdf, _ = self._make_sdf(tmp_path, table)
        for method in ("sum", "mean", "min", "max"):
            result = getattr(sdf.groupby("group"), method)("val")
            assert isinstance(result._pipeline[0], GroupByAggNode)
        count_result = sdf.groupby("group").count()
        assert isinstance(count_result._pipeline[0], GroupByAggNode)

    def test_groupby_node_not_blocked_by_guard(self, tmp_path):
        """GroupByAggNode in streaming pipeline should NOT raise from _run() guard."""
        from flint.planner.node import GroupByAggNode
        from flint.streaming.dataframe import StreamingDataFrame
        from unittest.mock import patch

        session = Session(local=True, temp_dir=str(tmp_path))
        source = FakeSource(pa.table({"group": ["a"], "val": [1]}))
        sdf = StreamingDataFrame(
            source=source,
            session=session,
            pipeline=[
                GroupByAggNode(
                    group_keys=["group"], aggregations=[("val", "sum", "val")]
                )
            ],
            sinks=[],
        )
        # Should not raise — guard only blocks ShuffleNode and JoinNode
        with patch.object(sdf, "_run"):
            sdf.write_stdio()

    def test_groupby_single_threaded_correctness(self, tmp_path):
        """Per-batch groupby produces correct aggregated result in single-partition path."""
        from flint.planner.node import GroupByAggNode

        table = pa.table({"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
        source = FakeSource(table)
        sink = CollectingSink()
        pipeline = [
            GroupByAggNode(group_keys=["group"], aggregations=[("val", "sum", "val")])
        ]

        loop = MicroBatchLoop(
            sources=[source],
            pipeline=pipeline,
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
        )
        loop._run_one_batch()

        assert len(sink.batches) == 1
        result = sink.batches[0].to_pandas().sort_values("group").reset_index(drop=True)
        assert result[result["group"] == "a"]["val"].iloc[0] == 3
        assert result[result["group"] == "b"]["val"].iloc[0] == 7

    def test_groupby_distributed_merge_reduce(self, tmp_path):
        """Distributed path with EvenPartitionSpec(n=3) gives correct global totals after merge."""
        from flint.executor.scheduler import Scheduler
        from flint.planner.node import EvenPartitionSpec, GroupByAggNode

        table = pa.table(
            {"group": ["a", "b", "a", "b", "a", "b"], "val": [1, 4, 2, 5, 3, 6]}
        )
        source = FakeSource(table)
        sink = CollectingSink()
        scheduler = Scheduler(local=True, n_workers=3)

        pipeline = [
            GroupByAggNode(group_keys=["group"], aggregations=[("val", "sum", "val")])
        ]
        loop = MicroBatchLoop(
            sources=[source],
            pipeline=pipeline,
            sinks=[sink],
            batch_size=10,
            temp_dir=str(tmp_path),
            partition_spec=EvenPartitionSpec(n_partitions=3),
            scheduler=scheduler,
        )
        loop._run_one_batch()
        scheduler.stop()

        assert len(sink.batches) == 1
        result = sink.batches[0].to_pandas().sort_values("group").reset_index(drop=True)
        assert result[result["group"] == "a"]["val"].iloc[0] == 6
        assert result[result["group"] == "b"]["val"].iloc[0] == 15
