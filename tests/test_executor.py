"""Unit tests for single-task execution via Executor + Driver (local mode)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flint.executor.driver import (
    _assign_partitions,
    _even_partition,
    _hash_partition,
    _user_defined_partition,
)
from flint.executor.executor import _partition_write_path
from flint.planner.node import (
    EvenPartitionSpec,
    HashPartitionSpec,
    UserDefinedPartitionSpec,
)
from flint.session import Session


@pytest.fixture()
def session(tmp_path):
    return Session(local=True, n_workers=2, temp_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# End-to-end local execution tests (Session → DataFrame → compute)
# ---------------------------------------------------------------------------


def test_from_arrow_compute(session):
    table = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
    df = session.from_arrow(table)
    result = df.compute()
    assert result.count() == 3


def test_filter_sql(session):
    table = pa.table({"a": [1, 2, 3, 4, 5]})
    df = session.from_arrow(table).filter("a > 2")
    result = df.to_arrow()
    assert len(result) == 3
    assert result.column("a").to_pylist() == [3, 4, 5]


def test_filter_callable(session):
    table = pa.table({"a": [1, 2, 3, 4, 5]})
    df = session.from_arrow(table).filter(lambda row: row["a"] > 2)
    result = df.to_arrow()
    assert len(result) == 3


def test_select(session):
    table = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df = session.from_arrow(table).select("a", "b")
    result = df.to_arrow()
    assert result.schema.names == ["a", "b"]


def test_limit(session):
    table = pa.table({"a": list(range(100))})
    df = session.from_arrow(table).limit(10)
    result = df.to_arrow()
    assert len(result) == 10


def test_map(session):
    table = pa.table({"a": [1, 2, 3]})
    df = session.from_arrow(table).map(
        lambda row: {"a": row["a"], "doubled": row["a"] * 2}
    )
    result = df.to_arrow()
    assert result.column("doubled").to_pylist() == [2, 4, 6]


def test_flatmap(session):
    table = pa.table({"a": [1, 2]})
    df = session.from_arrow(table).flatmap(lambda row: [row, row])
    result = df.to_arrow()
    assert len(result) == 4


def test_map_batches(session):
    table = pa.table({"a": [1, 2, 3, 4]})

    def double_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
        import pyarrow.compute as pc

        col = pc.multiply(batch.column("a"), 2)
        return pa.record_batch({"a": col})

    df = session.from_arrow(table).map_batches(double_batch)
    result = df.to_arrow()
    assert result.column("a").to_pylist() == [2, 4, 6, 8]


def test_groupby_agg(session):
    table = pa.table({"group": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]})
    df = session.from_arrow(table).groupby("group").agg({"val": "sum"})
    result = df.to_pandas().sort_values("group").reset_index(drop=True)
    assert result[result["group"] == "a"]["val"].iloc[0] == 3
    assert result[result["group"] == "b"]["val"].iloc[0] == 7


def test_sql_escape_hatch(session):
    table = pa.table({"a": [1, 2, 3]})
    df = session.from_arrow(table).sql("SELECT a, a * 2 AS double_a FROM this")
    result = df.to_arrow()
    assert "double_a" in result.schema.names
    assert result.column("double_a").to_pylist() == [2, 4, 6]


def test_count(session):
    table = pa.table({"a": list(range(50))})
    df = session.from_arrow(table)
    assert df.count() == 50


def test_to_pandas(session):
    import pandas as pd

    table = pa.table({"a": [1, 2, 3]})
    df = session.from_arrow(table)
    pdf = df.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    assert len(pdf) == 3


def test_chained_operations(session):
    """filter → select → map → limit chain produces correct results."""
    table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "charlie", "diana", "eve"],
            "score": [90, 50, 80, 40, 70],
        }
    )
    df = (
        session.from_arrow(table)
        .filter("score >= 70")
        .select("id", "name", "score")
        .map(lambda r: {**r, "grade": "pass"})
        .limit(2)
    )
    result = df.to_arrow()
    assert len(result) == 2
    assert "grade" in result.schema.names


def test_repartition(session):
    table = pa.table({"a": list(range(10))})
    df = session.from_arrow(table, n_partitions=1).repartition(2)
    datasets = df._compute()
    assert len(datasets) == 2


def test_join_inner(session):
    left = pa.table({"id": [1, 2, 3], "val_l": [10, 20, 30]})
    right = pa.table({"id": [1, 2, 4], "val_r": [100, 200, 400]})
    df_l = session.from_arrow(left, n_partitions=1)
    df_r = session.from_arrow(right, n_partitions=1)
    result = df_l.join(df_r, on="id", how="inner").to_pandas()
    assert len(result) == 2
    assert set(result["id"].tolist()) == {1, 2}


def test_compute_caches_result(session):
    """Calling compute() twice should not re-execute the plan."""
    table = pa.table({"a": [1, 2, 3]})
    df = session.from_arrow(table)
    r1 = df.compute()
    df.compute()  # second call should use cache
    # Cached datasets are set after first compute
    assert df._cached_datasets is r1._cached_datasets or df._cached_datasets is not None


# ---------------------------------------------------------------------------
# Partition assignment helpers
# ---------------------------------------------------------------------------


class TestEvenPartition:
    def test_basic(self):
        table = pa.table({"a": list(range(6))})
        pids = _even_partition(table, n_out=3)
        assert pids.type == pa.int32()
        assert len(pids) == 6
        assert pids.to_pylist() == [0, 0, 1, 1, 2, 2]

    def test_single_partition(self):
        table = pa.table({"a": list(range(4))})
        pids = _even_partition(table, n_out=1)
        assert all(p == 0 for p in pids.to_pylist())

    def test_more_partitions_than_rows(self):
        table = pa.table({"a": [1, 2]})
        pids = _even_partition(table, n_out=5)
        assert len(pids) == 2
        # All rows go to partition 0 (integer division rounds down)
        assert all(0 <= p < 5 for p in pids.to_pylist())


class TestHashPartition:
    def test_basic(self):
        table = pa.table({"key": ["a", "b", "c", "d"]})
        pids = _hash_partition(table, keys=["key"], n=3)
        assert pids.type == pa.int32()
        assert len(pids) == 4
        assert all(0 <= p < 3 for p in pids.to_pylist())

    def test_stable(self):
        table = pa.table({"key": ["x", "y", "z"]})
        pids1 = _hash_partition(table, keys=["key"], n=4)
        pids2 = _hash_partition(table, keys=["key"], n=4)
        assert pids1.to_pylist() == pids2.to_pylist()

    def test_multi_key(self):
        table = pa.table({"a": [1, 2], "b": [3, 4]})
        pids = _hash_partition(table, keys=["a", "b"], n=4)
        assert len(pids) == 2
        assert all(0 <= p < 4 for p in pids.to_pylist())


class TestUserDefinedPartition:
    def test_basic(self):
        table = pa.table({"a": [0, 1, 2, 3]})
        spec = UserDefinedPartitionSpec(
            fn=lambda batch: pa.array([v % 2 for v in batch.column("a").to_pylist()]),
            n_partitions=2,
        )
        pids = _user_defined_partition(table, spec)
        assert pids.type == pa.int32()
        assert pids.to_pylist() == [0, 1, 0, 1]

    def test_empty_table(self):
        table = pa.table({"a": pa.array([], type=pa.int64())})
        spec = UserDefinedPartitionSpec(fn=lambda batch: pa.array([]), n_partitions=2)
        pids = _user_defined_partition(table, spec)
        assert len(pids) == 0


class TestAssignPartitionsDispatch:
    def test_dispatches_even(self):
        table = pa.table({"a": list(range(4))})
        spec = EvenPartitionSpec(n_partitions=2)
        pids = _assign_partitions(table, spec)
        assert len(pids) == 4
        assert all(0 <= p < 2 for p in pids.to_pylist())

    def test_dispatches_hash(self):
        table = pa.table({"k": ["x", "y"]})
        spec = HashPartitionSpec(keys=["k"], n_partitions=3)
        pids = _assign_partitions(table, spec)
        assert len(pids) == 2

    def test_dispatches_user_defined(self):
        table = pa.table({"a": [0, 1, 2]})
        spec = UserDefinedPartitionSpec(
            fn=lambda batch: pa.array([v % 3 for v in batch.column("a").to_pylist()]),
            n_partitions=3,
        )
        pids = _assign_partitions(table, spec)
        assert pids.to_pylist() == [0, 1, 2]

    def test_unknown_spec_raises(self):
        table = pa.table({"a": [1]})
        with pytest.raises(NotImplementedError):
            _assign_partitions(table, object())


# ---------------------------------------------------------------------------
# _partition_write_path — unique file naming
# ---------------------------------------------------------------------------


class TestPartitionWritePath:
    def test_directory_path_includes_task_id(self):
        path = _partition_write_path("/out", partition_id=0, task_id="abc123")
        assert path == "/out/part-00000-abc123.parquet"

    def test_directory_path_without_task_id(self):
        path = _partition_write_path("/out", partition_id=0)
        assert path == "/out/part-00000.parquet"

    def test_file_path_includes_task_id(self):
        path = _partition_write_path("/out/result.parquet", partition_id=2, task_id="xyz")
        assert path == "/out/result-part-00002-xyz.parquet"

    def test_file_path_without_task_id(self):
        path = _partition_write_path("/out/result.parquet", partition_id=2)
        assert path == "/out/result-part-00002.parquet"

    def test_different_task_ids_produce_distinct_paths(self):
        p1 = _partition_write_path("/out", partition_id=0, task_id="aaa")
        p2 = _partition_write_path("/out", partition_id=0, task_id="bbb")
        assert p1 != p2

    def test_write_parquet_files_have_distinct_names(self, tmp_path, session):
        """Two write_parquet calls to the same dir produce non-colliding files."""
        import os
        table = pa.table({"x": [1, 2, 3]})
        out = str(tmp_path / "output")
        os.makedirs(out)
        session.from_arrow(table, n_partitions=1).write_parquet(out).compute()
        session.from_arrow(table, n_partitions=1).write_parquet(out).compute()
        files = os.listdir(out)
        assert len(files) == 2
        assert len(set(files)) == 2  # all unique
