"""Unit tests for DataFrame plan construction (no execution)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flint.dataframe import GroupedDataFrame
from flint.planner.node import (
    FilterNode,
    FlatMapNode,
    GroupByAggNode,
    JoinNode,
    LimitNode,
    MapBatchesNode,
    MapNode,
    RepartitionNode,
    SelectNode,
    SqlNode,
    WriteCsv,
    WriteParquet,
)
from flint.session import Session


@pytest.fixture()
def session(tmp_path):
    return Session(local=True, temp_dir=str(tmp_path))


@pytest.fixture()
def df(session):
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    return session.from_arrow(table)


# ---------------------------------------------------------------------------
# Plan construction — no execution happens
# ---------------------------------------------------------------------------


def test_filter_sql_builds_filter_node(df):
    result = df.filter("a > 1")
    assert isinstance(result._node, FilterNode)
    assert result._node.is_sql is True
    assert result._node.predicate == "a > 1"
    assert result._node.children[0] is df._node


def test_filter_callable_builds_filter_node(df):
    def fn(row):
        return row["a"] > 1

    result = df.filter(fn)
    assert isinstance(result._node, FilterNode)
    assert result._node.is_sql is False
    assert result._node.predicate is fn


def test_select_builds_select_node(df):
    result = df.select("a", "b")
    assert isinstance(result._node, SelectNode)
    assert result._node.columns == ["a", "b"]


def test_map_builds_map_node(df):
    def fn(row):
        return row

    result = df.map(fn)
    assert isinstance(result._node, MapNode)
    assert result._node.fn is fn


def test_flatmap_builds_flatmap_node(df):
    def fn(row):
        return [row]

    result = df.flatmap(fn)
    assert isinstance(result._node, FlatMapNode)
    assert result._node.fn is fn


def test_map_batches_builds_node(df):
    def fn(batch):
        return batch

    result = df.map_batches(fn, batch_size=512)
    assert isinstance(result._node, MapBatchesNode)
    assert result._node.batch_size == 512


def test_limit_builds_limit_node(df):
    result = df.limit(10)
    assert isinstance(result._node, LimitNode)
    assert result._node.limit == 10


def test_repartition_even_builds_repartition_node(df):
    from flint.planner.node import EvenPartitionSpec

    result = df.repartition(4)
    assert isinstance(result._node, RepartitionNode)
    assert isinstance(result._node.partition_spec, EvenPartitionSpec)
    assert result._node.partition_spec.n_partitions == 4


def test_repartition_hash_by_column(df):
    from flint.planner.node import HashPartitionSpec

    result = df.repartition(4, partition_by="a")
    assert isinstance(result._node, RepartitionNode)
    assert isinstance(result._node.partition_spec, HashPartitionSpec)
    assert result._node.partition_spec.keys == ["a"]


def test_groupby_returns_grouped_dataframe(df):
    grouped = df.groupby("a")
    assert isinstance(grouped, GroupedDataFrame)
    assert grouped._keys == ["a"]


def test_groupby_agg_builds_groupby_node(df):
    result = df.groupby("a").agg({"b": "sum"})
    assert isinstance(result._node, GroupByAggNode)
    assert result._node.group_keys == ["a"]
    assert ("b", "sum", "b") in result._node.aggregations


def test_join_builds_join_node(session):
    t1 = pa.table({"id": [1, 2], "v": [10, 20]})
    t2 = pa.table({"id": [1, 2], "w": [100, 200]})
    df1 = session.from_arrow(t1)
    df2 = session.from_arrow(t2)
    result = df1.join(df2, on="id")
    assert isinstance(result._node, JoinNode)
    assert result._node.left_keys == ["id"]
    assert result._node.right_keys == ["id"]
    assert result._node.how == "inner"


def test_sql_builds_sql_node(df):
    result = df.sql("SELECT *, a * 2 AS double_a FROM this")
    assert isinstance(result._node, SqlNode)
    assert "double_a" in result._node.sql


def test_write_parquet_builds_write_node(df):
    result = df.write_parquet("/tmp/out.parquet")
    assert isinstance(result._node, WriteParquet)
    assert result._node.path == "/tmp/out.parquet"


def test_write_csv_builds_write_node(df):
    result = df.write_csv("/tmp/out.csv")
    assert isinstance(result._node, WriteCsv)


def test_chaining_returns_new_dataframe_each_time(df):
    """Each operation must return a new DataFrame (immutable)."""
    r1 = df.filter("a > 0")
    r2 = r1.select("a")
    r3 = r2.limit(5)
    assert r1 is not df
    assert r2 is not r1
    assert r3 is not r2
    assert df._node is not r1._node


def test_plan_depth(df):
    """Verify parent-child linkage through the plan."""
    result = df.filter("a > 0").select("a").limit(5)
    assert isinstance(result._node, LimitNode)
    assert isinstance(result._node.children[0], SelectNode)
    assert isinstance(result._node.children[0].children[0], FilterNode)
    assert result._node.children[0].children[0].children[0] is df._node
