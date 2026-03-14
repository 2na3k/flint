"""Tests for Parquet / CSV readers and Hive-partitioned dataset handling."""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flint.session import Session


@pytest.fixture()
def session(tmp_path):
    return Session(local=True, temp_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hive_parquet(root: str, data: dict) -> None:
    """Write a Hive-partitioned Parquet dataset.

    data: {(part_col_values, ...): pa.Table}
    e.g. {("2024", "01"): table1, ("2024", "02"): table2}
    """
    for part_vals, table in data.items():
        if not isinstance(part_vals, tuple):
            part_vals = (part_vals,)
        # Build directory path from partition specs
        # caller passes (col=val, ...) strings already formed
        dir_path = os.path.join(root, *part_vals)
        os.makedirs(dir_path, exist_ok=True)
        pq.write_table(table, os.path.join(dir_path, "part-0.parquet"))


def make_hive_csv(root: str, data: dict) -> None:
    import pyarrow.csv as pcsv

    for part_vals, table in data.items():
        if not isinstance(part_vals, tuple):
            part_vals = (part_vals,)
        dir_path = os.path.join(root, *part_vals)
        os.makedirs(dir_path, exist_ok=True)
        with pa.OSFile(os.path.join(dir_path, "part-0.csv"), "wb") as f:
            pcsv.write_csv(table, f)


# ---------------------------------------------------------------------------
# Flat file reading
# ---------------------------------------------------------------------------


def test_read_single_parquet(session, tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = str(tmp_path / "data.parquet")
    pq.write_table(table, path)

    df = session.read_parquet(path)
    assert df.count() == 3


def test_read_parquet_glob(session, tmp_path):
    for i in range(3):
        pq.write_table(pa.table({"a": [i]}), str(tmp_path / f"part-{i}.parquet"))

    df = session.read_parquet(str(tmp_path / "*.parquet"))
    assert df.count() == 3


def test_read_single_csv(session, tmp_path):
    import pyarrow.csv as pcsv

    table = pa.table({"x": [10, 20, 30]})
    path = str(tmp_path / "data.csv")
    with pa.OSFile(path, "wb") as f:
        pcsv.write_csv(table, f)

    df = session.read_csv(path)
    assert df.count() == 3


# ---------------------------------------------------------------------------
# Hive-partitioned Parquet
# ---------------------------------------------------------------------------


def test_hive_parquet_detection(session, tmp_path):
    """Session should detect Hive partitioning from key=value directories."""
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2023",): pa.table({"id": [1, 2], "val": [10, 20]}),
            ("year=2024",): pa.table({"id": [3, 4], "val": [30, 40]}),
        },
    )

    df = session.read_parquet(root)
    node = df._node
    assert node.hive_partitioning is True
    assert "year" in node.partition_columns
    assert len(node.paths) == 2


def test_hive_parquet_columns_added(session, tmp_path):
    """Partition key-value columns must appear in every row of the result."""
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2023",): pa.table({"id": [1, 2]}),
            ("year=2024",): pa.table({"id": [3, 4]}),
        },
    )

    result = session.read_parquet(root).to_arrow()
    assert "year" in result.schema.names
    years = set(result.column("year").to_pylist())
    assert years == {2023, 2024}
    assert result.num_rows == 4


def test_hive_parquet_multi_level(session, tmp_path):
    """Multi-level Hive partitioning: year= / month=."""
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2024", "month=1"): pa.table({"id": [1]}),
            ("year=2024", "month=2"): pa.table({"id": [2]}),
            ("year=2025", "month=1"): pa.table({"id": [3]}),
        },
    )

    result = session.read_parquet(root).to_arrow()
    assert "year" in result.schema.names
    assert "month" in result.schema.names
    assert result.num_rows == 3


def test_hive_parquet_filter_reads_correct_data(session, tmp_path):
    """Filtering on partition columns should return only matching rows."""
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2023",): pa.table({"id": [1, 2]}),
            ("year=2024",): pa.table({"id": [3, 4]}),
        },
    )

    result = session.read_parquet(root).filter("year = 2024").to_arrow()
    assert result.num_rows == 2
    assert set(result.column("id").to_pylist()) == {3, 4}


def test_hive_parquet_partition_pruning(session, tmp_path):
    """PartitionPruning optimizer rule should reduce the file list before execution."""
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2023",): pa.table({"id": [1, 2]}),
            ("year=2024",): pa.table({"id": [3, 4]}),
            ("year=2025",): pa.table({"id": [5, 6]}),
        },
    )

    df = session.read_parquet(root).filter("year = 2024")
    df.explain("optimized")

    # After optimization, the ReadParquet node should have only 1 file
    from flint.planner.optimizer import Optimizer
    from flint.planner.node import FilterNode, ReadParquet

    optimized = Optimizer().optimize(df._node)
    # The optimized tree has FilterNode → ReadParquet(1 file)
    assert isinstance(optimized, FilterNode)
    assert isinstance(optimized.children[0], ReadParquet)
    assert len(optimized.children[0].paths) == 1


# ---------------------------------------------------------------------------
# Hive-partitioned CSV
# ---------------------------------------------------------------------------


def test_hive_csv_detection(session, tmp_path):
    root = str(tmp_path / "csv_dataset")
    make_hive_csv(
        root,
        {
            ("region=us",): pa.table({"id": [1, 2], "v": [10, 20]}),
            ("region=eu",): pa.table({"id": [3, 4], "v": [30, 40]}),
        },
    )

    df = session.read_csv(root)
    assert df._node.hive_partitioning is True
    assert "region" in df._node.partition_columns


def test_hive_csv_columns_added(session, tmp_path):
    root = str(tmp_path / "csv_dataset")
    make_hive_csv(
        root,
        {
            ("region=us",): pa.table({"id": [1, 2]}),
            ("region=eu",): pa.table({"id": [3, 4]}),
        },
    )

    result = session.read_csv(root).to_arrow()
    assert "region" in result.schema.names
    regions = set(result.column("region").to_pylist())
    assert regions == {"us", "eu"}
    assert result.num_rows == 4


# ---------------------------------------------------------------------------
# explain() shows partition info
# ---------------------------------------------------------------------------


def test_explain_shows_hive_info(session, tmp_path, capsys):
    root = str(tmp_path / "dataset")
    make_hive_parquet(
        root,
        {
            ("year=2024",): pa.table({"id": [1]}),
        },
    )

    df = session.read_parquet(root)
    df.explain("logical")
    captured = capsys.readouterr()
    assert "hive_cols" in captured.out


# ---------------------------------------------------------------------------
# Hive-partitioned writes
# ---------------------------------------------------------------------------


def test_write_parquet_hive_creates_directories(session, tmp_path):
    """write_parquet with partition_cols should create key=val/ directories."""
    out = str(tmp_path / "out")
    table = pa.table({"year": [2023, 2023, 2024], "id": [1, 2, 3]})

    session.from_arrow(table).write_parquet(out, partition_cols=["year"]).compute()

    assert os.path.isdir(os.path.join(out, "year=2023"))
    assert os.path.isdir(os.path.join(out, "year=2024"))


def test_write_parquet_hive_data_correct(session, tmp_path):
    """Partition column is dropped from file schema; data round-trips correctly."""
    out = str(tmp_path / "out")
    table = pa.table({"year": [2023, 2023, 2024], "id": [1, 2, 3]})

    session.from_arrow(table).write_parquet(out, partition_cols=["year"]).compute()

    result = session.read_parquet(out).to_arrow()
    assert result.num_rows == 3
    assert "year" in result.schema.names
    assert set(result.column("year").to_pylist()) == {2023, 2024}


def test_write_parquet_hive_multi_level(session, tmp_path):
    """Multi-level Hive partition: year= / month=."""
    out = str(tmp_path / "out")
    table = pa.table(
        {
            "year": [2024, 2024, 2025],
            "month": [1, 2, 1],
            "id": [10, 20, 30],
        }
    )

    session.from_arrow(table).write_parquet(
        out, partition_cols=["year", "month"]
    ).compute()

    assert os.path.isdir(os.path.join(out, "year=2024", "month=1"))
    assert os.path.isdir(os.path.join(out, "year=2024", "month=2"))
    assert os.path.isdir(os.path.join(out, "year=2025", "month=1"))

    result = session.read_parquet(out).to_arrow()
    assert result.num_rows == 3
    assert set(result.column("year").to_pylist()) == {2024, 2025}
