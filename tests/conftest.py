"""Shared test fixtures for Flint tests."""

from __future__ import annotations


import pyarrow as pa
import pytest

from flint.session import Session


@pytest.fixture()
def session(tmp_path):
    """A local Session using a temporary directory."""
    s = Session(local=True, n_workers=2, temp_dir=str(tmp_path))
    yield s
    s.stop()


@pytest.fixture()
def simple_table() -> pa.Table:
    """A small in-memory Arrow table for unit tests."""
    return pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "charlie", "diana", "eve"],
            "age": [25, 30, 22, 35, 28],
            "score": [90.0, 85.5, 78.0, 92.5, 88.0],
        }
    )


@pytest.fixture()
def parquet_file(tmp_path, simple_table) -> str:
    """Write the simple table to a Parquet file and return the path."""
    import pyarrow.parquet as pq

    path = str(tmp_path / "test.parquet")
    pq.write_table(simple_table, path)
    return path


@pytest.fixture()
def multi_parquet_files(tmp_path, simple_table) -> list[str]:
    """Split the simple table into two Parquet files and return paths."""
    import pyarrow.parquet as pq

    paths = []
    half = len(simple_table) // 2
    for i, slc in enumerate([simple_table.slice(0, half), simple_table.slice(half)]):
        path = str(tmp_path / f"part_{i}.parquet")
        pq.write_table(slc, path)
        paths.append(path)
    return paths
