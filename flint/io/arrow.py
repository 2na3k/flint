"""Arrow I/O helpers — load Dataset → pa.Table and write utilities."""

from __future__ import annotations

import os
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


def load_dataset(dataset: object) -> pa.Table:
    """Dispatch ``dataset.to_arrow()`` with a clear error for unknown types.

    Preferred over calling ``.to_arrow()`` directly because it provides a
    helpful error message.
    """
    from flint.dataframe import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(dataset)!r}")
    return dataset.to_arrow()


def write_parquet(
    table: pa.Table,
    path: str,
    filesystem: Optional[object] = None,
    compression: str = "snappy",
    partition_cols: Optional[list[str]] = None,
) -> None:
    """Write a pyarrow Table to a Parquet file.

    Parameters
    ----------
    table:
        Data to write.
    path:
        Destination file or directory path.
    filesystem:
        An fsspec filesystem, or ``None`` for local.
    compression:
        Parquet compression codec (``"snappy"``, ``"zstd"``, ``"none"``).
    partition_cols:
        If provided, write a partitioned dataset using these columns.
    """
    if partition_cols:
        import pyarrow.dataset as ds

        ds.write_dataset(
            table,
            base_dir=path,
            format="parquet",
            partitioning=partition_cols,
            filesystem=filesystem,
            existing_data_behavior="overwrite_or_ignore",
        )
    else:
        if filesystem is None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pq.write_table(table, path, compression=compression, filesystem=filesystem)


def write_csv(
    table: pa.Table,
    path: str,
    delimiter: str = ",",
    include_header: bool = True,
) -> None:
    """Write a pyarrow Table to a CSV file."""
    import pyarrow.csv as pcsv

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_options = pcsv.WriteOptions(
        delimiter=delimiter,
        include_header=include_header,
    )
    with pa.OSFile(path, "wb") as f:
        pcsv.write_csv(table, f, write_options=write_options)


def infer_schema_from_parquet(
    path: str, filesystem: Optional[object] = None
) -> pa.Schema:
    """Read only the schema from a Parquet file without loading data."""
    return pq.read_schema(path, filesystem=filesystem)
