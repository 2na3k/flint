"""Filesystem resolution and dataset discovery (including Hive partitioning)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Matches Hive partition directory segments like  year=2024  or  dt=2024-01-01
_HIVE_PART_RE = re.compile(r"^([^=]+)=(.+)$")


# ---------------------------------------------------------------------------
# Filesystem resolution (fsspec / pyarrow)
# ---------------------------------------------------------------------------


def resolve_filesystem(uri: str) -> Tuple[Optional[object], str]:
    """Resolve a URI to an ``(fsspec.AbstractFileSystem, path)`` pair.

    Returns ``(None, path)`` for local paths — pyarrow uses the local FS by default.
    """
    import fsspec

    if "://" not in uri:
        return None, uri

    scheme = uri.split("://")[0].lower()

    if scheme == "file":
        return None, uri[len("file://"):]

    if scheme in ("s3", "s3a"):
        return fsspec.filesystem("s3"), uri[len(f"{scheme}://"):]

    if scheme in ("gs", "gcs"):
        return fsspec.filesystem("gcs"), uri[len(f"{scheme}://"):]

    if scheme in ("az", "abfs", "abfss"):
        return fsspec.filesystem("abfs"), uri[len(f"{scheme}://"):]

    return fsspec.filesystem(scheme), uri[len(f"{scheme}://"):]


def _pyarrow_fs(protocol: Optional[str], path: str):
    """Return a ``(pyarrow.fs.FileSystem, clean_path)`` pair."""
    import pyarrow.fs as pafs

    if protocol is None and "://" not in path:
        return pafs.LocalFileSystem(), path

    uri = path if "://" in path else f"{protocol}://{path}"
    try:
        return pafs.FileSystem.from_uri(uri)
    except Exception:
        return pafs.LocalFileSystem(), path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def is_glob(path: str) -> bool:
    """Return True if the path contains glob wildcards."""
    return any(c in path for c in ("*", "?", "[", "]"))


def expand_paths(path: str, filesystem: Optional[object] = None) -> List[str]:
    """Expand glob patterns into a sorted list of concrete file paths."""
    if not is_glob(path):
        return [path]

    if filesystem is not None:
        return sorted(filesystem.glob(path))

    import glob as glob_mod

    return sorted(glob_mod.glob(path))


# ---------------------------------------------------------------------------
# Hive partition discovery
# ---------------------------------------------------------------------------


def discover_dataset(
    path: str,
    format: str = "parquet",
    filesystem: Optional[str] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Discover all data files under *path*, detecting Hive-style partitioning.

    Works with local paths, S3, GCS, and any other pyarrow-supported filesystem.

    Parameters
    ----------
    path:
        Root directory of the dataset, or a glob pattern.
    format:
        File format: ``"parquet"`` or ``"csv"``.
    filesystem:
        fsspec protocol string (``"s3"``, ``"gcs"``, …).  ``None`` = local.

    Returns
    -------
    (files, partition_cols, partition_values)
        files:             sorted list of concrete file paths
        partition_cols:    list of partition column names (empty if not partitioned)
        partition_values:  one dict per file — e.g. ``{"year": 2024, "month": 1}``
    """
    # Glob patterns → flat file list, no Hive discovery
    if is_glob(path):
        files = expand_paths(path, filesystem)
        return files, [], [{} for _ in files]

    fs_obj, clean_path = _pyarrow_fs(filesystem, path)

    # Check if this looks like a Hive-partitioned directory
    if _is_hive_partitioned(clean_path, fs_obj):
        return _discover_hive(clean_path, format, fs_obj, filesystem)

    # Plain directory or single file — list all matching files
    files = _list_files(clean_path, format, fs_obj)
    return files, [], [{} for _ in files]


def _is_hive_partitioned(path: str, fs_obj) -> bool:
    """Return True if any immediate child directory matches ``key=value``."""
    import pyarrow.fs as pafs

    try:
        selector = pafs.FileSelector(path, recursive=False)
        infos = fs_obj.get_file_info(selector)
        return any(
            info.type == pafs.FileType.Directory and _HIVE_PART_RE.match(info.base_name)
            for info in infos
        )
    except Exception:
        return False


def _discover_hive(
    base_path: str,
    format: str,
    fs_obj,
    protocol: Optional[str],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Walk a Hive-partitioned directory tree and collect files + partition values."""
    import pyarrow.dataset as pads

    ext = ".parquet" if format == "parquet" else ".csv"

    try:
        dataset = pads.dataset(
            base_path,
            format=format,
            filesystem=fs_obj,
            partitioning="hive",
        )
        partition_cols: List[str] = dataset.partitioning.schema.names
    except Exception:
        # Fallback — walk manually
        return _walk_hive_manually(base_path, ext, fs_obj)

    if not partition_cols:
        files = _list_files(base_path, format, fs_obj)
        return files, [], [{} for _ in files]

    files: List[str] = []
    partition_values: List[Dict[str, Any]] = []

    part_schema = dataset.partitioning.schema

    for fragment in sorted(dataset.get_fragments(), key=lambda f: f.path):
        fpath = fragment.path
        files.append(fpath)
        part_vals = _parse_hive_path(fpath, base_path, part_schema)
        partition_values.append(part_vals)

    return files, partition_cols, partition_values


def _walk_hive_manually(
    base_path: str,
    ext: str,
    fs_obj,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Manual recursive walk when pyarrow.dataset cannot parse the tree."""
    import pyarrow.fs as pafs

    files: List[str] = []
    partition_values: List[Dict[str, Any]] = []
    partition_cols_set: list = []

    try:
        selector = pafs.FileSelector(base_path, recursive=True)
        infos = fs_obj.get_file_info(selector)
    except Exception:
        return [], [], []

    for info in sorted(infos, key=lambda i: i.path):
        if info.type != pafs.FileType.File:
            continue
        if not info.path.endswith(ext):
            continue
        files.append(info.path)
        part_vals = _parse_hive_path_raw(info.path, base_path)
        partition_values.append(part_vals)
        for k in part_vals:
            if k not in partition_cols_set:
                partition_cols_set.append(k)

    return files, partition_cols_set, partition_values


def _list_files(path: str, format: str, fs_obj) -> List[str]:
    """List all files of the given format under *path* (non-recursive for single files)."""
    import pyarrow.fs as pafs

    ext = ".parquet" if format == "parquet" else ".csv"

    try:
        info = fs_obj.get_file_info(path)
        if info.type == pafs.FileType.File:
            return [path]
        # It's a directory — list recursively
        selector = pafs.FileSelector(path, recursive=True)
        infos = fs_obj.get_file_info(selector)
        return sorted(i.path for i in infos if i.type == pafs.FileType.File and i.path.endswith(ext))
    except Exception:
        return [path]


def _parse_hive_path(file_path: str, base_path: str, part_schema) -> Dict[str, Any]:
    """Extract and type-cast Hive partition values from a file path."""
    raw = _parse_hive_path_raw(file_path, base_path)
    result: Dict[str, Any] = {}
    for field in part_schema:
        raw_val = raw.get(field.name)
        if raw_val is not None:
            result[field.name] = _cast_value(raw_val, field.type)
    return result


def _parse_hive_path_raw(file_path: str, base_path: str) -> Dict[str, str]:
    """Extract raw (string) key=value pairs from path components."""
    rel = file_path.replace("\\", "/")
    base = base_path.rstrip("/") + "/"
    if rel.startswith(base):
        rel = rel[len(base):]
    parts = rel.split("/")[:-1]  # exclude filename
    result: Dict[str, str] = {}
    for part in parts:
        m = _HIVE_PART_RE.match(part)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def _cast_value(raw: str, pa_type) -> Any:
    """Cast a partition value string to the appropriate Python type."""
    import pyarrow as pa

    try:
        if pa.types.is_integer(pa_type):
            return int(raw)
        if pa.types.is_floating(pa_type):
            return float(raw)
        if pa.types.is_date(pa_type):
            from datetime import date

            return date.fromisoformat(raw)
        return str(raw)
    except (ValueError, TypeError):
        return str(raw)


# ---------------------------------------------------------------------------
# Partition pruning helper (used by optimizer)
# ---------------------------------------------------------------------------


def eval_partition_filter(
    partition_values: List[Dict[str, Any]],
    partition_cols: List[str],
    predicate: str,
) -> List[int]:
    """Return indices of partitions that satisfy *predicate*.

    Uses DuckDB to evaluate the SQL predicate against a small table of
    partition values.  Returns all indices if the predicate cannot be parsed
    (safe fallback — no pruning, no data loss).
    """
    if not partition_values:
        return []

    import duckdb
    import pyarrow as pa

    n = len(partition_values)
    data: Dict[str, Any] = {"__idx__": list(range(n))}
    for col in partition_cols:
        data[col] = [pv.get(col) for pv in partition_values]

    table = pa.table(data)
    conn = duckdb.connect()
    conn.register("__parts__", table)

    try:
        rows = conn.execute(f"SELECT __idx__ FROM __parts__ WHERE ({predicate})").fetchall()
        return [r[0] for r in rows]
    except Exception:
        # Predicate references non-partition columns or has syntax we can't evaluate
        return list(range(n))
