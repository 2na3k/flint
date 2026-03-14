import os
import tempfile
import uuid


def generate_id() -> str:
    """Generate a short unique ID for nodes, stages, and tasks."""
    return uuid.uuid4().hex[:12]


def generate_temp_path(
    temp_dir: str, prefix: str = "part", suffix: str = ".parquet"
) -> str:
    """Generate a unique temp file path inside temp_dir."""
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"{prefix}_{generate_id()}{suffix}")


def make_temp_dir(prefix: str = "flint_") -> str:
    """Create and return a new temporary directory."""
    return tempfile.mkdtemp(prefix=prefix)
