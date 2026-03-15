"""Flint structured logger — loguru-based, singleton per process."""

from __future__ import annotations

import sys

from loguru import logger as _loguru_logger

# Remove default handler so callers configure their own sinks
_loguru_logger.remove()

# -----------------------------------------------------------------------
# Default handler: human-readable stderr
# -----------------------------------------------------------------------
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[flint_context]}</cyan> | "
    "{message}"
)

_loguru_logger.add(
    sys.stderr,
    format=_DEFAULT_FORMAT,
    level="INFO",
    filter=lambda r: True,
    colorize=True,
)


def get_logger(context: str = "flint"):
    """Return a bound loguru logger with *context* attached.

    Parameters
    ----------
    context:
        Short label that appears in every log line, e.g. ``"executor"``,
        ``"streaming"``, ``"driver"``.

    Example
    -------
    ::

        log = get_logger("executor")
        log.info("task started", task_id="abc", partition=0)
    """
    return _loguru_logger.bind(flint_context=context)


def add_json_sink(path: str, level: str = "DEBUG") -> int:
    """Add a JSON-lines sink that writes structured records to *path*.

    Returns the sink id (pass to ``remove_sink`` to detach it).

    Example
    -------
    ::

        sid = add_json_sink("/var/log/flint.jsonl")
        # ... run workload ...
        remove_sink(sid)
    """
    import json

    def _serialiser(message) -> str:
        record = message.record
        payload = {
            "ts": record["time"].isoformat(),
            "level": record["level"].name,
            "context": record["extra"].get("flint_context", ""),
            "message": record["message"],
            **{k: v for k, v in record["extra"].items() if k != "flint_context"},
        }
        if record["exception"] is not None:
            payload["exception"] = str(record["exception"])
        return json.dumps(payload)

    return _loguru_logger.add(path, format=_serialiser, level=level, serialize=False)


def remove_sink(sink_id: int) -> None:
    """Detach a previously registered sink by its id."""
    _loguru_logger.remove(sink_id)
