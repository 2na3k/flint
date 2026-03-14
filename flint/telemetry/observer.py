"""StateObserver — unified job-event bus for batch and streaming pipelines.

Usage
-----
Batch (automatic via Executor):

    # Events are emitted automatically — just subscribe to observe them.
    from flint.telemetry.observer import StateObserver

    def my_handler(event):
        print(event.status, event.rows_out, event.duration_ms)

    StateObserver.get().subscribe(my_handler)

Streaming (automatic via MicroBatchLoop):

    # Same observer — streaming emits BatchEvent, batch emits TaskEvent.
    StateObserver.get().subscribe(my_handler)

Manual export / reset::

    obs = StateObserver.get()
    summary = obs.summary()          # dict with aggregate counters
    obs.reset()                      # clear counters (e.g. between test runs)
"""

from __future__ import annotations

import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from flint.telemetry.logger import get_logger

_log = get_logger("observer")


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class EventKind(str, Enum):
    TASK = "task"        # single executor task (batch)
    BATCH = "batch"      # one micro-batch cycle (streaming)


class EventStatus(str, Enum):
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class JobEvent:
    """Immutable record emitted by the observer at each lifecycle transition.

    Fields
    ------
    kind        ``"task"`` for batch executor tasks, ``"batch"`` for streaming.
    status      ``"started" | "success" | "failed"``
    job_id      Unique ID of the task or batch (streaming uses batch number).
    stage_id    Execution stage label (``"streaming"`` for streaming batches).
    partition_id
                Partition index (0 for streaming).
    rows_in     Rows consumed from input (populated on success/failure).
    rows_out    Rows produced in output (populated on success/failure).
    duration_ms Wall-clock milliseconds for the event (0 on ``"started"``).
    error       Exception instance if status is ``"failed"``, else ``None``.
    extra       Arbitrary key-value context attached by the emitting site.
    """

    kind: EventKind
    status: EventStatus
    job_id: str
    stage_id: str
    partition_id: int = 0
    rows_in: int = 0
    rows_out: int = 0
    duration_ms: float = 0.0
    error: Optional[Exception] = field(default=None, repr=False)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "job_id": self.job_id,
            "stage_id": self.stage_id,
            "partition_id": self.partition_id,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duration_ms": round(self.duration_ms, 3),
            "error": str(self.error) if self.error else None,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# StateObserver
# ---------------------------------------------------------------------------


class StateObserver:
    """Singleton event bus that tracks job state for batch and streaming.

    Subscribers receive every ``JobEvent`` synchronously in the calling thread.
    The observer also maintains running aggregate counters accessible via
    ``summary()``.
    """

    _instance: Optional[StateObserver] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: List[Callable[[JobEvent], None]] = []
        self._sub_lock = threading.Lock()

        # Aggregate counters
        self._total_tasks: int = 0
        self._failed_tasks: int = 0
        self._total_batches: int = 0
        self._failed_batches: int = 0
        self._total_rows_in: int = 0
        self._total_rows_out: int = 0
        self._total_duration_ms: float = 0.0
        self._counter_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> StateObserver:
        """Return the process-wide singleton observer."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = StateObserver()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (mainly for tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, handler: Callable[[JobEvent], None]) -> None:
        """Register *handler* to receive every ``JobEvent``.

        Handlers are called synchronously; keep them fast.
        """
        with self._sub_lock:
            self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[JobEvent], None]) -> None:
        with self._sub_lock:
            self._subscribers = [h for h in self._subscribers if h is not handler]

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: JobEvent) -> None:
        """Publish *event* to all subscribers and update aggregate counters."""
        self._update_counters(event)
        self._log_event(event)
        with self._sub_lock:
            handlers = list(self._subscribers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                _log.warning("subscriber raised an exception — skipping")

    # ------------------------------------------------------------------
    # Context managers — primary integration points
    # ------------------------------------------------------------------

    @contextmanager
    def observe_task(
        self,
        task_id: str,
        stage_id: str,
        partition_id: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Context manager that wraps one executor task (batch mode).

        Yields a mutable ``ctx`` dict; callers should populate:
            - ``ctx["rows_in"]``  — rows fed into the task
            - ``ctx["rows_out"]`` — rows produced by the task

        Example::

            with observer.observe_task(task.task_id, task.stage_id, task.partition_id) as ctx:
                result = executor.run(task)
                ctx["rows_in"]  = task.input_row_count
                ctx["rows_out"] = len(result)
        """
        ctx: Dict[str, Any] = {"rows_in": 0, "rows_out": 0}
        self.emit(
            JobEvent(
                kind=EventKind.TASK,
                status=EventStatus.STARTED,
                job_id=task_id,
                stage_id=stage_id,
                partition_id=partition_id,
                extra=extra or {},
            )
        )
        t0 = time.monotonic()
        try:
            yield ctx
            duration_ms = (time.monotonic() - t0) * 1000
            self.emit(
                JobEvent(
                    kind=EventKind.TASK,
                    status=EventStatus.SUCCESS,
                    job_id=task_id,
                    stage_id=stage_id,
                    partition_id=partition_id,
                    rows_in=ctx["rows_in"],
                    rows_out=ctx["rows_out"],
                    duration_ms=duration_ms,
                    extra=extra or {},
                )
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self.emit(
                JobEvent(
                    kind=EventKind.TASK,
                    status=EventStatus.FAILED,
                    job_id=task_id,
                    stage_id=stage_id,
                    partition_id=partition_id,
                    rows_in=ctx["rows_in"],
                    rows_out=ctx["rows_out"],
                    duration_ms=duration_ms,
                    error=exc,
                    extra=extra or {},
                )
            )
            raise

    @contextmanager
    def observe_batch(
        self,
        batch_num: int,
        source_label: str = "streaming",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Context manager that wraps one micro-batch cycle (streaming mode).

        Yields a mutable ``ctx`` dict; callers should populate:
            - ``ctx["rows_in"]``  — rows polled from source(s)
            - ``ctx["rows_out"]`` — rows written to sink(s)

        Example::

            with observer.observe_batch(batch_num, source_label="KafkaSource") as ctx:
                table = poll_sources()
                ctx["rows_in"] = len(table)
                result = run_pipeline(table)
                ctx["rows_out"] = len(result)
        """
        ctx: Dict[str, Any] = {"rows_in": 0, "rows_out": 0}
        batch_id = f"{source_label}#batch-{batch_num}"
        self.emit(
            JobEvent(
                kind=EventKind.BATCH,
                status=EventStatus.STARTED,
                job_id=batch_id,
                stage_id=source_label,
                extra=extra or {},
            )
        )
        t0 = time.monotonic()
        try:
            yield ctx
            duration_ms = (time.monotonic() - t0) * 1000
            self.emit(
                JobEvent(
                    kind=EventKind.BATCH,
                    status=EventStatus.SUCCESS,
                    job_id=batch_id,
                    stage_id=source_label,
                    rows_in=ctx["rows_in"],
                    rows_out=ctx["rows_out"],
                    duration_ms=duration_ms,
                    extra=extra or {},
                )
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self.emit(
                JobEvent(
                    kind=EventKind.BATCH,
                    status=EventStatus.FAILED,
                    job_id=batch_id,
                    stage_id=source_label,
                    rows_in=ctx["rows_in"],
                    rows_out=ctx["rows_out"],
                    duration_ms=duration_ms,
                    error=exc,
                    extra=extra or {},
                )
            )
            raise

    # ------------------------------------------------------------------
    # Summary / export
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of aggregate counters."""
        with self._counter_lock:
            return {
                "total_tasks": self._total_tasks,
                "failed_tasks": self._failed_tasks,
                "total_batches": self._total_batches,
                "failed_batches": self._failed_batches,
                "total_rows_in": self._total_rows_in,
                "total_rows_out": self._total_rows_out,
                "total_duration_ms": round(self._total_duration_ms, 3),
            }

    def clear_counters(self) -> None:
        """Zero out aggregate counters without removing subscribers."""
        with self._counter_lock:
            self._total_tasks = 0
            self._failed_tasks = 0
            self._total_batches = 0
            self._failed_batches = 0
            self._total_rows_in = 0
            self._total_rows_out = 0
            self._total_duration_ms = 0.0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_counters(self, event: JobEvent) -> None:
        if event.status == EventStatus.STARTED:
            return
        with self._counter_lock:
            if event.kind == EventKind.TASK:
                self._total_tasks += 1
                if event.status == EventStatus.FAILED:
                    self._failed_tasks += 1
            else:
                self._total_batches += 1
                if event.status == EventStatus.FAILED:
                    self._failed_batches += 1
            self._total_rows_in += event.rows_in
            self._total_rows_out += event.rows_out
            self._total_duration_ms += event.duration_ms

    def _log_event(self, event: JobEvent) -> None:
        base = dict(
            job_id=event.job_id,
            stage=event.stage_id,
            partition=event.partition_id,
            rows_in=event.rows_in,
            rows_out=event.rows_out,
            duration_ms=round(event.duration_ms, 1),
        )
        if event.status == EventStatus.STARTED:
            _log.debug(f"[{event.kind.value}] started  {event.job_id}", **base)
        elif event.status == EventStatus.SUCCESS:
            _log.info(f"[{event.kind.value}] success  {event.job_id}", **base)
        else:
            tb = (
                "".join(traceback.format_exception(type(event.error), event.error, event.error.__traceback__))
                if event.error
                else ""
            )
            _log.error(
                f"[{event.kind.value}] FAILED   {event.job_id} — {event.error}",
                **base,
                traceback=tb,
            )
