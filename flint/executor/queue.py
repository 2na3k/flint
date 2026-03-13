"""Thread-safe task queue for the Flint executor."""

from __future__ import annotations

import queue
from typing import Optional

from flint.executor.task import Task


class TaskQueue:
    """A thread-safe FIFO queue for ``Task`` objects.

    Wraps ``queue.Queue`` so that producers (Driver) and consumers (Scheduler)
    can operate concurrently without locking.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Task] = queue.Queue()

    def put(self, task: Task) -> None:
        """Enqueue a task."""
        self._queue.put(task)

    def get(self, timeout: Optional[float] = None) -> Task:
        """Dequeue and return the next task.  Blocks until one is available."""
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        """Signal that the last dequeued task has been processed."""
        self._queue.task_done()

    def join(self) -> None:
        """Block until all enqueued tasks have been processed."""
        self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
