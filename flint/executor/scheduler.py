"""Scheduler — dispatches Tasks to local threads or Ray workers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, List

from flint.executor.task import Task

if TYPE_CHECKING:
    pass


class Scheduler:
    """Submits tasks for execution and collects results.

    Modes
    -----
    local:
        Tasks run in a ``ThreadPoolExecutor`` (same process, parallel threads).
        Useful for testing and single-machine workloads without Ray overhead.
    ray:
        Tasks run as ``ray.remote`` functions on Ray workers.
        Enables true multi-process / multi-machine distribution.
    """

    def __init__(self, local: bool = True, n_workers: int = 4) -> None:
        self.local = local
        self.n_workers = n_workers
        self._executor: ThreadPoolExecutor | None = None

    def start(self) -> None:
        if self.local:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        else:
            import ray  # noqa: F401 — ensure Ray is available

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def submit_batch(self, tasks: List[Task]) -> List[Task]:
        """Submit a batch of tasks and return them with results populated.

        Blocks until all tasks in the batch are complete.
        """
        if self.local:
            return self._submit_local(tasks)
        return self._submit_ray(tasks)

    # ------------------------------------------------------------------
    # Local execution (ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _submit_local(self, tasks: List[Task]) -> List[Task]:
        from flint.executor.executor import Executor

        executor = Executor()
        if self._executor is None:
            self.start()
        assert self._executor is not None

        futures: List[Future] = [self._executor.submit(executor.run, task) for task in tasks]
        for task, future in zip(tasks, futures):
            try:
                task.output_dataset = future.result()
            except Exception as exc:
                task.error = exc
                raise
        return tasks

    # ------------------------------------------------------------------
    # Ray execution
    # ------------------------------------------------------------------

    def _submit_ray(self, tasks: List[Task]) -> List[Task]:
        import ray

        futures = [_ray_execute_task.remote(task) for task in tasks]
        results = ray.get(futures)
        for task, result in zip(tasks, results):
            task.output_dataset = result
        return tasks


# Module-level Ray remote function (must be at module level for pickling)
try:
    import ray

    @ray.remote
    def _ray_execute_task(task: Task) -> object:
        """Execute a Task on a Ray worker and return an in-memory Dataset.

        The executor writes intermediate results to a temp Parquet file on the
        *worker's* local filesystem.  Because the driver cannot access that path,
        we read the table back into memory and return an ``InMemoryDataset`` so
        Ray ships the data through its object store rather than a file pointer.
        """
        import shutil
        import tempfile

        from flint.dataframe import InMemoryDataset
        from flint.executor.executor import Executor

        # Give this worker its own scratch space that it can actually write to
        task.temp_dir = tempfile.mkdtemp(prefix="flint_worker_")
        try:
            result = Executor().run(task)
            table = result.to_arrow()
            return InMemoryDataset(table, task.partition_id)
        finally:
            shutil.rmtree(task.temp_dir, ignore_errors=True)

except ImportError:
    # Ray not installed — remote execution unavailable
    pass
