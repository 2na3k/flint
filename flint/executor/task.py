"""Task definition and lifecycle for the Flint executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    pass


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    """A single unit of work: apply a pipeline of nodes to one input partition.

    Fields
    ------
    task_id        Unique identifier.
    stage_id       The ExecutionStage this task belongs to.
    partition_id   Which output partition this task produces.
    pipeline       Ordered list of Node objects to apply in sequence.
    input_datasets One Dataset per input partition (usually one, two for joins).
    temp_dir       Directory for writing intermediate Parquet files.
    output_dataset Populated by the Executor after completion.
    status         Lifecycle state.
    error          Exception captured on failure.
    """

    task_id: str
    stage_id: str
    partition_id: int
    pipeline: List[Any] = field(default_factory=list)  # List[Node]
    input_datasets: List[Any] = field(default_factory=list)  # List[Dataset]
    temp_dir: str = ""
    output_dataset: Optional[Any] = None  # Optional[Dataset]
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[Exception] = field(default=None, compare=False, repr=False)
