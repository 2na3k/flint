"""Planner — converts a logical node DAG into an ExecutionPlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from flint.planner.node import (
    BroadcastNode,
    HashPartitionSpec,
    JoinNode,
    Node,
    ReadNode,
    RepartitionNode,
    ShuffleNode,
)
from flint.planner.optimizer import Optimizer
from flint.utils import generate_id


# ---------------------------------------------------------------------------
# Execution plan data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionStage:
    """A set of tasks that can run in parallel without a shuffle.

    All nodes in ``pipeline`` are applied in sequence on each input partition.
    Stage boundaries occur at ``ShuffleNode`` and ``JoinNode``.
    """

    stage_id: str
    pipeline: List[Node]  # nodes to execute in sequence per partition
    depends_on: List[str]  # stage_ids that must complete first
    n_partitions: int
    # For join stages: which prior stages provide left/right data
    left_stage_id: Optional[str] = None
    right_stage_id: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Topologically ordered list of execution stages."""

    stages: List[ExecutionStage] = field(default_factory=list)
    output_stage_id: str = ""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Converts a logical node DAG into an ``ExecutionPlan``.

    Algorithm
    ---------
    1. Run the ``Optimizer`` over the DAG.
    2. Walk the DAG recursively from the root.
    3. At ``ShuffleNode`` / ``RepartitionNode`` / ``JoinNode`` boundaries,
       start a new ``ExecutionStage``.
    4. Within a stage, consecutive non-boundary nodes are pipelined.
    5. For ``JoinNode``: insert ``ShuffleNode`` on both sides (unless broadcast).
    """

    def __init__(self) -> None:
        self._stages: List[ExecutionStage] = []
        self._optimizer = Optimizer()

    def build(self, root: Node) -> ExecutionPlan:
        """Build an ``ExecutionPlan`` from *root*."""
        self._stages = []
        optimized_root = self._optimizer.optimize(root)
        output_stage_id = self._process(optimized_root)
        return ExecutionPlan(stages=self._stages, output_stage_id=output_stage_id)

    # ------------------------------------------------------------------
    # Recursive processing
    # ------------------------------------------------------------------

    def _process(self, node: Node) -> str:
        """Return the stage_id whose output represents *node*'s result."""
        if isinstance(node, JoinNode):
            return self._process_join(node)
        if isinstance(node, (ShuffleNode, RepartitionNode)):
            return self._process_shuffle(node)
        if isinstance(node, ReadNode):
            return self._process_source(node)
        return self._process_transform(node)

    def _process_source(self, node: ReadNode) -> str:
        """Source nodes start a new stage on their own."""
        stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[node],
            depends_on=[],
            n_partitions=node.n_partitions or 1,
        )
        self._stages.append(stage)
        return stage.stage_id

    def _process_transform(self, node: Node) -> str:
        """Non-boundary transforms are appended to the parent stage when possible."""
        if not node.children:
            # Leaf that isn't a source — treat as single-partition source
            stage = ExecutionStage(
                stage_id=generate_id(),
                pipeline=[node],
                depends_on=[],
                n_partitions=1,
            )
            self._stages.append(stage)
            return stage.stage_id

        child_stage_id = self._process(node.children[0])
        child_stage = self._get_stage(child_stage_id)

        # Append to the child stage if it hasn't been "closed" by a shuffle
        last_node = child_stage.pipeline[-1] if child_stage.pipeline else None
        if last_node is not None and not isinstance(
            last_node, (ShuffleNode, RepartitionNode)
        ):
            child_stage.pipeline.append(node)
            return child_stage_id

        # Otherwise, start a new stage
        stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[node],
            depends_on=[child_stage_id],
            n_partitions=child_stage.n_partitions,
        )
        self._stages.append(stage)
        return stage.stage_id

    def _process_shuffle(self, node: Node) -> str:
        """Shuffle / repartition always creates a new stage (boundary)."""
        child_stage_id = self._process(node.children[0])
        child_stage = self._get_stage(child_stage_id)

        spec = node.partition_spec
        n = (
            spec.n_partitions
            if (spec and spec.n_partitions > 0)
            else child_stage.n_partitions
        )

        # Materialise n into the spec so the executor knows the target count
        if spec and spec.n_partitions == 0:
            import copy

            spec = copy.copy(spec)
            spec.n_partitions = n

        shuffle_node = ShuffleNode(children=[node.children[0]], partition_spec=spec)
        stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[shuffle_node],
            depends_on=[child_stage_id],
            n_partitions=n,
        )
        self._stages.append(stage)
        return stage.stage_id

    def _process_join(self, node: JoinNode) -> str:
        """Join — insert shuffles on both sides, then a local join stage."""
        left_source_stage_id = self._process(node.children[0])
        right_source_stage_id = self._process(node.children[1])

        left_stage = self._get_stage(left_source_stage_id)
        right_stage = self._get_stage(right_source_stage_id)

        if node.broadcast:
            # No shuffle — right side is broadcast by the Driver
            broadcast_node = BroadcastNode(children=[node.children[1]])
            right_broadcast_stage = ExecutionStage(
                stage_id=generate_id(),
                pipeline=[broadcast_node],
                depends_on=[right_source_stage_id],
                n_partitions=right_stage.n_partitions,
            )
            self._stages.append(right_broadcast_stage)

            join_stage = ExecutionStage(
                stage_id=generate_id(),
                pipeline=[node],
                depends_on=[left_source_stage_id, right_broadcast_stage.stage_id],
                n_partitions=left_stage.n_partitions,
                left_stage_id=left_source_stage_id,
                right_stage_id=right_broadcast_stage.stage_id,
            )
            self._stages.append(join_stage)
            return join_stage.stage_id

        # Determine target partition count — use max of both sides if auto
        spec = node.partition_spec
        n = (
            spec.n_partitions
            if (spec and spec.n_partitions > 0)
            else max(left_stage.n_partitions, right_stage.n_partitions)
        )

        # Insert hash shuffle on left side
        left_spec = HashPartitionSpec(keys=node.left_keys, n_partitions=n)
        left_shuffle = ShuffleNode(
            children=[node.children[0]], partition_spec=left_spec
        )
        left_shuffle_stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[left_shuffle],
            depends_on=[left_source_stage_id],
            n_partitions=n,
        )

        # Insert hash shuffle on right side (same n and same hash fn)
        right_spec = HashPartitionSpec(keys=node.right_keys, n_partitions=n)
        right_shuffle = ShuffleNode(
            children=[node.children[1]], partition_spec=right_spec
        )
        right_shuffle_stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[right_shuffle],
            depends_on=[right_source_stage_id],
            n_partitions=n,
        )

        self._stages.extend([left_shuffle_stage, right_shuffle_stage])

        # Local join stage — each partition joins its co-located pair
        join_stage = ExecutionStage(
            stage_id=generate_id(),
            pipeline=[node],
            depends_on=[left_shuffle_stage.stage_id, right_shuffle_stage.stage_id],
            n_partitions=n,
            left_stage_id=left_shuffle_stage.stage_id,
            right_stage_id=right_shuffle_stage.stage_id,
        )
        self._stages.append(join_stage)
        return join_stage.stage_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_stage(self, stage_id: str) -> ExecutionStage:
        for stage in self._stages:
            if stage.stage_id == stage_id:
                return stage
        raise ValueError(f"Stage '{stage_id}' not found in plan")
