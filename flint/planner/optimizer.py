"""Logical plan optimizer — rewrites the node DAG before execution."""

from __future__ import annotations

from typing import List

from typing import Optional

from flint.planner.node import (
    FilterNode,
    LimitNode,
    Node,
    ReadCsv,
    ReadParquet,
    SelectNode,
)


# ---------------------------------------------------------------------------
# Optimization rule base
# ---------------------------------------------------------------------------


class OptimizationRule:
    """Visitor-style rule that rewrites a single node (and its subtree)."""

    def apply(self, node: Node) -> Node:
        """Return a (possibly rewritten) node.  Default: no change."""
        return node


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


class FilterFusion(OptimizationRule):
    """Merge consecutive SQL FilterNodes into a single node.

    Before:  FilterNode("a > 0") → FilterNode("b < 10")
    After:   FilterNode("(a > 0) AND (b < 10)")
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, FilterNode) or not node.is_sql:
            return node
        if not node.children:
            return node

        child = node.children[0]
        if isinstance(child, FilterNode) and child.is_sql:
            merged_pred = f"({child.predicate}) AND ({node.predicate})"
            return FilterNode(
                children=child.children,
                predicate=merged_pred,
                is_sql=True,
            )
        return node


class SelectFusion(OptimizationRule):
    """Merge consecutive SelectNodes — keep only the outermost projection.

    Before:  SelectNode(["a","b","c"]) → SelectNode(["a","b"])
    After:   SelectNode(["a","b"])
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, SelectNode):
            return node
        if not node.children:
            return node

        child = node.children[0]
        if isinstance(child, SelectNode):
            # Outer projection wins; inner columns are irrelevant if outer is subset
            return SelectNode(children=child.children, columns=node.columns)
        return node


class PredicatePushdown(OptimizationRule):
    """Push SQL FilterNodes down past SelectNodes.

    Before:  SelectNode(cols) → FilterNode(pred, is_sql=True)
    After:   FilterNode(pred, is_sql=True) → SelectNode(cols)

    This brings the filter closer to the source so fewer rows pass through
    the projection step.
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, FilterNode) or not node.is_sql:
            return node
        if not node.children:
            return node

        child = node.children[0]
        if isinstance(child, SelectNode):
            # Swap: filter goes below select
            new_filter = FilterNode(
                children=child.children,
                predicate=node.predicate,
                is_sql=True,
            )
            return SelectNode(children=[new_filter], columns=child.columns)
        return node


class ProjectionPushdown(OptimizationRule):
    """Push SelectNodes down as far as possible (toward source nodes).

    Currently handles the case where a SelectNode sits above a FilterNode —
    moves the select below the filter so only needed columns travel upward.

    Before:  SelectNode(cols) → FilterNode → ...
    After:   FilterNode → SelectNode(cols) → ...
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, SelectNode):
            return node
        if not node.children:
            return node

        child = node.children[0]
        # Only push past non-Python filters (SQL filters don't change column set)
        if isinstance(child, FilterNode) and child.is_sql:
            new_select = SelectNode(children=child.children, columns=node.columns)
            return FilterNode(
                children=[new_select],
                predicate=child.predicate,
                is_sql=True,
            )
        return node


class LimitPushdown(OptimizationRule):
    """Push LimitNodes down past FilterNodes.

    Applying the limit earlier reduces the number of rows that subsequent
    operations need to process.  Only safe when the order is irrelevant (which
    is always true in Flint — we make no ordering guarantees).

    Before:  LimitNode(n) → FilterNode → ...
    After:   FilterNode → LimitNode(n) → ...
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, LimitNode):
            return node
        if not node.children:
            return node

        child = node.children[0]
        if isinstance(child, FilterNode):
            new_limit = LimitNode(children=child.children, limit=node.limit)
            return FilterNode(
                children=[new_limit],
                predicate=child.predicate,
                is_sql=child.is_sql,
            )
        return node


class PartitionPruning(OptimizationRule):
    """Skip Hive partition files that cannot match a SQL filter predicate.

    Before:
        FilterNode("year = 2024") → ReadParquet(files=[yr=2023/*, yr=2024/*])
    After:
        FilterNode("year = 2024") → ReadParquet(files=[yr=2024/*])

    The FilterNode is kept — it still handles row-level filtering for
    non-partition columns.  Only files from non-matching partitions are dropped.
    """

    def apply(self, node: Node) -> Node:
        if not isinstance(node, FilterNode) or not node.is_sql:
            return node
        if not node.children:
            return node

        child = node.children[0]
        if not isinstance(child, (ReadParquet, ReadCsv)):
            return node
        if not child.hive_partitioning or not child.partition_columns:
            return node

        from flint.io.fs import eval_partition_filter

        kept = eval_partition_filter(
            child.partition_values, child.partition_columns, str(node.predicate)
        )

        if len(kept) == len(child.partition_values):
            return node  # nothing to prune

        import copy

        new_child = copy.copy(child)
        new_child.paths = [child.paths[i] for i in kept]
        new_child.partition_values = [child.partition_values[i] for i in kept]
        new_child.n_partitions = max(1, len(new_child.paths))

        return FilterNode(
            children=[new_child],
            predicate=node.predicate,
            is_sql=True,
        )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

_DEFAULT_RULES: List[OptimizationRule] = [
    PartitionPruning(),  # run first — prunes files before other rewrites
    FilterFusion(),
    PredicatePushdown(),
    ProjectionPushdown(),
    LimitPushdown(),
    SelectFusion(),
]


class Optimizer:
    """Applies a sequence of optimisation rules to the logical plan DAG.

    Rules are applied bottom-up (children first, then parent) in multiple
    passes until the plan stabilises (fixed-point iteration, capped at 10).
    """

    def __init__(self, rules: Optional[List[OptimizationRule]] = None) -> None:
        self.rules = rules if rules is not None else _DEFAULT_RULES

    def optimize(self, root: Node) -> Node:
        for _ in range(10):
            new_root = self._rewrite(root)
            if self._equal_structure(new_root, root):
                break
            root = new_root
        return root

    def _rewrite(self, node: Node) -> Node:
        # Rewrite children first (bottom-up)
        new_children = [self._rewrite(c) for c in node.children]
        node = self._replace_children(node, new_children)

        # Apply all rules in sequence
        for rule in self.rules:
            node = rule.apply(node)
        return node

    @staticmethod
    def _replace_children(node: Node, new_children: List[Node]) -> Node:
        """Return a shallow copy of *node* with updated children."""
        import copy

        new_node = copy.copy(node)
        new_node.children = new_children
        return new_node

    @staticmethod
    def _equal_structure(a: Node, b: Node) -> bool:
        """Compare two node trees by node_id (identity-based equality)."""
        if a.node_id != b.node_id:
            return False
        if len(a.children) != len(b.children):
            return False
        return all(
            Optimizer._equal_structure(ca, cb) for ca, cb in zip(a.children, b.children)
        )
