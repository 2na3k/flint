"""Unit tests for the Planner (stage splitting) and Optimizer."""

from __future__ import annotations

import pyarrow as pa
import pytest

from flint.planner.node import (
    EvenPartitionSpec,
    FilterNode,
    HashPartitionSpec,
    JoinNode,
    LimitNode,
    MapNode,
    ReadArrow,
    RepartitionNode,
    SelectNode,
)
from flint.planner.optimizer import (
    FilterFusion,
    LimitPushdown,
    Optimizer,
    PredicatePushdown,
    SelectFusion,
)
from flint.planner.planner import Planner
from flint.session import Session


@pytest.fixture()
def session(tmp_path):
    return Session(local=True, temp_dir=str(tmp_path))


def make_source(n_partitions: int = 4) -> ReadArrow:
    table = pa.table({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    return ReadArrow(table=table, n_partitions=n_partitions)


# ---------------------------------------------------------------------------
# Planner: stage splitting
# ---------------------------------------------------------------------------


def test_linear_pipeline_is_single_stage():
    """Source + transforms with no shuffle should produce one stage."""
    source = make_source(4)
    filter_node = FilterNode(children=[source], predicate="a > 0", is_sql=True)
    select_node = SelectNode(children=[filter_node], columns=["a"])

    plan = Planner().build(select_node)
    assert len(plan.stages) == 1
    stage = plan.stages[0]
    assert stage.n_partitions == 4


def test_repartition_creates_two_stages():
    """Source → Repartition should produce two stages."""
    source = make_source(4)
    repart = RepartitionNode(
        children=[source],
        partition_spec=EvenPartitionSpec(n_partitions=2),
    )
    plan = Planner().build(repart)
    assert len(plan.stages) == 2
    assert plan.stages[-1].n_partitions == 2


def test_repartition_with_transform_after_creates_three_stages():
    """Source → Repartition → Map should produce three stages."""
    source = make_source(4)
    repart = RepartitionNode(
        children=[source],
        partition_spec=EvenPartitionSpec(n_partitions=2),
    )
    map_node = MapNode(children=[repart], fn=lambda r: r)

    plan = Planner().build(map_node)
    assert len(plan.stages) == 3


def test_join_creates_five_stages():
    """Left source + Right source + left shuffle + right shuffle + join = 5 stages."""
    left_source = make_source(4)
    right_source = make_source(4)
    join_node = JoinNode(
        children=[left_source, right_source],
        left_keys=["a"],
        right_keys=["a"],
        how="inner",
        partition_spec=HashPartitionSpec(keys=["a"], n_partitions=4),
    )
    plan = Planner().build(join_node)
    assert len(plan.stages) == 5


def test_output_stage_id_is_last_stage(session):
    source = make_source(2)
    filter_node = FilterNode(children=[source], predicate="a > 0", is_sql=True)
    plan = Planner().build(filter_node)
    assert plan.output_stage_id == plan.stages[-1].stage_id


def test_stage_dependencies_are_set():
    source = make_source(4)
    repart = RepartitionNode(
        children=[source],
        partition_spec=EvenPartitionSpec(n_partitions=2),
    )
    plan = Planner().build(repart)
    source_stage, shuffle_stage = plan.stages
    assert source_stage.stage_id in shuffle_stage.depends_on


# ---------------------------------------------------------------------------
# Optimizer rules
# ---------------------------------------------------------------------------


def test_filter_fusion():
    """Two consecutive SQL FilterNodes should be merged into one."""
    source = make_source()
    f1 = FilterNode(children=[source], predicate="a > 0", is_sql=True)
    f2 = FilterNode(children=[f1], predicate="b < 10", is_sql=True)

    rule = FilterFusion()
    result = rule.apply(f2)
    assert isinstance(result, FilterNode)
    assert "(a > 0)" in result.predicate
    assert "(b < 10)" in result.predicate


def test_filter_fusion_skips_python_filters():
    """Python callable filters should NOT be fused."""
    source = make_source()
    f1 = FilterNode(children=[source], predicate=lambda r: r["a"] > 0, is_sql=False)
    f2 = FilterNode(children=[f1], predicate="b < 10", is_sql=True)

    rule = FilterFusion()
    result = rule.apply(f2)
    # f2 is SQL but child (f1) is Python — no fusion should happen
    assert result is f2


def test_predicate_pushdown():
    """SQL FilterNode above SelectNode should be pushed below it."""
    source = make_source()
    sel = SelectNode(children=[source], columns=["a", "b"])
    filt = FilterNode(children=[sel], predicate="a > 0", is_sql=True)

    rule = PredicatePushdown()
    result = rule.apply(filt)
    assert isinstance(result, SelectNode)
    assert isinstance(result.children[0], FilterNode)


def test_select_fusion():
    """Two consecutive SelectNodes should collapse into the outer one."""
    source = make_source()
    sel1 = SelectNode(children=[source], columns=["a", "b"])
    sel2 = SelectNode(children=[sel1], columns=["a"])

    rule = SelectFusion()
    result = rule.apply(sel2)
    assert isinstance(result, SelectNode)
    assert result.columns == ["a"]
    assert result.children[0] is source


def test_limit_pushdown():
    """LimitNode above FilterNode should be pushed below it."""
    source = make_source()
    filt = FilterNode(children=[source], predicate="a > 0", is_sql=True)
    lim = LimitNode(children=[filt], limit=5)

    rule = LimitPushdown()
    result = rule.apply(lim)
    assert isinstance(result, FilterNode)
    assert isinstance(result.children[0], LimitNode)
    assert result.children[0].limit == 5


def test_optimizer_runs_multiple_rules():
    """Optimizer should apply all rules in a multi-pass fixed-point loop.

    Plan: SelectNode → FilterNode(b<10) → FilterNode(a>0) → source

    After FilterFusion: the two consecutive SQL filters merge into one.
    After ProjectionPushdown: SelectNode is pushed below the merged FilterNode.
    Final shape: FilterNode(merged) → SelectNode → source
    """
    source = make_source()
    f1 = FilterNode(children=[source], predicate="a > 0", is_sql=True)
    f2 = FilterNode(children=[f1], predicate="b < 10", is_sql=True)
    sel = SelectNode(children=[f2], columns=["a", "b"])

    optimizer = Optimizer()
    result = optimizer.optimize(sel)
    # ProjectionPushdown moves SelectNode below the (fused) FilterNode
    assert isinstance(result, FilterNode)
    assert isinstance(result.children[0], SelectNode)
