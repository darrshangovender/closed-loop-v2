"""Per-life coverage, the stratum, unit means over orders, Gap_B and policy outcomes, on hand-made
bounds whose answers are worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from closed_loop_v2.estimator import Prediction
from closed_loop_v2.metrics import (
    UnitTable,
    gap_b,
    life_metrics,
    pair_loops,
    policy_outcomes,
    unit_coverage,
)
from closed_loop_v2.replay import LifeRun, ReplayRun
from closed_loop_v2.types import Trajectory

INF = math.inf


def life(m: list[float]) -> tuple[Trajectory, Prediction]:
    n = len(m)
    t = Trajectory("FD002", 1, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 0)
    return t, Prediction(np.asarray(m, dtype=np.float64), np.ones(n))


def run_of(*lives: LifeRun) -> ReplayRun:
    return ReplayRun(tuple(lives))


def lr(end: str, lo: list[float], hi: list[float]) -> LifeRun:
    return LifeRun(end, len(lo) - 1, np.asarray(lo, float), np.asarray(hi, float))  # type: ignore[arg-type]


# life A: 4 rows, true remaining life 3, 2, 1, 0; m 10, 6, 4, 1 (stratum M = 5: rows 2 and 3)
A = life([10.0, 6.0, 4.0, 1.0])
A_OPEN = lr("failed", [2.0, 2.5, -INF, 0.0], [4.0, 3.0, 1.0, INF])
# life B: 3 rows, remaining life 2, 1, 0; m 9, 9, 0 (stratum: row 2 only); replaced at row 1
B = life([9.0, 9.0, 0.0])
B_CLOSED = lr("replaced", [1.0, 1.5], [3.0, 2.0])
B_OPEN = lr("failed", [1.0, 1.5, 0.5], [3.0, 2.0, 0.5])


def test_life_metrics_by_hand() -> None:
    got = life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)
    # A: rul >= lo at rows 0, 2 (-inf covers), 3; lo <= rul <= hi at rows 0, 2, 3 (+inf covers)
    assert got.cover_lo[0] == 0.75 and got.cover[0] == 0.75
    assert got.stratum_rows[0] == 2
    assert got.stratum_cover[0] == 1.0 and got.stratum_cover_lo[0] == 1.0
    assert not got.replaced[0] and got.tau[0] == 3 and got.rul_at_tau[0] == 0.0
    assert got.mean_width[0] == pytest.approx((2.0 + 0.5) / 2)
    assert got.finite_rows[0] == 2 and got.infinite_rows[0] == 2
    # B: rows 0..1 contributed; rul 2, 1 against [1, 3], [1.5, 2]
    assert got.rows[1] == 2
    assert got.cover_lo[1] == 0.5 and got.cover[1] == 0.5
    assert got.stratum_rows[1] == 0 and math.isnan(got.stratum_cover_lo[1])
    assert got.replaced[1] and got.tau[1] == 1 and got.rul_at_tau[1] == 1.0
    assert got.mean_width[1] == pytest.approx(1.25) and got.infinite_rows[1] == 0


def test_an_upper_side_below_a_finite_lower_side_counts_as_infinite_width_only_when_infinite() -> (
    None
):
    lone = life([3.0])
    got = life_metrics([lone], run_of(lr("failed", [INF], [-INF])), stratum_m=10.0)
    assert got.cover_lo[0] == 0.0 and got.cover[0] == 0.0  # +inf lower bound covers nothing
    assert got.infinite_rows[0] == 1 and math.isnan(got.mean_width[0])


def test_unit_coverage_over_orders_and_the_every_order_stratum_rule() -> None:
    order1 = life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)
    order2 = life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)
    cov = unit_coverage([order1, order2])
    assert cov.n_orders == 2
    assert cov.cover_lo[0] == 0.75
    # B: 0.5 in order 1; in order 2 rul 2, 1, 0 against lo 1, 1.5, 0.5 gives 1/3
    assert cov.cover_lo[1] == pytest.approx((0.5 + 1.0 / 3.0) / 2)
    # B has a stratum row in order 2 only, so it is out of the stratum
    assert list(cov.in_stratum) == [True, False]
    assert cov.stratum_excluded == 1
    assert cov.stratum_cover_lo[0] == 1.0 and math.isnan(cov.stratum_cover_lo[1])


def test_gap_b_pairs_by_unit_and_counts_the_stratum_drop() -> None:
    closed = unit_coverage([life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)])
    open_ = unit_coverage([life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)])
    g = pair_loops(closed, open_)
    np.testing.assert_allclose(g.pooled, [0.0, 0.5 - 1.0 / 3.0])
    # B is in the open-loop stratum (row 2) but not the closed one: dropped
    assert list(g.stratum_units) == [0] and g.stratum_dropped == 1
    np.testing.assert_array_equal(g.stratum, [0.0])
    table = UnitTable(["FD002/1", "FD002/2"])
    table.add("V1", "closed", [life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)])
    table.add("V1", "open", [life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)])
    via_table = gap_b(table, "V1")
    np.testing.assert_array_equal(via_table.pooled, g.pooled)
    assert table.methods() == ["V1"]


def test_unit_table_rejects_a_wrong_size_or_loop() -> None:
    table = UnitTable(["only one"])
    one = life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)
    with pytest.raises(ValueError, match="units"):
        table.add("V1", "open", [one])
    with pytest.raises(ValueError, match="loop"):
        table.add("V1", "sideways", [one])


def test_policy_outcomes_by_hand() -> None:
    order1 = life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)
    order2 = life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)
    p = policy_outcomes([order1, order2])
    assert p.failed_fraction == pytest.approx((0.5 + 1.0) / 2)
    assert p.replacements == pytest.approx(0.5)
    assert p.mean_rul_at_replacement == 1.0
    # widths: A 1.25 in both orders; B 1.25 then (2 + 0.5 + 0) / 3
    assert p.mean_width == pytest.approx((1.25 + (1.25 + 2.5 / 3.0) / 2) / 2)
    assert p.infinite_rows == 2.0 and p.rows == pytest.approx((6 + 7) / 2)
    assert p.n_orders == 2


def test_a_static_replay_stands_for_every_order() -> None:
    one = life_metrics([A, B], run_of(A_OPEN, B_CLOSED), stratum_m=5.0)
    single, repeated = unit_coverage([one]), unit_coverage([one] * 5)
    np.testing.assert_array_equal(single.cover_lo, repeated.cover_lo)
    assert policy_outcomes([one]).replacements == policy_outcomes([one] * 5).replacements


def test_no_replacement_and_no_lives() -> None:
    one = life_metrics([A], run_of(A_OPEN), stratum_m=5.0)
    assert policy_outcomes([one]).mean_rul_at_replacement is None
    empty = life_metrics([], run_of(), stratum_m=5.0)
    assert unit_coverage([empty]).n == 0
    p = policy_outcomes([empty])
    assert p.mean_width is None and p.rows == 0.0


def test_mismatched_inputs_raise() -> None:
    with pytest.raises(ValueError, match="replayed lives"):
        life_metrics([A, B], run_of(A_OPEN), stratum_m=5.0)
    short = LifeRun("failed", 3, np.zeros(2), np.ones(2))  # tau 3 needs 4 bounds
    with pytest.raises(ValueError, match="bounds"):
        life_metrics([A], run_of(short), stratum_m=5.0)
    one = life_metrics([A], run_of(A_OPEN), stratum_m=5.0)
    two = life_metrics([A, B], run_of(A_OPEN, B_OPEN), stratum_m=5.0)
    with pytest.raises(ValueError, match="same lives"):
        unit_coverage([one, two])
    with pytest.raises(ValueError, match="at least one"):
        unit_coverage([])


def test_coverage_uses_the_capped_truth_and_the_cost_uses_the_uncapped_life() -> None:
    n = 6
    t = Trajectory("FD002", 1, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 0)
    lives = [(t.capped(3), Prediction(np.full(n, 3.0), np.ones(n)))]
    # replaced at row 1: truth min(5, 3) = 3 and min(4, 3) = 3; bounds [2.5, 3.5] cover 3
    run = run_of(lr("replaced", [2.5, 2.5], [3.5, 3.5]))
    m = life_metrics(lives, run, stratum_m=10.0)
    assert m.cover[0] == 1.0  # the uncapped 5 and 4 would both be above 3.5
    assert m.rul_at_tau[0] == 4.0  # the remaining life at replacement, uncapped
