"""Coverage, Gap_B and policy outcomes (docs/adr/0001-design.md, section 8; 0002, section 4).

Truth is ``Trajectory.rul()`` at each row: the dataset's remaining life, capped at ``cap`` when the
trajectory is capped (ADR 0001, amendment of 10 November 2026). The remaining life at replacement,
a cost of the policy, uses the uncapped value. For one life in one replay:

* ``cover`` is the mean over its contributed rows of ``lo <= rul <= hi`` (``c_u``), and
  ``cover_lo`` the mean of ``rul >= lo`` (``c_u^lo``). The open loop contributes every row; the
  closed loop the rows up to the replacement.
* The **decision stratum** is the rows whose shared point estimate ``m`` is at most ``M`` cycles.
  ``m`` comes from the estimator, so the stratum is the same set of rows for every method in the
  open loop. ``stratum_cover`` and ``stratum_cover_lo`` are the same means over the contributed
  rows in the stratum, ``NaN`` for a life with none.
* **Infinite bounds.** A side at ``-inf`` (lower) or ``+inf`` (upper) covers every remaining life,
  and the comparisons above count it so. ``mean_width`` is the mean of ``hi - lo`` over the rows
  whose two sides are finite (``NaN`` if there is none), and ``infinite_rows`` counts the rows with
  an infinite side.

``unit_coverage`` turns the replays of one method in one loop (one per order) into per-unit means
over the orders. A unit with no stratum row in an order is missing from the stratum for that order,
and it counts in the stratum only if it has stratum rows in every order; the units left out are
counted. ``gap_b`` pairs the closed and the open loop of one method by unit.

A method whose quantiles never change (V1, VW) gives the same per-life outcome in every order, so
its one ``replay_static`` stands for all ``R`` orders: averaging ``R`` equal copies changes nothing.
Policy outcomes are therefore stated per order (means over orders), so that both kinds of method
are on one scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop_v2.replay import Lives, ReplayRun, check_lives

__all__ = [
    "LOOPS",
    "GapB",
    "LifeMetrics",
    "PolicyOutcomes",
    "UnitCoverage",
    "UnitTable",
    "gap_b",
    "life_metrics",
    "pair_loops",
    "policy_outcomes",
    "unit_coverage",
]


@dataclass(frozen=True, eq=False)
class LifeMetrics:
    """Per-life outcomes of one replay; every array is indexed by life."""

    rows: NDArray[np.int64]
    cover: NDArray[np.float64]
    cover_lo: NDArray[np.float64]
    stratum_rows: NDArray[np.int64]
    stratum_cover: NDArray[np.float64]
    stratum_cover_lo: NDArray[np.float64]
    replaced: NDArray[np.bool_]
    tau: NDArray[np.int64]
    rul_at_tau: NDArray[np.float64]
    mean_width: NDArray[np.float64]
    finite_rows: NDArray[np.int64]
    infinite_rows: NDArray[np.int64]

    @property
    def n(self) -> int:
        return int(self.rows.size)


def life_metrics(lives: Lives, run: ReplayRun, *, stratum_m: float) -> LifeMetrics:
    """The per-life metrics of ``run``, whose lives are ``lives`` in the same order."""
    check_lives(lives)
    if len(run.lives) != len(lives):
        raise ValueError(f"{len(run.lives)} replayed lives for {len(lives)} lives")
    n = len(lives)
    rows = np.zeros(n, dtype=np.int64)
    cover = np.zeros(n)
    cover_lo = np.zeros(n)
    s_rows = np.zeros(n, dtype=np.int64)
    s_cover = np.full(n, np.nan)
    s_cover_lo = np.full(n, np.nan)
    replaced = np.zeros(n, dtype=bool)
    tau = np.zeros(n, dtype=np.int64)
    rul_tau = np.zeros(n)
    width = np.full(n, np.nan)
    finite = np.zeros(n, dtype=np.int64)
    infinite = np.zeros(n, dtype=np.int64)
    for u, ((traj, pred), lr) in enumerate(zip(lives, run.lives, strict=True)):
        end = lr.tau + 1
        rul = traj.rul()[:end].astype(np.float64)
        lo, hi = lr.lo, lr.hi
        if lo.shape != (end,) or hi.shape != (end,):
            raise ValueError(f"life {u}: bounds do not match its {end} contributed rows")
        hit_lo = rul >= lo
        hit = hit_lo & (rul <= hi)
        rows[u] = end
        cover[u] = float(hit.mean())
        cover_lo[u] = float(hit_lo.mean())
        in_s = pred.m[:end] <= stratum_m
        s_rows[u] = int(in_s.sum())
        if s_rows[u] > 0:
            s_cover[u] = float(hit[in_s].mean())
            s_cover_lo[u] = float(hit_lo[in_s].mean())
        replaced[u] = lr.end == "replaced"
        tau[u] = lr.tau
        rul_tau[u] = float(traj.rul_uncapped()[lr.tau])
        both = np.isfinite(lo) & np.isfinite(hi)
        finite[u] = int(both.sum())
        infinite[u] = end - finite[u]
        if finite[u] > 0:
            width[u] = float((hi[both] - lo[both]).mean())
    return LifeMetrics(
        rows,
        cover,
        cover_lo,
        s_rows,
        s_cover,
        s_cover_lo,
        replaced,
        tau,
        rul_tau,
        width,
        finite,
        infinite,
    )


def _same_lives(runs: Sequence[LifeMetrics]) -> int:
    if not runs:
        raise ValueError("need at least one replay")
    n = runs[0].n
    if any(r.n != n for r in runs):
        raise ValueError("every replay must cover the same lives")
    return n


@dataclass(frozen=True, eq=False)
class UnitCoverage:
    """One method in one loop: per-unit means over the orders.

    ``stratum_cover`` and ``stratum_cover_lo`` are ``NaN`` for a unit outside the stratum
    (``in_stratum`` false): a unit with no stratum row in at least one order.
    """

    cover: NDArray[np.float64]
    cover_lo: NDArray[np.float64]
    stratum_cover: NDArray[np.float64]
    stratum_cover_lo: NDArray[np.float64]
    in_stratum: NDArray[np.bool_]
    n_orders: int

    @property
    def n(self) -> int:
        return int(self.cover.size)

    @property
    def stratum_excluded(self) -> int:
        return int((~self.in_stratum).sum())


def unit_coverage(runs: Sequence[LifeMetrics]) -> UnitCoverage:
    """Per-unit means over the orders, one ``LifeMetrics`` per order."""
    n = _same_lives(runs)
    cover = np.mean([r.cover for r in runs], axis=0) if n else np.zeros(0)
    cover_lo = np.mean([r.cover_lo for r in runs], axis=0) if n else np.zeros(0)
    in_s = np.all([r.stratum_rows > 0 for r in runs], axis=0) if n else np.zeros(0, dtype=bool)
    s_cover = np.full(n, np.nan)
    s_cover_lo = np.full(n, np.nan)
    if in_s.any():
        s_cover[in_s] = np.mean([r.stratum_cover[in_s] for r in runs], axis=0)
        s_cover_lo[in_s] = np.mean([r.stratum_cover_lo[in_s] for r in runs], axis=0)
    return UnitCoverage(cover, cover_lo, s_cover, s_cover_lo, in_s, len(runs))


@dataclass(frozen=True, eq=False)
class GapB:
    """``c^lo`` of one method, closed loop against open loop, paired by unit.

    ``pooled_*`` hold every unit. ``stratum_*`` hold the units in the stratum in both loops, listed
    in ``stratum_units``; ``stratum_dropped`` counts the units left out.
    """

    pooled_closed: NDArray[np.float64]
    pooled_open: NDArray[np.float64]
    stratum_closed: NDArray[np.float64]
    stratum_open: NDArray[np.float64]
    stratum_units: NDArray[np.intp]
    stratum_dropped: int

    @property
    def pooled(self) -> NDArray[np.float64]:
        """Per-unit ``Gap_B``, closed minus open, every unit."""
        return self.pooled_closed - self.pooled_open

    @property
    def stratum(self) -> NDArray[np.float64]:
        """Per-unit ``Gap_B`` in the stratum, for the units in ``stratum_units``."""
        return self.stratum_closed - self.stratum_open


Loop = str  # "open" or "closed"
LOOPS: tuple[Loop, Loop] = ("open", "closed")


class UnitTable:
    """Per unit, method and loop: the ``UnitCoverage`` of every cell, over the same units."""

    def __init__(self, units: Sequence[str]) -> None:
        self.units: tuple[str, ...] = tuple(units)
        self._cells: dict[tuple[str, Loop], UnitCoverage] = {}

    def add(self, method: str, loop: Loop, runs: Sequence[LifeMetrics]) -> UnitCoverage:
        """Average ``runs`` (one per order) into the cell ``(method, loop)`` and store it."""
        if loop not in LOOPS:
            raise ValueError(f"loop must be one of {LOOPS}, got {loop!r}")
        cell = unit_coverage(runs)
        if cell.n != len(self.units):
            raise ValueError(f"{cell.n} units in the replays, {len(self.units)} in the table")
        self._cells[(method, loop)] = cell
        return cell

    def __getitem__(self, key: tuple[str, Loop]) -> UnitCoverage:
        return self._cells[key]

    def methods(self) -> list[str]:
        """Methods with a cell, in the order they were added."""
        seen: list[str] = []
        for method, _ in self._cells:
            if method not in seen:
                seen.append(method)
        return seen


def gap_b(table: UnitTable, method: str) -> GapB:
    """``c^lo`` closed against open for ``method``, paired by unit, pooled and in the stratum."""
    return pair_loops(table[(method, "closed")], table[(method, "open")])


def pair_loops(closed: UnitCoverage, open_: UnitCoverage) -> GapB:
    """Pair the closed and the open loop of one method by unit."""
    if closed.n != open_.n:
        raise ValueError("the two loops must cover the same units")
    both = closed.in_stratum & open_.in_stratum
    idx = np.flatnonzero(both).astype(np.intp)
    return GapB(
        closed.cover_lo.copy(),
        open_.cover_lo.copy(),
        closed.stratum_cover_lo[idx],
        open_.stratum_cover_lo[idx],
        idx,
        int(closed.n - idx.size),
    )


@dataclass(frozen=True)
class PolicyOutcomes:
    """Descriptive outcomes of one method in one loop, as means over the orders.

    * ``failed_fraction``: the fraction of lives that failed unreplaced.
    * ``replacements``: lives replaced per order.
    * ``mean_rul_at_replacement``: the remaining life at the replacement row, over every replaced
      life of every order (``None`` with no replacement).
    * ``mean_width``: the mean over units of each unit's mean finite width, the unit's value being
      averaged over the orders in which it has a finite row (``None`` if no unit has one).
    * ``infinite_rows`` and ``rows``: rows with an infinite side, and all rows, per order.
    """

    failed_fraction: float
    replacements: float
    mean_rul_at_replacement: float | None
    mean_width: float | None
    infinite_rows: float
    rows: float
    n_orders: int


def policy_outcomes(runs: Sequence[LifeMetrics]) -> PolicyOutcomes:
    """The descriptive policy outcomes of one method in one loop, one ``LifeMetrics`` per order."""
    n = _same_lives(runs)
    r = len(runs)
    if n == 0:
        return PolicyOutcomes(0.0, 0.0, None, None, 0.0, 0.0, r)
    failed = float(np.mean([float((~x.replaced).mean()) for x in runs]))
    replacements = float(np.mean([float(x.replaced.sum()) for x in runs]))
    ruls = np.concatenate([x.rul_at_tau[x.replaced] for x in runs])
    mean_rul = float(ruls.mean()) if ruls.size else None
    widths = np.array([x.mean_width for x in runs])  # (r, n), NaN where a life has no finite row
    has = np.isfinite(widths)
    per_unit = np.full(n, np.nan)
    any_unit = has.any(axis=0)
    if any_unit.any():
        sums = np.where(has, widths, 0.0).sum(axis=0)
        per_unit[any_unit] = sums[any_unit] / has.sum(axis=0)[any_unit]
    mean_width = float(per_unit[any_unit].mean()) if any_unit.any() else None
    infinite = float(np.mean([float(x.infinite_rows.sum()) for x in runs]))
    rows = float(np.mean([float(x.rows.sum()) for x in runs]))
    return PolicyOutcomes(failed, replacements, mean_rul, mean_width, infinite, rows, r)
