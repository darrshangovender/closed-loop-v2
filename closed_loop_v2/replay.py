"""The fleet replay (docs/adr/0001-design.md, section 7; 0002, section 4).

``k`` engines run at once. Each slot holds one life (a trajectory and its shared estimates). The
schedule, which ``replay`` follows exactly:

1. At the start, slots ``0..k-1`` take the first ``k`` lives of ``order``.
2. On each global step the slots are taken in order ``0..k-1``. Each occupied slot processes its
   life's next row ``i``, with bounds from ``method.quantiles()`` as they stand at the start of the
   step. In the closed loop the life ends at ``tau = i`` if ``lo_i <= h``; otherwise, at its last
   row, it ends in a failure. The replacement test comes first, so a rule that fires on the last row
   counts as a replacement there.
3. At the end of the step every life that ended is resolved in slot order:
   ``method.observe_life(life_labels(...))`` is called, then that slot takes the next life of
   ``order``, which processes its row 0 on the next step. A slot with nothing left stays empty.
4. The run ends when every slot is empty.

**Speed.** A method's quantiles change only when a life is resolved, so ``replay`` does not walk the
schedule one step at a time. From the current quantiles it finds, for every occupied slot, the row
at which its life would end if nothing changed (one vectorised pass over the remaining rows), jumps
to the earliest such step, writes the bounds of every row processed up to it, resolves the lives
that end there in slot order, and re-reads the quantiles. A slot's plan is recomputed only when the
quantiles differ from those it was made with. The result is the step-by-step schedule above, which
``tests/test_replay.py`` checks against a literal step-by-step loop. Two assumptions make it exact:
``quantiles()`` has no side effects (it is read once per resolution, not once per step), and the
rule fires row by row (``BoundRule``), so the first trigger in a run of rows is the first row whose
lower bound is at most ``h``.

``replay_static`` is the path for a method whose quantiles never change (V1 and VW). Then each
life's outcome does not depend on ``k``, on ``order`` or on the other lives, so it is computed
once per life. A test shows it gives the same lives as ``replay`` for every ``k`` and order.

**Infinite bounds.** A quantile of ``-inf`` or ``+inf`` (ACI with ``alpha_t`` outside ``(0, 2)``,
or a calibration set too small for a finite quantile) gives an infinite bound at every row, since
``s > 0``. A lower bound of ``-inf`` fires the bound rule at once; one of ``+inf`` does not fire it.
The replay keeps infinite bounds as they are; ``metrics`` counts them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from closed_loop_v2.conformal import (
    Method,
    Quantiles,
    bounds,
    cap_of,
    interval_ends,
    life_labels,
)
from closed_loop_v2.estimator import Prediction
from closed_loop_v2.policy import BoundRule
from closed_loop_v2.types import Trajectory

__all__ = [
    "End",
    "LifeRun",
    "Lives",
    "ReplayRun",
    "check_lives",
    "orders",
    "replay",
    "replay_static",
]

End = Literal["replaced", "failed"]
Lives = Sequence[tuple[Trajectory, Prediction]]


@dataclass(frozen=True, eq=False)
class LifeRun:
    """One life in a replay: the rows it contributed and the bounds used at each.

    ``tau`` is the last row contributed: the replacement row, or the last row after a failure.
    ``lo`` and ``hi`` have ``tau + 1`` entries. ``slot``, ``start_step`` and ``end_step`` describe
    where the life ran in the schedule; ``replay_static`` leaves them at ``-1``.
    """

    end: End
    tau: int
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    slot: int = -1
    start_step: int = -1
    end_step: int = -1

    @property
    def rows(self) -> int:
        return self.tau + 1

    def same_outcome(self, other: LifeRun) -> bool:
        """Equal ends, ``tau`` and bounds, bit for bit; the schedule fields are not compared."""
        return (
            self.end == other.end
            and self.tau == other.tau
            and np.array_equal(self.lo, other.lo)
            and np.array_equal(self.hi, other.hi)
        )


@dataclass(frozen=True, eq=False)
class ReplayRun:
    """Every life of one replay, by life index, and the order in which lives were resolved.

    ``resolved`` lists life indices in the order ``observe_life`` was called (empty for
    ``replay_static``); ``n_steps`` is the number of global steps the schedule took.
    """

    lives: tuple[LifeRun, ...]
    resolved: tuple[int, ...] = ()
    n_steps: int = 0
    quantile_changes: int = field(default=0)

    def same_lives(self, other: ReplayRun) -> bool:
        """The per-life outcomes are equal; scheduling details are ignored."""
        return len(self.lives) == len(other.lives) and all(
            a.same_outcome(b) for a, b in zip(self.lives, other.lives, strict=True)
        )


def check_lives(lives: Lives) -> None:
    """Every prediction has one entry per trajectory row."""
    for traj, pred in lives:
        if pred.n != traj.n:
            raise ValueError(f"{traj.key}: {pred.n} predictions for {traj.n} rows")


def orders(n: int, r: int, seed: int) -> NDArray[np.intp]:
    """``r`` permutations of ``range(n)``; row ``j`` from ``default_rng([seed, j])``."""
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    out = np.empty((r, n), dtype=np.intp)
    for j in range(r):
        out[j] = np.random.default_rng([seed, j]).permutation(n)
    return out


def _check_order(order: Sequence[int] | NDArray[np.intp], n: int) -> list[int]:
    seq = [int(i) for i in order]
    if sorted(seq) != list(range(n)):
        raise ValueError(f"order must be a permutation of range({n})")
    return seq


def _first_end(lo: NDArray[np.float64], rule: BoundRule | None, last: int) -> tuple[int, End]:
    """Offset of the row where a life ends, from bounds that start at its current row."""
    if rule is not None:
        hit = rule.first_trigger(lo)
        if hit is not None:
            return hit, "replaced"
    return last, "failed"


def replay_static(lives: Lives, quantiles: Quantiles, rule: BoundRule | None) -> ReplayRun:
    """The replay of a method whose quantiles never change: every life on its own."""
    check_lives(lives)
    out = []
    for traj, pred in lives:
        lo, hi = bounds(pred, quantiles, cap_of(traj))
        tau, end = _first_end(lo, rule, traj.n - 1)
        out.append(LifeRun(end, tau, lo[: tau + 1].copy(), hi[: tau + 1].copy()))
    return ReplayRun(tuple(out))


@dataclass(eq=False)
class _Slot:
    life: int
    cur: int  # the row this slot processes at the next step
    start_step: int
    lo: NDArray[np.float64]  # full length, NaN until a row is processed
    hi: NDArray[np.float64]
    plan_q: Quantiles | None = None  # the quantiles the plan below was made with
    offset: int = 0  # steps from now to the row at which the life ends under plan_q
    end: End = "failed"


def replay(
    lives: Lives,
    method: Method,
    rule: BoundRule | None,
    *,
    k: int,
    order: Sequence[int] | NDArray[np.intp],
) -> ReplayRun:
    """Run the fleet schedule of the module docstring and return every life's rows and bounds.

    ``rule=None`` is the open loop: no life is replaced, every life ends in a failure at its last
    row, and the method sees every label. ``method`` is updated in place.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    check_lives(lives)
    n_lives = len(lives)
    queue = _check_order(order, n_lives)
    ms = [pred.m for _, pred in lives]
    ss = [pred.s for _, pred in lives]
    ns = [traj.n for traj, _ in lives]
    caps = [cap_of(traj) for traj, _ in lives]
    done: list[LifeRun | None] = [None] * n_lives
    resolved: list[int] = []
    slots: list[_Slot | None] = [None] * k
    nxt = 0
    step = 0

    def start(slot: int, at: int) -> None:
        nonlocal nxt
        if nxt < n_lives:
            life = queue[nxt]
            nxt += 1
            n = ns[life]
            slots[slot] = _Slot(life, 0, at, np.full(n, np.nan), np.full(n, np.nan))
        else:
            slots[slot] = None

    for j in range(k):
        start(j, 0)

    q = method.quantiles()
    changes = 0
    while True:
        active = [(j, s) for j, s in enumerate(slots) if s is not None]
        if not active:
            break
        for _, s in active:
            if s.plan_q != q:
                m, sd = ms[s.life], ss[s.life]
                lo_rest = interval_ends(m[s.cur :], sd[s.cur :], q, caps[s.life])[0]
                s.offset, s.end = _first_end(lo_rest, rule, ns[s.life] - 1 - s.cur)
                s.plan_q = q
        jump = min(s.offset for _, s in active)
        ended: list[int] = []
        for j, s in active:
            rows = slice(s.cur, s.cur + jump + 1)
            m, sd = ms[s.life], ss[s.life]
            s.lo[rows], s.hi[rows] = interval_ends(m[rows], sd[rows], q, caps[s.life])
            if s.offset == jump:
                ended.append(j)
            else:
                s.cur += jump + 1
                s.offset -= jump + 1
        step += jump  # the step at which the earliest lives end
        for j in ended:  # slot order
            s_end = slots[j]
            assert s_end is not None
            s = s_end
            tau = s.cur + jump
            traj, pred = lives[s.life]
            labels = life_labels(
                traj.rul(),
                pred,
                s.lo,
                s.hi,
                replaced_at=tau if s.end == "replaced" else None,
                cap=math.inf if traj.rul_cap is None else float(traj.rul_cap),
            )
            method.observe_life(labels)
            done[s.life] = LifeRun(
                s.end,
                tau,
                s.lo[: tau + 1].copy(),
                s.hi[: tau + 1].copy(),
                slot=j,
                start_step=s.start_step,
                end_step=step,
            )
            resolved.append(s.life)
            start(j, step + 1)
        step += 1
        new_q = method.quantiles()
        if new_q != q:
            changes += 1
        q = new_q

    out = tuple(d for d in done if d is not None)
    if len(out) != n_lives:  # pragma: no cover - the schedule resolves every life
        raise RuntimeError("the replay did not resolve every life")
    return ReplayRun(out, tuple(resolved), step, changes)
