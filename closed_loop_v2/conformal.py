"""Conformal methods V1, V2L, V3L and VW (docs/adr/0001-design.md, sections 5 to 7).

Every method turns scores ``r = (RUL - m) / s`` into a lower and an upper quantile ``(q_lo, q_hi)``.
The interval at a checkpoint is ``[m + s q_lo, m + s q_hi]``, with both ends limited to the cap of
the truth (``bounds``).

* **V1**, split conformal, takes one uniformly drawn row from each calibration unit. Its two-sided
  coverage is at least ``1 - alpha``, and its lower-bound coverage at least ``1 - alpha / 2``,
  when the calibration units and a new unit are exchangeable, as a marginal probability over a
  random unit and a random row drawn the same way (the split-conformal argument applied to one
  score per unit). It says nothing about rows that a policy chose.
* **VW**, selection-aware weighted conformal, draws each unit's row uniformly from the rows the
  policy reaches (Tibshirani, Barber, Candes and Ramdas, 2019, covariate shift with a known
  likelihood ratio). The policy depends on the quantiles, so they come from a fixed number of
  fixed-point steps that start at V1's. A fixed policy gives an exact statement; the iterated
  version is approximate, and the history of the steps is kept so that convergence can be shown.
* **V2L** and **V3L** are adaptive conformal inference (Gibbs and Candes, 2021) with one update per
  resolved life. That guarantee is about the long-run average miss rate of the stream of labels
  the method sees, whatever the sequence; it is not about oracle coverage when a policy decides
  which labels exist.

Random streams. Unit ``j`` of a calibration set has one uniform draw ``u_j`` from
``default_rng([seed, j])``. V1 uses row ``floor(u_j n)`` and VW reuses the same ``u_j`` at every
step, so VW with a rule that never fires equals V1 exactly. The adaptive methods draw from their
own stream, so it cannot collide with the unit streams.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from closed_loop_v2.estimator import Prediction
from closed_loop_v2.policy import TriggerRule
from closed_loop_v2.types import Trajectory

CalibPairs = Sequence[tuple[Trajectory, Prediction]]

# An index within this relative distance of a whole number is taken to be that whole number, so
# that float error in ``(n + 1) * a`` cannot move a quantile by one place at an exact multiple.
_INDEX_TOL = 1e-9
_ACI_KEY = (4001, 7)  # appended to the seed for the adaptive methods' stream


# ---------------------------------------------------------------- quantiles


def _near_int(x: float) -> int | None:
    r = round(x)
    return int(r) if abs(x - r) <= _INDEX_TOL * max(1.0, abs(x)) else None


def _floor_index(x: float) -> int:
    r = _near_int(x)
    return math.floor(x) if r is None else r


def _ceil_index(x: float) -> int:
    r = _near_int(x)
    return math.ceil(x) if r is None else r


def _sorted(scores: NDArray[np.float64] | Sequence[float]) -> NDArray[np.float64]:
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")
    return s


def _kth_smallest(sorted_scores: NDArray[np.float64], k: int) -> float:
    """The k-th smallest score (1-based); ``-inf`` below the first and ``+inf`` past the last.

    That is the k-th smallest of the scores with ``-inf`` and ``+inf`` added at the two ends.
    """
    if k < 1:
        return -math.inf
    if k > sorted_scores.size:
        return math.inf
    return float(sorted_scores[k - 1])


def _check_level(a: float) -> None:
    if not math.isfinite(a):
        raise ValueError("the level must be finite")


def conformal_lower(scores: NDArray[np.float64] | Sequence[float], a: float) -> float:
    """The ``floor((n + 1) a)``-th smallest score, or ``-inf`` when that index is below 1.

    An index above ``n`` (a level of about 1 or more) gives ``+inf``. For ``n`` scores and a new
    exchangeable score ``S``, ``P(S < result) <= a``.
    """
    _check_level(a)
    s = _sorted(scores)
    return _kth_smallest(s, _floor_index((s.size + 1) * a))


def conformal_upper(scores: NDArray[np.float64] | Sequence[float], a: float) -> float:
    """The ``ceil((n + 1)(1 - a))``-th smallest score, or ``+inf`` when that index is above ``n``.

    An index below 1 (a level of about 1 or more) gives ``-inf``. For ``n`` scores and a new
    exchangeable score ``S``, ``P(S > result) <= a``.
    """
    _check_level(a)
    s = _sorted(scores)
    return _kth_smallest(s, _ceil_index((s.size + 1) * (1.0 - a)))


@dataclass(frozen=True)
class Quantiles:
    q_lo: float
    q_hi: float


def interval_ends(
    m: NDArray[np.float64], s: NDArray[np.float64], q: Quantiles, cap: float | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(min(m + s q_lo, cap), min(m + s q_hi, cap))`` row by row; no limit when ``cap`` is None.

    The truth is ``min(remaining life, cap)`` (ADR 0001 section 5), so it never exceeds ``cap``, and
    both ends are limited to ``cap``: the interval is intersected with the truth's range. That can
    only turn a miss into a hit, so every coverage guarantee stands, and every lower bound is then
    at most ``cap``, where covering the capped truth and covering the remaining life are the same
    event (review D, finding S1). A ``q_lo`` of ``-inf`` stays ``-inf``; a ``q_hi`` of ``+inf``
    becomes ``cap``.
    """
    lo = m + s * q.q_lo
    hi = m + s * q.q_hi
    if cap is None:
        return lo, hi
    return np.minimum(lo, cap), np.minimum(hi, cap)


def cap_of(traj: Trajectory) -> float | None:
    """The cap of the trajectory's truth, as a float, or None for an uncapped trajectory."""
    return None if traj.rul_cap is None else float(traj.rul_cap)


def bounds(
    pred: Prediction, q: Quantiles, cap: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The interval ``(m + s q_lo, m + s q_hi)`` at every row, both ends limited to ``cap``.

    See ``interval_ends``. With ``cap=None`` an infinite ``q`` stays infinite.
    """
    return interval_ends(pred.m, pred.s, q, cap)


def score(rul: NDArray[np.float64] | NDArray[np.int64], pred: Prediction, i: int) -> float:
    """``(rul[i] - m[i]) / s[i]``."""
    return float((rul[i] - pred.m[i]) / pred.s[i])


def scores(rul: NDArray[np.float64] | NDArray[np.int64], pred: Prediction) -> NDArray[np.float64]:
    """The score at every row."""
    return (np.asarray(rul, dtype=np.float64) - pred.m) / pred.s


def _quantiles_at(sc: NDArray[np.float64] | Sequence[float], a: float) -> Quantiles:
    s = _sorted(sc)
    return Quantiles(
        _kth_smallest(s, _floor_index((s.size + 1) * a)),
        _kth_smallest(s, _ceil_index((s.size + 1) * (1.0 - a))),
    )


# ---------------------------------------------------------------- the methods


class Method(Protocol):
    """What the replay needs from a conformal method."""

    def quantiles(self) -> Quantiles: ...

    def observe_life(self, labels: LifeLabels) -> None: ...


def _check_pairs(calib: CalibPairs) -> None:
    if len(calib) == 0:
        raise ValueError("need at least one calibration unit")
    for traj, pred in calib:
        if pred.n != traj.n:
            raise ValueError(f"{traj.key}: {pred.n} predictions for {traj.n} rows")


def _unit_draws(seed: int, n_units: int) -> NDArray[np.float64]:
    """One ``u_j ~ U[0, 1)`` per unit from ``default_rng([seed, j])``, ``j`` the unit's position."""
    return np.array([np.random.default_rng([seed, j]).random() for j in range(n_units)])


def draw_row(u: float, n_rows: int) -> int:
    """``floor(u * n_rows)``, clipped to the last row so that the index is valid for any ``u``."""
    return min(int(u * n_rows), n_rows - 1)


def calibration_scores(calib: CalibPairs, *, seed: int) -> NDArray[np.float64]:
    """V1's scores: one uniformly drawn row per calibration unit."""
    _check_pairs(calib)
    u = _unit_draws(seed, len(calib))
    return np.array(
        [score(traj.rul(), pred, draw_row(u[j], traj.n)) for j, (traj, pred) in enumerate(calib)]
    )


@dataclass(frozen=True)
class SplitConformal:
    """Fixed quantiles. ``observe_life`` does nothing.

    ``history`` and ``moved_last_step`` are filled in only by ``fit_vw``: the quantiles at step 0
    (V1's) to step ``steps``, and whether the last step changed them.
    """

    q: Quantiles
    history: tuple[Quantiles, ...] = ()
    moved_last_step: bool | None = None

    def quantiles(self) -> Quantiles:
        return self.q

    def observe_life(self, labels: LifeLabels) -> None:
        return None


def fit_v1(calib: CalibPairs, *, alpha: float = 0.10, seed: int) -> SplitConformal:
    """Split conformal on one uniformly drawn row per calibration unit, ``alpha / 2`` per side."""
    return SplitConformal(_quantiles_at(calibration_scores(calib, seed=seed), alpha / 2.0))


def fit_vw(
    calib: CalibPairs,
    rule: TriggerRule,
    *,
    alpha: float = 0.10,
    seed: int,
    steps: int,
) -> SplitConformal:
    """Selection-aware weighted conformal by fixed-point iteration from V1's quantiles.

    Each step: take every unit's lower bounds under the current ``q_lo``, find the row ``tau_j``
    where ``rule`` first fires (the last row when it never does), draw row
    ``floor(u_j (tau_j + 1))`` with the same ``u_j`` as before, and recompute the quantiles from
    those scores. ``history`` has ``steps + 1`` entries, from V1's quantiles to the last step.

    With a lower bound of ``-inf`` (``q_lo = -inf`` from a very small calibration set) a
    ``BoundRule`` fires at row 0 for every ``h``, which makes VW degenerate; use at least
    ``2 / alpha - 1`` calibration units.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    _check_pairs(calib)
    u = _unit_draws(seed, len(calib))
    ruls = [traj.rul().astype(np.float64) for traj, _ in calib]
    a = alpha / 2.0
    q = _quantiles_at(
        [score(ruls[j], pred, draw_row(u[j], traj.n)) for j, (traj, pred) in enumerate(calib)], a
    )
    history = [q]
    moved = False
    for _ in range(steps):
        sc = []
        for j, (traj, pred) in enumerate(calib):
            tau = rule.first_trigger(bounds(pred, q, cap_of(traj))[0])
            reach = traj.n if tau is None else tau + 1
            sc.append(score(ruls[j], pred, draw_row(u[j], reach)))
        new = _quantiles_at(sc, a)
        moved = new != q
        q = new
        history.append(q)
    return SplitConformal(q, history=tuple(history), moved_last_step=moved)


# ---------------------------------------------------------------- labels


class Status(enum.IntEnum):
    """Label of one checkpoint once its life has ended."""

    HIT = 0
    MISS = 1
    LOWER_HIT = 2
    UNKNOWN = 3


@dataclass(frozen=True, eq=False)
class LifeLabels:
    """Statuses of one resolved life. ``scores`` is set only after an observed failure."""

    status: NDArray[np.int8]
    observed_failure: bool
    scores: NDArray[np.float64] | None


def life_labels(
    rul: NDArray[np.float64] | NDArray[np.int64],
    pred: Prediction,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    *,
    replaced_at: int | None = None,
    cap: float = math.inf,
) -> LifeLabels:
    """Statuses of one life from its truth, estimates and bounds (full length).

    ``rul`` is the truth (``Trajectory.rul()``, capped at ``cap`` when the trajectory is).
    ``replaced_at=None`` is an observed failure: every row is ``HIT`` (``lo <= rul <= hi``) or
    ``MISS``, and the scores are returned. ``replaced_at=tau`` is a replacement at row ``tau``;
    rows ``0..tau`` are labelled with ``b_i = tau - i``, the cycles the unit is known to have
    lasted beyond row ``i``. The truth is ``min(remaining life, cap)``, so it is known to be at
    least ``min(b_i, cap)``, and exactly ``cap`` once ``b_i >= cap``:

    * rows with ``b_i >= cap``: ``HIT`` if ``lo_i <= cap <= hi_i``, otherwise ``MISS``;
    * otherwise ``MISS`` if ``b_i > hi_i`` or ``lo_i > cap`` (the truth is at most ``cap``),
      ``LOWER_HIT`` if ``lo_i <= b_i <= hi_i`` and ``UNKNOWN`` if ``b_i < lo_i``.

    With ``cap = inf`` (the default) the first case and ``lo_i > cap`` never arise. Bounds from
    ``bounds`` with the trajectory's cap are never above ``cap``.
    """
    r = np.asarray(rul, dtype=np.float64)
    n = r.size
    if not (pred.n == n and lo.shape == (n,) and hi.shape == (n,)):
        raise ValueError("rul, pred, lo and hi must have the same length")
    if replaced_at is None:
        hit = (lo <= r) & (r <= hi)
        status = np.where(hit, Status.HIT, Status.MISS).astype(np.int8)
        return LifeLabels(status, True, scores(r, pred))
    if not 0 <= replaced_at < n:
        raise ValueError(f"replaced_at must be in 0..{n - 1}")
    end = replaced_at + 1
    b = (replaced_at - np.arange(end)).astype(np.float64)
    lo_e, hi_e = lo[:end], hi[:end]
    status = np.full(end, Status.UNKNOWN, dtype=np.int8)
    status[(b >= lo_e) & (b <= hi_e)] = Status.LOWER_HIT
    status[b > hi_e] = Status.MISS
    status[lo_e > cap] = Status.MISS  # the truth never exceeds cap: a certain miss (review D, S1)
    exact = b >= cap  # the capped truth is known: it equals cap
    status[exact] = np.where((lo_e[exact] <= cap) & (cap <= hi_e[exact]), Status.HIT, Status.MISS)
    return LifeLabels(status, False, None)


# ---------------------------------------------------------------- adaptive conformal


class AdaptiveConformal:
    """Adaptive conformal inference with one update per resolved life (V2L and V3L).

    ``rule="naive"`` (V2L) counts ``UNKNOWN`` and ``LOWER_HIT`` as hits. ``rule="skip"`` (V3L)
    leaves ``UNKNOWN`` out and counts ``LOWER_HIT`` as a hit. After a life, with ``miss`` the mean
    miss indicator over the rows counted, ``alpha_t += gamma (alpha - miss)``; a life with no row
    counted changes nothing. ``alpha_t`` is not clipped: ``alpha_t / 2`` outside ``(0, 1)`` gives
    infinite quantiles (see ``conformal_lower`` and ``conformal_upper``). After an observed failure
    one uniformly drawn score of that life joins the pool, which keeps the latest ``window``.
    """

    def __init__(
        self,
        initial_scores: NDArray[np.float64] | Sequence[float],
        *,
        alpha: float = 0.10,
        gamma: float = 0.02,
        window: int = 500,
        rule: Literal["naive", "skip"],
        seed: int,
    ) -> None:
        if rule not in ("naive", "skip"):
            raise ValueError(f"rule must be 'naive' or 'skip', not {rule!r}")
        if window < 1:
            raise ValueError("window must be at least 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        _sorted(initial_scores)  # NaN check
        self._pool: deque[float] = deque(
            (float(v) for v in np.asarray(initial_scores, dtype=np.float64).ravel()),
            maxlen=window,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.rule = rule
        self._alpha_t = alpha
        self._rng = np.random.default_rng([seed, *_ACI_KEY])

    @property
    def alpha_t(self) -> float:
        return self._alpha_t

    @property
    def pool(self) -> tuple[float, ...]:
        """The scores now in the pool, oldest first."""
        return tuple(self._pool)

    def quantiles(self) -> Quantiles:
        return _quantiles_at(np.fromiter(self._pool, dtype=np.float64), self._alpha_t / 2.0)

    def observe_life(self, labels: LifeLabels) -> None:
        st = labels.status
        counted = np.ones(st.shape, dtype=bool) if self.rule == "naive" else st != Status.UNKNOWN
        n_counted = int(counted.sum())
        if n_counted > 0:
            miss = float(((st == Status.MISS) & counted).sum()) / n_counted
            self._alpha_t += self.gamma * (self.alpha - miss)
        if labels.observed_failure:
            if labels.scores is None or labels.scores.size == 0:
                raise ValueError("an observed failure needs its scores")
            self._pool.append(float(labels.scores[int(self._rng.integers(labels.scores.size))]))


def make_v2l(
    calib: CalibPairs,
    *,
    seed: int,
    alpha: float = 0.10,
    gamma: float = 0.02,
    window: int = 500,
) -> AdaptiveConformal:
    """V2L: the pool starts from V1's draw, and ``UNKNOWN`` counts as a hit."""
    return AdaptiveConformal(
        calibration_scores(calib, seed=seed),
        alpha=alpha,
        gamma=gamma,
        window=window,
        rule="naive",
        seed=seed,
    )


def make_v3l(
    calib: CalibPairs,
    *,
    seed: int,
    alpha: float = 0.10,
    gamma: float = 0.02,
    window: int = 500,
) -> AdaptiveConformal:
    """V3L: the pool starts from V1's draw, and ``UNKNOWN`` is skipped."""
    return AdaptiveConformal(
        calibration_scores(calib, seed=seed),
        alpha=alpha,
        gamma=gamma,
        window=window,
        rule="skip",
        seed=seed,
    )
