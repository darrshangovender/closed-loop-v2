"""The shared remaining-life estimator (docs/adr/0001-design.md, section 5).

Two gradient-boosted regressors share one feature matrix:

* ``m`` predicts ``min(RUL, cap)``;
* ``s`` predicts ``|min(RUL, cap) - m_oof|``, where ``m_oof`` is the out-of-fold prediction of the
  same ``m`` model from cross-fitting with folds that hold whole units. ``s`` therefore never sees
  ``m``'s in-sample errors. ``predict`` floors ``s`` at ``s_min``.

Both targets use the capped truth (ADR 0001, amendment of 10 November 2026). With the uncapped
remaining life, ``s`` learned the large errors of early life, when the engine is healthy and its
remaining life cannot be told from the sensors, and the symmetric score then pulled every lower
bound down by that scale. ``fit`` caps the truth itself, so it does not matter whether the
trajectories passed in are already capped; trajectories capped at another value are refused.

Early stopping is off, so the boosting has no internal random split, and the folds come from a
generator keyed by the seed. A fit repeats bit for bit for the same seed, data and library build.

Threads. Every fit and prediction runs with one OpenMP thread. On the Windows development laptop
(scikit-learn 1.9.1, Python 3.12) the multi-threaded split search stalled for minutes on a 1,300-row
fit at 2 and at 4 threads, and finished in 3 s at 1; one thread also takes the thread count out of
the determinism claim. ``threadpoolctl`` is a dependency of scikit-learn, not a new one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from closed_loop_v2.features import ConditionModel, window_features
from closed_loop_v2.types import Trajectory

_FOLD_KEY = (23, 3)  # appended to the seed; keeps the fold stream apart from the other modules
_HGB_KEY = (23, 5)  # the same, for scikit-learn's random_state


def _sklearn_seed(seed: int) -> int:
    """A 32-bit ``random_state`` derived from any non-negative ``seed``.

    scikit-learn takes seeds below 2**32 only; the run's seeds are 64-bit (``seeds.py``). The map
    goes through ``numpy.random.SeedSequence``, so distinct seeds give unrelated states and the same
    seed always gives the same one. Review of 10 November, finding 5.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    state = np.random.SeedSequence([seed, *_HGB_KEY]).generate_state(1, dtype=np.uint32)
    return int(state[0])


@dataclass(frozen=True)
class EstimatorParams:
    """Free choices of the estimator. Tuned on dev units and then frozen."""

    window: int
    cap: float
    s_min: float
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int

    def __post_init__(self) -> None:
        if self.window < 1 or self.max_iter < 1 or self.max_leaf_nodes < 2:
            raise ValueError("window and max_iter must be at least 1, max_leaf_nodes at least 2")
        if self.min_samples_leaf < 1 or self.learning_rate <= 0.0:
            raise ValueError("min_samples_leaf must be at least 1 and learning_rate positive")
        if self.cap <= 0.0 or self.s_min <= 0.0:
            raise ValueError("cap and s_min must be positive")


DEFAULT_PARAMS = EstimatorParams(
    window=30,
    cap=125.0,
    s_min=5.0,
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
)


@dataclass(frozen=True, eq=False)
class Prediction:
    """Point estimate ``m`` and scale ``s`` at each row of one trajectory (float64, length n)."""

    m: NDArray[np.float64]
    s: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.m.ndim != 1 or self.m.shape != self.s.shape:
            raise ValueError("m and s must be one-dimensional and of equal length")
        if not (np.isfinite(self.m).all() and np.isfinite(self.s).all() and (self.s > 0).all()):
            raise ValueError("m must be finite and s finite and positive")

    @property
    def n(self) -> int:
        return int(self.m.shape[0])


def _regressor(params: EstimatorParams, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=params.max_iter,
        learning_rate=params.learning_rate,
        max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf,
        early_stopping=False,
        random_state=_sklearn_seed(seed),
    )


def unit_folds(n_units: int, n_folds: int, seed: int) -> NDArray[np.intp]:
    """Fold number of each unit (by position): a seeded shuffle dealt out round-robin, so the fold
    sizes differ by at most one. Whole units share a fold."""
    rng = np.random.default_rng([seed, *_FOLD_KEY])
    folds = np.empty(n_units, dtype=np.intp)
    folds[rng.permutation(n_units)] = np.arange(n_units) % n_folds
    return folds


@dataclass(frozen=True, eq=False)
class Estimator:
    """A fitted condition model plus the ``m`` and ``s`` regressors."""

    condition_model: ConditionModel
    params: EstimatorParams
    m_model: HistGradientBoostingRegressor
    s_model: HistGradientBoostingRegressor
    seed: int

    def features(self, traj: Trajectory) -> NDArray[np.float64]:
        """Feature matrix of one trajectory; row ``i`` uses rows ``0..i`` only."""
        return window_features(self.condition_model.normalise(traj), self.params.window)

    @classmethod
    def fit(
        cls,
        trajs: Sequence[Trajectory],
        params: EstimatorParams,
        *,
        seed: int,
        n_folds: int = 5,
    ) -> Estimator:
        """Fit on ``trajs`` (training units, whose true remaining life is known at every row).

        Folds are by position in ``trajs``, not by unit number, because unit numbers repeat across
        subsets. With fewer units than ``n_folds`` the number of folds is the number of units.
        """
        n_units = len(trajs)
        if n_units < 2:
            raise ValueError("cross-fitting needs at least two units")
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        n_folds = min(n_folds, n_units)
        if not float(params.cap).is_integer():
            raise ValueError(f"cap must be a whole number of cycles, got {params.cap!r}")
        other = sorted({t.rul_cap for t in trajs if t.rul_cap not in (None, int(params.cap))})
        if other:
            raise ValueError(f"trajectories capped at {other}, the estimator's cap is {params.cap}")
        cond = ConditionModel.fit(trajs, seed=seed)
        feats = [window_features(cond.normalise(t), params.window) for t in trajs]
        x = np.concatenate(feats)
        rul = np.concatenate([t.rul_uncapped() for t in trajs]).astype(np.float64)
        unit_of_row = np.repeat(np.arange(n_units), [t.n for t in trajs])
        target_m = np.minimum(rul, params.cap)  # the capped truth: the target of m and of s

        row_fold = unit_folds(n_units, n_folds, seed)[unit_of_row]
        m_oof = np.empty_like(rul)
        with threadpool_limits(limits=1, user_api="openmp"):
            for f in range(n_folds):
                held = row_fold == f
                fold_model = _regressor(params, seed).fit(x[~held], target_m[~held])
                m_oof[held] = fold_model.predict(x[held])

            m_model = _regressor(params, seed).fit(x, target_m)
            s_model = _regressor(params, seed).fit(x, np.abs(target_m - m_oof))
        return cls(condition_model=cond, params=params, m_model=m_model, s_model=s_model, seed=seed)

    def predict(self, traj: Trajectory) -> Prediction:
        """``m`` and ``s`` at every row, with ``s`` floored at ``s_min``.

        It reads the sensors and settings only, never ``rul_at_end``, and row ``i`` is unchanged
        when later rows are cut off or altered.
        """
        x = self.features(traj)
        with threadpool_limits(limits=1, user_api="openmp"):
            m = np.asarray(self.m_model.predict(x), dtype=np.float64)
            s_raw = np.asarray(self.s_model.predict(x), dtype=np.float64)
        s = np.maximum(s_raw, self.params.s_min)
        return Prediction(m=m, s=s)
