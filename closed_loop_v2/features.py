"""Features for the remaining-life estimator (docs/adr/0001-design.md, section 5).

Every feature at row ``i`` uses rows ``0..i`` of its own trajectory and nothing later.

* ``ConditionModel`` assigns each row's three settings to the nearest of ``k`` centroids (k-means
  on the settings, numpy only) and z-scores the sensors within that condition.
* ``window_features`` turns the z-scores into a rolling mean and a rolling least-squares slope.

Empty clusters. k-means++ starts every centroid on a data row, so a cluster can only empty out when
rows repeat or when there are fewer distinct rows than ``k``. An empty cluster is re-seeded, in
order of cluster index, on the row that lies farthest from the centroid it is currently assigned to
(ties go to the lowest row index). A row is used to re-seed at most one cluster per step. When no
row is left to move (every distance is zero) the cluster keeps a duplicate centroid, stays empty,
and gets mean 0 and standard deviation 0, so it contributes z-scores of 0 if it is ever hit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop_v2.types import N_SENSORS, N_SETTINGS, Trajectory

MAX_LLOYD_STEPS = 100
_KMEANS_KEY = (17, 5)  # appended to the seed; keeps this stream apart from the other modules


def _sq_dist(x: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Euclidean distance from every row of ``x`` to every centroid, shape (n, k)."""
    d = np.empty((x.shape[0], centroids.shape[0]), dtype=np.float64)
    for c in range(centroids.shape[0]):
        d[:, c] = ((x - centroids[c]) ** 2).sum(axis=1)
    return d


def _kmeans_pp(x: NDArray[np.float64], k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """k-means++ seeding: each new centroid is a row drawn with probability proportional to its
    squared distance from the nearest centroid already chosen."""
    n = x.shape[0]
    chosen = [int(rng.integers(n))]
    d2 = ((x - x[chosen[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        if total > 0.0:
            cum = np.cumsum(d2)
            idx = min(int(np.searchsorted(cum, rng.random() * total, side="right")), n - 1)
        else:  # every row sits on a chosen centroid: the duplicate will be an empty cluster
            idx = int(rng.integers(n))
        chosen.append(idx)
        d2 = np.minimum(d2, ((x - x[idx]) ** 2).sum(axis=1))
    return x[chosen].copy()


def _lloyd(
    x: NDArray[np.float64], centroids: NDArray[np.float64], max_steps: int = MAX_LLOYD_STEPS
) -> NDArray[np.float64]:
    """Lloyd steps from ``centroids``; stops when the assignment stops changing or at ``max_steps``.

    Empty clusters are re-seeded as described in the module docstring.
    """
    cen = centroids.copy()
    labels: NDArray[np.intp] | None = None
    rows = np.arange(x.shape[0])
    for _ in range(max_steps):
        d = _sq_dist(x, cen)
        new = d.argmin(axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        own = d[rows, labels]
        used = np.zeros(x.shape[0], dtype=bool)
        for c in range(cen.shape[0]):
            members = labels == c
            if members.any():
                cen[c] = x[members].mean(axis=0)
            else:
                cand = np.where(used, -1.0, own)
                row = int(np.argmax(cand))
                cen[c] = x[row]
                used[row] = True
    return cen


@dataclass(frozen=True, eq=False)
class ConditionModel:
    """Operating-condition centroids and per-condition sensor statistics.

    ``means`` and ``stds`` have shape ``(k, 21)`` (all sensors); ``kept`` lists the sensors that
    are used. A standard deviation below ``min_std`` in a condition is stored as 0.0, and a zero
    standard deviation gives a z-score of 0 in that condition.
    """

    centroids: NDArray[np.float64]
    means: NDArray[np.float64]
    stds: NDArray[np.float64]
    kept: tuple[int, ...]

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])

    @classmethod
    def fit(
        cls, trajs: Sequence[Trajectory], *, seed: int, k: int = 6, min_std: float = 1e-6
    ) -> ConditionModel:
        """Fit on every row of ``trajs``.

        The k-means runs on the raw settings (the C-MAPSS conditions are far apart on that scale).
        Pass ``k=1`` for a single-condition subset: with ``k > 1`` the noise would be split up.
        A sensor is dropped when its largest within-condition standard deviation is below
        ``min_std``, which is the same as being below it in every condition.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if len(trajs) == 0:
            raise ValueError("need at least one trajectory")
        settings = np.concatenate([t.settings for t in trajs]).astype(np.float64, copy=False)
        sensors = np.concatenate([t.sensors for t in trajs]).astype(np.float64, copy=False)
        if not (np.isfinite(settings).all() and np.isfinite(sensors).all()):
            raise ValueError("settings and sensors must be finite")
        rng = np.random.default_rng([seed, *_KMEANS_KEY])
        centroids = _lloyd(settings, _kmeans_pp(settings, k, rng))
        cond = _sq_dist(settings, centroids).argmin(axis=1)
        means = np.zeros((k, N_SENSORS))
        stds = np.zeros((k, N_SENSORS))
        for c in range(k):
            members = cond == c
            if members.any():
                means[c] = sensors[members].mean(axis=0)
                stds[c] = sensors[members].std(axis=0)
        kept = tuple(int(j) for j in np.flatnonzero(stds.max(axis=0) >= min_std))
        if not kept:
            raise ValueError("no sensor varies by at least min_std within a condition")
        stds[stds < min_std] = 0.0
        return cls(centroids=centroids, means=means, stds=stds, kept=kept)

    def condition(self, settings: NDArray[np.float64]) -> NDArray[np.intp]:
        """Index of the nearest centroid for each row (ties go to the lowest index)."""
        if settings.ndim != 2 or settings.shape[1] != N_SETTINGS:
            raise ValueError(f"settings must have shape (n, {N_SETTINGS})")
        return _sq_dist(settings, self.centroids).argmin(axis=1)

    def normalise(self, traj: Trajectory) -> NDArray[np.float64]:
        """Kept sensors, z-scored within each row's own condition. Shape ``(n, len(kept))``."""
        cond = self.condition(traj.settings)
        cols = list(self.kept)
        mean = self.means[cond][:, cols]
        std = self.stds[cond][:, cols]
        diff = traj.sensors[:, cols] - mean
        z = np.zeros_like(diff)
        np.divide(diff, std, out=z, where=std > 0.0)
        return z


def window_features(z: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Rolling mean and least-squares slope of each column, then the 1-based cycle count.

    For row ``i`` the window is rows ``max(0, i - window + 1)..i``. The slope is against the row
    index and is 0 for a one-row window. The output has shape ``(n, 2k + 1)``: ``k`` means, ``k``
    slopes, the cycle count. Row ``i`` uses no later row. The sums come from cumulative sums, so
    truncating ``z`` leaves the earlier rows' features bit for bit unchanged.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be two-dimensional")
    if not np.isfinite(z).all():
        raise ValueError("z must be finite")
    n, k = z.shape
    idx = np.arange(n)
    length = np.minimum(idx + 1, window)
    start = idx + 1 - length
    zero = np.zeros((1, k))
    cs = np.concatenate([zero, np.cumsum(z, axis=0)])
    csx = np.concatenate([zero, np.cumsum(idx[:, None] * z, axis=0)])
    sy = cs[idx + 1] - cs[start]
    sxy = csx[idx + 1] - csx[start]
    mean = sy / length[:, None]
    xbar = idx - (length - 1) / 2.0
    sxx = length * (length.astype(np.float64) ** 2 - 1.0) / 12.0
    slope = np.zeros((n, k))
    multi = length > 1
    slope[multi] = (sxy[multi] - xbar[multi, None] * sy[multi]) / sxx[multi, None]
    return np.concatenate([mean, slope, (idx + 1.0)[:, None]], axis=1)
