"""Fixed-sample intervals for the verdicts (docs/adr/0001-design.md, section 11).

The evaluation set is fixed in advance and analysed once, so v0.2 uses fixed-sample intervals.

* **Means and paired differences** use the Student-t interval
  ``mean +/- t(1 - alpha/2, n - 1) s / sqrt(n)``, clipped to the range the values can take, only
  when all three of v0.1's guard conditions hold: at least ``SPREAD_MIN_PAIRS`` (100) values, some
  spread (largest value differs from the smallest), and a sample kurtosis of at most
  ``n / SPREAD_KURTOSIS_DIVISOR`` (``n / 50``). The kurtosis is computed as v0.1's
  ``closed_loop.stats.confseq.paired_cs`` computes it: deviations scaled to a peak of 1, the fourth
  central moment over the square of the second, both with ``n`` in the denominator. The two
  constants are imported from v0.1, not copied.
* **Otherwise** the fixed-sample betting interval is used (``betting_interval``, ADR 0004): the
  hedged-capital interval of Waudby-Smith and Ramdas, "Estimating means of bounded random variables
  by betting", JRSS B 86(1), 2024, with predictable plug-in bets tuned for the fixed ``n``, on the
  values rescaled to ``[0, 1]`` and mapped back (differences in ``[-w, w]`` likewise). It is valid
  at every sample size and for any distribution on the range, provided the values are independent
  with one mean and are fed in an order drawn at random independently of them. Here the order is
  the fixed canonical order below, and V2L's and V3L's per-unit values are coupled through their
  shared ``alpha_t``, so for those rows the level is nominal, not proven (review D, S2; ADR 0004
  section 3). The t route assumes independence too. Until ADR 0004 (10 November 2026, before the
  freeze) this route used v0.1's ``betting_cs``, a confidence sequence; read at one ``n`` its
  half-width had a floor of 0.06 at ``n = 373``, above the C2-PERC band, so a C2-PERC loss could not
  be reached.
* **A binary rate** uses the exact Clopper-Pearson interval (v0.1's ``clopper_pearson``).

**Canonical order for the betting interval** (review of packet C, finding S1). The betting interval
is built from predictable bets, so its bounds depend on the order of the values, and the per-unit
arrays arrive in subset blocks (FD001, then FD002, ...), whose means differ. Before any call to
``betting_interval`` the values are therefore put into an order that depends only on their
multiset:

* for a mean, the values are sorted;
* for paired data, the pairs are sorted lexicographically by ``(a - b, a, b)``;
* then a fixed pseudo-random permutation is applied, from
  ``numpy.random.default_rng([*ORDER_KEY, n])`` with ``ORDER_KEY = (4099, 1)``. For paired data the
  same permutation moves both arms, so pairs stay together.

The betting interval is then exactly invariant to the input order, bit for bit, and the sequence it
reads is not sorted (a sorted sequence would put every low value first). The permutation is fixed
by the key and ``n`` alone, so nobody can choose it after seeing data; it is a fixed arrangement,
not a random draw per analysis. The betting argument is exact for an order drawn at random
independently of the values. A fixed order is not such a draw, and the argument as written does not
cover it, so the stated level for the betting route is nominal (ADR 0004 section 3). The
t route needs no such step: its mean, standard deviation and
kurtosis do not depend on the order beyond floating-point rounding, and it is left as it was.

``Interval.method`` names the rule that produced each interval (``"t"``, ``"betting"`` or
``"clopper-pearson"``) so that the report can show it. ``Interval.estimate`` is the sample mean (or
``k / n``); 0002 lists ``Interval(low, high, method, n)``, and the estimate is added so that the
report can print it from the same object. With no values the interval is the whole range, the
method is ``"betting"`` (the betting interval of no data) and the estimate is ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from closed_loop.stats import clopper_pearson
from closed_loop.stats.confseq import SPREAD_KURTOSIS_DIVISOR, SPREAD_MIN_PAIRS
from numpy.typing import NDArray
from scipy.stats import t as student_t

__all__ = [
    "BET_TRUNCATION",
    "ORDER_KEY",
    "Interval",
    "IntervalMethod",
    "betting_interval",
    "binomial_interval",
    "canonical_order",
    "canonical_pairs",
    "kurtosis",
    "mean_interval",
    "paired_interval",
    "use_t",
]

IntervalMethod = Literal["t", "betting", "clopper-pearson"]
# Keys the fixed permutation that follows the canonical sort (module docstring). Changing it changes
# every betting interval, so it is inside the code hash like the verdict constants.
ORDER_KEY: Final[tuple[int, int]] = (4099, 1)
# The truncation c of the betting interval (ADR 0004): every bet keeps each factor of the capital at
# least 1 - c, so the capital stays positive.
BET_TRUNCATION: Final[float] = 0.75
_BISECTION_STEPS: Final[int] = 64  # halves [0, 1] to below float spacing near any end


@dataclass(frozen=True)
class Interval:
    """A two-sided interval at level ``1 - alpha`` and the rule that produced it."""

    low: float
    high: float
    method: IntervalMethod
    n: int
    estimate: float | None
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """The interval as ``results.json`` stores it, at full precision."""
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "method": self.method,
            "n": self.n,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Interval:
        """The inverse of ``to_dict``, for an interval read back from ``results.json``."""
        method = str(d["method"])
        if method not in ("t", "betting", "clopper-pearson"):
            raise ValueError(f"unknown interval method {method!r}")
        est = d.get("estimate")
        return cls(
            float(d["low"]),
            float(d["high"]),
            cast(IntervalMethod, method),
            int(d["n"]),
            None if est is None else float(est),
            float(d["alpha"]),
        )


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _values(x: Sequence[float] | NDArray[np.float64], lo: float, hi: float) -> NDArray[np.float64]:
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"need finite lo < hi, got lo={lo!r}, hi={hi!r}")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < lo) or np.any(arr > hi):
        raise ValueError(f"every value must be finite and within [{lo}, {hi}]")
    return arr


def kurtosis(x: NDArray[np.float64]) -> float | None:
    """Sample kurtosis as v0.1's ``paired_cs`` computes it, or ``None`` with no spread.

    Deviations from the mean are scaled to a peak of 1 first, so that powers of a very small or very
    large deviation cannot underflow or overflow; the moments have ``n`` in the denominator.
    """
    if x.size == 0 or float(np.max(x)) == float(np.min(x)):
        return None
    dev = x - float(x.mean())
    peak = float(np.max(np.abs(dev)))
    z = dev / peak
    m2 = float(np.mean(z**2))
    m4 = float(np.mean(z**4))
    if m2 <= 0.0:
        return None
    return m4 / (m2 * m2)


def use_t(x: NDArray[np.float64]) -> bool:
    """v0.1's guard: ``n >= 100``, some spread, and a kurtosis of at most ``n / 50``."""
    n = int(x.size)
    if n < SPREAD_MIN_PAIRS:
        return False
    k = kurtosis(x)
    return k is not None and k <= n / SPREAD_KURTOSIS_DIVISOR


def _fixed_permutation(n: int) -> NDArray[np.intp]:
    """The fixed permutation of ``n`` positions, from ``ORDER_KEY`` and ``n`` alone."""
    perm: NDArray[np.intp] = np.random.default_rng([*ORDER_KEY, n]).permutation(n)
    return perm


def canonical_order(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``x`` sorted, then put through the fixed permutation: a function of the multiset of ``x``."""
    ordered: NDArray[np.float64] = np.sort(x, kind="stable")[_fixed_permutation(int(x.size))]
    return ordered


def canonical_pairs(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pairs ``(a_i, b_i)`` sorted by ``(a - b, a, b)``, then both arms put through one permutation.

    The result depends only on the multiset of pairs, and every pair stays together.
    """
    by = np.lexsort((b, a, a - b))  # the last key is the primary one
    perm = _fixed_permutation(int(a.size))
    return a[by][perm], b[by][perm]


def _plug_in_bets(x: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """The bets ``lambda_t`` for ``t = 1..n``; each one uses ``x_1..x_{t-1}`` only (ADR 0004).

    ``mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)``, ``var_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) /
    (t + 1)`` with ``var_0 = 1/4``, and ``lambda_t = sqrt(2 log(2 / alpha) / (n var_{t-1}))``.
    """
    n = int(x.size)
    t = np.arange(1, n + 1, dtype=np.float64)
    mu = (0.5 + np.cumsum(x)) / (t + 1.0)
    sq = np.cumsum((x - mu) ** 2)
    var_before = np.empty(n)
    var_before[0] = 0.25
    var_before[1:] = (0.25 + sq[:-1]) / t[1:]
    bets: NDArray[np.float64] = np.sqrt(2.0 * math.log(2.0 / alpha) / (n * var_before))
    return bets


def _log_capital_up(x: NDArray[np.float64], bets: NDArray[np.float64], m: float) -> float:
    """``log K^+(m)``: betting that the mean is above ``m``. Never increases with ``m``."""
    lam = bets if m <= 0.0 else np.minimum(bets, BET_TRUNCATION / m)
    return float(np.sum(np.log1p(lam * (x - m))))


def _log_capital_down(x: NDArray[np.float64], bets: NDArray[np.float64], m: float) -> float:
    """``log K^-(m)``: betting that the mean is below ``m``. Never decreases with ``m``."""
    lam = bets if m >= 1.0 else np.minimum(bets, BET_TRUNCATION / (1.0 - m))
    return float(np.sum(np.log1p(-lam * (x - m))))


def betting_interval(x: NDArray[np.float64], alpha: float) -> tuple[float, float]:
    """Fixed-sample betting interval ``(L, U)`` for the mean of ``x`` in ``[0, 1]`` (ADR 0004).

    The hedged-capital interval of Waudby-Smith and Ramdas (JRSS B 86(1), 2024) with predictable
    plug-in bets tuned for ``n = len(x)`` and truncation ``BET_TRUNCATION``: ``m`` is excluded when
    ``max(K^+(m), K^-(m)) / 2 >= 1 / alpha``. The values are read in the order given; callers put
    them into the canonical order first. ``K^+`` never increases and ``K^-`` never decreases with
    ``m``, so the kept set is an interval; each end is found by bisection and returned at the
    excluded end of its final bracket, so the result can only be wider than the exact set. With
    probability at most ``alpha`` it is empty (``L > U``). With no values it is ``(0, 1)``.
    """
    _check_alpha(alpha)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("betting_interval needs a one-dimensional array of values in [0, 1]")
    if arr.size == 0:
        return 0.0, 1.0
    bets = _plug_in_bets(arr, alpha)
    limit = math.log(2.0 / alpha)

    # Lower end: K^+ is excluded at small m. m = 1 is always kept (every factor is at most 1).
    if _log_capital_up(arr, bets, 0.0) < limit:
        low = 0.0
    else:
        out, keep = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (out + keep)
            if _log_capital_up(arr, bets, mid) >= limit:
                out = mid
            else:
                keep = mid
        low = out

    # Upper end: K^- is excluded at large m. m = 0 is always kept (every factor is at most 1).
    if _log_capital_down(arr, bets, 1.0) < limit:
        high = 1.0
    else:
        keep, out = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (keep + out)
            if _log_capital_down(arr, bets, mid) >= limit:
                out = mid
            else:
                keep = mid
        high = out
    return low, high


def _t_interval(
    x: NDArray[np.float64], lo: float, hi: float, alpha: float
) -> tuple[float, float, float]:
    """``mean +/- t(1 - alpha/2, n - 1) s / sqrt(n)``, clipped to ``[lo, hi]``; and the mean."""
    n = int(x.size)
    mean = float(x.mean())
    s = float(np.std(x, ddof=1))
    half = float(student_t.ppf(1.0 - alpha / 2.0, n - 1)) * s / math.sqrt(n)
    return max(lo, mean - half), min(hi, mean + half), mean


def mean_interval(
    x: Sequence[float] | NDArray[np.float64], *, lo: float, hi: float, alpha: float
) -> Interval:
    """Interval for the mean of ``x``, whose values lie in ``[lo, hi]``."""
    _check_alpha(alpha)
    arr = _values(x, lo, hi)
    n = int(arr.size)
    if n == 0:
        return Interval(lo, hi, "betting", 0, None, alpha)
    if use_t(arr):
        low, high, mean = _t_interval(arr, lo, hi, alpha)
        return Interval(low, high, "t", n, mean, alpha)
    width = hi - lo
    canon = canonical_order(arr)  # the betting interval reads the values in this order only
    unit = np.clip((canon - lo) / width, 0.0, 1.0)
    lower, upper = betting_interval(unit, alpha)
    return Interval(
        lo + lower * width,
        lo + upper * width,
        "betting",
        n,
        float(canon.mean()),
        alpha,
    )


def paired_interval(
    a: Sequence[float] | NDArray[np.float64],
    b: Sequence[float] | NDArray[np.float64],
    *,
    lo: float,
    hi: float,
    alpha: float,
) -> Interval:
    """Interval for ``mean(a - b)``, with ``a`` and ``b`` in ``[lo, hi]``, paired by position.

    The differences lie in ``[-w, w]`` with ``w = hi - lo``; a t interval is clipped to that range.
    """
    _check_alpha(alpha)
    arr_a = _values(a, lo, hi)
    arr_b = _values(b, lo, hi)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"a and b differ in length: {arr_a.shape} and {arr_b.shape}")
    n = int(arr_a.size)
    w = hi - lo
    if n == 0:
        return Interval(-w, w, "betting", 0, None, alpha)
    d = arr_a - arr_b
    if use_t(d):
        low, high, mean = _t_interval(d, -w, w, alpha)
        return Interval(low, high, "t", n, mean, alpha)
    canon_a, canon_b = canonical_pairs(arr_a, arr_b)  # the same order for both arms
    diffs = canon_a - canon_b
    unit = np.clip((diffs + w) / (2.0 * w), 0.0, 1.0)
    lower, upper = betting_interval(unit, alpha)
    return Interval(
        -w + lower * 2.0 * w, -w + upper * 2.0 * w, "betting", n, float(diffs.mean()), alpha
    )


def binomial_interval(k: int, n: int, alpha: float) -> Interval:
    """The exact Clopper-Pearson interval for ``k`` successes in ``n`` trials."""
    low, high = clopper_pearson(k, n, alpha)
    return Interval(float(low), float(high), "clopper-pearson", int(n), k / n if n else None, alpha)
