"""Quantiles, V1, VW, label statuses and the adaptive methods, on made-up scores."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from closed_loop_v2.conformal import (
    AdaptiveConformal,
    LifeLabels,
    Quantiles,
    SplitConformal,
    Status,
    bounds,
    calibration_scores,
    conformal_lower,
    conformal_upper,
    draw_row,
    fit_v1,
    fit_vw,
    life_labels,
    make_v2l,
    make_v3l,
    score,
)
from closed_loop_v2.estimator import Prediction
from closed_loop_v2.policy import BoundRule
from closed_loop_v2.types import Trajectory

INF = math.inf


def make_traj(n: int, unit: int = 1) -> Trajectory:
    """A trajectory whose failure is at its last row; only its length matters here."""
    return Trajectory(
        "FD002", unit, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 0
    )


def make_pair(rng: np.random.Generator, unit: int, lo_n: int = 20, hi_n: int = 60):
    """One synthetic unit: a unit-level bias plus row noise, so rows of a unit are correlated and
    units are independent and identically distributed."""
    n = int(rng.integers(lo_n, hi_n))
    traj = make_traj(n, unit)
    s = rng.uniform(2.0, 6.0, size=n)
    bias = rng.normal(0.0, 3.0)
    drift = rng.normal(0.0, 1.0) * np.linspace(0.0, 1.0, n)
    m = traj.rul() + bias + drift + rng.normal(0.0, 2.0, size=n)
    return traj, Prediction(m, s)


def make_calib(seed: int, n_units: int, **kw: int):
    rng = np.random.default_rng(seed)
    return [make_pair(rng, j, **kw) for j in range(n_units)]


def score_pair(traj: Trajectory, targets: NDArray[np.float64]) -> tuple[Trajectory, Prediction]:
    """A unit whose score at row i is ``targets[i]`` (s = 1, m = rul - target)."""
    return traj, Prediction(
        traj.rul().astype(np.float64) - np.asarray(targets, dtype=np.float64), np.ones(traj.n)
    )


# ------------------------------------------------------------------ quantile indices


def test_lower_index_edges() -> None:
    sc = np.arange(1.0, 20.0)  # n = 19
    assert conformal_lower(sc, 0.05) == 1.0  # (n+1) a = 1.0 exactly, not 0.9999
    assert conformal_lower(sc, 0.10) == 2.0
    assert conformal_lower(sc, 0.0499) == -INF  # floor(0.998) = 0 < 1
    assert conformal_lower(sc, 0.0) == -INF
    assert conformal_lower(sc, 0.5) == 10.0
    assert conformal_lower(sc, 0.9999) == 19.0  # floor(19.998) = 19
    assert conformal_lower(sc, 1.0) == INF  # index 20 > n


def test_upper_index_edges() -> None:
    sc = np.arange(1.0, 20.0)
    assert conformal_upper(sc, 0.05) == 19.0  # ceil(20 * 0.95) = 19, not 20
    assert conformal_upper(sc, 0.10) == 18.0
    assert conformal_upper(sc, 0.0499) == INF  # ceil(19.002) = 20 > n
    assert conformal_upper(sc, 0.0) == INF
    assert conformal_upper(sc, 0.5) == 10.0
    assert conformal_upper(sc, 0.9999) == 1.0  # ceil(0.002) = 1
    assert conformal_upper(sc, 1.0) == -INF  # index 0 < 1


def test_small_n() -> None:
    assert conformal_lower([3.0], 0.5) == 3.0  # floor(1.0) = 1
    assert conformal_lower([3.0], 0.4) == -INF
    assert conformal_upper([3.0], 0.5) == 3.0  # ceil(1.0) = 1
    assert conformal_upper([3.0], 0.4) == INF
    assert conformal_lower([], 0.05) == -INF
    assert conformal_upper([], 0.05) == INF


def test_unsorted_input_and_ties() -> None:
    assert conformal_lower([5.0, 1.0, 3.0, 1.0, 9.0], 0.34) == 1.0  # floor(2.04) = 2nd smallest
    assert conformal_upper([5.0, 1.0, 3.0, 1.0, 9.0], 0.34) == 5.0  # ceil(3.96) = 4th smallest


def test_infinite_and_nan_inputs() -> None:
    with pytest.raises(ValueError):
        conformal_lower([1.0, math.nan], 0.1)
    with pytest.raises(ValueError):
        conformal_upper([1.0], math.nan)
    with pytest.raises(ValueError):
        conformal_lower([1.0], INF)


def test_indices_match_exact_arithmetic_at_every_level_to_three_places() -> None:
    """floor((n+1)a) and ceil((n+1)(1-a)) in exact fractions, for a = j / 1000."""
    for n in list(range(1, 41)) + [99, 100, 199, 200, 999, 1000]:
        sc = np.arange(1.0, n + 1)  # the k-th smallest is k
        for j in range(0, 1001, 1 if n <= 40 else 7):
            a = j / 1000
            exact_lo = math.floor((n + 1) * Fraction(j, 1000))
            exact_hi = math.ceil((n + 1) * (1 - Fraction(j, 1000)))
            want_lo = -INF if exact_lo < 1 else (INF if exact_lo > n else float(exact_lo))
            want_hi = -INF if exact_hi < 1 else (INF if exact_hi > n else float(exact_hi))
            assert conformal_lower(sc, a) == want_lo, (n, j)
            assert conformal_upper(sc, a) == want_hi, (n, j)


def test_computed_levels_hit_the_exact_index() -> None:
    # alpha / 2 with alpha = 0.1 is the level fit_v1 uses; 1 - 0.95 is another float route
    sc = np.arange(1.0, 20.0)
    assert conformal_lower(sc, 0.1 / 2) == 1.0
    assert conformal_lower(sc, 1 - 0.95) == 1.0
    assert conformal_upper(sc, 0.1 / 2) == 19.0
    sc39 = np.arange(1.0, 40.0)
    assert conformal_lower(sc39, 0.05) == 2.0 and conformal_upper(sc39, 0.05) == 38.0


def test_bounds_and_score() -> None:
    pred = Prediction(np.array([10.0, 20.0]), np.array([2.0, 4.0]))
    lo, hi = bounds(pred, Quantiles(-1.0, 0.5))
    np.testing.assert_array_equal(lo, [8.0, 16.0])
    np.testing.assert_array_equal(hi, [11.0, 22.0])
    lo, hi = bounds(pred, Quantiles(-INF, INF))
    assert (lo == -INF).all() and (hi == INF).all()
    assert score(np.array([14, 8]), pred, 0) == 2.0
    assert score(np.array([14, 8]), pred, 1) == -3.0


# ------------------------------------------------------------------ V1


def test_v1_takes_one_row_per_unit_and_quantiles_of_those() -> None:
    calib = [score_pair(make_traj(10 + j % 7, j), np.full(10 + j % 7, float(j))) for j in range(39)]
    assert sorted(calibration_scores(calib, seed=0)) == [float(j) for j in range(39)]
    q = fit_v1(calib, seed=0).quantiles()
    assert q == Quantiles(1.0, 37.0)  # 2nd smallest, and ceil(40 * 0.95) = 38th smallest


def test_v1_draw_is_keyed_by_seed_and_unit_index() -> None:
    calib = make_calib(1, 30)
    full = calibration_scores(calib, seed=7)
    np.testing.assert_array_equal(full, calibration_scores(calib, seed=7))
    np.testing.assert_array_equal(calibration_scores(calib[:12], seed=7), full[:12])
    assert not np.array_equal(full, calibration_scores(calib, seed=8))
    # the row is floor(u * n) with u from default_rng([seed, j])
    for j in (0, 5, 29):
        u = np.random.default_rng([7, j]).random()
        traj, pred = calib[j]
        assert full[j] == score(traj.rul(), pred, int(u * traj.n))


def test_v1_rows_are_uniform() -> None:
    traj = make_traj(10)
    calib = [score_pair(traj, np.arange(10.0)) for _ in range(5000)]  # score = row index
    counts = np.bincount(calibration_scores(calib, seed=3).astype(int), minlength=10)
    chi2 = float(((counts - 500) ** 2 / 500).sum())
    assert chi2 < 27.9  # 9 degrees of freedom, p = 0.001


def test_draw_row_is_clipped() -> None:
    assert draw_row(0.0, 5) == 0 and draw_row(0.5, 4) == 2 and draw_row(0.999, 1) == 0
    assert draw_row(1.0, 5) == 4  # not a value of u in [0, 1), but the guard holds
    u_max = float(np.nextafter(1.0, 0.0))
    assert all(draw_row(u_max, n) == n - 1 for n in range(1, 3000))


def v1_miss_rates(n_cal: int, trials: int, seed: int) -> tuple[float, float]:
    """Lower-side miss rate and two-sided miss rate of V1 on fresh exchangeable units."""
    rng = np.random.default_rng(seed)
    lower = both = 0
    for t in range(trials):
        calib = [make_pair(rng, j) for j in range(n_cal)]
        q = fit_v1(calib, alpha=0.10, seed=t).quantiles()
        traj, pred = make_pair(rng, 10_000)
        i = int(rng.integers(traj.n))  # a uniformly drawn row of a new unit
        lo, hi = bounds(pred, q)
        rul = float(traj.rul()[i])
        lower += rul < lo[i]
        both += (rul < lo[i]) or (rul > hi[i])
    return lower / trials, both / trials


def check_rates(lower: float, both: float, trials: int, z: float) -> None:
    sd_lower = math.sqrt(0.05 * 0.95 / trials)
    sd_both = math.sqrt(0.10 * 0.90 / trials)
    assert abs(lower - 0.05) < z * sd_lower, lower
    assert abs(both - 0.10) < z * sd_both, both


def test_v1_monte_carlo_fast() -> None:
    """n = 39 makes (n + 1) * 0.05 = 2 exactly, so continuous scores give a lower miss of exactly
    alpha / 2 and a two-sided miss of alpha. 400 trials, 4.5 standard deviations."""
    lower, both = v1_miss_rates(39, 400, seed=11)
    check_rates(lower, both, 400, z=4.5)


@pytest.mark.slow
def test_v1_monte_carlo_2000_trials() -> None:
    lower, both = v1_miss_rates(39, 2000, seed=12)
    check_rates(lower, both, 2000, z=4.0)


def test_v1_is_conservative_when_the_index_is_not_a_whole_number() -> None:
    """n = 30: (n + 1) * 0.05 = 1.55 so k = 1 and the miss rate is at most 1/31, below alpha / 2."""
    rng = np.random.default_rng(13)
    trials, lower = 300, 0
    for t in range(trials):
        calib = [make_pair(rng, j) for j in range(30)]
        q = fit_v1(calib, seed=t).quantiles()
        traj, pred = make_pair(rng, 10_000)
        i = int(rng.integers(traj.n))
        lower += float(traj.rul()[i]) < bounds(pred, q)[0][i]
    assert lower / trials < 0.05 + 4.5 * math.sqrt(0.05 * 0.95 / trials)
    assert lower / trials < 1 / 31 + 4.5 * math.sqrt(1 / 31 * 30 / 31 / trials)


def test_split_conformal_does_nothing_on_observe() -> None:
    sc = SplitConformal(Quantiles(-1.0, 2.0))
    labels = LifeLabels(np.array([Status.MISS], dtype=np.int8), True, np.array([0.5]))
    sc.observe_life(labels)
    assert sc.quantiles() == Quantiles(-1.0, 2.0)


# ------------------------------------------------------------------ VW


class FixedRule:
    """Fires at row ``tau`` whatever the bounds (``None``: never)."""

    def __init__(self, tau: int | None) -> None:
        self.tau = tau

    def first_trigger(self, lo: NDArray[np.float64]) -> int | None:
        return self.tau


def exact_quantiles(sc: list[float], alpha: str) -> Quantiles:
    """Quantiles by exact fractions, for the reference loop (alpha is a decimal string)."""
    a = Fraction(alpha) / 2
    s = sorted(sc)
    n = len(s)
    k_lo, k_hi = math.floor((n + 1) * a), math.ceil((n + 1) * (1 - a))

    def kth(k: int) -> float:
        return -INF if k < 1 else (INF if k > n else s[k - 1])

    return Quantiles(kth(k_lo), kth(k_hi))


def vw_reference(calib, h: float, alpha: str, seed: int, steps: int) -> list[Quantiles]:
    """The fixed-point iteration written out with plain loops."""
    u = [np.random.default_rng([seed, j]).random() for j in range(len(calib))]

    def pick(q: Quantiles | None) -> list[float]:
        out = []
        for j, (traj, pred) in enumerate(calib):
            n = traj.n
            tau = n - 1
            if q is not None:
                for i in range(n):
                    if pred.m[i] + pred.s[i] * q.q_lo <= h:
                        tau = i
                        break
            row = min(math.floor(u[j] * (tau + 1)), n - 1)
            out.append(float((traj.rul()[row] - pred.m[row]) / pred.s[row]))
        return out

    q = exact_quantiles(pick(None), alpha)
    history = [q]
    for _ in range(steps):
        q = exact_quantiles(pick(q), alpha)
        history.append(q)
    return history


def rising_calib(seed: int, n_units: int):
    """Scores rise with the row (late rows are the high ones), so a policy that cuts a life short
    changes the scores that get drawn."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_units):
        n = int(rng.integers(40, 80))
        traj = make_traj(n, j)
        s = rng.uniform(3.0, 5.0, n)
        target = 4.0 * np.arange(n) / n + rng.normal(0.0, 0.5)
        m = traj.rul() - s * target
        out.append((traj, Prediction(m, s)))
    return out


def test_vw_equals_v1_when_the_rule_never_fires() -> None:
    calib = make_calib(2, 60)
    v1 = fit_v1(calib, seed=5)
    for h in (-INF, -1e12):
        vw = fit_vw(calib, BoundRule(h), seed=5, steps=6)
        assert vw.quantiles() == v1.quantiles()
        assert all(q == v1.quantiles() for q in vw.history)
        assert len(vw.history) == 7
        assert vw.moved_last_step is False
    assert fit_vw(calib, FixedRule(None), seed=5, steps=3).quantiles() == v1.quantiles()


def test_vw_with_no_steps_is_v1() -> None:
    calib = make_calib(3, 40)
    vw = fit_vw(calib, BoundRule(1e9), seed=1, steps=0)
    assert vw.quantiles() == fit_v1(calib, seed=1).quantiles()
    assert vw.history == (vw.quantiles(),) and vw.moved_last_step is False
    with pytest.raises(ValueError):
        fit_vw(calib, BoundRule(0.0), seed=1, steps=-1)


def test_vw_takes_floor_u_times_tau_plus_one_with_one_u_per_unit() -> None:
    tau = 3
    calib = [score_pair(make_traj(12, j), np.arange(12.0)) for j in range(199)]  # score = row
    u = np.array([np.random.default_rng([4, j]).random() for j in range(199)])
    want = exact_quantiles([float(math.floor(x * (tau + 1))) for x in u], "0.1")
    for steps in (1, 2, 5):  # the same u at every step: the answer does not move
        got = fit_vw(calib, FixedRule(tau), seed=4, steps=steps)
        assert got.quantiles() == want
        assert got.history[1:] == (want,) * steps
    # a rule firing at row 0 draws row 0 for everyone
    zero = fit_vw(calib, FixedRule(0), seed=4, steps=2).quantiles()
    assert zero == Quantiles(0.0, 0.0)


def test_vw_matches_the_loop_reference_and_the_policy_moves_the_quantiles() -> None:
    calib = rising_calib(5, 120)
    got = fit_vw(calib, BoundRule(40.0), seed=9, steps=8)
    ref = vw_reference(calib, 40.0, "0.1", seed=9, steps=8)
    assert list(got.history) == ref
    assert got.history[1] != got.history[0]  # the rule is live: it is not V1 in disguise
    assert got.quantiles().q_hi < fit_v1(calib, seed=9).quantiles().q_hi  # late rows are cut off


def test_vw_reaches_a_fixed_point_on_a_synthetic_case() -> None:
    calib = rising_calib(6, 150)
    got = fit_vw(calib, BoundRule(25.0), seed=3, steps=25)
    assert got.moved_last_step is False
    q = got.quantiles()
    # one more step from q gives q: scores are re-drawn with the same u at the bounds q implies
    again = vw_reference(calib, 25.0, "0.1", seed=3, steps=26)
    assert again[-1] == q and again[-2] == q
    assert list(got.history) == again[:26]
    # and the iteration did move on the way
    assert len({(x.q_lo, x.q_hi) for x in got.history}) > 1


def test_vw_infinite_lower_quantile_makes_the_rule_fire_at_once() -> None:
    calib = make_calib(7, 5)  # too few units for a finite 5% quantile
    vw = fit_vw(calib, BoundRule(-INF), seed=0, steps=1)
    assert vw.history[0].q_lo == -INF
    ref = vw_reference(calib, -INF, "0.1", seed=0, steps=1)
    assert list(vw.history) == ref


def test_vw_input_checks() -> None:
    calib = make_calib(8, 3)
    bad = [(calib[0][0], Prediction(np.ones(2), np.ones(2)))]
    with pytest.raises(ValueError):
        fit_vw(bad, BoundRule(0.0), seed=0, steps=1)
    with pytest.raises(ValueError):
        fit_v1([], seed=0)


# ------------------------------------------------------------------ label statuses


def hand_pred(n: int) -> Prediction:
    return Prediction(np.zeros(n), np.ones(n))


def test_replaced_life_statuses_by_hand() -> None:
    # replaced at row 5, so b = 5, 4, 3, 2, 1, 0 for rows 0..5; row 6 is never labelled
    lo = np.array([6.0, 4.0, 0.0, 0.0, 1.01, -3.0, 0.0])
    hi = np.array([9.0, 9.0, 3.0, 1.99, 5.0, 4.0, 0.0])
    rul = np.arange(6.0, -1.0, -1.0)
    lab = life_labels(rul, hand_pred(7), lo, hi, replaced_at=5)
    assert not lab.observed_failure and lab.scores is None
    want = [
        Status.UNKNOWN,  # b = 5 < lo = 6
        Status.LOWER_HIT,  # b = 4 == lo
        Status.LOWER_HIT,  # b = 3 == hi
        Status.MISS,  # b = 2 > hi = 1.99
        Status.UNKNOWN,  # b = 1 < lo = 1.01
        Status.LOWER_HIT,  # b = 0 inside
    ]
    assert lab.status.tolist() == [int(s) for s in want]
    assert lab.status.dtype == np.int8


def test_replacement_at_row_zero_and_unbounded_sides() -> None:
    lo = np.array([-INF, -INF, -INF])
    hi = np.array([INF, INF, INF])
    lab = life_labels(np.array([2.0, 1.0, 0.0]), hand_pred(3), lo, hi, replaced_at=0)
    assert lab.status.tolist() == [int(Status.LOWER_HIT)]
    lab = life_labels(np.array([2.0, 1.0, 0.0]), hand_pred(3), lo, hi, replaced_at=2)
    assert set(lab.status.tolist()) == {int(Status.LOWER_HIT)}


def test_observed_failure_rows_are_hit_or_miss_only() -> None:
    rul = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    pred = Prediction(
        rul + np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    lo = np.array([0.0, 0.0, 2.5, 0.0, 0.0])
    hi = np.array([10.0, 2.9, 3.0, 1.0, 0.0])
    lab = life_labels(rul, pred, lo, hi)
    assert lab.observed_failure
    assert lab.status.tolist() == [
        int(s) for s in (Status.HIT, Status.MISS, Status.MISS, Status.HIT, Status.HIT)
    ]
    assert lab.scores is not None
    np.testing.assert_allclose(lab.scores, -1.0)  # (rul - m) / s = -1 in every row


def test_life_labels_input_checks() -> None:
    rul = np.arange(3.0)
    with pytest.raises(ValueError):
        life_labels(rul, hand_pred(3), np.zeros(3), np.zeros(3), replaced_at=3)
    with pytest.raises(ValueError):
        life_labels(rul, hand_pred(3), np.zeros(3), np.zeros(3), replaced_at=-1)
    with pytest.raises(ValueError):
        life_labels(rul, hand_pred(3), np.zeros(2), np.zeros(3))


# ------------------------------------------------------------------ adaptive conformal


def labels(
    *statuses: Status, failure: bool = False, scores: list[float] | None = None
) -> LifeLabels:
    return LifeLabels(
        np.array([int(s) for s in statuses], dtype=np.int8),
        failure,
        None if scores is None else np.array(scores),
    )


POOL = [float(x) for x in range(1, 40)]  # n = 39: quantiles at alpha / 2 = 0.05 are 2 and 38


def test_initial_quantiles_equal_v1() -> None:
    calib = make_calib(9, 39)
    v1 = fit_v1(calib, seed=2).quantiles()
    assert make_v2l(calib, seed=2).quantiles() == v1
    assert make_v3l(calib, seed=2).quantiles() == v1
    np.testing.assert_array_equal(
        np.array(make_v2l(calib, seed=2).pool), calibration_scores(calib, seed=2)
    )


def test_update_arithmetic() -> None:
    aci = AdaptiveConformal(POOL, alpha=0.10, gamma=0.02, rule="naive", seed=0)
    assert aci.quantiles() == Quantiles(2.0, 38.0)
    aci.observe_life(
        labels(Status.HIT, Status.HIT, Status.MISS, Status.HIT, failure=True, scores=[1.0])
    )
    assert aci.alpha_t == pytest.approx(0.10 + 0.02 * (0.10 - 0.25))
    aci.observe_life(labels(Status.HIT, Status.HIT, failure=True, scores=[1.0]))
    assert aci.alpha_t == pytest.approx(0.097 + 0.02 * 0.10)  # no misses: alpha_t rises
    aci.observe_life(labels(Status.MISS, failure=True, scores=[1.0]))
    assert aci.alpha_t == pytest.approx(0.099 + 0.02 * (0.10 - 1.0))


def test_a_life_that_counts_no_row_changes_nothing() -> None:
    skip = AdaptiveConformal(POOL, rule="skip", seed=0)
    skip.observe_life(labels(Status.UNKNOWN, Status.UNKNOWN))
    assert skip.alpha_t == 0.10 and skip.pool == tuple(POOL)
    naive = AdaptiveConformal(POOL, rule="naive", seed=0)
    naive.observe_life(labels())  # an empty life counts nothing either
    assert naive.alpha_t == 0.10


def test_naive_counts_unknown_and_lower_hit_as_hits_and_skip_leaves_unknown_out() -> None:
    life = labels(Status.UNKNOWN, Status.LOWER_HIT, Status.MISS, Status.UNKNOWN)
    naive = AdaptiveConformal(POOL, rule="naive", seed=0)
    naive.observe_life(life)
    assert naive.alpha_t == pytest.approx(0.10 + 0.02 * (0.10 - 1 / 4))
    skip = AdaptiveConformal(POOL, rule="skip", seed=0)
    skip.observe_life(life)
    assert skip.alpha_t == pytest.approx(0.10 + 0.02 * (0.10 - 1 / 2))


def test_censored_lives_add_nothing_to_the_pool_but_failures_do() -> None:
    aci = AdaptiveConformal(POOL, rule="naive", seed=0)
    aci.observe_life(labels(Status.LOWER_HIT, Status.MISS))
    assert aci.pool == tuple(POOL)
    aci.observe_life(labels(Status.HIT, failure=True, scores=[123.0]))
    assert aci.pool == (*POOL, 123.0)


def test_pool_keeps_the_latest_window_and_drops_the_oldest() -> None:
    aci = AdaptiveConformal([10.0, 20.0, 30.0], window=3, rule="naive", seed=0)
    aci.observe_life(labels(Status.HIT, failure=True, scores=[99.0]))
    assert aci.pool == (20.0, 30.0, 99.0)
    aci.observe_life(labels(Status.HIT, failure=True, scores=[98.0]))
    assert aci.pool == (30.0, 99.0, 98.0)
    # an initial pool longer than the window keeps its last entries
    assert AdaptiveConformal(POOL, window=5, rule="skip", seed=0).pool == tuple(POOL[-5:])


def test_added_score_is_a_uniform_draw_from_the_life() -> None:
    life_scores = [float(x) for x in range(10)]
    a = AdaptiveConformal([0.0], window=10_000, rule="naive", seed=5)
    b = AdaptiveConformal([0.0], window=10_000, rule="naive", seed=5)
    c = AdaptiveConformal([0.0], window=10_000, rule="naive", seed=6)
    for _ in range(3000):
        for aci in (a, b, c):
            aci.observe_life(labels(Status.HIT, failure=True, scores=life_scores))
    assert a.pool == b.pool and a.pool != c.pool
    counts = np.bincount(np.array(a.pool[1:]).astype(int), minlength=10)
    chi2 = float(((counts - 300) ** 2 / 300).sum())
    assert chi2 < 27.9  # 9 degrees of freedom, p = 0.001


def test_alpha_t_is_not_clipped_and_out_of_range_levels_give_infinite_sides() -> None:
    low = AdaptiveConformal(POOL, alpha=0.10, gamma=0.2, rule="naive", seed=0)
    low.observe_life(labels(Status.MISS, Status.MISS))
    assert low.alpha_t == pytest.approx(0.10 + 0.2 * (0.10 - 1.0))  # -0.08, left as it is
    assert low.alpha_t < 0
    assert low.quantiles() == Quantiles(-INF, INF)
    high = AdaptiveConformal(POOL, alpha=0.90, gamma=1.5, rule="naive", seed=0)
    high.observe_life(labels(Status.HIT))
    assert high.alpha_t == pytest.approx(0.90 + 1.5 * 0.90)  # 2.25, so a = 1.125 per side
    assert high.quantiles() == Quantiles(INF, -INF)


def test_rule_and_argument_validation() -> None:
    with pytest.raises(ValueError):
        AdaptiveConformal(POOL, rule="other", seed=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AdaptiveConformal(POOL, window=0, rule="naive", seed=0)
    with pytest.raises(ValueError):
        AdaptiveConformal(POOL, alpha=1.0, rule="naive", seed=0)
    aci = AdaptiveConformal(POOL, rule="naive", seed=0)
    with pytest.raises(ValueError):
        aci.observe_life(labels(Status.HIT, failure=True))  # an observed failure needs scores


def test_end_to_end_labels_into_aci() -> None:
    """life_labels feeds observe_life: a failure life with one miss in four rows."""
    rul = np.array([3.0, 2.0, 1.0, 0.0])
    pred = Prediction(rul.copy(), np.ones(4))  # scores are all 0
    lo, hi = bounds(pred, Quantiles(-1.0, 1.0))
    hi[1] = 1.5  # rul 2.0 above 1.5: a miss
    aci = make_v3l([score_pair(make_traj(4), np.zeros(4))] * 39, seed=1, gamma=0.1)
    aci.observe_life(life_labels(rul, pred, lo, hi))
    assert aci.alpha_t == pytest.approx(0.10 + 0.1 * (0.10 - 0.25))
    assert aci.pool[-1] == 0.0 and len(aci.pool) == 40


def test_bounds_are_limited_to_the_cap_of_the_truth() -> None:
    from closed_loop_v2.conformal import Quantiles, bounds, cap_of, interval_ends

    pred = Prediction(np.array([130.0, 100.0, 124.0]), np.array([2.0, 10.0, 1.0]))
    q = Quantiles(-1.0, 1.0)
    lo, hi = bounds(pred, q, 125.0)
    assert list(lo) == [125.0, 90.0, 123.0] and list(hi) == [125.0, 110.0, 125.0]
    lo, hi = bounds(pred, q)  # no cap: as before
    assert list(lo) == [128.0, 90.0, 123.0] and list(hi) == [132.0, 110.0, 125.0]
    lo, hi = interval_ends(pred.m, pred.s, Quantiles(-math.inf, math.inf), 125.0)
    assert np.all(lo == -math.inf) and np.all(hi == 125.0)  # +inf becomes the cap
    n = 3
    t = Trajectory("FD001", 1, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 0)
    assert cap_of(t) is None and cap_of(t.capped(125)) == 125.0


def test_a_lower_bound_above_the_cap_is_a_certain_miss() -> None:
    """The truth min(RUL, cap) never exceeds cap, so lo > cap misses whatever b is (review D S1)."""
    from closed_loop_v2.conformal import Status, life_labels

    n = 3
    pred = Prediction(np.full(n, 3.0), np.ones(n))
    lo = np.array([5.0, 5.0, 5.0])
    hi = np.array([6.0, 6.0, 6.0])
    out = life_labels(np.zeros(n), pred, lo, hi, replaced_at=2, cap=4.0)
    # b = 2, 1, 0, all below cap and below lo: UNKNOWN without the rule, MISS with it
    assert list(out.status) == [Status.MISS, Status.MISS, Status.MISS]
    plain = life_labels(np.zeros(n), pred, lo, hi, replaced_at=2)
    assert list(plain.status) == [Status.UNKNOWN, Status.UNKNOWN, Status.UNKNOWN]


def test_labels_with_a_cap_are_exact_once_the_survival_reaches_it() -> None:
    from closed_loop_v2.conformal import Status, life_labels

    n = 8
    pred = Prediction(np.full(n, 3.0), np.ones(n))
    lo = np.array([2.0, 2.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    hi = np.array([5.0, 4.0, 9.0, 1.5, 3.0, 3.0, 3.0, 3.0])
    out = life_labels(np.zeros(n), pred, lo, hi, replaced_at=6, cap=4.0)
    # b = 6, 5, 4, 3, 2, 1, 0: rows 0 to 2 have b >= 4 (truth exactly 4): 4 in [2, 5] HIT,
    # 4 in [2, 4] HIT, 4 < 6 MISS; then 3 > 1.5 MISS, 2 in [1, 3] LOWER_HIT, 1 in [1, 3]
    # LOWER_HIT, 0 < 1 UNKNOWN
    assert list(out.status) == [
        Status.HIT, Status.HIT, Status.MISS, Status.MISS,
        Status.LOWER_HIT, Status.LOWER_HIT, Status.UNKNOWN,
    ]  # fmt: skip
    same = life_labels(np.zeros(n), pred, lo, hi, replaced_at=6)  # no cap: the old rule
    assert list(same.status[:3]) == [
        Status.MISS,
        Status.MISS,
        Status.UNKNOWN,
    ]  # 6 > 5, 5 > 4, 4 < 6
