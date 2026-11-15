"""The interval rule: the t/betting guard at n = 99/100 and at the kurtosis limit, the t interval
against scipy, the fixed-sample betting fallback (ADR 0004: a case worked by hand, a brute-force
grid, its floor, its coverage), its canonical order (the betting interval does not depend on the
order of the values), and Clopper-Pearson."""

from __future__ import annotations

import math

import numpy as np
import pytest
from closed_loop.stats import betting_cs, clopper_pearson
from scipy import stats as sps

from closed_loop_v2.stats import (
    BET_TRUNCATION,
    Interval,
    betting_interval,
    binomial_interval,
    canonical_order,
    canonical_pairs,
    kurtosis,
    mean_interval,
    paired_interval,
    use_t,
)

A = 0.05
PRIMARY = 0.05 / 6


def alternating(n: int) -> np.ndarray:
    """0, 1, 0, 1, ...: kurtosis about 1, far below n / 50 once n >= 100."""
    return (np.arange(n) % 2).astype(np.float64)


def three_point(n: int, k: int) -> np.ndarray:
    """k values at 0, k at 1, the rest at 0.5: kurtosis exactly n / (2k), mean exactly 0.5."""
    return np.array([0.0] * k + [1.0] * k + [0.5] * (n - 2 * k))


# ----------------------------------------------------------------------------- the guard


def test_the_guard_switches_between_99_and_100() -> None:
    assert mean_interval(alternating(99), lo=0, hi=1, alpha=A).method == "betting"
    assert mean_interval(alternating(100), lo=0, hi=1, alpha=A).method == "t"
    assert not use_t(alternating(99)) and use_t(alternating(100))


def test_the_guard_switches_at_the_kurtosis_limit() -> None:
    at_limit = three_point(100, 25)  # kurtosis 100 / 50 = 2.0 exactly: at most the limit
    past = three_point(100, 24)  # kurtosis 100 / 48, just past it
    assert kurtosis(at_limit) == 2.0
    assert kurtosis(past) == pytest.approx(100 / 48)
    assert mean_interval(at_limit, lo=0, hi=1, alpha=A).method == "t"
    assert mean_interval(past, lo=0, hi=1, alpha=A).method == "betting"


def test_no_spread_goes_to_betting() -> None:
    assert mean_interval(np.full(500, 0.25), lo=0, hi=1, alpha=A).method == "betting"
    assert kurtosis(np.full(5, 1.0)) is None


def test_kurtosis_is_the_biased_moment_ratio() -> None:
    x = np.random.default_rng(3).gamma(2.0, size=400)
    assert kurtosis(x) == pytest.approx(float(sps.kurtosis(x, fisher=False, bias=True)))
    tiny = 1e-90 * x  # the peak scaling keeps powers of tiny deviations from underflowing
    assert kurtosis(tiny) == pytest.approx(kurtosis(x))


def test_the_paired_guard_reads_the_differences() -> None:
    a = alternating(100)
    assert paired_interval(a, np.zeros(100), lo=0, hi=1, alpha=A).method == "t"
    assert paired_interval(a[:99], np.zeros(99), lo=0, hi=1, alpha=A).method == "betting"
    same = paired_interval(a, a, lo=0, hi=1, alpha=A)  # differences all 0: no spread
    assert same.method == "betting"


# ----------------------------------------------------------------------------- the t interval


def test_the_t_interval_matches_scipy() -> None:
    x = np.random.default_rng(5).uniform(0.3, 0.9, size=250)
    got = mean_interval(x, lo=0, hi=1, alpha=0.01)
    low, high = sps.t.interval(0.99, len(x) - 1, loc=x.mean(), scale=sps.sem(x))
    assert got.method == "t"
    assert got.low == pytest.approx(low, rel=1e-12) and got.high == pytest.approx(high, rel=1e-12)
    assert got.estimate == pytest.approx(x.mean()) and got.n == 250 and got.alpha == 0.01


def test_the_paired_t_interval_matches_scipy() -> None:
    rng = np.random.default_rng(6)
    a, b = rng.uniform(0.4, 1.0, 300), rng.uniform(0.3, 0.9, 300)
    got = paired_interval(a, b, lo=0, hi=1, alpha=A)
    d = a - b
    low, high = sps.t.interval(1 - A, len(d) - 1, loc=d.mean(), scale=sps.sem(d))
    assert got.method == "t"
    assert (got.low, got.high) == pytest.approx((low, high), rel=1e-12)


def test_the_t_interval_is_clipped_to_the_range() -> None:
    x = np.array([0.6] * 30 + [1.0] * 90)  # kurtosis 2.33, under 120 / 50
    alpha = 1e-9
    got = mean_interval(x, lo=0, hi=1, alpha=alpha)
    unclipped = x.mean() + sps.t.ppf(1 - alpha / 2, 119) * sps.sem(x)
    assert got.method == "t" and unclipped > 1.0
    assert got.high == 1.0 and got.low < x.mean()


# ----------------------------------------------------------------------------- the fallback


def reference_capital(x: np.ndarray, alpha: float, m: float) -> tuple[float, float]:
    """``(log K^+(m), log K^-(m))`` written out from ADR 0004 section 2, one step at a time."""
    n = len(x)
    total, sq, var_before = 0.0, 0.0, 0.25
    log_up = log_down = 0.0
    for t, v in enumerate(x, start=1):
        bet = math.sqrt(2.0 * math.log(2.0 / alpha) / (n * var_before))
        up = bet if m <= 0.0 else min(bet, BET_TRUNCATION / m)
        down = bet if m >= 1.0 else min(bet, BET_TRUNCATION / (1.0 - m))
        log_up += math.log(1.0 + up * (v - m))
        log_down += math.log(1.0 - down * (v - m))
        total += v
        mu = (0.5 + total) / (t + 1)
        sq += (v - mu) ** 2
        var_before = (0.25 + sq) / (t + 1)
    return log_up, log_down


def excluded(x: np.ndarray, alpha: float, m: float) -> bool:
    """``max(K^+, K^-) / 2 >= 1 / alpha``, from the reference capital."""
    return max(reference_capital(x, alpha, m)) >= math.log(2.0 / alpha)


def test_one_value_worked_by_hand() -> None:
    # n = 1, alpha = 1/2: the bet is sqrt(2 log 4 / (1/4)) = sqrt(8 log 4), about 3.33. For x = 1,
    # K^+(m) = 1 + 3.33 (1 - m) reaches 2 / alpha = 4 for m <= 1 - 3 / 3.33 (the bet is not
    # truncated there: 0.75 / m > 3.33 for m < 0.225), and K^- never exceeds 1.
    bet = math.sqrt(8.0 * math.log(4.0))
    low, high = betting_interval(np.array([1.0]), 0.5)
    assert low == pytest.approx(1.0 - 3.0 / bet, abs=1e-12) and high == 1.0
    low, high = betting_interval(np.array([0.0]), 0.5)  # the mirror image
    assert low == 0.0 and high == pytest.approx(3.0 / bet, abs=1e-12)


def zero_inflated(n: int, seed: int) -> np.ndarray:
    """Mostly 0.5 (a gap of 0 on the paired scale) with a few values anywhere in [0, 1]."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 0.5)
    k = rng.choice(n, max(1, n // 12), replace=False)
    x[k] = rng.uniform(0.0, 1.0, k.size)
    return x


def test_the_ends_match_a_brute_force_grid_and_sit_on_the_excluded_side() -> None:
    x = zero_inflated(60, seed=3)
    low, high = betting_interval(x, A)
    assert 0.0 < low < x.mean() < high < 1.0
    assert excluded(x, A, low) and excluded(x, A, high)  # each end is an excluded point
    step = 1e-3
    grid = np.arange(0.0, 1.0 + step / 2, step)
    out = np.array([excluded(x, A, float(m)) for m in grid])
    inside = (grid > low + 1e-9) & (grid < high - 1e-9)
    assert not out[inside].any()  # everything strictly between the ends is kept
    assert out[grid <= low].all() and out[grid >= high].all()  # and nothing outside them
    assert not excluded(x, A, low + 1e-9) and not excluded(x, A, high - 1e-9)  # the ends are tight


def test_all_equal_values_reach_inside_the_band_but_not_below_any_valid_floor() -> None:
    # 373 pairs with no difference at the primary level. Any interval valid for every distribution
    # on [-1, 1] must contain +-(1 - alpha^(1/n)): a mass p at -1 goes unseen with probability
    # (1 - p)^n. A C2-PERC loss needs the interval inside +-0.03.
    a = np.full(373, 0.95)
    got = paired_interval(a, a, lo=0.0, hi=1.0, alpha=PRIMARY)
    floor = 1.0 - PRIMARY ** (1.0 / 373)
    assert got.method == "betting" and got.estimate == 0.0
    assert got.low <= -floor and got.high >= floor
    assert -0.03 < got.low and got.high < 0.03
    assert got.low == pytest.approx(-got.high, abs=1e-12)  # symmetric for a symmetric sample


def test_it_is_narrower_than_v01s_sequence_at_the_registered_size() -> None:
    # Spread values (sd about 0.08): narrower. Values with almost no spread, like the dev gaps,
    # where v0.1's sequence sits on its range floor: less than half as wide.
    for x, ratio in ((blocked_values(), 1.0), (zero_inflated(373, seed=5), 0.5)):
        canon = canonical_order(x)
        new = betting_interval(canon, PRIMARY)
        old_low, old_high = betting_cs(canon, PRIMARY)
        assert new[1] - new[0] < ratio * (float(old_high[-1]) - float(old_low[-1]))


def test_the_mean_and_paired_routes_rescale_the_betting_interval() -> None:
    x = np.array([2.0, 3.0, 5.0, 4.0, 4.5])
    got = mean_interval(x, lo=1.0, hi=6.0, alpha=A)
    low, high = betting_interval((canonical_order(x) - 1.0) / 5.0, A)  # in the canonical order
    assert got.method == "betting"
    assert (got.low, got.high) == (1.0 + low * 5.0, 1.0 + high * 5.0)
    assert got.low <= x.mean() <= got.high
    a, b = np.array([0.9, 0.8, 1.0]), np.array([0.5, 0.7, 0.6])
    got = paired_interval(a, b, lo=0, hi=1, alpha=A)
    ca, cb = canonical_pairs(a, b)
    low, high = betting_interval((ca - cb + 1.0) / 2.0, A)
    assert (got.low, got.high) == (-1.0 + low * 2.0, -1.0 + high * 2.0)
    assert got.estimate == pytest.approx(float((a - b).mean()))


def test_the_betting_interval_refuses_bad_input() -> None:
    assert betting_interval(np.array([]), A) == (0.0, 1.0)
    for bad in ([1.5], [-0.1], [math.nan], [[0.5, 0.5]]):
        with pytest.raises(ValueError):
            betting_interval(np.array(bad), A)
    with pytest.raises(ValueError):
        betting_interval(np.array([0.5]), 1.0)


def coverage_draws() -> dict[str, tuple[float, object]]:
    """Distributions on [0, 1] with their means: rare events, a coverage-like spike, skew, a point.

    Each draw function takes a generator and a size.
    """

    def rare(rng: np.random.Generator, n: int) -> np.ndarray:
        return (rng.uniform(size=n) < 0.02).astype(np.float64)

    def spike(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.where(rng.uniform(size=n) < 0.05, 0.0, 0.99)

    def arcsine(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.beta(0.5, 0.5, n)

    def gaps(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.where(rng.uniform(size=n) < 0.97, 0.5, rng.uniform(size=n))

    def point(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.full(n, 0.3)

    return {
        "rare": (0.02, rare),
        "spike": (0.95 * 0.99, spike),
        "arcsine": (0.5, arcsine),
        "gaps": (0.5, gaps),
        "point": (0.3, point),
    }


def miss_count(name: str, n: int, alpha: float, reps: int, seed: int) -> int:
    mean, draw = coverage_draws()[name]
    rng = np.random.default_rng(seed)
    misses = 0
    for _ in range(reps):
        x = draw(rng, n)  # type: ignore[operator]
        low, high = betting_interval(canonical_order(x), alpha)
        misses += not (low <= mean <= high)
    return misses


def allowed_misses(alpha: float, reps: int) -> float:
    """The level plus three binomial standard deviations: a check, not a proof."""
    return alpha * reps + 3.0 * math.sqrt(alpha * (1.0 - alpha) * reps)


@pytest.mark.parametrize("name", ["rare", "spike", "arcsine", "gaps", "point"])
@pytest.mark.parametrize("n", [20, 120])
def test_the_betting_interval_covers_the_mean(name: str, n: int) -> None:
    reps = 400
    assert miss_count(name, n, 0.1, reps, seed=n) <= allowed_misses(0.1, reps)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rare", "spike", "arcsine", "gaps", "point"])
@pytest.mark.parametrize("n", [20, 120, 373])
@pytest.mark.parametrize("alpha", [0.1, PRIMARY])
def test_the_betting_interval_covers_the_mean_slow(name: str, n: int, alpha: float) -> None:
    reps = 2000
    assert miss_count(name, n, alpha, reps, seed=7 * n) <= allowed_misses(alpha, reps)


# ----------------------------------------------------------------------------- canonical order


def blocked_values(n: int = 373) -> np.ndarray:
    """Per-unit values in four subset blocks with different means and a heavy left tail.

    Like the review's probe (finding S1): the kurtosis guard sends them to the betting route.
    """
    rng = np.random.default_rng(17)
    sizes = (54, 137, 49, n - 54 - 137 - 49)
    means = (0.99, 0.85, 0.99, 0.82)
    parts = [np.clip(rng.normal(m, 0.02, k), 0.0, 1.0) for m, k in zip(means, sizes, strict=True)]
    x = np.concatenate(parts)
    x[rng.choice(n, 4, replace=False)] = 0.0  # the tail
    return x


def test_the_betting_mean_does_not_depend_on_the_order_of_the_values() -> None:
    x = blocked_values()
    assert not use_t(x)  # these values take the betting route
    raw = [betting_interval(v, PRIMARY) for v in (x, x[::-1])]
    assert raw[0] != raw[1]  # the betting interval alone does depend on the order
    want = mean_interval(x, lo=0, hi=1, alpha=PRIMARY)
    assert want.method == "betting"
    rng = np.random.default_rng(23)
    for order in (x[::-1], np.sort(x), *(rng.permutation(x) for _ in range(5))):
        assert mean_interval(order, lo=0, hi=1, alpha=PRIMARY) == want  # bit for bit


def blocked_pairs(n: int = 373) -> tuple[np.ndarray, np.ndarray]:
    """Pairs whose differences sit in four subset blocks with different means, and a heavy tail.

    The kurtosis guard sends the differences to the betting route, and the betting interval on
    them, read in the given order, moves with the order.
    """
    rng = np.random.default_rng(37)
    sizes = (54, 137, 49, n - 54 - 137 - 49)
    shifts = (0.0, 0.12, 0.0, 0.15)
    a = np.clip(rng.normal(0.95, 0.02, n), 0.0, 1.0)
    d = np.concatenate([rng.normal(s, 0.02, k) for s, k in zip(shifts, sizes, strict=True)])
    b = np.clip(a - d, 0.0, 1.0)
    tail = rng.choice(n, 4, replace=False)
    a[tail], b[tail] = 1.0, 0.0
    # four exact ties in a - b (binary fractions) with unequal a, so every sort key is used
    a[:4] = (0.5, 0.75, 0.875, 0.625)
    b[:4] = a[:4] - 0.125
    return a, b


def test_the_betting_paired_interval_does_not_depend_on_the_order_of_the_pairs() -> None:
    a, b = blocked_pairs()
    assert not use_t(a - b)  # these differences take the betting route
    raw = [betting_interval((x - y + 1.0) / 2.0, PRIMARY) for x, y in ((a, b), (a[::-1], b[::-1]))]
    assert raw[0] != raw[1]  # the betting interval alone does depend on the order
    want = paired_interval(a, b, lo=0, hi=1, alpha=PRIMARY)
    assert want.method == "betting"
    rng = np.random.default_rng(29)
    perms = [np.arange(a.size)[::-1], *(rng.permutation(a.size) for _ in range(5))]
    for perm in perms:
        assert paired_interval(a[perm], b[perm], lo=0, hi=1, alpha=PRIMARY) == want  # bit for bit


def test_the_canonical_order_keeps_the_values_and_their_pairs_and_is_not_sorted() -> None:
    x = blocked_values()
    canon = canonical_order(x)
    assert np.array_equal(np.sort(canon), np.sort(x))  # the same multiset
    assert not np.all(np.diff(canon) >= 0)  # not left sorted
    assert np.array_equal(canonical_order(x[::-1]), canon)
    rng = np.random.default_rng(31)
    a, b = rng.uniform(size=50), rng.uniform(size=50)
    ca, cb = canonical_pairs(a, b)
    assert sorted(zip(ca, cb, strict=True)) == sorted(zip(a, b, strict=True))  # pairs kept
    perm = rng.permutation(50)
    pa, pb = canonical_pairs(a[perm], b[perm])
    assert np.array_equal(pa, ca) and np.array_equal(pb, cb)


def test_empty_inputs_give_the_whole_range() -> None:
    e = mean_interval([], lo=0, hi=1, alpha=A)
    assert (e.low, e.high, e.n, e.estimate) == (0, 1, 0, None)
    p = paired_interval([], [], lo=0, hi=1, alpha=A)
    assert (p.low, p.high) == (-1, 1)


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="within"):
        mean_interval([0.5, 1.5], lo=0, hi=1, alpha=A)
    with pytest.raises(ValueError, match="within"):
        mean_interval([math.nan], lo=0, hi=1, alpha=A)
    with pytest.raises(ValueError, match="alpha"):
        mean_interval([0.5], lo=0, hi=1, alpha=0.0)
    with pytest.raises(ValueError, match="lo < hi"):
        mean_interval([0.5], lo=1, hi=1, alpha=A)
    with pytest.raises(ValueError, match="length"):
        paired_interval([0.1, 0.2], [0.1], lo=0, hi=1, alpha=A)


# ----------------------------------------------------------------------------- binomial


def test_binomial_is_clopper_pearson() -> None:
    got = binomial_interval(690, 707, 0.05 / 6)
    assert (got.low, got.high) == clopper_pearson(690, 707, 0.05 / 6)
    assert got.method == "clopper-pearson" and got.estimate == 690 / 707
    assert binomial_interval(0, 0, A).estimate is None


def test_interval_round_trip() -> None:
    iv = mean_interval(alternating(120), lo=0, hi=1, alpha=A)
    assert Interval.from_dict(iv.to_dict()) == iv
    with pytest.raises(ValueError, match="method"):
        Interval.from_dict({**iv.to_dict(), "method": "z"})
