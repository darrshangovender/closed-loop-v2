# 0004: the fixed-sample betting interval, chosen before the freeze

Status: accepted, 10 November 2026. Written before the freeze, before any calib, eval or test value was used or looked at (the loader parses calib rows together with dev rows, and the data tests hash the held-out files' bytes; pre-registration section 12). It amends 0001 section 11 and 0002 section 4 (`stats.py`). The evidence behind it is the reachability check on dev units only.

## 1. What was wrong

0001 section 11 says that v0.2 "uses fixed-sample intervals, not v0.1's confidence sequences". Its fallback for rows that fail the t guard was nevertheless v0.1's `betting_cs` (and `paired_betting_cs`), and those are confidence sequences:
- They are valid at every sample size at once, as a running intersection, and v0.2 reads them at one fixed `n`.
- That uniformity costs width. The half-width has a floor set by the range of the values, not by their spread (v0.1's note A16).

The reachability check (`scripts/reachability.py`, `docs/REACHABILITY.md`, 10 November) resampled the per-unit values of a dev-only `--quick` run to the registered sizes:
- Every C2-PERC and C2-SELECT row took the betting route. The dev values are mostly near zero with a few large ones, with sample kurtosis 19.15 (V1, pooled), 16.69 (V2L, pooled) and 23.93 (the C2-SELECT difference), against the guard's 7.46 at `n = 373`.
- The median half-width at `n = 373` was 0.0600, above the C2-PERC band of 0.03. So a C2-PERC loss, an interval inside the band, could not be reached at all.
- A C2-PERC win needed a pooled gap of -0.090 or beyond, and a C2-SELECT win a difference of at least 0.080.

0001 section 12 requires "a check that every label is reachable" before the freeze. The check failed on that requirement, and the cause was the interval, not the claims.

## 2. Decision

The fallback becomes the hedged-capital betting interval of Waudby-Smith and Ramdas ("Estimating means of bounded random variables by betting", JRSS B 86(1), 2024), with predictable plug-in bets tuned for the fixed sample size. It is implemented in `closed_loop_v2/stats.py`; v0.1's code is not changed.

For values `x_1, ..., x_n` in `[0, 1]`, put into the canonical order that `stats.py` already uses (sorted, then a fixed permutation from the key `(4099, 1)` and `n`):
- **Running estimates:** `mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)` and `var_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)`, with `var_0 = 1/4`.
- **Bets:** `lambda_t = sqrt(2 log(2 / alpha) / (n var_{t-1}))`. Each bet uses `x_1, ..., x_{t-1}` only.
- **Truncation:** for a candidate mean `m`, `lambda_t^+(m) = min(lambda_t, c / m)` and `lambda_t^-(m) = min(lambda_t, c / (1 - m))`, with `c = 3/4`.
- **Capital:** `K^+(m) = prod_t (1 + lambda_t^+(m) (x_t - m))` and `K^-(m) = prod_t (1 - lambda_t^-(m) (x_t - m))`.
- **The interval:** `m` is excluded when `max(K^+(m), K^-(m)) / 2 >= 1 / alpha`.
  - `K^+` never increases with `m` and `K^-` never decreases, so the kept set is an interval `[L, U]`.
  - Each end is found by bisection and returned at the excluded end of its final bracket, so rounding can only widen the interval.

Means of values in `[lo, hi]` are rescaled to `[0, 1]` and back, and paired differences in `[-w, w]` likewise, as before. The t route, its guard, Clopper-Pearson for C2-EXT, the canonical order and the treatment of an empty interval (inconclusive) are unchanged.

## 3. Why it is valid

Suppose the `n` values are independent with a common mean `mu`. Given the fitted estimator and the calibration, the per-unit values of the eval units are: the units' roles come from a keyed hash, not from their data. This is the assumption that the t route and v0.1's interval already make.
- At `m = mu` every factor of `K^+` and `K^-` has conditional mean 1, given the earlier values, because the bet is fixed before `x_t` is seen.
- Every factor is at least `1 - c = 1/4`, so it is positive.
- `K^+(mu)` and `K^-(mu)` are therefore non-negative martingales that start at 1. By Markov's inequality each reaches `2 / alpha` with probability at most `alpha / 2`, so `mu` is excluded with probability at most `alpha`.
- This holds for every `n` and every distribution on `[0, 1]`. Tuning the bets for `n` changes the width, not the validity.

**The order.** The martingale argument needs an order fixed independently of the values. The canonical order is a sort followed by one fixed permutation, chosen by a key written down before any data. A uniformly random order applied to the sorted values would make the argument exact. The fixed key is the same for every analysis, and no one chose it after seeing data. The same caveat applied to v0.1's interval under the canonical order, and the review of packet C chose this order (finding S1) so that a result cannot depend on how units are listed.

**Amendment, 10 November 2026, after review D (finding S2), before the freeze.** Two conditions of the argument above are not met, and the statement is scoped accordingly:
- **The order is not random.** Under the canonical order `x_t` is a fixed order statistic of the sample put through one fixed permutation, so the bets are not fixed before a random `x_t` is seen in the sense the martingale argument needs. The argument as written does not cover this order; the paragraph above that called the fixed key a stand-in for a random draw gave an analogy, not a proof. This applies to every row that takes the betting route.
- **V2L's and V3L's per-unit values are not independent.** For V1 and VW a unit's bounds depend only on that unit, given the estimator and the calibration. For V2L and V3L they also depend on the shared `alpha_t`, which every earlier unit in the queue has moved. On dev units at the tuned point the correlation between queue position and a unit's `Gap_B` was -0.4007 (review D, `probe_v2l_coupling.py`). This applies to the V2L and V3L rows whichever interval rule they take, the t route included.
- **What is claimed.** For the rows these touch the stated level is nominal, not proven. The pre-registration (section 7), 0001 section 11 and the report say so. Review D's coverage simulation on independent synthetic values found misses far below `alpha` in the canonical, a random and the iid order alike (for example 0.0080, 0.0037 and 0.0030 at `alpha` = 0.1, n = 74), so no coverage failure was shown; that is evidence, not a proof, and it does not cover the coupling.

## 4. What it can show at these sizes

The reachability check is rerun with this interval after it is implemented and tested. Its numbers are recorded in `docs/REACHABILITY.md` and the pre-registration's section 7, not here.

## 5. What does not change

- The claims, the band (0.03), the C2-SELECT margin (0.02) and floor (0.92), the C2-EXT range, the levels (0.05 / 6 primary, 0.05 secondary) and the verdict rules in `closed_loop_v2/verdicts.py`.
- The tuned values. The interval does not enter tuning, so tuning is not rerun for this change. (Tuning was rerun later for another reason, review D's finding S1: ADR 0003 section 8.)
- v0.1's code, which stays untouched. `closed_loop.stats.betting_cs` is no longer called by v0.2, and Clopper-Pearson is still imported from it.

## 6. Tests

`tests/test_stats.py` checks the following:
- A one-value case worked by hand: `x = (1)` at `alpha = 1/2` gives `L = 1 - 3 / sqrt(8 log 4)` and `U = 1`.
- Agreement with a brute-force evaluation of the capital on a fine grid.
- That the ends sit on the excluded side.
- A floor for all-equal values.
- That the interval is narrower than v0.1's sequence at the registered size.
- Order invariance under the canonical order.
- Refusal of bad input.
- A coverage simulation on rare-event, skewed and point-mass distributions. The slow marker runs the larger version.
