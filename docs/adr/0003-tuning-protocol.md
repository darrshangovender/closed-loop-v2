# 0003: how the free choices are tuned, written before any tuning

Status: accepted, 10 November 2026, before any tuning run. It implements 0001 section 5 ("tuned on dev ... then frozen in `benchmarks/tuned.json`") and section 7 ("`h` is tuned on dev against a stated trade-off"). Only dev units are used: 195 units across the four subsets. Tuning code asks for dev units only and uses no calib value (0001 section 4); the loader parses the calib units' rows together with the dev rows of each training file and keeps them in a cache that a dev request does not return (review D, finding N1). Eval units and NASA's test files stay closed.

## 1. Fixed in advance, not tuned

| Choice | Value | Why |
|---|---|---|
| Two-sided level | 0.90 (0.05 each side) | 0001 section 5; the same as v0.1 |
| Operating conditions `k` | 6 | the six settings clusters in C-MAPSS; on dev the clusters are at most 0.009 across, with centroids at least 7.0 apart, at every seed tried |
| ACI step `gamma`, pool window | 0.02, 500 | v0.1's V2L, not retuned on new data |
| Fleet size `K` | 20 engines at once | a fleet, not one engine at a time, so labels resolve while other lives run; 373 eval lives give about 19 generations. Matters only for V2L and V3L |
| Orders `R` | 20 | per-unit values are averaged over orders; 20 keeps `--full` inside its time budget |
| VW fixed-point steps | 10 | the report shows each step's quantiles and whether the last step moved them |
| HGB settings | `DEFAULT_PARAMS` (300 iterations, learning rate 0.05, 31 leaves, 50 per leaf) | kept fixed so the grid below stays small; one thread |

## 2. The estimator: `window`, `cap`, `s_min`

- **Grid:** `window` in {15, 30, 50}, `cap` in {125, 150} and `s_min` in {2, 5}. That is 12 settings.
- **Predictions:** for each setting, unit-grouped 5-fold cross-fitting on dev gives every dev unit predictions from an estimator not fitted on any of its rows.
- **V1:** on those out-of-fold predictions, V1's quantiles come from one uniformly drawn row per unit (195 rows).
- **Criterion:** the mean interval width `s (q_hi - q_lo)` over every row of every dev unit. The smallest wins. A tie within 1% goes to the smaller `window`, then the smaller `cap`, then the larger `s_min`.
- **Reporting:** every setting's width, its pooled coverage and the RMSE of `m` against `min(RUL, cap)` are published in `docs/TUNING.md`.

The pooled coverage is not part of the criterion. Conformal calibration sets the coverage, so width is the quantity left to choose.

## 3. The policy threshold `h`

- **Replay:** with the chosen estimator, the closed loop is replayed on the dev out-of-fold predictions with V1. The fleet and orders are as in section 1.
- **Grid:** `h` in {0, 5, 10, 15, 20, 25, 30} cycles.
- **Trade-off:** the smallest mean remaining life at replacement, subject to an unplanned-failure fraction of at most 0.05 (units that reach failure unreplaced, averaged over orders).
- **If no grid value meets the limit,** the one with the lowest unplanned-failure fraction is chosen, and that is written down.

## 4. The decision stratum `M`

`M` is the smallest value in 5-cycle steps such that at least 90% of dev units have a checkpoint with `m <= M` before replacement in every order. It is measured in the closed-loop replay of section 3 at the chosen `h`. This puts the stratum around the policy's decisions while keeping exclusions near one unit in ten.

## 5. What tuning may and may not touch

- **It writes:** `benchmarks/tuned.json`, holding the chosen values, the grid, every setting's numbers and the dev unit list's hash, and `docs/TUNING.md`, rendered from it.
- **It may not change:** the claims, bands, levels or verdict rules in 0001 section 9 and `closed_loop_v2/verdicts.py`, the role formula, or anything in section 1.
- **Selection is mechanical:** if a result looks wrong, the code or this protocol is fixed and the whole tuning is rerun, with the reason added to this file. Picking a different grid value by hand is not allowed.
- **Afterwards:** `tuned.json` is frozen with the pre-registration.

## 6. Implementation details, fixed before the first run (10 November 2026)

`scripts/tune.py` implements sections 2 to 4. These details were written down before it ran on any real unit:
- **`M` grid:** 5, 10, 15 and so on up to the chosen `cap`. If no value reaches 90%, the script stops with an error instead of picking one.
- **`h` choice:** ties in mean remaining life go to the smaller `h`. A value with no replacement has no mean remaining life and cannot qualify. In the fallback, ties in the failure fraction go to the smaller `h`.
- **Out-of-fold predictions:**
  - The outer 5 folds hold whole dev units. Their own seeded stream is the fit seed plus a fixed key.
  - Each outer fold fits a full `Estimator`, including its own inner cross-fitting for `s`, on the other four folds.
- **`s_min`:** this is a floor applied in `Estimator.predict` only, so the two `s_min` values share one fit. `tests/test_tune.py` checks that raising the floor at prediction gives the same predictions as fitting with it.
- **Seeds:** the key `"tune"` takes the place of the pre-registration hash (`closed_loop_v2.seeds`), with the labels `fit-nominal` for the fits and folds and `calib-draw` for V1's drawn rows. So tuning shares no seed with `--quick` or the full run.
- **Order independence:** V1's quantiles never change, so the closed-loop replays of sections 3 and 4 are static (`replay_static`). Each life's outcome is then the same in every order, and "averaged over orders" and "in every order" reduce to one replay.
- **Outputs:** `benchmarks/tuned.json` carries the chosen values under `estimator`, `h` and `stratum_m`, the text fields `tuned_on`, `criterion`, `script` and `notes`, and the full grids under `record`. `docs/TUNING.md` is rendered from it. `benchmarks/run.py` reads `stratum_m` from `tuned.json` in place of its provisional design value.

## 7. First tuning run stopped; the truth is capped and `cap` is fixed (10 November 2026)

**What happened.** The first run of `scripts/tune.py` on the 195 dev units (its log is kept outside this repository, as `tune-run1-uncapped.log` in the working folder's run directory) finished the estimator grid and the `h` grid. It then stopped by design at step 3: no `M` up to the cap reached 90%. Its numbers:
- **Widths:** mean interval widths of 151.2107 to 193.8741 cycles across the 12 settings, with pooled coverage 0.8883 to 0.9126.
- **Choice:** the criterion chose window 30, cap 150, s_min 5.
- **`h` grid:** every `h` from 0 to 30 gave an unplanned-failure fraction of 0.0000. The mean remaining life at replacement ran from 119.8923 cycles (`h` = 0) to 185.1282 (`h` = 30).

The policy was replacing every unit with about 120 cycles or more still to run, so the closed loop was degenerate. That is not a test of anything.

**Why.** The truth was the uncapped remaining life:
- Early in a C-MAPSS life the engine is healthy, and its remaining life cannot be told from the sensors. So `m` saturates at the cap while the truth is often far above it.
- The scale model `s` learned those large upside errors.
- The symmetric score `(RUL - m) / s` then gave the lower bound `m + s q_lo` the same large scale, so lower bounds fell to `h` or below while many cycles remained.

**The fix, made before any calib, eval or test value was used or looked at:**
- **Capped truth.** The truth is `min(remaining life, cap)` everywhere: the estimator's two targets, the conformal scores, VW, the labels, every coverage metric and the external check (ADR 0001, amendment of the same date). This is the usual convention for C-MAPSS.
  - A lower bound `L <= cap` covers `min(RUL, cap)` exactly when it covers `RUL`, so the lower-bound claims mean the same thing for every bound at or below the cap. With `h <= 30` that includes every bound near a decision.
  - A replaced unit known to have survived at least `cap` cycles beyond a row has an exactly known truth there (`cap`), and is labelled `HIT` or `MISS` accordingly.
  - The remaining life at replacement, a cost, stays uncapped.
- **`cap` fixed at 125, not tuned.** With a capped truth, widths under different caps measure different quantities: a smaller cap is narrower by construction, so width cannot choose the cap. 125 cycles is a common choice in C-MAPSS work and was the grid's smaller value. The estimator grid of section 2 becomes `window` in {15, 30, 50} x `s_min` in {2, 5}, 6 settings. Everything else in sections 2 to 6 is unchanged.
- **Rerun.** The whole tuning is rerun from the start. Nothing from the first run is carried over.

## 8. Tuning rerun after review D (10 November 2026, before the freeze)

**Why.** Review D (finding S1) showed that where `m` is above the cap, `m + s q_lo` could exceed 125, and such a lower bound was then counted as a miss of the capped truth even when it covered the remaining life. Both ends of every interval are now limited to the cap (ADR 0001 section 5, second amendment). Tune's criterion, the mean unlimited width `s (q_hi - q_lo)`, does not change, and neither does V1's draw, but the trigger and the pooled-coverage record use the bounds, so tuning was rerun from the start on the final code, as section 7 did. Nothing from the earlier run was carried over.

**What changed.** The rerun (`scripts/tune.py`, 608 s, the same 195 dev units, dev list SHA-256 unchanged) chose the same values: window 50, `s_min` 2, `h` = 5 and `M` = 35, with the same V1 quantiles (-2.279140370793755, 2.1601501985273517). Every mean width, every RMSE, the whole `h` grid and the whole `M` grid are identical. Only the pooled interval coverage moved, on the three `s_min` = 2 settings, where some lower bounds had been above the cap (now limited to 125, they cover rows whose remaining life is 125 or more; limiting an upper bound cannot change coverage of a truth that is at most 125): 0.913596105739977 to 0.9140468302796746 (window 15), 0.9117255989002321 to 0.9121537872129448 (window 30) and 0.9240078426069908 to 0.9251346539562346 (window 50). Coverage is not part of the criterion.

**The chosen window is at the edge of the grid** (review D, finding N4). With `s_min` 2 the mean widths are 48.48247929622987 at window 15, 41.12953283411823 at 30 and 38.23938716681448 at 50: still falling at the largest window tried. The rule of section 2 was followed and is not changed; a wider window might be narrower still, and v0.2 does not test it. `docs/TUNING.md` is rendered from `tuned.json` by `tune.py`, so this note is kept here rather than in the generated file.
