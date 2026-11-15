# Pre-registration

Repository: `closed-loop-v2`. The data are NASA's C-MAPSS turbofan trajectories, **simulated by NASA's C-MAPSS model, not field data**.

**How this file becomes binding:**
- `benchmarks/run.py --freeze` hashes it: SHA-256 over the bytes after CRLF becomes LF and trailing spaces are stripped, the rule of `closed_loop.audit.prereg`.
- `--freeze` writes the ledger's first `freeze` entry, with that hash, the hash of `benchmarks/tuned.json`, the code hash (`closed_loop_v2/` and `benchmarks/run.py`), the hash of `data/MANIFEST.json` and the library versions. These include v0.1's version and the hash of its code, which the code hash does not cover; v0.1 supplies the t guard, Clopper-Pearson, the ledger and the guard.
- Until that entry exists this file binds nothing, and the held-out data cannot be read (`HeldoutAccess`, ADR 0001 section 4).
- Only the first freeze entry counts. A changed document cannot be frozen again to reopen the data.
- The run's seeds are derived from this file's hash, so they do not exist before the freeze.

## 1. Claims and what would count against them

| ID | Question | Comparison | Primary |
|---|---|---|---|
| C2-PERC (4 verdicts) | Does open-loop coverage carry over to the closed loop? | `Gap_B^lo` = closed-loop minus open-loop lower-bound coverage, paired by unit, for V1 and for V2L, pooled and in the decision stratum | yes |
| C2-SELECT | Does calibrating on what the policy lets through restore closed-loop coverage? | closed-loop `c^lo`, VW against V1, paired by unit | yes |
| C2-EXT | Does V1's calibration carry over to NASA's own test units at their last observed cycle? | lower-bound coverage on all 707 test units against 0.95 | yes |
| C2-CENSOR | Does online recalibration on policy-censored labels add to the gap? | `Gap_B(V2L) - Gap_B(V1)` and `Gap_B(V3L) - Gap_B(V1)`, paired | secondary |
| C2-SHIFT | Under a fault mode the calibration never saw, does online recalibration help? | shifted condition, closed-loop `c^lo`, V2L and V3L against V1 | secondary |

The thesis C2-PERC tests is v0.1's: open-loop calibration does not carry over to the closed loop.
- **For it:** an interval that excludes the band `[-0.03, +0.03]`.
- **Against it:** an interval inside the band.
- Section 7 states which of these each row can reach at the registered sizes.
- C2-PERC is the feedback-loop question v0.1's C-PERC asked, now asked on real degradation trajectories rather than a synthetic plant.

## 2. Data, roles and counts

- **Subsets:** FD001 to FD004 (ADR 0001 section 3).
  - Training units run to failure: 100, 260, 100 and 249.
  - NASA's test units stop before failure: 100, 259, 100 and 248, with the true remaining life at the last cycle in `RUL_FD00x.txt`.
  - The dataset's readme swaps FD004's pair; the loader checks the training count, and the first held-out read checks the test count.
- **Roles:** set by a keyed hash of subset and unit number, independent of the rows (ADR 0001 section 4).

| Role | FD001 | FD002 | FD003 | FD004 | Total | Use |
|---|---|---|---|---|---|---|
| dev | 25 (12.8%) | 64 (32.8%) | 35 (17.9%) | 71 (36.4%) | 195 | fitting and every tuned choice |
| calib | 21 (14.9%) | 59 (41.8%) | 16 (11.3%) | 45 (31.9%) | 141 | conformal calibration and VW's replay |
| eval | 54 (14.5%) | 137 (36.7%) | 49 (13.1%) | 133 (35.7%) | 373 | held out until the freeze, read once |
| NASA test | 100 | 259 | 100 | 248 | 707 | held out until the freeze (C2-EXT) |

- **Subset mix:** the mix differs between calib and eval by chance, for example FD002 is 41.8% of calib and 36.7% of eval. Roles were not stratified by subset. The hash makes each unit's role independent of its data, so calib and eval units are exchangeable draws from the same pool of units. A per-subset breakdown is reported (section 8).
- **Shifted condition:**
  - The estimator is fitted on FD001 and FD002 dev units (89) and calibrated on FD001 and FD002 calib units (80).
  - It is evaluated on FD003 and FD004 eval units (182).

## 3. Quantities and levels

- **Checkpoint:** every cycle `t` of every unit. Features use rows up to `t` only.
- **Truth:** `min(RUL_t, cap)`, where `RUL_t = failure cycle - t` is the remaining life and `cap` = 125 cycles, fixed, not tuned (ADR 0001 section 5 and ADR 0003 section 7, both amended 10 November before any calib, eval or test value was used or looked at).
  - The estimator's targets, the scores, VW, the labels, every coverage metric and the C2-EXT check all use this capped truth.
  - The cycles left at replacement, a cost, are uncapped.
  - The report and the README block state the capped truth next to the data note.
- **Interval:** `[min(m + s q_lo, 125), min(m + s q_hi, 125)]`, two-sided level 0.90 with 0.05 on each side. **Nominal interval coverage 0.90; nominal lower-bound coverage 0.95.**
  - Both ends are limited to the cap (ADR 0001 section 5, amended after review D). The capped truth never exceeds 125, so the limit only removes the part of an interval where the truth cannot be.
  - Every lower bound is therefore at or below 125, and lower-bound coverage of the capped truth equals lower-bound coverage of the remaining life at every checkpoint, C2-EXT's included.
  - The trigger, VW's reach and the labels use the limited bounds.
- **Decision stratum:** checkpoints with `m <= M`, `M` = 35 cycles (tuned, ADR 0003 section 4).

## 4. Methods and fixed values

- **Estimator:** six operating conditions by k-means on dev settings; per-condition z-scores; window mean and slope over `window` = 50 cycles (tuned), plus the cycle count.
  - Two `HistGradientBoostingRegressor` models, fitted on one thread: `m` on `min(RUL, cap)`, and `s` on `|min(RUL, cap) - m|` from unit-grouped 5-fold cross-fitting, floored at `s_min` = 2 cycles (tuned).
  - Boosting settings: 300 iterations, learning rate 0.05, 31 leaves, at least 50 rows per leaf.
- **V1:** split conformal on one uniformly drawn checkpoint per calibration unit.
- **VW:** selection-aware weighted conformal (Tibshirani, Barber, Candès and Ramdas 2019).
  - The draw is uniform over the checkpoints the policy reaches on each calibration unit.
  - There are 10 fixed-point steps from V1's quantiles, with the same per-unit uniforms at every step. The step history is published.
  - On the dev-only `--quick` run the steps alternated between two pairs of quantiles and had not settled after 10 steps. The report states how the steps ended, from the published history:
    - **settled:** the last step did not move the quantiles;
    - **two-cycle:** the last step returned them to the pair of two steps earlier. Each step is the same deterministic map with the same per-unit uniforms, so the steps then alternate between those two pairs for ever, and which pair the run ends on depends only on the parity of the step count;
    - **unsettled:** anything else.
  - For a two-cycle, the run also replays the other pair (the one before the last) in the closed loop, and C2-SELECT is judged with both pairs (section 8). An unsettled VW makes C2-SELECT inconclusive, since its quantiles would then depend on where the iteration stopped.
- **V2L:** adaptive conformal inference with one update per resolved life (Gibbs and Candès 2021), `gamma = 0.02` and a pool of 500. Censored labels count as hits.
- **V3L:** as V2L, except that labels with an unknown outcome are skipped.
- **Policy (bound rule):** replace at the first checkpoint whose lower bound is at most `h` = 5 cycles (tuned).
- **Fleet and orders:** `K = 20` engines at once and `R = 20` random orders. Open and closed loop use the same orders.
- **Seeds:** labels `fit-nominal`, `fit-shifted`, `calib-draw` and `orders`, each derived from this file's hash (`closed_loop_v2/seeds.py`).
  - V1, VW, V2L and V3L share `calib-draw`.
  - One `calib-draw` and one `orders` seed serve both conditions.
  - V2L and V3L draw from the same adaptive stream in every order and in both loops.
  - These are common random numbers: they tighten the paired comparisons, and no registered comparison crosses the two conditions.

- **Registered values:** the table printed by `python benchmarks/run.py --registered-table`, pasted here verbatim. `--freeze` refuses this document unless that exact table is in it. The table is recomputed and checked against this document again at every `--full`, before any held-out read (a mismatch refuses the run), and at every `--check` (a mismatch fails it). So the numbers here and in the code agree at the freeze and at each of those checks.
  - The tuned values come from `scripts/tune.py` on the 195 dev units (ADR 0003). The record of every setting is `benchmarks/tuned.json`, rendered in `docs/TUNING.md`.

| Value | Registered | From |
|---|---|---|
| `k` | 20 | `DESIGN` in `benchmarks/run.py` |
| `r` | 20 | `DESIGN` in `benchmarks/run.py` |
| `stratum_m` | 35.0 | `benchmarks/tuned.json` |
| `alpha` | 0.1 | `DESIGN` in `benchmarks/run.py` |
| `gamma` | 0.02 | `DESIGN` in `benchmarks/run.py` |
| `aci_window` | 500 | `DESIGN` in `benchmarks/run.py` |
| `vw_steps` | 10 | `DESIGN` in `benchmarks/run.py` |
| `n_folds` | 5 | `DESIGN` in `benchmarks/run.py` |
| `h` | 5.0 | `benchmarks/tuned.json` |
| `estimator.window` | 50 | `benchmarks/tuned.json` |
| `estimator.cap` | 125.0 | `benchmarks/tuned.json` |
| `estimator.s_min` | 2.0 | `benchmarks/tuned.json` |
| `estimator.max_iter` | 300 | `benchmarks/tuned.json` |
| `estimator.learning_rate` | 0.05 | `benchmarks/tuned.json` |
| `estimator.max_leaf_nodes` | 31 | `benchmarks/tuned.json` |
| `estimator.min_samples_leaf` | 50 | `benchmarks/tuned.json` |
| `PERC_BAND` | 0.03 | `closed_loop_v2/verdicts.py` |
| `SELECT_MARGIN` | 0.02 | `closed_loop_v2/verdicts.py` |
| `SELECT_FLOOR` | 0.92 | `closed_loop_v2/verdicts.py` |
| `EXT_LO` | 0.92 | `closed_loop_v2/verdicts.py` |
| `EXT_HI` | 0.98 | `closed_loop_v2/verdicts.py` |
| `PRIMARY_ALPHA` | 0.008333333333333333 | `closed_loop_v2/verdicts.py` |
| `SECONDARY_ALPHA` | 0.05 | `closed_loop_v2/verdicts.py` |
| `SPREAD_MIN_PAIRS` | 100 | `closed_loop.stats.confseq` (v0.1) |
| `SPREAD_KURTOSIS_DIVISOR` | 50.0 | `closed_loop.stats.confseq` (v0.1) |
| `BET_TRUNCATION` | 0.75 | `closed_loop_v2/stats.py` |
| `ORDER_KEY.0` | 4099 | `closed_loop_v2/stats.py` |
| `ORDER_KEY.1` | 1 | `closed_loop_v2/stats.py` |

## 5. The replay and the labels

- **Open loop:** no replacements. Every life ends in failure, and the online methods see every exact label.
- **Closed loop:** a unit replaced at `tau` contributes checkpoints up to `tau`. With `b = tau - t`, the cycles it is known to have survived past `t`, its labels are:
  - `MISS` if `b > hi`;
  - `LOWER_HIT` if `lo <= b <= hi`;
  - `UNKNOWN` if `b < lo`.
  - Once `b >= cap`, the capped truth is known exactly (`cap`): the label is `HIT` if `lo <= cap <= hi` and `MISS` otherwise. This rule overrides the three above.
  - A row whose lower bound is above `cap` is a certain `MISS`, since the capped truth is at most `cap`. With bounds limited to the cap (section 3) this does not arise in the run; the label code keeps the rule for any caller that passes unlimited bounds.
- **Failures:** a unit that reaches its last cycle unreplaced is an unplanned failure, with exact labels. A trigger on the last cycle itself counts as a replacement.
- **Online updates:** the online methods update when a life ends, in slot order.
- **Truth:** coverage metrics use the capped truth of section 3, computed from the dataset's true remaining life, not the labels.
- **No physics feedback:** a replacement does not change any other engine. That is why VW's covariate-shift argument applies here (ADR 0001 sections 6 and 13).

## 6. Metrics

- **Per unit:** `c_u` and `c_u^lo`, the interval and lower-bound coverage over the checkpoints the unit contributes, averaged over the 20 orders.
  - Fleet values weight units equally.
  - A unit enters the stratum only if it has stratum checkpoints in every order. Exclusions are counted and reported.
- **Primary, one per claim:** as in section 1.
- **Descriptive, with no decision rule:**
  - the unplanned-failure fraction;
  - the mean remaining life at replacement, uncapped;
  - the mean finite width, and the number of checkpoints with an infinite side;
  - VW's convergence;
  - a per-subset breakdown.

## 7. Statistical method

- **Means of per-unit values and paired differences:** Student-t interval, used only when `n >= 100`, the values are not all equal, and the sample kurtosis is at most `n / 50`. These are v0.1's guard and constants, imported from `closed_loop.stats.confseq`.
  - Otherwise the fixed-sample betting interval of ADR 0004 is used (`betting_interval` in `closed_loop_v2/stats.py`). It is the hedged-capital interval of Waudby-Smith and Ramdas (JRSS B 86(1), 2024), with predictable bets tuned for `n` and truncation 3/4.
  - It is valid at every size for independent values with one mean, fed in a uniformly random order.
  - Until 10 November this route used v0.1's confidence sequence; ADR 0004 says why it was replaced before the freeze.
  - Its result depends on the order the values are fed in, so the values are first put into a canonical order that depends only on their multiset, then permuted by a fixed key (`closed_loop_v2/stats.py`). This removes any dependence on the subset order in which units are listed.
  - The report names the rule used for each row.
- **Scope of the stated level** (review D, finding S2; ADR 0004 section 3). Two conditions of that validity are not met here:
  - The betting interval is fed the values in the fixed canonical order above, not a random one, and the martingale argument as written does not cover a fixed order. This applies to every row that takes the betting route.
  - V2L's and V3L's per-unit values are coupled through the shared `alpha_t` that every earlier unit in the queue has moved, so they are not independent, whichever rule is used. On dev units at the tuned point the correlation between queue position and a unit's `Gap_B` was -0.4007. This applies to the two V2L rows of C2-PERC and to the secondary V2L and V3L rows.
  - For the rows these touch, the level `0.05 / 6` is nominal, not proven. The report says so.
- **The binomial C2-EXT:** exact Clopper-Pearson interval.
- **Levels:** the six primary verdicts at `0.05 / 6`, two-sided. Secondary rows at 0.05, descriptive.
- **What a bound can show at these sizes:** `scripts/reachability.py`, recorded in `docs/REACHABILITY.md` and `benchmarks/reachability.json`.
  - **The inputs are made up from dev units only,** at two operating points. They test the rules and the sizes; they are not results.
    - *Quick:* the per-unit values of the 35 evaluation units of a `--quick` run, which re-splits dev units and fits its own estimator.
    - *Dev point:* all 195 dev units at the tuned point (`scripts/dev_point.py`), from tune's out-of-fold predictions, split into two seeded halves, each evaluated through `run.py`'s own evaluation with the methods calibrated on the other half. Its V1 quantiles reproduce `tuned.json`'s.
    - Each point's values are resampled 20 times to the registered sizes and run through this interval and verdict code at `0.05 / 6`.
  - **Two ways a gap is placed:** uniformly (every unit moves by the same amount) and concentrated (the whole gap carried by 10% of the units, each floored at 0). The dev values of V1 and of the C2-SELECT difference are mostly near zero with a few large ones, so a real gap may be concentrated, and the concentrated thresholds are the ones to read. Values lie in `[0, 1]`, so a concentrated shift can move the mean by at most the carriers' share, `ceil(0.10 n) / n`, a little over 0.10.
  - **The route:** every V1 pooled C2-PERC row and every C2-SELECT difference took the betting route. Sample kurtosis 19.15 (quick) and 35.21 (dev point) for V1 pooled, 23.93 and 19.29 for the C2-SELECT difference, against the guard's 7.46 at `n = 373`. V2L pooled took the betting route at quick (kurtosis 16.69) and the `t` route at the dev point (kurtosis 6.16).
  - **C2-PERC, pooled:**
    - The median half-width at `n = 373` is 0.0197 (quick, V1), 0.0197 (quick, V2L), 0.0199 (dev point, V1) and 0.0058 (dev point, V2L).
    - At a gap of 0 every resample gave a loss, at both points. At a gap of -0.03 every resample gave inconclusive.
    - The smallest gap tried that won in every resample: V1 -0.055 uniform and -0.060 concentrated at quick, -0.055 and -0.060 at the dev point; V2L -0.055 and -0.060 at quick, -0.040 and -0.040 at the dev point.
  - **C2-PERC, stratum:** the stratum sizes are whatever the evaluation units give, and are reported with the exclusions.
    - At quick only 7 (V1) and 11 (V2L) of the 35 units had stratum checkpoints in both loops, because the policy often replaced a unit before `m` reached 35. Resampled to 74 and 117 units, the half-widths are 0.0943 and 0.0610. A loss cannot be reached at those sizes. A uniform win needs -0.140 (V1) or -0.100 (V2L); a concentrated one did not win in every resample at any gap tried.
    - At the dev point 127 (V1) and 140 (V2L) of the 195 units had them. Resampled to 242 and 267 units, the half-widths are 0.0302 and 0.0293. At a gap of 0: inconclusive in every resample (V1) and inconclusive in 10 and loss in 10 (V2L). A win needs -0.065 (V1) or -0.065 (V2L) uniform, -0.080 or -0.065 concentrated.
  - **C2-SELECT:**
    - The difference's median half-width is 0.0197 (quick) and 0.0200 (dev point). At a difference of 0 every resample gave a loss at quick and inconclusive at the dev point, where in no resample was the upper end below the 0.02 margin (the half-widths there are close to 0.02).
    - A win needs a difference of at least 0.045 uniform at both points; concentrated, 0.045 (quick) and 0.050 (dev point).
    - VW's closed-loop `c^lo` had a median lower end at `n = 373` of 0.9760 (quick) and 0.9429 (dev point), above the floor of 0.92.
    - These rows use VW's last pair. VW's steps ended in a two-cycle at quick and in one dev-point half, and settled in the other half (`docs/REACHABILITY.md`). So in the full run a C2-SELECT win or loss also needs VW settled, or both pairs of a two-cycle giving the same label (section 8).
  - **C2-EXT:** Clopper-Pearson at 707 units.
    - A coverage of 0.95 gives 0.9249 to 0.9695: a win.
    - 0.92 gives 0.8886 to 0.9441: inconclusive.
    - 0.85 gives 0.8115 to 0.8836: a loss.

## 8. Decision rules (as coded in `closed_loop_v2/verdicts.py`)

- **C2-PERC, each of the four:**
  - win if the interval's low end is above 0.03 or its high end is below -0.03;
  - loss if the interval lies within `[-0.03, 0.03]`, ends included;
  - otherwise inconclusive.
- **C2-SELECT:**
  - the rule for one VW pair: win if the paired difference's low end is at least 0.02 and VW's `c^lo` low end is at least 0.92; loss if the difference's high end is below 0.02, or VW's `c^lo` high end is below 0.92; otherwise inconclusive;
  - VW settled (section 4): the rule with its pair;
  - VW in a two-cycle: the rule with each of the two pairs (the difference and VW's `c^lo` computed for each, at the same level); the label if both pairs give the same one, otherwise inconclusive;
  - VW unsettled: inconclusive, whatever the intervals;
  - the row records VW's state and, for a two-cycle, both pairs' intervals, so `--check` can rebuild the label.
- **C2-EXT:**
  - win if the Clopper-Pearson interval lies within `[0.92, 0.98]`;
  - loss if it is disjoint from that range;
  - otherwise inconclusive.
- **C2-SELECT is read on pooled coverage:**
  - The quantity is each unit's lower-bound coverage over every checkpoint it contributes in the closed loop.
  - That is the population VW's weighting targets: a uniform draw over the checkpoints the policy reaches.
  - The stratum version is reported, without a label.
- **C2-CENSOR and C2-SHIFT have no label.** Their intervals are reported at 0.05 as estimates. No win, loss or inconclusive label is attached, and they do not change any primary verdict.
- **General:** an interval too wide to decide is inconclusive, never a loss. Every result is published whatever it shows. If a sophisticated method (VW, V2L or V3L) does not beat V1, that is the result.

## 9. Analyses registered in advance but not primary

C2-CENSOR, C2-SHIFT, the stratum version of C2-SELECT, the descriptive metrics of section 6, and the per-subset breakdown (coverage, lower-bound coverage and `Gap_B` per subset, for each method and loop, at 0.05). All of them are reported without a label.

## 10. Exploratory

Anything not listed above is exploratory and is labelled so where it appears.

## 11. Process

- **One full run:** `python benchmarks/run.py --full` after the freeze.
  - Every held-out read writes a ledger line first.
  - An error after the first held-out read writes `full-failed`.
  - A code fix after the freeze is recorded with its reason, and any rerun is reported beside the first, as in v0.1.
  - Each full run's `results.json` is also kept as `benchmarks/runs/results-run-{n}.json`, which is never overwritten.
- **No tuning after the freeze:** nothing in `tuned.json`, the verdict rules or this file changes after the freeze.
- **Before the freeze:** an adversarial review of the replay, metrics, statistics and verdicts, and the reachability check.
- **After the run:** a claim check of the README against the records, and a clean-install proof.

## 12. What had been seen before the freeze

- **Dev units:** fully. They were used for building, the condition check and tuning.
- **Calib units:** loaded by packet A's data tests to check their counts, shapes, roles, and that the remaining life is 0 at the last row. Those tests last ran on 10 November. The loader also parses the calib units' rows whenever dev units are asked for (tuning, `--quick`, `scripts/dev_point.py`), because they share the training files, and keeps them in a cache that a dev request does not return (`closed_loop_v2/data/load.py`). No coverage, score or model output on calib units had been computed or looked at. VW's and V1's calibration use them for the first time in the full run.
- **Eval units, NASA's test files and the RUL files:** no value used or looked at. The data tests check every file's SHA-256 against the manifest, the held-out files included, so their bytes have been hashed; nothing was parsed from them. The FD004 test count (248) comes from the literature, not from the files.
  - Eval units' rows sit in the same `train_FD00x.txt` files as dev and calib units. The loader checks each whole file's SHA-256 against the manifest. Of an eval unit's rows it reads only the unit-number column, to check that the units run 1 to N. It does not convert their settings or sensor values (`closed_loop_v2/data/parse.py`, `parse_training`).
- **Tuning** (`scripts/tune.py`, ADR 0003) read dev units only.
  - Its first run used the uncapped truth. It stopped by design, having shown a degenerate loop that replaced every unit early.
  - The truth was then capped (section 3), and tuning was rerun from the start (ADR 0003 section 7).
  - It was rerun once more on the final code after review D limited both bounds to the cap (ADR 0003 section 8). The choices and quantiles were the same; only the pooled coverage of the `s_min` 2 settings changed.
- **The reachability check** (`scripts/reachability.py`) read the per-unit values of a `--quick` run, which re-splits dev units only.
  - Its first version showed that v0.1's confidence sequence could not reach a C2-PERC loss.
  - The interval was replaced before the freeze (ADR 0004), and the check was rerun.
  - The claims, bands, levels and verdict rules were not changed.
- **The adversarial reviews' probes** read dev units only.
  - Review D, the fourth review, computed dev values at the tuned point and looked at them: V1's pooled `Gap_B` mean -0.001963986125596948 and V2L's -0.049553660818287715; VW's closed-loop `c^lo` below V1's (0.9248243723487124 against 0.9520682907131467); lower bounds above the cap counted as misses (finding S1); and V2L's queue effect (finding S2, section 7).
- **`scripts/dev_point.py`** read dev units only. It evaluates all 195 at the tuned point, in two halves each calibrated on the other, through `run.py`'s own evaluation, for the reachability check (section 7).
  - What it showed, and was looked at, is in `docs/REACHABILITY.md`: on dev units at the tuned point V2L's pooled `Gap_B` was below zero (mean -0.0342) and VW's closed-loop `c^lo` below V1's (difference mean -0.0052); resampled at `n = 373` as it was, C2-SELECT gave a loss in every resample.
  - VW's fixed-point steps settled in one half and alternated between two pairs in the other. On the `--quick` run they also alternated.
- **Decisions made in light of these dev numbers,** all before the freeze and all recorded where they apply:
  - Both bounds limited to the cap (review D, finding S1; ADR 0001 section 5), then tuning rerun (ADR 0003 section 8).
  - The betting interval's level scoped as nominal for the fixed order and for V2L and V3L (review D, finding S2; section 7 and ADR 0004).
  - VW's end state (review D, finding N3; sections 4 and 8). It was first registered as "inconclusive unless VW settled". VW settled in only one of the three dev fits, so on 10 November it was changed to judge a two-cycle with both pairs. A settled VW gets the same label under both rules; the change only lets a two-cycle reach a win or a loss when both pairs agree.
  - None of these changed a claim, band, margin, floor or level.
