# 0001: closed-loop v0.2, the perception study on run-to-failure data

Status: accepted for the build, 9 November 2026. Written before any training or test row was parsed: so far only the dataset's `readme.txt` and its unit counts have been read. The pre-registration (`docs/PREREGISTRATION.md`) fixes the numbers this record leaves open; where they differ, the frozen pre-registration wins.

## 1. The question

closed-loop v0.1 asked whether conformal remaining-life intervals calibrated open loop keep their coverage once a maintenance policy acts on them. On its simulated plant the answer was inconclusive. In the trend-estimated stratum under the bound rule, oracle lower-bound coverage fell at nominal by 0.0509 (V1) and 0.0570 (V2L) against open loop. The intervals excluded zero but did not clear the registered band of 0.03, and under a load shift the gap reversed.

v0.1's plant was its own simulator, so it could not say whether the effect belongs to real degradation or to the simulator that produced it. v0.2 asks the same question on run-to-failure trajectories that this project did not generate:

> When a replacement policy acts on conformal remaining-life bounds, does the coverage measured on the checkpoints the fleet actually lives through match the coverage measured open loop? If not, does calibrating on what the policy lets through restore it?

## 2. Where v0.2 lives: a new repository

v0.2 is a new repository, `closed-loop-v2`, with the package `closed_loop_v2`. The alternative, `closed_loop/v2` inside v0.1, would break v0.1's record:

- v0.1's code hash covers everything under `closed_loop/`, plus `benchmarks/run.py` and two spec caches. Any file added under `closed_loop/` changes the hash that its published numbers carry.
- The package version lives in the hashed `closed_loop/__init__.py`, so v0.1 cannot become 0.2.0 without the same break.
- v0.2 needs its own pre-registration, ledger, held-out rule and run script, and none of v0.1's may change.

So v0.1 stays frozen and publishable as it is, and v0.2 depends on it as a library. From `closed_loop.audit` it takes:

- `Ledger`, the hash-chained append-only record;
- `prereg_sha256` and `code_sha256`;
- `HeavyRunLock`, `commit_charge_percent` and `require_headroom`, the shared lock and the 88% commit-charge guard.

From `closed_loop.stats` it takes `clopper_pearson`, `betting_cs` and `paired_betting_cs` (since ADR 0004, `clopper_pearson` only). v0.1's plant, perception and planning code is not used.

The dependency is pinned. During development it is v0.1's checkout installed editable. Once v0.1 is published it becomes `closed-loop @ git+https://github.com/darrshangovender/closed-loop@v0.1.0`. If v0.1 is never published, the five files used are small enough to vendor with their hashes recorded.

## 3. Data

**Chosen: NASA C-MAPSS turbofan degradation, subsets FD001 to FD004.**

| Subset | Train units | Test units | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, fan) |
| FD004 | 249 | 248 | 6 | 2 (HPC, fan) |

*Amended 10 November, before any held-out read. The dataset's `readme.txt` gives FD004 as 248 train and 249 test. Packet A's loader reads the unit column only, and it found units 1 to 249 in `train_FD004.txt`. The published literature gives 249 and 248; for example, the dataset table in arXiv:2112.05372. The test count of 248 is checked when the test files are first read after the freeze, and a mismatch stops the run. The extra unit, FD004 unit 249, has the role dev, so the calib and eval counts in section 4 are unchanged.*

- **Format:** 26 columns per row: unit, cycle, three operational settings and 21 sensors.
- **Training units** run until failure, so the true remaining life is known at every cycle.
- **Test units** stop before failure, and NASA publishes their true remaining life at the last cycle.

**What the data is, said plainly:** NASA generated these trajectories with its C-MAPSS engine model. They are simulated, not field data. What v0.2 gains is that the degradation comes from a model this project did not write, built by people who were not testing this question, and benchmarked by many published studies.

**Why not the alternatives:**

- **IMS bearings** (University of Cincinnati, via the NASA repository): real rig data, but three tests hold four bearing failures in all. That is too few run-to-failure units to estimate coverage.
- **FEMTO/PRONOSTIA bearings**: real rig data with 17 bearings. No licence is stated, and the original host no longer serves it.
- **N-CMAPSS** (2021): more realistic flight profiles, but about 14 GB of HDF5 for roughly 90 units, which this laptop cannot hold in memory.

**Terms.**

- **Source:** NASA PCoE repository, item 6. The download link and checksums are in `data/MANIFEST.json`.
- **Conditions of use:** the repository asks users to acknowledge it and the data donors. It says users take the data at their own risk and that neither NASA nor the donors accept liability.
- **Licence:** none is stated. NASA generated the data, so it is a work of the US Government.
- **Redistribution:** the repository does not redistribute the data. `scripts/fetch_cmapss.py` downloads the zip, checks its sha256 (`c9c5dec1...d3b2`) and its 14 member files against the manifest, and never runs anything from it.
- **Citations:** Saxena and Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository; Saxena, Goebel, Simon and Eklund (2008), "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.

## 4. Who is held out, decided by unit number alone

Each training unit gets a role from a keyed hash of its subset and number, independent of anything in its rows:

```
u = int.from_bytes(sha256(f"closed-loop-v2/roles/{subset}/{unit}").digest()[:8], "big") / 2**64
role = "dev" if u < 0.30 else "calib" if u < 0.50 else "eval"
```

Computed from unit numbers only, before any row was parsed. FD004's dev count was corrected from 70 to 71 on 10 November, after the loader found unit 249 (section 3):

| Subset | dev | calib | eval |
|---|---|---|---|
| FD001 | 25 | 21 | 54 |
| FD002 | 64 | 59 | 137 |
| FD003 | 35 | 16 | 49 |
| FD004 | 71 | 45 | 133 |
| **All** | **195** | **141** | **373** |

What each role is for:

- **dev:** fits the estimator and tunes every free choice. It may be read any number of times.
- **calib:** calibrates the conformal methods and replays the policy for the selection-aware method. Its rows reach calibration code only, and tuning code never reads them.
- **eval:** held out until the freeze, then used once.
- **The 707 official test units and their true remaining lives** are a second held-out set, also read only after the freeze. They are the external check.

The loader enforces all of this, as v0.1's seed guard did:

- **Rows:** eval rows and the test and RUL files are parsed only through a `HeldoutAccess` object. It refuses to open unless the ledger holds a freeze entry whose pre-registration hash matches the file on disk.
- **Every held-out read** writes a ledger line.
- **Tests:**
  - One test proves that development code paths raise on an eval unit.
  - Another proves that the development loader does not open the test or RUL files. It records every file Python opens through the interpreter's audit hook.
  - The fetch script and the general `verify()` do hash every file, which reads its bytes but parses no row. The development loader hashes the training files only.
- **Since the review of 10 November:**
  - Only the first freeze entry counts, so an edited pre-registration cannot be re-frozen to reopen the data.
  - With the real data, the gate accepts only the repository's own ledger and pre-registration.

## 5. The estimator (shared by every method)

Each checkpoint is one cycle `t` of one unit, and the features use rows up to `t` only:

- **Operating condition:** each row's three settings are assigned to the nearest of six centroids, found by k-means on dev rows. FD001 and FD003 fall into one of them.
- **Normalised sensors:** each sensor is z-scored within its condition with dev statistics. Sensors whose dev variance is near zero are dropped, and the dropped set is recorded.
- **Window features:** for each kept sensor, the mean and least-squares slope over the last `W` cycles (fewer at the start), plus the cycle count `t`.

Two gradient-boosted regressors (scikit-learn `HistGradientBoostingRegressor`) are fitted on dev units only:

- **The point estimate `m`** has the target `min(RUL, cap)`.
- **The scale `s`** has the target `|min(RUL, cap) - m|`. Its residuals come from unit-grouped cross-fitting on dev, so `s` never sees `m`'s in-sample errors. `s` is floored at `s_min`.

The score at a checkpoint is `r = (min(RUL, cap) - m) / s`. Every method shares `m` and `s` and differs only in how it turns scores into a lower and an upper quantile. The interval is `[m + s q_lo, m + s q_hi]`, each end limited to the cap (amendment below), with nominal two-sided level 0.90 (0.05 each side), so the nominal lower-bound coverage is 0.95, as in v0.1.

`W`, `s_min` and the regressors' sizes are tuned on dev by unit-grouped cross-validation, then frozen in `benchmarks/tuned.json`. `cap` is fixed at 125 cycles.

*Amended 10 November, before any calib, eval or test value was used or looked at (ADR 0003 section 7). As first written, `s` had the target `|RUL - m|`, the score used the uncapped remaining life, and `cap` was tuned. The first dev tuning run showed why that fails: early in a life `m` saturates at the cap while the uncapped truth is often far above it, so `s` learned large upside errors, the symmetric score widened the lower bound by the same amount, and the bound rule replaced every dev unit with about 120 cycles or more still to run. The amendment:*
- *The truth is `min(RUL, cap)` wherever this document compares a bound or an estimate with the remaining life: both targets, the score, the guarantees of section 6, the labels of section 7, the coverage metrics of section 8 and the claims of section 9.*
- *A lower bound `L <= cap` covers `min(RUL, cap)` exactly when it covers `RUL`, so for every bound at or below the cap, the lower-bound claims mean what they meant before. A lower bound above the cap is always a miss under the capped truth, but under the uncapped truth only when `RUL < L`.*
- *In the closed loop, a replaced unit known to have lived at least `cap` more cycles past a checkpoint has an exactly known truth there, `cap`. Its label is `HIT` if the interval contains `cap` and `MISS` otherwise.*
- *The cycles left at replacement, a cost, stay uncapped.*
- *`cap` is fixed, not tuned: with a capped truth a smaller cap is narrower by construction, so width cannot choose it.*

*Amended again 10 November, after review D (finding S1), before the freeze. Where `m` is above the cap, `m + s q_lo` can exceed 125, and the second bullet above made such a lower bound a miss of the capped truth even when it covered the remaining life. Now:*
- *Both ends are limited to the cap: `L = min(m + s q_lo, cap)` and `U = min(m + s q_hi, cap)`. The capped truth never exceeds the cap, so this only removes the part of an interval where the truth cannot be.*
- *Every lower bound is then at or below the cap, so lower-bound coverage of the capped truth equals lower-bound coverage of the remaining life at every checkpoint, and the trigger, VW's reach, the labels and C2-EXT all use the limited bounds.*
- *The label code also treats a lower bound above the cap as a certain miss (the capped truth is at most the cap), for any caller that passes unlimited bounds.*
- *The width that chose the estimator setting (ADR 0003) is still the unlimited `s (q_hi - q_lo)`, so the criterion did not change; tuning was rerun on the final code (ADR 0003 section 8).*

## 6. The methods

| Name | What it is | Source |
|---|---|---|
| V1 | Split conformal. For each calibration unit, one checkpoint is drawn uniformly from its whole life with a registered seed. `q_lo` and `q_hi` are the finite-sample conformal quantiles of those scores. Never updated. | v0.1's V1, with one checkpoint per unit so the guarantee below is exact |
| V2L | Adaptive conformal inference with one update per resolved life: `alpha_t` steps by `gamma (alpha - miss)`, with `gamma = 0.02` and `miss` the life's mean interval-miss indicator. The pool is a rolling window of 500 scores, started from the calibration draw. Only observed failures add scores. A censored label counts as a hit (V2's rule). | v0.1's V2L; Gibbs and Candès (2021) |
| V3L | V2L with V3's rule instead: labels whose outcome is unknown are skipped, not counted as hits. | v0.1's V3L |
| VW | Selection-aware weighted conformal. The deployment policy is replayed on each calibration unit's full trajectory. The unit's draw is taken uniformly from the checkpoints that policy reaches, not from its whole life. The policy depends on the intervals, so the quantiles are found by a fixed number of fixed-point steps from V1's. | Weighted conformal prediction under covariate shift, Tibshirani, Barber, Candès and Ramdas (2019), with the policy's reach as the weight |

**Why VW is the right new method here.** In the replay a replacement only truncates a trajectory. It does not change it, so given the history up to `t`, the remaining life has the same distribution whether or not the policy is acting. What the policy changes is which checkpoints are lived through: a selection that depends on the history alone. That is a covariate shift with an exactly known likelihood ratio, the indicator that the policy reaches the checkpoint. Weighted conformal prediction is the method built for this case, and here it needs no density estimate.

**The guarantees, stated with their scope.**

- **V1:** for a new unit exchangeable with the calibration units, and a checkpoint drawn uniformly from its life, `P(RUL >= lower bound) >= 0.95` and `P(RUL in interval) >= 0.90`. This is the split-conformal theorem applied to one score per unit.
  - It says nothing about checkpoints chosen by a policy.
  - It says nothing about NASA's test truncation points, which are not uniform draws.
- **VW:** the same statement for a checkpoint drawn uniformly from the part of a new unit's life that a fixed policy reaches. It is exact for a policy fixed before the calibration draw. With the fixed-point steps the policy depends on the calibration data, so VW's guarantee is approximate, and v0.2 measures how far it holds.
- **V2L and V3L:** ACI's long-run guarantee is about the average miss rate of the label stream they see, under any sequence. It is not about oracle coverage when the policy decides which labels exist.

## 7. The replay

- **The fleet:** `K` engines run at once. When one ends, by failure or by replacement, the next unit in a registered random order starts in its place. `R` orders are run, and per-unit results are averaged over them.
- **The policy (bound rule):** replace at the first checkpoint where the lower bound is at most `h` cycles. `h` is tuned on dev against a stated trade-off and then frozen.
- **Open loop:** no replacements. Every life ends in failure, and the online methods see every label.
- **Closed loop:** a unit replaced at `tau` contributes checkpoints `1..tau`. It is known to have lived at least `tau - t` more cycles past checkpoint `t`, and its labels carry v0.1's statuses plus one:
  - `LOWER_HIT` where the lower bound is at most `tau - t` and the upper bound is at least `tau - t`;
  - `MISS` where the upper bound is below `tau - t`, because the unit outlived the interval (amended 9 November, before any code was written; 0002 section 3);
  - `UNKNOWN` where the lower bound is above `tau - t`.
  - A unit that reaches its last cycle unreplaced is an unplanned failure, and all its labels are exact.
  - If the rule first triggers on the last cycle itself, the unit counts as replaced at that cycle, not as an unplanned failure. The replacement is decided on that cycle's bound, before the failure is seen. Its labels are the replaced-life statuses with `tau - t` cycles survived.
- **Online updates:** labels resolve when a life ends, and the online methods update then.
- **No physics feedback:** a replacement does not change any other engine. v0.2 says so; it is the reason the covariate-shift argument above holds.

## 8. Metrics

- **Per-unit coverage:** `c_u` and `c_u^lo`, the interval and lower-bound coverage averaged over the checkpoints the unit contributes. Open loop counts all of the unit's checkpoints. Closed loop counts those up to replacement. Truth is `min(remaining life, cap)`, with the dataset's remaining life (amended 10 November; section 5).
- **Fleet value:** the mean over units, each unit weighted equally.
- **Decision stratum:** checkpoints where the shared point estimate `m` is at most `M` cycles, defined by the estimator so it is the same set for every method open loop. This is the analogue of v0.1's trend-estimated stratum. A unit with no checkpoint in the stratum is excluded and counted.
- **Gap_B:** paired by unit, `Gap_B = c^lo(closed) - c^lo(open)` for the same method, pooled and in the stratum.
- **Policy outcomes, descriptive:** the fraction of units with an unplanned failure, the mean cycles left at replacement (uncapped), and mean interval width.

## 9. Claims (numbers fixed in the pre-registration)

| Claim | Question | Primary |
|---|---|---|
| C2-PERC | Does open-loop coverage carry over to the closed loop? `Gap_B^lo` for V1 and for V2L, pooled and in the decision stratum: four verdicts, nominal condition. Win (thesis supported): the interval excludes `[-0.03, +0.03]`. Loss (not supported): it lies inside the band for V1 and V2L. | yes |
| C2-SELECT | Does calibrating on what the policy lets through restore closed-loop coverage? Closed-loop `c^lo`, VW against V1. Win: lower bound of the paired difference at least 0.02 and lower bound of `c^lo(VW)` at least 0.92. Loss: upper bound of the paired difference below 0.02, or VW's upper bound below 0.92. How VW's fixed-point steps ended decides which pairs are judged: settled (the last step did not move the quantiles), the rule with that pair; a two-cycle (the last step returned them to the pair of two steps earlier, so the end pair depends only on the parity of the step count), the rule with each of the two pairs, the label kept only if both give it, otherwise inconclusive; unsettled, inconclusive whatever the intervals (amended 10 November after review D, finding N3, before the freeze). | yes |
| C2-EXT | Does V1's calibration carry over to NASA's own test units, at their last observed cycle? Lower-bound coverage on all 707 against 0.95, exact Clopper-Pearson interval. Win: inside `[0.92, 0.98]`. Loss: outside it. The truncation points are NASA's, not uniform, so this is a measurement, not a consequence of V1's guarantee. | yes |
| C2-CENSOR | Does online recalibration on policy-censored labels add to the gap? Paired `Gap_B(V2L) - Gap_B(V1)`, and the same for V3L. | secondary |
| C2-SHIFT | Under the shifted condition (section 10), V2L and V3L against V1, closed-loop `c^lo`. The analogue of v0.1's C-ADAPT. | secondary |

As in v0.1, the labels are **win**, **loss** and **inconclusive**. An interval too wide to decide is inconclusive, never a loss. Every label is checked for reachability on made-up data built from dev spreads before the freeze, and an unreachable label is written down, not tuned away.

## 10. Conditions

- **Nominal:** the estimator is fitted on all dev units, calibrated on all calib units, and evaluated on all eval units.
- **Shifted:** the estimator is fitted and calibrated on single-fault subsets only (FD001 and FD002 dev and calib units, 89 and 80), and evaluated on the two-fault subsets' eval units (FD003 and FD004, 182). A fault mode the calibration never saw appears in deployment. That is a real shift and not a covariate shift, so VW's argument does not cover it.

## 11. Statistics

The evaluation set is fixed in advance and analysed once, so v0.2 uses fixed-sample intervals, not v0.1's confidence sequences:

- **Means of per-unit values and paired differences:** Student-t intervals, used only when `n >= 100` and the sample kurtosis is at most `n / 50`, v0.1's guard. Otherwise a fixed-sample betting interval is used. It is valid at any size for independent values with one mean, fed in a uniformly random order. *(Amended 10 November, before the freeze, by ADR 0004. As first written this was v0.1's `betting_cs`, a confidence sequence, whose floor of 0.06 at n = 373 made a C2-PERC loss unreachable.)*
  - *Scope (amended 10 November after review D, finding S2, before the freeze): the betting interval is fed the values in a fixed canonical order, not a random one, and V2L's and V3L's per-unit values are coupled through their shared `alpha_t`, so they are not independent whichever rule is used. For the rows these touch, the stated level is nominal, not proven (ADR 0004 section 3).*
- **Binary external coverage:** Clopper-Pearson.
- **Level:** primary verdicts at `0.05 / 6` (four C2-PERC verdicts, C2-SELECT, C2-EXT). Secondary rows at 0.05, descriptive.
- **What a bound can show at this size:** the pre-registration states it from dev spreads, as v0.1's section 6 did.

## 12. Discipline (as v0.1)

- **One command regenerates everything:** `python benchmarks/run.py`.
  - `--quick` runs on dev units only and writes nothing to the ledger.
  - `--full` takes the freeze, then the held-out sets.
  - `--check` compares hashes and documents.
- **What the ledger records:** the pre-registration hash, the code hash (all of `closed_loop_v2/` and `benchmarks/run.py`), the library versions (with v0.1's version and the hash of its code, since v0.1 supplies the t guard, Clopper-Pearson, the ledger and the guard) and every held-out read.
- **The registered values** are checked against the frozen document at the freeze, and again at every `--full` (before any held-out read; a mismatch refuses the run) and `--check` (a mismatch fails it). Amended 10 November after review D, finding S3.
- **Frozen files:** `docs/PREREGISTRATION.md` and `benchmarks/tuned.json`.
- **Heavy runs** take the shared lock and stop at 88% commit charge.
- **Tests and checks:** ruff, mypy and pytest must be green. Mutation spot checks prove the guard and the verdict rules fail when broken.
- **Before the freeze:** an adversarial review (Diogenes) of the data guard, the conformal methods and the replay, and a check that every label is reachable.
- **After the run:** a claim check of the README against the records, and a clean-install proof.

## 13. What v0.2 cannot say

- **Not field data.** The trajectories come from a NASA simulator. A positive or negative result here is about C-MAPSS degradation.
- **No physics feedback.** Replacing one engine does not change another. In a plant where maintenance changes the load on the remaining equipment, the covariate-shift argument for VW would not hold as it stands.
- **One policy family.** The bound rule is tested at one threshold, `h`, which is tuned on dev. Other policies select other checkpoints.
- **Small strata.** FD003 contributes 16 calibration units. The shifted condition calibrates on 80 units and evaluates on 182.
