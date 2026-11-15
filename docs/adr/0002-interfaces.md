# 0002: interfaces and build packets

Status: accepted for the build, 9 November 2026. It implements `0001-design.md`, and 0001 wins on any conflict. Each packet owns the files listed under it and touches no other file. When it needs a change elsewhere, it says so in its report.

## 1. Shared types (`closed_loop_v2/types.py`, written)

- `Trajectory(subset, unit, source, cycles, settings, sensors, rul_at_end)`, frozen.
  - `cycles` is `1..n`. `settings` has shape `(n, 3)` and `sensors` `(n, 21)`, both float64.
  - `rul_at_end` is 0 for a training unit (its last row is the failure cycle) and NASA's value for a test unit.
  - `failure_cycle = cycles[-1] + rul_at_end`, and `rul()` is `failure_cycle - cycles`.
  - *Amended 10 November (0001 section 5, capped truth):* a further field `rul_cap: int | None = None`. `rul_uncapped()` is `failure_cycle - cycles`; `rul()` is that, capped at `rul_cap` when it is set; `capped(cap)` returns a copy with `rul_cap = cap`. The estimator caps `rul_uncapped()` at its own `cap` and refuses trajectories capped at another value. Labels and coverage metrics read the capped `rul()`. The cycles left at replacement read `rul_uncapped()`.
- `SUBSETS = ("FD001", "FD002", "FD003", "FD004")`, `Role = Literal["dev", "calib", "eval"]`.

## 2. Packet A: data, roles, guard, audit paths

**Owns:** `closed_loop_v2/data/*.py`, `closed_loop_v2/audit.py`, `scripts/fetch_cmapss.py`, `data/MANIFEST.json`, `tests/data/`, `tests/test_audit.py`, `tests/conftest.py`, `tests/test_packaging.py`, `Makefile`, `.github/workflows/tests.yml`.

- **`data/roles.py`:** `role_of(subset: str, unit: int) -> Role`, exactly the formula in 0001 section 4.
  - Also `ROLE_COUNTS: dict[str, dict[Role, int]]`, the table in 0001 section 4, which a test recomputes.
  - Also `UNITS_PER_FILE = {"train": {...}, "test": {...}}`, from the dataset's readme.
- **`data/MANIFEST.json`** holds:
  - the source URL;
  - the zip's size, sha256 and md5;
  - the name of the inner zip;
  - a mapping from each of the 14 member file names to its sha256. Copy these from `Documents\Career\datasets\cmapss\SHA256SUMS`, and include the zip's own line from `PROVENANCE.md`.
- **`data/paths.py`:**
  - `data_dir() -> Path` is the environment variable `CLOSED_LOOP_V2_DATA` if set, otherwise `<repo>/data/cmapss`.
  - `verify(directory) -> None` checks each member's sha256 against the manifest and raises `DataError`, naming the file, on a missing file or a mismatch.
  - `data_available() -> bool`.
- **`scripts/fetch_cmapss.py [--dest DIR] [--from-zip PATH]`:**
  - It downloads the zip, or uses a local copy, and checks its sha256 before opening it.
  - It reads the inner zip in memory and writes each member by its bare name only: a name with a directory part, `..`, a drive or a leading dot raises.
  - It checks every member's sha256 and never executes or imports anything from the archive.
  - Standard library only.
- **`data/parse.py` (private):** parses one whitespace-separated file into rows of 26 floats. The unit and cycle columns are checked to be integers, and rows are grouped by unit into `Trajectory` objects.
- **`data/load.py`:** `development_units(role, subsets=SUBSETS) -> list[Trajectory]` for `role` in `{"dev", "calib"}`.
  - It reads only `train_FD00x.txt`. Rows of units with any other role are dropped as they are grouped, before any array is built for them.
  - Passing `"eval"` raises `HeldoutError`.
  - It never opens a `test_` or `RUL_` file.
  - It verifies the files first, and caches results per process.
  - It also checks that each training file holds unit numbers `1..N` with `N` from the readme. That check reads the unit column only.
- **`data/heldout.py`:** `HeldoutAccess(ledger: closed_loop.audit.ledger.Ledger, prereg_path: Path)`.
  - `is_frozen()`: the ledger holds an entry of kind `"freeze"` whose field `prereg_sha256` equals `closed_loop.audit.prereg.prereg_sha256(prereg_path)`.
  - `eval_units(subsets=SUBSETS) -> list[Trajectory]` returns the training units whose role is `"eval"`.
  - `test_units(subsets=SUBSETS) -> list[Trajectory]` returns test units with `rul_at_end` read from `RUL_FD00x.txt`, one value per test unit in unit order.
  - Each of these two raises `HeldoutError` unless frozen, and otherwise appends a ledger entry of kind `"heldout-use"` with `n=1`, `what` (`"eval"` or `"test"`), `subsets` and `units`.
  - v0.1's `Ledger` accepts only the kinds `freeze`, `full`, `full-failed` and `heldout-use`. A `heldout-use` entry must carry an integer `n`, which adds to the running `heldout_evaluations`. So in v0.2 that counter counts reads of held-out data.
  - The entry is written before the arrays are returned. A read that fails after the entry is written still counts.
- **`audit.py`:**
  - `REPO`, `PREREG = REPO/"docs/PREREGISTRATION.md"`, `TUNED = REPO/"benchmarks/tuned.json"`, `LEDGER = REPO/"benchmarks/ledger.jsonl"`, `RESULTS = REPO/"benchmarks/results.json"`, `CACHE = REPO/"benchmarks/.cache"`.
  - `code_sha256_v2() -> str` is `closed_loop.audit.prereg.code_sha256([REPO/"closed_loop_v2", REPO/"benchmarks"/"run.py"])`. That function accepts a directory or a single file as a root.
  - No lock path is written into the code, because the shared lock file is a feature of one laptop, not of the repository. The lock path is read from `CLOSED_LOOP_V2_LOCK`. Without it there is no shared lock, but there is still the commit-charge guard from `closed_loop.audit.guard`.
- **Tests:**
  - The role counts.
  - The development loader refuses `"eval"`, and it never opens a `test_` or `RUL_` file: patch `open` and `Path.read_text`/`read_bytes` and record every path.
  - `HeldoutAccess` refuses before a freeze, works after a matching freeze entry, refuses when the pre-registration bytes change, and writes its ledger lines. Use a temporary ledger and pre-registration.
  - `verify` catches one changed byte.
  - The fetch script refuses a zip-slip name and a wrong sha256. Build the test zips in memory.
  - Tests marked `data` run on the real files. They check shapes, dev and calib counts per subset, the role of every loaded unit, and RUL at the last row = 0. **They must not load eval units or any test or RUL file.**
  - `conftest.py` skips `data` tests when `data_available()` is false.
  - `test_packaging.py`:
    - the version matches pyproject;
    - closed-loop is referenced by direct URL and tag only;
    - `.gitignore` hides `data/*` but not `data/MANIFEST.json`;
    - no machine path in any tracked text file.
- **CI** (`.github/workflows/tests.yml`): Ubuntu, Python 3.11 and 3.12. It installs closed-loop from its URL, then `pip install -e ".[dev]"`, then runs `ruff check .`, `mypy` and `pytest tests/ -q`. Data tests skip in CI because there is no data.

## 3. Packet B: features, estimator, methods

**Owns:** `closed_loop_v2/features.py`, `closed_loop_v2/estimator.py`, `closed_loop_v2/conformal.py`, `closed_loop_v2/policy.py`, `tests/test_features.py`, `tests/test_estimator.py`, `tests/test_conformal.py`, `tests/test_policy.py`. Tests build synthetic `Trajectory` objects. They read no data files.

- **`features.py`:**
  - `ConditionModel` (frozen dataclass):
    - `centroids`, shape `(k, 3)`;
    - `means` and `stds`, shape `(k, 21)`, per condition;
    - `kept`, a tuple of sensor indices.
  - `ConditionModel.fit(trajs, *, seed, k=6, min_std=1e-6) -> ConditionModel`:
    - k-means on the rows' settings, with a deterministic k-means++ seeded by `seed` and at most 100 Lloyd steps, numpy only;
    - per-condition means and standard deviations;
    - drops each sensor whose largest within-condition standard deviation is below `min_std`, or whose within-condition standard deviation is below `min_std` in every condition.
  - `condition(settings) -> NDArray[int]` assigns each row to its nearest centroid.
  - `normalise(traj) -> NDArray (n, len(kept))` z-scores within each row's condition. A zero standard deviation in a condition makes the z-score 0.
  - `window_features(z, window) -> NDArray (n, 2k+1)`: for each row `i`, over rows `max(0, i-window+1)..i`, the mean and the least-squares slope against the row index (0 for one row), followed by the 1-based cycle count. It is causal: row `i` uses no later row. Vectorise it with cumulative sums, and test it against a loop.
- **`estimator.py`:**
  - `EstimatorParams(window, cap, s_min, max_iter, learning_rate, max_leaf_nodes, min_samples_leaf)`, frozen. `DEFAULT_PARAMS` is a starting point: window 30, cap 125, s_min 5.0, max_iter 300, learning_rate 0.05, max_leaf_nodes 31, min_samples_leaf 50.
  - `Estimator.fit(trajs, params, *, seed, n_folds=5) -> Estimator`:
    - fits the condition model;
    - fits `m` with `HistGradientBoostingRegressor` on `min(RUL, cap)`;
    - fits `s` on `|min(RUL, cap) - m_oof|` (amended 10 November; it was `|RUL - m_oof|`), where `m_oof` comes from unit-grouped `n_folds`-fold cross-fitting of the same `m` model;
    - sets `random_state` from `seed`, so a fit is deterministic.
  - `Prediction(m, s)` holds float64 arrays of length `n`.
  - `Estimator.predict(traj) -> Prediction` floors `s` at `s_min`. It is causal: truncating the trajectory at row `i` leaves the predictions for rows `<= i` unchanged, and a test proves it.
- **`conformal.py`:**
  - **Quantiles.** `conformal_lower(scores, a)` is the `floor((n+1) a)`-th smallest, or `-inf` when that index is below 1. `conformal_upper(scores, a)` is the `ceil((n+1)(1-a))`-th smallest, or `+inf` when the index is above `n`. `Quantiles(q_lo, q_hi)` is frozen, and `bounds(pred, q) -> (lo, hi)` is `m + s q_lo` and `m + s q_hi`.
    - *Amended 10 November after review D (finding S1), before the freeze:* `bounds(pred, q, cap=None)` limits both ends to `cap` when it is given (`interval_ends(m, s, q, cap)` does the same for arrays), and `cap_of(traj)` is the trajectory's `rul_cap`. Every caller passes the trajectory's cap: the VW trigger, `replay_static`, the online replay, the external check and tune's coverage record. Tune's width criterion stays the unlimited `s (q_hi - q_lo)`.
  - **Score.** `score(rul, pred, i) = (rul[i] - m[i]) / s[i]`.
  - **`SplitConformal(q: Quantiles)`:** `quantiles()` returns `q`, and `observe_life(...)` does nothing.
  - **`fit_v1(calib, *, alpha=0.10, seed) -> SplitConformal`.** `calib` is a sequence of `(Trajectory, Prediction)` pairs. One uniform row per unit comes from `numpy.random.default_rng([seed, unit_index])`, and the level is `alpha/2` on each side.
  - **`fit_vw(calib, rule, *, alpha=0.10, seed, steps) -> SplitConformal`.**
    - Draw one `u_j ~ U[0,1)` per unit, once, from a stream keyed by seed.
    - Start from V1's quantiles.
    - At each step, compute each unit's lower bounds and `tau_j = rule.first_trigger(lo)`, using its last index when the rule never triggers.
    - Take the row `floor(u_j (tau_j + 1))` and recompute the quantiles from those scores.
    - Return the quantiles after `steps` steps, and record each step's `(q_lo, q_hi)` and whether the last step moved them, so the report can show convergence.
  - **Label statuses (closed loop),** for one life with true remaining life `rul`, bounds `lo` and `hi`, and either an observed failure or a replacement at row `tau`:
    - With an observed failure, each row is `HIT` or `MISS` against `[lo, hi]`.
    - With a replacement, let `b_i = tau - i`, the cycles the unit is known to have survived past row `i`:
      - `MISS` if `b_i > hi_i` (the upper side is known to have missed);
      - `LOWER_HIT` if `lo_i <= b_i <= hi_i`;
      - `UNKNOWN` if `b_i < lo_i`.
      - *Amended 10 November (capped truth, ADR 0001 section 5), recorded here after review D (finding N8):* once `b_i >= cap` the capped truth at row `i` is known exactly (`cap`), and the label is `HIT` if `lo_i <= cap <= hi_i` and `MISS` otherwise; this overrides the three rules above.
      - *Amended 10 November after review D (finding S1):* a row with `lo_i > cap` is a certain `MISS`, since the capped truth is at most `cap`. With bounds limited to the cap this does not arise in the replay; it guards callers that pass unlimited bounds.
    - `life_labels(...) -> LifeLabels` holds the per-row statuses, the observed-failure flag, and, only when the failure was observed, the scores.
  - **`AdaptiveConformal(initial_scores, *, alpha=0.10, gamma=0.02, window=500, rule, seed)`.**
    - `rule` is `"naive"` (V2L: `UNKNOWN` and `LOWER_HIT` count as hits) or `"skip"` (V3L: `UNKNOWN` is skipped, `LOWER_HIT` counts as a hit).
    - `quantiles()` uses the pool at `alpha_t / 2` on each side. `alpha_t` is not clipped: a level outside `(0, 1)` gives an infinite side.
    - `observe_life(labels)`:
      - `miss` is the mean miss indicator over the rows the rule counts, with no update if it counts none;
      - `alpha_t += gamma (alpha - miss)`;
      - after an observed failure, one uniformly drawn score from that life joins the pool, which keeps the most recent `window`.
  - **Constructors:** `make_v2l(calib, *, seed)` and `make_v3l(calib, *, seed)` start their pools from the same draw as `fit_v1`.
- **`policy.py`:** `BoundRule(h: float)`, frozen. `first_trigger(lo) -> int | None` is the first index with `lo <= h`.
- **Tests:**
  - the quantile index edge cases (`n` small, `a` near 0 and 1);
  - V1's coverage on exchangeable synthetic units is near nominal: a Monte Carlo over 2,000 trials with a binomial tolerance, marked `slow` if it takes more than 20 s;
  - VW equals V1 when the rule never triggers;
  - VW's fixed point on a synthetic case;
  - each label status on hand-made lives;
  - the ACI update arithmetic;
  - the estimator's causality and determinism;
  - the window features against a loop.

## 4. Packet C: replay, metrics, statistics, verdicts, run script (after A and B)

**Owns:**
- `closed_loop_v2/replay.py`, `metrics.py`, `stats.py`, `verdicts.py`, `seeds.py`, `report.py`;
- `benchmarks/run.py`;
- `tests/test_replay.py`, `test_metrics.py`, `test_stats.py`, `test_verdicts.py`, `test_report.py`, `test_run.py`.

It builds on packets A and B's signatures as built. Their reports are the reference where those differ from section 3.

- **`seeds.py`:** `run_seed(prereg_sha: str, label: str) -> int` is the first 8 bytes, big-endian, of `sha256(f"closed-loop-v2/seed/{prereg_sha}/{label}")`.
  - The labels are `fit-nominal`, `fit-shifted`, `calib-draw` and `orders`.
  - V1, VW, V2L and V3L all take `calib-draw`: VW reproduces V1 only with V1's draws, and the adaptive methods start their pools from V1's draw. Their own update stream is keyed apart inside `conformal.py` (review of 10 November, finding 4).
  - Seeds are 64-bit. The estimator maps them to scikit-learn's 32-bit `random_state` itself (finding 5).
  - Seeds therefore exist only once the pre-registration is fixed, as in v0.1.
  - `--quick` uses `prereg_sha = "quick"`.
- **`replay.py`:** the methods share the protocol `Method`: `quantiles() -> Quantiles` and `observe_life(labels) -> None`.
  - **`replay(lives, method, rule, *, k, order) -> ReplayRun`.**
    - `lives` is a sequence of `(Trajectory, Prediction)` pairs, `rule` is a `BoundRule` (or `None` for the open loop), and `order` is a permutation of life indices.
    - Every one of the `k` slots takes the next life in `order`.
    - On each global step, the slots are taken in order `0..k-1`. Each occupied slot processes its life's next row `i`, with bounds from `method.quantiles()` as they stand at the start of the step. In the closed loop the life ends at `tau = i` if `lo_i <= h`; otherwise it ends at its last row, which is a failure.
    - At the end of the step, each life that ended is resolved in slot order: `method.observe_life(life_labels(...))` is called, then the slot takes the next life. A slot with nothing left stays empty.
    - The run ends when every slot is empty.
    - `ReplayRun` holds, per life, the rows contributed (`0..tau`, or every row), the `lo` and `hi` used at each, and the end (`"replaced"` at `tau`, or `"failed"`).
  - **`replay_static(lives, quantiles, rule) -> ReplayRun`** is the fast path for a method whose quantiles never change (V1 and VW). A test shows it equals `replay` for every `k` and order.
  - **`orders(n, r, seed) -> NDArray (r, n)`** gives `r` permutations from `numpy.random.default_rng([seed, j])`.
- **`metrics.py`:**
  - **Per life, from the dataset's true remaining life:**
    - `cover = lo <= rul <= hi` and `cover_lo = rul >= lo` over the contributed rows;
    - the same over the stratum rows (`m <= M`);
    - the end and `tau`, the remaining life at `tau`, and the mean finite width;
    - the count of rows with an infinite side.
  - **`UnitTable`:** per unit, method and loop, the mean over the `R` orders of `c_u`, `c_u^lo`, the stratum `c_u` and the stratum `c_u^lo`. A unit with no stratum row in an order is missing from the stratum for that order, and it counts in the stratum only if it has stratum rows in every order. The number of units excluded is counted.
  - **`gap_b(table, method) -> paired arrays`:** `c^lo` closed against open for the same units, pooled and in the stratum. The stratum uses units present in both loops, and the units it drops are counted.
  - **Policy outcomes (descriptive):**
    - the fraction of units that failed unreplaced, averaged over orders;
    - the mean remaining life at replacement;
    - the mean width;
    - the number of rows with an infinite side.
- **`stats.py`:**
  - **`mean_interval(x, *, lo, hi, alpha) -> Interval(low, high, method, n)`.** It uses the Student-t interval `mean +/- t(1 - alpha/2, n-1) s / sqrt(n)`, clipped to `[lo, hi]`, only when all of these hold:
    - `n >= 100`;
    - the spread is not zero, judged by max != min;
    - the sample kurtosis is at most `n / 50`. Compute it exactly as v0.1's `closed_loop.stats.confseq.paired_cs`: deviations scaled to a peak of 1, moments with `n` in the denominator. Import v0.1's constants, don't copy them.

    Otherwise it uses the betting interval: `closed_loop.stats.betting_cs` on values rescaled to `[0, 1]`, final bounds, mapped back.
  - **`paired_interval(a, b, *, lo, hi, alpha)`** is the same rule on `a - b`. Its fallback is `closed_loop.stats.paired_betting_cs`.
  - *Amended 10 November by ADR 0004:* both fallbacks are now v0.2's `betting_interval`, a fixed-sample hedged-capital betting interval, on values rescaled to `[0, 1]`. v0.1's two functions are no longer called.
  - **`binomial_interval(k, n, alpha)`** is `closed_loop.stats.clopper_pearson`.
  - `Interval.method` names which rule was used, so the report can show it.
- **`verdicts.py`:** the constants live in this module, so they are inside the code hash, and the pre-registration repeats them.
  - **`perc_verdict(gap, band=0.03)`:**
    - win if `low > band` or `high < -band`;
    - loss if `-band <= low` and `high <= band`;
    - inconclusive otherwise.
  - **`select_verdict(diff, vw, margin=0.02, floor=0.92)`:**
    - win if `diff.low >= margin` and `vw.low >= floor`;
    - loss if `diff.high < margin` or `vw.high < floor`;
    - inconclusive otherwise.
  - **`vw_state(history)` and `select_claim_verdict(state, rows)`** (amended 10 November after review D, finding N3, before the freeze):
    - `vw_state` gives `settled` if the last step did not move VW's quantiles (or there was no step), `two-cycle` if the last pair equals the one two steps back, and `unsettled` otherwise;
    - `select_claim_verdict` takes one `(diff, vw)` row for `settled` and two for `two-cycle` (the last pair and the one before it), and keeps a label only if every row gives it; `unsettled` gives inconclusive; a wrong number of rows or an unknown state raises `ValueError`;
    - `benchmarks/run.py` replays a two-cycle's other pair in the closed loop as the table cell `("VW-alt", "closed")`, not a reported method. The claim row records `vw_state` and, for a two-cycle, `diff_alt` and `vw_alt`, so `--check` can rebuild the label; `--check` also refuses a `vw_state` that is not what VW's published history gives.
  - **`ext_verdict(ci, lo=0.92, hi=0.98)`:**
    - win if `lo <= ci.low` and `ci.high <= hi`;
    - loss if the interval is disjoint from `[lo, hi]`;
    - inconclusive otherwise.
  - **Levels:** `PRIMARY_ALPHA = 0.05 / 6` and `SECONDARY_ALPHA = 0.05`.
- **`report.py`:** `render_results_md(results) -> str` and `render_readme_block(results) -> str` are pure functions of `results.json`, so `--check` can regenerate them and compare. Displayed decimals are fixed (4 for coverages and gaps, 1 for cycles), and `results.json` holds full precision. Follow v0.1's `benchmarks/run.py` report code for the layout and the simulated-data labelling, here "C-MAPSS: simulated by NASA's C-MAPSS model, not field data".
- **`benchmarks/run.py`**, the same shape and exit codes as v0.1's:
  - **`--quick`:**
    - re-splits dev units by `sha256("closed-loop-v2/quick/{subset}/{unit}")` into fit (< 0.5), calib (< 0.75) and eval;
    - runs the whole pipeline, with both conditions, every method, both loops and every verdict;
    - writes `benchmarks/.cache/quick.json`;
    - never touches the ledger or held-out data;
    - uses `benchmarks/tuned.json` if present, otherwise defaults.
  - **`--freeze`:**
    - refuses if a freeze entry exists, or if `docs/PREREGISTRATION.md` or `benchmarks/tuned.json` is missing;
    - appends `freeze` with `prereg_sha256`, `tuned_sha256` (`file_sha256`), `code_sha256` and `manifest_sha256`.
  - **`--full`** (the default):
    - refuses unless the prereg and tuned hashes equal the freeze entry's;
    - takes the heavy lock (`--lock` or `CLOSED_LOOP_V2_LOCK`, none if neither) and `require_headroom`;
    - fits the estimator on dev (nominal) and on FD001 and FD002 dev (shifted);
    - calibrates on calib;
    - reads eval units and test units through `HeldoutAccess`, which writes ledger lines;
    - replays both conditions, with `R` orders in each loop;
    - computes metrics, intervals and verdicts;
    - writes `benchmarks/results.json`, `benchmarks/RESULTS.md` and the README block between markers;
    - appends `full` with the results' sha256 and the code hash (a warning if it differs from the freeze).

    An exception after the first held-out read appends `full-failed` with the error, then re-raises.
  - **`--check`** needs no data. It:
    - verifies the chain;
    - compares the current prereg, tuned and code hashes with the freeze, saying "changed since the freeze" for code without failing, as v0.1 does;
    - checks the results' sha256 against the last `full` entry;
    - regenerates `RESULTS.md` and the README block from `results.json` and compares them byte for byte.
- **Performance:** a `--full` run in one process under about 15 minutes and 2 GB on this laptop. `--quick` under 3 minutes. Tests use small synthetic fleets.
- **Tests:**
  - `replay` with hand-made lives and a stub method that records the calls to `observe_life` and their order;
  - `replay_static` equals `replay`;
  - the open loop never replaces;
  - the closed loop stops at the first `lo <= h`;
  - `metrics` on hand-made bounds;
  - `stats`: the guard switches at `n = 99`/`100` and at the kurtosis limit, and the t interval matches scipy;
  - each verdict's edges, at and just past each boundary;
  - `report` is deterministic;
  - `run.py`:
    - `--freeze` refuses twice;
    - `--full` refuses on changed prereg bytes;
    - `--check` catches an edited `RESULTS.md`, using a temporary root with synthetic data written in the real layout;
    - `--quick` writes no ledger line.
