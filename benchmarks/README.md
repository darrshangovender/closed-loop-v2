# benchmarks

`run.py` is the single entry point. C-MAPSS is simulated by NASA's C-MAPSS engine model; it is not
field data. The data directory comes from `CLOSED_LOOP_V2_DATA`, and the heavy-run lock from
`--lock` or `CLOSED_LOOP_V2_LOCK`. Every data run first checks commit charge and stops at 88%.

| Mode | What it does | Writes |
|---|---|---|
| `--quick` | Re-splits the development units by a hash of subset and unit number into fit, calibration and evaluation, then runs the whole pipeline on them. Its external check uses the quick evaluation units cut at a hashed row as a stand-in for NASA's test units. Reads no held-out file and no ledger. Labelled "QUICK RUN, NOT A RESULT". | `.cache/quick.json`, `.cache/quick.md` |
| `--freeze` | Records the hashes of `docs/PREREGISTRATION.md`, `tuned.json`, the code and the data manifest, plus the design constants (with the tuned `M`) and the seed labels. Refuses if the ledger has any entry; if the pre-registration or a complete `tuned.json` is missing; if the pre-registration still contains `[T]`, `[R]`, `DRAFT` or `Placeholders:`; or if it does not contain, verbatim, the registered-values table (the refusal prints the table to paste). | one `freeze` line in `ledger.jsonl` |
| `--full` (default) | Refuses unless the pre-registration, `tuned.json` and manifest still match the freeze. Stops as "not run" (exit 4, no ledger line) if the README's result markers are broken, or on a missing data folder, an invalid `tuned.json` or another `OSError` before the first held-out read. Fits, calibrates, then reads the evaluation units and NASA's test units through `HeldoutAccess` (one ledger line each), replays both conditions and both loops, keeps the results text in `.cache/full-unpublished.json`, and publishes. Any failure once held-out data has been read, including one while publishing or writing the `full` line, is recorded as `full-failed`. | `results.json`, `RESULTS.md`, the README results block, `runs/results-run-{n}.json`, `heldout-use` and `full` lines |
| `--check` | Needs no data. Verifies the ledger chain and the hashes against the freeze (a code change is reported, not a failure), the results' sha256 against the last `full` line, each archived run against its `full` line, regenerates `RESULTS.md` and the README block byte for byte, lints them and rebuilds every verdict from its stored intervals. | nothing |

`python benchmarks/run.py --registered-table` (not listed in `--help`) prints the registered-values
table from `DESIGN`, `tuned.json` and the verdict constants, with every number at full precision. It
reads no data and no ledger. That table is pasted into the pre-registration before `--freeze`.

Exit codes: 0 done; 1 a check failed or a run was refused; 2 usage; 3 refused or paused (commit
charge, lock held); 4 not run (for example, missing data or broken README markers, before any
held-out read).

**Reruns.** Each full run's `results.json` is also kept as `runs/results-run-{n}.json`, where `n`
counts the `full` lines of the ledger, this run included. A file that already exists is not
replaced; the copy takes an `-again` suffix instead, as in v0.1. `RESULTS.md` lists the earlier full
runs, and each `full` line records its run number and archive path.

`tuned.json` holds the estimator's parameters, `h`, the decision stratum `M` (`stratum_m`),
free-text provenance (`tuned_on`, `criterion`, `script`, `notes`) and the tuning grids under
`record`, the shape `scripts/tune.py` writes. `run.py` reads `stratum_m` in place of
`DESIGN.stratum_m`, which is only the fallback when the file lacks it; `record` must be an object
and is not read. Any other key is refused. The design constants (`k`, `R`, the VW step count and
the rest) sit in `run.py` as `DESIGN`.
