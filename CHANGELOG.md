# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-15

First public release. The data are NASA's C-MAPSS turbofan trajectories, which are simulated by
NASA's C-MAPSS engine model, not field data. The repository does not redistribute them.

### Added
- The data guard, keyed-hash roles, estimator, the four interval methods (V1, VW, V2L, V3L), the
  bound-rule replay, the statistics and the verdict rules, in `closed_loop_v2/`.
- `benchmarks/run.py` with `--quick`, `--freeze`, `--full` and `--check`.
- `docs/PREREGISTRATION.md`, frozen on 10 November 2026, with the ledger in `benchmarks/ledger.jsonl`.
- One full run after the freeze, on 10 November 2026, archived as `benchmarks/runs/results-run-1.json`
  and reported in `benchmarks/RESULTS.md` and the README.
- `scripts/fetch_cmapss.py`, which downloads NASA's zip and checks it against `data/MANIFEST.json`.

### Changed after the run
None of these touches the hashed code, the pre-registration or `benchmarks/tuned.json`.
- The README's prose around the generated results block, checked claim by claim against the records.
- The README gives the timings and the Python 3.11 result of a clean install on 11 November 2026.
- CI now ends with `python benchmarks/run.py --check`.
