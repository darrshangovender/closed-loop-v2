#!/usr/bin/env python3
"""Dev-only per-unit values at the tuned operating point, for the reachability check (review D, B1).

    python scripts/dev_point.py [--lock PATH] [--out PATH]

C-MAPSS is simulated by NASA's C-MAPSS engine model, not field data.

What it reads: the 195 dev units, through ``development_units("dev")``, and
``benchmarks/tuned.json``. It never asks for calib units, eval units or NASA's test files.

Why: ``--quick`` runs at a small re-split of the dev units (35 evaluation units) with the quick
operating point, so its per-unit values do not show what the tuned design does at the registered
sizes (review D found more units in the stratum, a reachable stratum loss and t-routes at the
tuned point). This script builds per-unit values at the tuned point from every dev unit:

1. **Predictions:** ``scripts/tune.py``'s out-of-fold predictions for the chosen setting
   (``window``, ``cap`` and ``s_min`` from ``tuned.json``), with tune's seeds and outer folds, so
   that no unit is predicted by an estimator fitted on it. V1's quantiles on all 195 out-of-fold
   lives are compared with ``tuned.json``'s ``chosen_quantiles`` as a reproduction check.
2. **Two halves, cross-calibrated:** the dev units are split in two by a seeded shuffle. Each half
   is evaluated with V1, VW, V2L and V3L calibrated on the other half, through the full run's own
   ``evaluate_condition`` (``benchmarks/run.py``: the same replay, ``K``, ``R``, ``h``, ``M`` and
   per-unit metrics). Calibration therefore never sees the units it is evaluated on, as in the full
   run, but each half calibrates on about 97 units (the full run: 141) and V2L's fleet holds about
   97 lives per order (the full run: 373).
3. **Output:** ``benchmarks/.cache/dev-point.json`` with ``mode`` ``"dev-point"``, the per-unit
   values of both halves in the layout ``scripts/reachability.py`` reads from ``--quick``
   (``conditions.nominal.cells[method][loop].per_unit`` and ``eval_units``), each half's VW step
   history, policy outcomes and final ``alpha_t``, and the provenance: the dev list's SHA-256,
   ``tuned.json``'s SHA-256, the code hash, the seeds' key and the library versions.

Seeds use the key ``"dev-point"`` for the calibration draw, the orders and the halves (the halves
take ``HALF_KEY`` appended to the ``orders`` seed, a stream of their own); the out-of-fold fit uses
tune's key so that the predictions are tune's. Nothing here shares a stream with the full run.
The values are dev numbers made to check the rules and the sizes; they are not results.

The heavy-run lock comes from ``--lock`` or ``CLOSED_LOOP_V2_LOCK``; the 88% commit-charge guard
applies before the lock is taken and again before each fit and each half.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "scripts"):
    if str(extra) not in sys.path:  # run as a script from a source checkout
        sys.path.insert(0, str(extra))

import tune  # noqa: E402
from closed_loop.audit.guard import (  # noqa: E402
    GuardError,
    HeavyRunLock,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.prereg import code_sha256, file_sha256  # noqa: E402

from closed_loop_v2.audit import TUNED, heavy_run  # noqa: E402
from closed_loop_v2.conformal import fit_v1, fit_vw  # noqa: E402
from closed_loop_v2.data.load import development_units  # noqa: E402
from closed_loop_v2.estimator import Estimator, Prediction  # noqa: E402
from closed_loop_v2.policy import BoundRule  # noqa: E402
from closed_loop_v2.seeds import run_seed  # noqa: E402
from closed_loop_v2.types import Trajectory  # noqa: E402

KEY = "dev-point"
OWNER = "closed-loop-v2"
HALF_KEY = (31, 1)  # appended to the dev-point "orders" seed: the halves' own stream
OUT = REPO / "benchmarks" / ".cache" / "dev-point.json"
PER_UNIT_KEYS = ("cover", "cover_lo", "stratum_cover", "stratum_cover_lo")
Log = Callable[[str], None]


def load_run() -> ModuleType:
    """``benchmarks/run.py`` as a module (it is a script, not part of the package)."""
    name = "closed_loop_v2_run_for_dev_point"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "benchmarks" / "run.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load benchmarks/run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def halves(n_units: int, seed: int) -> NDArray[np.intp]:
    """Half (0 or 1) of each dev unit: a seeded shuffle dealt alternately (sizes within one)."""
    rng = np.random.default_rng([seed, *HALF_KEY])
    out = np.empty(n_units, dtype=np.intp)
    out[rng.permutation(n_units)] = np.arange(n_units) % 2
    return out


@dataclasses.dataclass(frozen=True)
class OutOfFold:
    """Stands in for the fitted estimator in ``evaluate_condition``: each unit's prediction comes
    from the out-of-fold estimator that was not fitted on it."""

    by_name: dict[str, Prediction]
    condition_model: Any

    def predict(self, t: Trajectory) -> Prediction:
        return self.by_name[tune.unit_name(t)]


def merge_cells(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The per-unit values of the halves, concatenated in half order, per method and loop."""
    methods = parts[0]["cells"]
    cells: dict[str, Any] = {}
    for method, loops in methods.items():
        cells[method] = {}
        for loop in loops:
            cells[method][loop] = {
                "per_unit": {
                    key: [v for p in parts for v in p["cells"][method][loop]["per_unit"][key]]
                    for key in PER_UNIT_KEYS
                },
                "policy_by_half": [p["cells"][method][loop]["policy"] for p in parts],
                "stratum_units_by_half": [p["cells"][method][loop]["stratum_units"] for p in parts],
            }
    return cells


def dev_point(
    units: Sequence[Trajectory],
    tuned_path: Path,
    guard: Callable[[], None],
    log: Log,
) -> dict[str, Any]:
    run = load_run()
    tuned = run.load_tuned(tuned_path)
    design = run.run_design(tuned)
    cap = run.truth_cap(tuned)
    params = tuned.params
    # tune.out_of_fold fits with tune's boosting settings; they must be the ones tuned.json records
    tuned_choices = ("window", "cap", "s_min")
    differ = [
        f.name
        for f in dataclasses.fields(params)
        if f.name not in tuned_choices
        and getattr(params, f.name) != getattr(tune.BASE_PARAMS, f.name)
    ]
    if differ:
        raise ValueError(
            f"tuned.json's boosting settings differ from tune's in {', '.join(differ)}; the "
            "out-of-fold fit would not be tune's"
        )
    capped = [t.capped(cap) for t in units]

    # 1. tune's out-of-fold predictions for the chosen setting
    fit_seed = run_seed(tune.TUNE_KEY, "fit-nominal")
    folds = tune.outer_folds(len(capped), fit_seed)
    ests: list[Estimator | None] = tune.out_of_fold(
        capped, params.window, float(params.cap), seed=fit_seed, folds=folds, guard=guard, log=log
    )
    preds = tune.predictions(capped, ests, params.s_min)
    lives = list(zip(capped, preds, strict=True))
    q_all = fit_v1(lives, alpha=tune.ALPHA, seed=run_seed(tune.TUNE_KEY, "calib-draw")).q
    record = json.loads(tuned_path.read_text(encoding="utf-8")).get("record", {})
    want = record.get("chosen_quantiles", {})
    reproduces = q_all.q_lo == want.get("q_lo") and q_all.q_hi == want.get("q_hi")
    log(f"V1 on all out-of-fold lives: ({q_all.q_lo!r}, {q_all.q_hi!r}); tuned.json {want}")
    first = next(e for e in ests if e is not None)
    oof = OutOfFold({tune.unit_name(t): p for t, p in lives}, first.condition_model)

    # 2. two halves, each evaluated with methods calibrated on the other
    half = halves(len(lives), run_seed(KEY, "orders"))  # its own stream: HALF_KEY appended
    calib_seed = run_seed(KEY, "calib-draw")
    parts: list[dict[str, Any]] = []
    calib_names: list[list[str]] = []
    for h in (0, 1):
        guard()
        evaluate = [t for t, g in zip(capped, half, strict=True) if g == h]
        calib = [lv for lv, g in zip(lives, half, strict=True) if g != h]
        v1 = fit_v1(calib, alpha=design.alpha, seed=calib_seed)
        vw = fit_vw(
            calib, BoundRule(tuned.h), alpha=design.alpha, seed=calib_seed, steps=design.vw_steps
        )
        fitted = run.Fitted(oof, calib, v1, vw)
        summary, _ = run.evaluate_condition("nominal", fitted, evaluate, tuned, design, KEY, log)
        parts.append(summary)
        calib_names.append([tune.unit_name(t) for t, _ in calib])
        log(f"half {h}: evaluated {len(evaluate)} units, calibrated on {len(calib)}")

    names = [n for p in parts for n in p["eval_units"]]
    return {
        "mode": "dev-point",
        "data": "C-MAPSS, simulated by NASA's C-MAPSS model; dev units only (no calib, eval or "
        "NASA test unit)",
        "operating_point": {
            "window": params.window,
            "cap": params.cap,
            "s_min": params.s_min,
            "h": tuned.h,
            "stratum_m": tuned.stratum_m,
            "k": design.k,
            "r": design.r,
            "alpha": design.alpha,
            "gamma": design.gamma,
            "aci_window": design.aci_window,
            "vw_steps": design.vw_steps,
        },
        "reproduction": {
            "v1_all_oof": [q_all.q_lo, q_all.q_hi],
            "tuned_json_chosen_quantiles": want,
            "reproduces_tuned_json": bool(reproduces),
        },
        "halves": [
            {
                "eval_units": len(p["eval_units"]),
                "calib_units": len(c),
                "calib_unit_names": c,
                "methods": p["methods"],
            }
            for p, c in zip(parts, calib_names, strict=True)
        ],
        "conditions": {"nominal": {"eval_units": names, "cells": merge_cells(parts)}},
        "provenance": {
            "dev_list_sha256": tune.dev_list_sha256(units),
            "tuned_sha256": file_sha256(tuned_path),
            "code_sha256": code_sha256(
                [REPO / "closed_loop_v2", REPO / "benchmarks" / "run.py", REPO / "scripts"]
            ),
            "code_roots": ["closed_loop_v2/", "benchmarks/run.py", "scripts/"],
            "seed_key": KEY,
            "fit_seed_key": tune.TUNE_KEY,
            "versions": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "scipy": version("scipy"),
                "scikit-learn": version("scikit-learn"),
                "closed-loop": version("closed-loop"),
            },
        },
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Dev-only values at the tuned point (B1).")
    parser.add_argument("--lock", default=None, help="heavy-run lock file")
    parser.add_argument("--tuned", type=Path, default=TUNED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def guard() -> None:
        require_headroom(commit_charge_percent)

    lock: AbstractContextManager[object]
    try:
        require_headroom(commit_charge_percent)
        task = "scripts/dev_point.py"
        lock = (
            HeavyRunLock(Path(args.lock), OWNER, task)
            if args.lock
            else heavy_run(OWNER, task, reader=commit_charge_percent)
        )
        with lock:
            units = development_units("dev")
            log(f"{len(units)} dev units loaded; dev list sha256 {tune.dev_list_sha256(units)}")
            out = dev_point(units, args.tuned, guard, log)
    except GuardError as exc:
        print(f"refused or stopped: {exc}", file=sys.stderr)
        return 3
    text = json.dumps(out, indent=1) + "\n"
    _write(args.out, text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log(f"wrote {args.out} (sha256 {digest})")
    if not out["reproduction"]["reproduces_tuned_json"]:
        print("error: V1's quantiles do not reproduce tuned.json (file kept)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
