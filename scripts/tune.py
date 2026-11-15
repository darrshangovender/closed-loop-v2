#!/usr/bin/env python3
"""Tune the free choices on dev units only (docs/adr/0003-tuning-protocol.md).

    python scripts/tune.py [--lock PATH] [--json PATH] [--md PATH]

C-MAPSS is simulated by NASA's C-MAPSS engine model, not field data.

What it reads: the 195 dev units of FD001 to FD004, through ``development_units("dev")``. It never
asks for calib units, eval units or NASA's test files. What it writes: ``benchmarks/tuned.json``
(the chosen values, the grid, every setting's numbers and the hash of the dev unit list) and
``docs/TUNING.md``, rendered from it. Both are written only after every step has finished.

The steps, each mechanical (ADR 0003 sections 2 to 4):

1. **Estimator grid.** ``window`` in {15, 30, 50} x ``s_min`` in {2, 5}, with ``cap`` fixed at 125
   (ADR 0003 section 7: the truth is ``min(remaining life, cap)``, so widths under different caps
   measure different quantities and cannot choose the cap). Every unit is capped before fitting.
   Every dev unit gets out-of-fold predictions from unit-grouped 5-fold cross-fitting: for each
   outer fold an ``Estimator`` is fitted on the other four folds. ``s_min`` is a floor applied at
   prediction time, so the two ``s_min`` values share one fit and differ only in ``predict``.
   V1's quantiles come from one uniformly drawn row per unit. The criterion is the mean of
   ``s (q_hi - q_lo)`` over every row of every dev unit; the smallest wins, and a tie within 1% goes
   to the smaller ``window``, then the smaller ``cap``, then the larger ``s_min``.
2. **Threshold ``h``** in {0, 5, ..., 30}. The closed loop is replayed with V1 on the chosen
   setting's out-of-fold predictions. The choice is the smallest mean remaining life at replacement
   among the values whose unplanned-failure fraction is at most 0.05; if none qualifies, the value
   with the lowest fraction (ties to the smaller ``h``), and the file says so.
3. **Stratum ``M``** in {5, 10, ...} up to ``cap``: the smallest value at which at least 90% of dev
   units have a row with ``m <= M`` before replacement, in the replay of step 2 at the chosen ``h``.

V1's quantiles never change, so each life's outcome does not depend on the fleet order
(``closed_loop_v2.replay.replay_static``); the "every order" and "averaged over orders" of ADR 0003
therefore reduce to one static replay, as in ``closed_loop_v2.metrics``.

Seeds use the key ``"tune"`` in place of the pre-registration hash (``closed_loop_v2.seeds``), so
tuning shares no stream with the full run. The outer folds have a stream of their own.

The heavy-run lock comes from ``--lock`` or ``CLOSED_LOOP_V2_LOCK``; the 88% commit-charge guard
applies before the lock is taken and again before each fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a script from a source checkout
    sys.path.insert(0, str(REPO))

from closed_loop.audit.guard import (  # noqa: E402
    GuardError,
    HeavyRunLock,
    commit_charge_percent,
    require_headroom,
)

from closed_loop_v2.audit import TUNED, heavy_run  # noqa: E402
from closed_loop_v2.conformal import Quantiles, bounds, cap_of, fit_v1  # noqa: E402
from closed_loop_v2.data.load import development_units  # noqa: E402
from closed_loop_v2.estimator import DEFAULT_PARAMS, Estimator, Prediction  # noqa: E402
from closed_loop_v2.metrics import life_metrics, policy_outcomes  # noqa: E402
from closed_loop_v2.policy import BoundRule  # noqa: E402
from closed_loop_v2.replay import replay_static  # noqa: E402
from closed_loop_v2.seeds import run_seed  # noqa: E402
from closed_loop_v2.types import Trajectory  # noqa: E402

TUNE_KEY = "tune"
OWNER = "closed-loop-v2"
ALPHA = 0.10
N_FOLDS = 5
WINDOWS: tuple[int, ...] = (15, 30, 50)
CAPS: tuple[float, ...] = (125.0,)  # fixed, not tuned (ADR 0003 section 7)
S_MINS: tuple[float, ...] = (2.0, 5.0)
TIE_REL = 0.01
H_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
FAIL_LIMIT = 0.05
M_STEP = 5.0
M_COVER = 0.90
_OUTER_KEY = (29, 1)  # appended to the fit seed: the outer folds' own stream
TUNING_MD = REPO / "docs" / "TUNING.md"
BASE_PARAMS = DEFAULT_PARAMS  # the fixed boosting settings (ADR 0003 section 1); tests shrink them

Lives = list[tuple[Trajectory, Prediction]]
Log = Callable[[str], None]


def unit_name(t: Trajectory) -> str:
    return f"{t.subset}/{t.unit}"


def dev_list_sha256(units: Sequence[Trajectory]) -> str:
    """SHA-256 of the dev unit names (``FD00x/unit``), one per line in load order."""
    text = "".join(f"{unit_name(t)}\n" for t in units)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def outer_folds(n_units: int, seed: int) -> NDArray[np.intp]:
    """Outer fold of each dev unit: a seeded shuffle dealt round-robin (fold sizes within one)."""
    rng = np.random.default_rng([seed, *_OUTER_KEY])
    folds = np.empty(n_units, dtype=np.intp)
    folds[rng.permutation(n_units)] = np.arange(n_units) % N_FOLDS
    return folds


def out_of_fold(
    units: Sequence[Trajectory],
    window: int,
    cap: float,
    *,
    seed: int,
    folds: NDArray[np.intp],
    guard: Callable[[], None],
    log: Log,
) -> list[Estimator | None]:
    """One fitted estimator per unit position: the one fitted on the four folds without it.

    The estimators are fitted with the smallest ``s_min`` of the grid; ``predictions`` raises the
    floor for a larger ``s_min`` without refitting (``s_min`` is applied in ``predict`` only).
    """
    params = dataclasses.replace(BASE_PARAMS, window=window, cap=cap, s_min=min(S_MINS))
    by_unit: list[Estimator | None] = [None] * len(units)
    for f in range(N_FOLDS):
        guard()
        train = [t for t, g in zip(units, folds, strict=True) if g != f]
        t0 = time.perf_counter()
        est = Estimator.fit(train, params, seed=seed, n_folds=N_FOLDS)
        for i, g in enumerate(folds):
            if g == f:
                by_unit[i] = est
        log(
            f"  window={window} cap={cap:g} fold {f}: {len(train)} units, "
            f"{time.perf_counter() - t0:.1f} s"
        )
    return by_unit


def predictions(
    units: Sequence[Trajectory], ests: Sequence[Estimator | None], s_min: float
) -> list[Prediction]:
    out = []
    for t, est in zip(units, ests, strict=True):
        if est is None:
            raise RuntimeError(f"{unit_name(t)} has no out-of-fold estimator")
        floored = dataclasses.replace(est, params=dataclasses.replace(est.params, s_min=s_min))
        out.append(floored.predict(t))
    return out


def setting_numbers(lives: Lives, q: Quantiles, cap: float) -> dict[str, float]:
    """Mean width, pooled interval coverage and RMSE of ``m`` against ``min(RUL, cap)``, by row."""
    widths, hits, sq = [], [], []
    for traj, pred in lives:
        rul = traj.rul().astype(np.float64)
        lo, hi = bounds(pred, q, cap_of(traj))  # limited to the cap, as in the replay
        widths.append(pred.s * (q.q_hi - q.q_lo))  # the criterion: the unlimited width (ADR 0003)
        hits.append((lo <= rul) & (rul <= hi))
        sq.append((pred.m - np.minimum(rul, cap)) ** 2)
    return {
        "mean_width": float(np.concatenate(widths).mean()),
        "pooled_coverage": float(np.concatenate(hits).mean()),
        "rmse_m": float(math.sqrt(np.concatenate(sq).mean())),
        "rows": int(sum(len(w) for w in widths)),
    }


def choose_setting(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Smallest mean width; within 1% of it, smaller window, then smaller cap, then larger s_min."""
    best = min(r["mean_width"] for r in rows)
    near = [r for r in rows if r["mean_width"] <= best * (1.0 + TIE_REL)]
    return min(near, key=lambda r: (r["window"], r["cap"], -r["s_min"]))


def choose_h(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """``(row, met)``: the smallest mean remaining life at replacement with failures at most 5%.

    Ties go to the smaller ``h``. With no qualifying value, the lowest failure fraction (ties to the
    smaller ``h``) and ``met = False``. A value with no replacement has no mean remaining life and
    cannot qualify.
    """
    ok = [
        r
        for r in rows
        if r["failed_fraction"] <= FAIL_LIMIT and r["mean_rul_at_replacement"] is not None
    ]
    if ok:
        return min(ok, key=lambda r: (r["mean_rul_at_replacement"], r["h"])), True
    return min(rows, key=lambda r: (r["failed_fraction"], r["h"])), False


def m_grid(cap: float) -> list[float]:
    return [M_STEP * i for i in range(1, int(cap // M_STEP) + 1)]


def tune(units: Sequence[Trajectory], guard: Callable[[], None], log: Log) -> dict[str, Any]:
    fit_seed = run_seed(TUNE_KEY, "fit-nominal")
    draw_seed = run_seed(TUNE_KEY, "calib-draw")
    folds = outer_folds(len(units), fit_seed)

    grid: list[dict[str, Any]] = []
    lives_of: dict[tuple[int, float, float], Lives] = {}
    q_of: dict[tuple[int, float, float], Quantiles] = {}
    for window in WINDOWS:
        for cap in CAPS:
            capped = [t.capped(int(cap)) for t in units]  # the truth is min(remaining life, cap)
            ests = out_of_fold(
                capped, window, cap, seed=fit_seed, folds=folds, guard=guard, log=log
            )
            for s_min in S_MINS:
                lives = list(zip(capped, predictions(capped, ests, s_min), strict=True))
                q = fit_v1(lives, alpha=ALPHA, seed=draw_seed).q
                nums = setting_numbers(lives, q, cap)
                grid.append(
                    {
                        "window": window,
                        "cap": cap,
                        "s_min": s_min,
                        "q_lo": q.q_lo,
                        "q_hi": q.q_hi,
                        **nums,
                    }
                )
                lives_of[(window, cap, s_min)] = lives
                q_of[(window, cap, s_min)] = q
                log(
                    f"window={window} cap={cap:g} s_min={s_min:g}: width {nums['mean_width']:.4f}, "
                    f"coverage {nums['pooled_coverage']:.4f}, RMSE {nums['rmse_m']:.4f}"
                )
    chosen = choose_setting(grid)
    key = (chosen["window"], chosen["cap"], chosen["s_min"])
    lives, q = lives_of[key], q_of[key]
    log(f"chosen: window={key[0]} cap={key[1]:g} s_min={key[2]:g}")

    h_rows: list[dict[str, Any]] = []
    runs = {}
    for h in H_GRID:
        run = replay_static(lives, q, BoundRule(h))
        po = policy_outcomes([life_metrics(lives, run, stratum_m=0.0)])
        runs[h] = run
        h_rows.append(
            {
                "h": h,
                "failed_fraction": po.failed_fraction,
                "replacements": po.replacements,
                "mean_rul_at_replacement": po.mean_rul_at_replacement,
            }
        )
        log(
            f"h={h:g}: failed {po.failed_fraction:.4f}, mean RUL at replacement "
            f"{po.mean_rul_at_replacement}"
        )
    h_row, met = choose_h(h_rows)
    h = float(h_row["h"])

    m_rows: list[dict[str, Any]] = []
    stratum_m = None
    for m in m_grid(chosen["cap"]):
        lm = life_metrics(lives, runs[h], stratum_m=m)
        frac = float((lm.stratum_rows > 0).mean())
        m_rows.append(
            {"M": m, "units_with_stratum_rows": int((lm.stratum_rows > 0).sum()), "fraction": frac}
        )
        if frac >= M_COVER:
            stratum_m = m
            break
    if stratum_m is None:
        raise RuntimeError(f"no M up to cap {chosen['cap']} gives {M_COVER:.0%} of dev units")
    log(f"h={h:g} (limit met: {met}); M={stratum_m:g}")

    params = dataclasses.replace(BASE_PARAMS, window=key[0], cap=key[1], s_min=key[2])
    return {
        "estimator": dataclasses.asdict(params),
        "h": h,
        "stratum_m": stratum_m,
        "tuned_on": f"{len(units)} dev units of FD001-FD004 (C-MAPSS, simulated by NASA); "
        f"dev list sha256 {dev_list_sha256(units)}",
        "criterion": "ADR 0003: estimator by minimum mean V1 width on dev out-of-fold predictions "
        "(1% tie rule); h by minimum mean remaining life at replacement with an "
        "unplanned-failure fraction of at most 0.05; M by the smallest 5-cycle value "
        "with at least 90% of dev units having a stratum row before replacement",
        "script": "scripts/tune.py",
        "notes": (
            "h limit met"
            if met
            else "no h met the 0.05 failure limit; the lowest failure fraction was taken"
        ),
        "record": {
            "protocol": "docs/adr/0003-tuning-protocol.md",
            "data": "C-MAPSS, simulated by NASA's C-MAPSS model, not field data",
            "dev_units": len(units),
            "dev_list_sha256": dev_list_sha256(units),
            "seed_key": TUNE_KEY,
            "alpha": ALPHA,
            "n_folds": N_FOLDS,
            "fixed": {
                k: v
                for k, v in dataclasses.asdict(BASE_PARAMS).items()
                if k not in ("window", "cap", "s_min")
            },
            "estimator_grid": grid,
            "chosen_quantiles": {"q_lo": q.q_lo, "q_hi": q.q_hi},
            "h_grid": h_rows,
            "h_limit_met": met,
            "fail_limit": FAIL_LIMIT,
            "m_grid": m_rows,
            "m_cover": M_COVER,
        },
    }


def _num(x: float | None, places: int = 4) -> str:
    return "none" if x is None else f"{x:.{places}f}"


def render_md(tuned: dict[str, Any]) -> str:
    rec = tuned["record"]
    est = tuned["estimator"]
    lines = [
        "# Tuning on dev units",
        "",
        "Rendered from `benchmarks/tuned.json` by `scripts/tune.py`; the protocol is "
        "`docs/adr/0003-tuning-protocol.md`, written before this run. "
        "C-MAPSS is simulated by NASA's C-MAPSS model, not field data.",
        "",
        f"- **Data:** {rec['dev_units']} dev units of FD001 to FD004 only. "
        f"Dev list SHA-256 `{rec['dev_list_sha256']}`. No calib, eval or NASA test data was read.",
        f"- **Seeds:** key `{rec['seed_key']}`, apart from the full run's.",
        f"- **Chosen:** window {est['window']}, cap {est['cap']:g}, s_min {est['s_min']:g}; "
        f"h {tuned['h']:g}; M {tuned['stratum_m']:g}.",
        f"- **Note:** {tuned['notes']}.",
        "",
        "## Estimator grid (V1 on out-of-fold predictions; one drawn row per unit)",
        "",
        "Width and coverage are means over every row of every dev unit. Coverage is not part of "
        "the criterion.",
        "",
        "| window | cap | s_min | mean width | pooled coverage | RMSE of m | q_lo | q_hi |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rec["estimator_grid"]:
        lines.append(
            f"| {r['window']} | {r['cap']:g} | {r['s_min']:g} | {r['mean_width']:.4f} | "
            f"{r['pooled_coverage']:.4f} | {r['rmse_m']:.4f} | {r['q_lo']:.4f} | {r['q_hi']:.4f} |"
        )
    lines += [
        "",
        "## Threshold h (closed loop, V1, chosen estimator)",
        "",
        f"Limit: unplanned-failure fraction at most {rec['fail_limit']}. "
        f"Limit met: {'yes' if rec['h_limit_met'] else 'no'}.",
        "",
        "| h | failed fraction | replacements | mean RUL at replacement |",
        "|---|---|---|---|",
    ]
    for r in rec["h_grid"]:
        lines.append(
            f"| {r['h']:g} | {r['failed_fraction']:.4f} | {r['replacements']:g} | "
            f"{_num(r['mean_rul_at_replacement'])} |"
        )
    lines += [
        "",
        f"## Stratum M (closed loop at h = {tuned['h']:g})",
        "",
        f"The smallest value with at least {rec['m_cover']:.0%} of dev units having a row with "
        "m <= M before replacement. Values are listed up to the one chosen.",
        "",
        "| M | units with stratum rows | fraction |",
        "|---|---|---|",
    ]
    for r in rec["m_grid"]:
        lines.append(f"| {r['M']:g} | {r['units_with_stratum_rows']} | {r['fraction']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tune on dev units (ADR 0003).")
    parser.add_argument("--lock", default=None, help="heavy-run lock file")
    parser.add_argument("--json", type=Path, default=TUNED)
    parser.add_argument("--md", type=Path, default=TUNING_MD)
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def guard() -> None:
        require_headroom(commit_charge_percent)

    lock: AbstractContextManager[object]
    try:
        require_headroom(commit_charge_percent)
        task = "scripts/tune.py"
        lock = (
            HeavyRunLock(Path(args.lock), OWNER, task)
            if args.lock
            else heavy_run(OWNER, task, reader=commit_charge_percent)
        )
        with lock:
            units = development_units("dev")
            log(f"{len(units)} dev units loaded; dev list sha256 {dev_list_sha256(units)}")
            tuned = tune(units, guard, log)
    except GuardError as exc:
        print(f"refused or stopped: {exc}", file=sys.stderr)
        return 3
    text = json.dumps(tuned, indent=2, sort_keys=False) + "\n"
    _write(args.json, text)
    _write(args.md, render_md(tuned))
    log(f"wrote {args.json} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
