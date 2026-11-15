#!/usr/bin/env python3
"""Can each primary verdict be reached at the registered sizes? (pre-registration section 7)

    python scripts/reachability.py [--input PATH ...] [--json PATH] [--md PATH]

C-MAPSS is simulated by NASA's C-MAPSS engine model, not field data. Everything here is built from
dev units only. No calib, eval or NASA test unit is read. There are two inputs, each an operating
point (review D, B1):

* **quick**: the per-unit values of a ``benchmarks/run.py --quick`` run (35 evaluation units from a
  re-split of the dev units, at the quick run's own quantiles);
* **dev-point**: the per-unit values of ``scripts/dev_point.py`` (all 195 dev units at the tuned
  operating point, from tune's out-of-fold predictions, each half calibrated on the other).

Each input's SHA-256, mode and code hash are recorded with its results (review D, N7).

For each primary row the script takes the dev per-unit values, resamples them with replacement to
the registered size (373 eval units; the stratum rows at 373 times the dev share of units kept in
the stratum), and runs the same interval and verdict code the full run uses
(``closed_loop_v2.stats``, ``closed_loop_v2.verdicts``) at the primary level ``0.05 / 6``. Shifted
inputs are made by lowering one side of each pair (never raising, so nothing is clipped at 1), in
two ways (review D, S5):

* **uniform**: every unit moves by the same amount (floored at 0). This turns the dev spread into a
  near-constant vector, so its win thresholds sit near the interval's floor plus the band;
* **concentrated**: the whole shift is carried by 10% of the units (``CONCENTRATED_SHARE``, chosen
  once per resample), floored at 0, so the mean can stop short of the target; the median mean
  reached is reported with each threshold.

The labels they get are reported:

* **C2-PERC** (four rows): the label with the values as they are, at a gap of 0, at a gap of
  ``-band``, the smallest gap (in steps of 0.005) that gets a win in every resample, and the labels
  at the fixed gaps -0.055, -0.08 and -0.1.
* **C2-SELECT**: the label with VW's values as they are, with the difference moved to 0, and the
  smallest difference (steps of 0.005) that gets a win in every resample with VW's values as they
  are.
* **C2-EXT**: Clopper-Pearson at 707 units for a coverage of 0.95, 0.92 and 0.85 (no dev values
  needed).

Each row also gives the dev mean, standard deviation and kurtosis of the differences, whether v0.1's
t-guard would hold at the registered size, and the median half-width over the resamples. A
threshold "in every resample" is over ``RESAMPLES`` resamples; the label counts are given with it.
These are made-up inputs for checking the rules and the sizes; they are not results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a script from a source checkout
    sys.path.insert(0, str(REPO))

from closed_loop_v2.stats import (  # noqa: E402
    Interval,
    binomial_interval,
    kurtosis,
    mean_interval,
    paired_interval,
)
from closed_loop_v2.verdicts import (  # noqa: E402
    PERC_BAND,
    PRIMARY_ALPHA,
    SELECT_FLOOR,
    SELECT_MARGIN,
    ext_verdict,
    perc_verdict,
    select_verdict,
    vw_state,
)

N_EVAL = 373
N_EXTERNAL = 707
RESAMPLES = 20
STEP = 0.005
MAX_SHIFT = 0.5
SEED = 20261110
CONCENTRATED_SHARE = 0.10
FIXED_GAPS = (-0.055, -0.08, -0.1)  # the gaps of review D's S5 table
MODES = ("quick", "dev-point")
QUICK = REPO / "benchmarks" / ".cache" / "quick.json"
DEV_POINT = REPO / "benchmarks" / ".cache" / "dev-point.json"
OUT_JSON = REPO / "benchmarks" / "reachability.json"
OUT_MD = REPO / "docs" / "REACHABILITY.md"

Pairs = tuple[NDArray[np.float64], NDArray[np.float64]]
Shifter = Callable[[int, NDArray[np.float64], NDArray[np.float64], float], Pairs]


def _vec(values: Sequence[float | None]) -> NDArray[np.float64]:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def per_unit(quick: dict[str, Any], method: str, loop: str, key: str) -> NDArray[np.float64]:
    cell = quick["conditions"]["nominal"]["cells"][method][loop]
    return _vec(cell["per_unit"][key])


def perc_pairs(quick: dict[str, Any], method: str, row: str) -> Pairs:
    """Closed and open lower-bound coverage of ``method`` by unit (stratum: kept in both loops)."""
    key = "cover_lo" if row == "pooled" else "stratum_cover_lo"
    a = per_unit(quick, method, "closed", key)
    b = per_unit(quick, method, "open", key)
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def lower_to_mean(a: NDArray[np.float64], b: NDArray[np.float64], target: float) -> Pairs:
    """Move ``mean(a - b)`` to ``target`` by lowering ``a`` (to go down) or ``b`` (to go up).

    Values are only lowered, and floored at 0; with a floor hit the mean misses the target, which
    the caller sees in the interval's estimate.
    """
    shift = target - float(np.mean(a - b))
    if shift < 0:
        return np.maximum(a + shift, 0.0), b.copy()
    return a.copy(), np.maximum(b - shift, 0.0)


def concentrate(
    a: NDArray[np.float64], b: NDArray[np.float64], target: float, carriers: NDArray[np.intp]
) -> Pairs:
    """As ``lower_to_mean``, but only the units at ``carriers`` move, each by the whole shift scaled
    up by ``n / len(carriers)``; floored at 0, so the mean can stop short of ``target``."""
    shift = (target - float(np.mean(a - b))) * a.size / carriers.size
    a2, b2 = a.copy(), b.copy()
    if shift < 0:
        a2[carriers] = np.maximum(a[carriers] + shift, 0.0)
    else:
        b2[carriers] = np.maximum(b[carriers] - shift, 0.0)
    return a2, b2


def resample(pairs: Pairs, n: int, rng: np.random.Generator) -> Pairs:
    idx = rng.integers(0, pairs[0].size, size=n)
    return pairs[0][idx], pairs[1][idx]


def carriers_for(n: int, rng: np.random.Generator) -> NDArray[np.intp]:
    """The units that carry a concentrated shift: ``ceil(share * n)`` of ``n``, without repeats."""
    m = max(1, math.ceil(CONCENTRATED_SHARE * n))
    return np.sort(rng.choice(n, size=m, replace=False)).astype(np.intp)


def shifters(samples: Sequence[Pairs], rng: np.random.Generator) -> dict[str, Shifter]:
    """The two ways of moving resample ``i`` to a target mean difference (fixed carriers per i)."""
    carriers = [carriers_for(a.size, rng) for a, _ in samples]

    def uniform(i: int, a: NDArray[np.float64], b: NDArray[np.float64], target: float) -> Pairs:
        return lower_to_mean(a, b, target)

    def concentrated(
        i: int, a: NDArray[np.float64], b: NDArray[np.float64], target: float
    ) -> Pairs:
        return concentrate(a, b, target, carriers[i])

    return {"uniform": uniform, "concentrated": concentrated}


def _half(iv: Interval) -> float:
    return (iv.high - iv.low) / 2.0


def _median_mean(shifted: Sequence[Pairs]) -> float:
    return float(np.median([float(np.mean(a - b)) for a, b in shifted]))


def describe(d: NDArray[np.float64], n_target: int) -> dict[str, Any]:
    k = kurtosis(d)
    return {
        "dev_units": int(d.size),
        "dev_mean": float(d.mean()) if d.size else None,
        "dev_sd": float(d.std(ddof=1)) if d.size > 1 else None,
        "dev_kurtosis": k,
        "n_target": n_target,
        "guard_limit": n_target / 50.0,
        "guard_holds_at_dev_kurtosis": bool(
            n_target >= 100 and k is not None and k <= n_target / 50.0
        ),
    }


def _labels_over(
    samples: Sequence[Pairs], fn: Callable[[NDArray[np.float64], NDArray[np.float64]], Any]
) -> dict[str, int]:
    out: dict[str, int] = {}
    for a, b in samples:
        lab = fn(a, b)
        out[lab] = out.get(lab, 0) + 1
    return out


def perc_row(name: str, pairs: Pairs, n_target: int, rng: np.random.Generator) -> dict[str, Any]:
    samples = [resample(pairs, n_target, rng) for _ in range(RESAMPLES)]
    moves = shifters(samples, rng)

    def iv(a: NDArray[np.float64], b: NDArray[np.float64]) -> Interval:
        return paired_interval(a, b, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)

    def verdict(a: NDArray[np.float64], b: NDArray[np.float64]) -> str:
        return perc_verdict(iv(a, b))

    def variant(move: Shifter) -> dict[str, Any]:
        def shifted(gap: float) -> list[Pairs]:
            return [move(i, a, b, gap) for i, (a, b) in enumerate(samples)]

        smallest_win = None
        for i in range(1, int(MAX_SHIFT / STEP) + 1):
            gap = -STEP * i
            if _labels_over(shifted(gap), verdict).get("win", 0) == RESAMPLES:
                smallest_win = gap
                break
        at_fixed = {}
        for gap in FIXED_GAPS:
            got = shifted(gap)
            at_fixed[f"{gap:g}"] = {
                "labels": _labels_over(got, verdict),
                "median_mean_reached": _median_mean(got),
                "median_half_width": float(np.median([_half(iv(a, b)) for a, b in got])),
            }
        return {
            "labels_at_gap_0": _labels_over(shifted(0.0), verdict),
            "labels_at_gap_minus_band": _labels_over(shifted(-PERC_BAND), verdict),
            "smallest_gap_all_wins": smallest_win,
            "median_mean_reached_at_smallest": (
                None if smallest_win is None else _median_mean(shifted(smallest_win))
            ),
            "at_fixed_gaps": at_fixed,
        }

    as_is = [iv(a, b) for a, b in samples]
    uniform = variant(moves["uniform"])
    return {
        "row": name,
        **describe(pairs[0] - pairs[1], n_target),
        "median_half_width": float(np.median([_half(x) for x in as_is])),
        "methods_used": sorted({x.method for x in as_is}),
        "labels_as_is": _labels_over(samples, verdict),
        "labels_at_gap_0": uniform["labels_at_gap_0"],
        "labels_at_gap_minus_band": uniform["labels_at_gap_minus_band"],
        "smallest_gap_all_wins": uniform["smallest_gap_all_wins"],
        "uniform": uniform,
        "concentrated": variant(moves["concentrated"]),
    }


def select_row(
    vw: NDArray[np.float64], v1: NDArray[np.float64], n_target: int, rng: np.random.Generator
) -> dict[str, Any]:
    samples = [resample((vw, v1), n_target, rng) for _ in range(RESAMPLES)]
    moves = shifters(samples, rng)

    def label(a: NDArray[np.float64], b: NDArray[np.float64]) -> str:
        diff = paired_interval(a, b, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
        own = mean_interval(a, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
        return select_verdict(diff, own)

    diffs = [paired_interval(a, b, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA) for a, b in samples]
    owns = [mean_interval(a, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA) for a, _ in samples]
    # Raising the difference leaves VW's values as they are and cutting it lowers them, so with VW's
    # own lower end under the floor in any resample no difference can win in every resample.
    floor_blocks = any(x.low < SELECT_FLOOR for x in owns)

    def variant(move: Shifter) -> dict[str, Any]:
        def shifted(diff: float) -> list[Pairs]:
            # VW's values stay as they are: only V1's side is lowered to raise the difference, and
            # only VW's side is lowered to cut it (which also lowers VW's own interval)
            return [move(i, a, b, diff) for i, (a, b) in enumerate(samples)]

        smallest_win = None
        first = 0 if not floor_blocks else int(MAX_SHIFT / STEP) + 1
        for i in range(first, int(MAX_SHIFT / STEP) + 1):
            target = STEP * i
            if _labels_over(shifted(target), label).get("win", 0) == RESAMPLES:
                smallest_win = target
                break
        return {
            "labels_at_diff_0": _labels_over(shifted(0.0), label),
            "smallest_diff_all_wins": smallest_win,
            "median_mean_reached_at_smallest": (
                None if smallest_win is None else _median_mean(shifted(smallest_win))
            ),
        }

    uniform = variant(moves["uniform"])
    return {
        "row": "VW minus V1, closed-loop c^lo",
        **describe(vw - v1, n_target),
        "median_half_width_diff": float(np.median([_half(x) for x in diffs])),
        "median_vw_low": float(np.median([x.low for x in owns])),
        "methods_used": sorted({x.method for x in diffs} | {x.method for x in owns}),
        "labels_as_is": _labels_over(samples, label),
        "labels_at_diff_0": uniform["labels_at_diff_0"],
        "smallest_diff_all_wins": uniform["smallest_diff_all_wins"],
        "uniform": uniform,
        "concentrated": variant(moves["concentrated"]),
        "vw_floor_blocks_a_win": floor_blocks,
        "vw_dev_mean": float(vw.mean()),
    }


def ext_row() -> dict[str, Any]:
    out = {}
    for p in (0.95, 0.92, 0.85):
        k = round(p * N_EXTERNAL)
        ci = binomial_interval(k, N_EXTERNAL, PRIMARY_ALPHA)
        out[f"{p:.2f}"] = {"k": k, "low": ci.low, "high": ci.high, "label": ext_verdict(ci)}
    return {"row": "V1 lower-bound coverage, external units", "n": N_EXTERNAL, "cases": out}


def input_code_sha256(quick: dict[str, Any]) -> str | None:
    """The code hash the input was made with: ``hashes`` of a quick run, ``provenance`` of a
    dev-point file."""
    for section in ("hashes", "provenance"):
        value = quick.get(section, {}).get("code_sha256")
        if value is not None:
            return str(value)
    return None


def vw_states(data: dict[str, Any]) -> list[str]:
    """How VW's fixed-point steps ended in the input: one state per half for a dev-point input."""
    if data.get("mode") == "dev-point":
        fits = [h.get("methods", {}).get("VW", {}) for h in data.get("halves", [])]
    else:
        fits = [data.get("conditions", {}).get("nominal", {}).get("methods", {}).get("VW", {})]
    return [vw_state(f["history"]) for f in fits if f.get("history")]


def reachability(
    quick: dict[str, Any], source: str = "benchmarks/.cache/quick.json"
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    n_quick = len(quick["conditions"]["nominal"]["eval_units"])
    for method in ("V1", "V2L"):
        for row in ("pooled", "stratum"):
            pairs = perc_pairs(quick, method, row)
            n_target = N_EVAL if row == "pooled" else math.floor(N_EVAL * pairs[0].size / n_quick)
            rows.append(perc_row(f"C2-PERC {method}, {row}", pairs, n_target, rng))
    vw = per_unit(quick, "VW", "closed", "cover_lo")
    v1 = per_unit(quick, "V1", "closed", "cover_lo")
    select = select_row(vw, v1, N_EVAL, rng)
    mode = quick.get("mode")
    return {
        "point": mode,
        "data": f"C-MAPSS, simulated by NASA's C-MAPSS model; dev units only (a {mode} input)",
        "source": source,
        "input_code_sha256": input_code_sha256(quick),
        "quick_eval_units": n_quick,
        "primary_alpha": PRIMARY_ALPHA,
        "perc_band": PERC_BAND,
        "select_margin": SELECT_MARGIN,
        "select_floor": SELECT_FLOOR,
        "resamples": RESAMPLES,
        "concentrated_share": CONCENTRATED_SHARE,
        "seed": SEED,
        "perc": rows,
        "select": select,
        "vw_states": vw_states(quick),
        "ext": ext_row(),
    }


def _f(x: float | None, places: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def _labels(d: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))


def _point_md(rep: dict[str, Any]) -> list[str]:
    lines = [
        f"## Operating point: {rep['point']}",
        "",
        f"- Input `{rep['source']}`, SHA-256 `{rep.get('input_sha256', 'n/a')}`; made with code "
        f"hash `{rep['input_code_sha256'] or 'n/a'}`.",
        f"- {rep['quick_eval_units']} dev units' per-unit values, resampled to the registered size "
        f"{rep['resamples']} times. Labels are counts over the resamples.",
        "",
        "### C2-PERC (band +-0.03)",
        "",
        "Uniform shift (every unit moves by the same amount):",
        "",
        "| row | dev units | dev mean | dev sd | kurtosis | n | guard (n/50) | median half-width "
        "| method | as is | gap 0 | gap -0.03 | smallest winning gap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["perc"]:
        lines.append(
            f"| {r['row']} | {r['dev_units']} | {_f(r['dev_mean'])} | {_f(r['dev_sd'])} | "
            f"{_f(r['dev_kurtosis'], 2)} | {r['n_target']} | {r['guard_limit']:.2f} "
            f"({'holds' if r['guard_holds_at_dev_kurtosis'] else 'fails'}) | "
            f"{_f(r['median_half_width'])} | {'/'.join(r['methods_used'])} | "
            f"{_labels(r['labels_as_is'])} | "
            f"{_labels(r['labels_at_gap_0'])} | {_labels(r['labels_at_gap_minus_band'])} | "
            f"{_f(r['smallest_gap_all_wins'], 3)} |"
        )
    share = f"{rep['concentrated_share']:.0%}"
    lines += [
        "",
        f"Concentrated shift (the whole gap carried by {share} of the units, floored at 0; "
        "'reached' is the median mean difference the shift reached):",
        "",
        "| row | gap 0 | gap -0.03 | smallest winning gap (reached) | at -0.055 | at -0.08 "
        "| at -0.1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["perc"]:
        c = r["concentrated"]

        def fixed(gap: str, c: dict[str, Any] = c) -> str:
            g = c["at_fixed_gaps"][gap]
            return (
                f"{_labels(g['labels'])} (reached {_f(g['median_mean_reached'])}, "
                f"half-width {_f(g['median_half_width'])})"
            )

        lines.append(
            f"| {r['row']} | {_labels(c['labels_at_gap_0'])} | "
            f"{_labels(c['labels_at_gap_minus_band'])} | {_f(c['smallest_gap_all_wins'], 3)} "
            f"({_f(c['median_mean_reached_at_smallest'])}) | {fixed('-0.055')} | "
            f"{fixed('-0.08')} | {fixed('-0.1')} |"
        )
    s = rep["select"]
    cs = s["concentrated"]
    lines += [
        "",
        "### C2-SELECT (margin 0.02, floor 0.92)",
        "",
        f"- Dev units {s['dev_units']}; difference mean {_f(s['dev_mean'])}, sd {_f(s['dev_sd'])}, "
        f"kurtosis {_f(s['dev_kurtosis'], 2)}; guard at n = {s['n_target']}: "
        f"{'holds' if s['guard_holds_at_dev_kurtosis'] else 'fails'}.",
        f"- VW closed-loop c^lo, dev mean {_f(s['vw_dev_mean'])}; median lower end at n = "
        f"{s['n_target']}: {_f(s['median_vw_low'])}. Median half-width of the difference: "
        f"{_f(s['median_half_width_diff'])} ({'/'.join(s['methods_used'])}).",
        f"- Labels as they are: {_labels(s['labels_as_is'])}.",
        f"- Uniform shift: at a difference of 0, {_labels(s['labels_at_diff_0'])}; smallest "
        f"difference with a win in every resample: {_f(s['smallest_diff_all_wins'], 3)}.",
        f"- Concentrated shift ({share} of units): at a difference of 0, "
        f"{_labels(cs['labels_at_diff_0'])}; smallest difference with a win in every resample: "
        f"{_f(cs['smallest_diff_all_wins'], 3)} (reached "
        f"{_f(cs['median_mean_reached_at_smallest'])}).",
    ]
    states = ", ".join(rep.get("vw_states", [])) or "not recorded"
    lines.append(
        f"- VW's fixed-point steps in this input ended: {states}. These rows use VW's last pair; "
        "a C2-SELECT win or loss also needs VW settled, or both pairs of a two-cycle giving the "
        "same label."
    )
    if s["vw_floor_blocks_a_win"]:
        lines.append(
            "- VW's own lower end is under the 0.92 floor in some resample, so no difference wins "
            "in every resample."
        )
    lines.append("")
    return lines


def render_md(full: dict[str, Any]) -> str:
    points = full["points"]
    lines = [
        "# Reachability of the primary verdicts",
        "",
        "Made-up inputs built from dev units only (C-MAPSS, simulated by NASA's C-MAPSS model). "
        "The per-unit values of each operating point are resampled to the registered size and run "
        f"through the full run's interval and verdict code at level {full['primary_alpha']:.6f} "
        "(0.05 / 6). These check the rules and the sizes; they are not results.",
        "",
    ]
    for rep in points:
        lines += _point_md(rep)
    ext = points[0]["ext"]
    lines += [
        "## C2-EXT ([0.92, 0.98], Clopper-Pearson at 707 units; no dev values needed)",
        "",
        "| coverage | k | low | high | label |",
        "|---|---|---|---|---|",
    ]
    for p, c in ext["cases"].items():
        lines.append(f"| {p} | {c['k']} | {c['low']:.4f} | {c['high']:.4f} | {c['label']} |")
    lines.append("")
    return "\n".join(lines)


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Reachability of the primary verdicts.")
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--md", type=Path, default=OUT_MD)
    args = parser.parse_args(argv)
    inputs = args.input or [QUICK, DEV_POINT]
    points = []
    for path in inputs:
        raw = path.read_bytes()
        quick = json.loads(raw.decode("utf-8"))
        if quick.get("mode") not in MODES:
            print(f"error: {path.name} is not a --quick or dev-point input", file=sys.stderr)
            return 1
        rep = reachability(quick, _shown(path))
        rep["input_sha256"] = hashlib.sha256(raw).hexdigest()
        points.append(rep)
    full = {
        "data": "C-MAPSS, simulated by NASA's C-MAPSS model; dev units only",
        "primary_alpha": PRIMARY_ALPHA,
        "points": points,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_md(full), encoding="utf-8", newline="\n")
    print(f"wrote {args.json} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
