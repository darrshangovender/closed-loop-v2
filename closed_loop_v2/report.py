"""``results.json`` to text: ``RESULTS.md`` and the README block (0002, section 4).

Both documents are **pure functions of the parsed ``results.json``**: every number in them is read
from it (the unit counts in the limits section included) and printed with a fixed number of decimals
(4 for coverages, gaps and quantiles, 1 for cycles). ``benchmarks/run.py --check`` regenerates them
and compares them byte for byte with the files on disk, so a hand-edited number is caught.
``results.json`` holds full precision and is not rounded.

``RESULTS.md`` also carries the registered per-subset breakdown (coverage, lower-bound coverage,
stratum lower-bound coverage and ``Gap_B`` per subset, method and loop), the external check's
``k`` and ``n`` per subset, and, for a full run, the run number and the earlier full runs.

Infinite numbers. JSON has no infinity, so ``to_json`` writes ``+inf`` and ``-inf`` as the strings
``"inf"`` and ``"-inf"`` (a quantile of ACI out of range, or a step of VW with a calibration set too
small for a finite quantile), and a missing number as ``null``. ``NaN`` is not written.

The text avoids absolutes. ``FORBIDDEN`` lists the words a generated sentence may not use unless a
proof backs it; ``absolutes_in`` finds them, ignoring free text that comes from ``tuned.json``
(data, not the report's own words). Every generated document says that C-MAPSS is simulated by
NASA's C-MAPSS model and is not field data.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from closed_loop_v2.verdicts import VW_SETTLED, VW_TWO_CYCLE, VW_UNSETTLED, vw_state

__all__ = [
    "DATA_NOTE",
    "FORBIDDEN",
    "QUICK_BANNER",
    "README_BEGIN",
    "README_END",
    "SCHEMA",
    "MarkerError",
    "absolutes_in",
    "data_strings",
    "from_json_number",
    "readme_block_of",
    "render_readme_block",
    "render_results_md",
    "splice_readme",
    "to_json",
]

SCHEMA: Final[str] = "closed-loop-v2/results/1"
DATA_NOTE: Final[str] = "C-MAPSS: simulated by NASA's C-MAPSS model, not field data"
QUICK_BANNER: Final[str] = "QUICK RUN, NOT A RESULT"
README_BEGIN: Final[str] = "<!-- RESULTS:BEGIN -->"
README_END: Final[str] = "<!-- RESULTS:END -->"
FORBIDDEN: Final[re.Pattern[str]] = re.compile(
    r"\b(never|always|guaranteed|guarantees)\b|100%", re.IGNORECASE
)
DASH: Final[str] = "-"
METHODS: Final[tuple[str, ...]] = ("V1", "VW", "V2L", "V3L")
LOOPS: Final[tuple[str, ...]] = ("open", "closed")
CONDITIONS: Final[tuple[str, ...]] = ("nominal", "shifted")
SECONDARY_NOTE: Final[str] = (
    "Secondary rows, at the descriptive level of 0.05. They have no registered rule and decide "
    "nothing."
)


class MarkerError(ValueError):
    """The README does not carry the two marker comments, once each and in order."""


# ----------------------------------------------------------------------------- JSON


def _encode(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    raise TypeError(f"cannot write {type(value).__name__} to results.json")


def to_json(results: Mapping[str, Any]) -> str:
    """The canonical text of ``results.json``: UTF-8, LF line ends, final newline, full precision.

    Python writes the shortest decimal that reads back as the same float, so no digit is lost.
    """
    return json.dumps(_encode(results), indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def from_json_number(value: Any) -> float | None:
    """A number as ``results.json`` stores it: a float, ``"inf"``, ``"-inf"`` or ``null``."""
    if value is None:
        return None
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


# ----------------------------------------------------------------------------- formatting


def num(value: Any, places: int = 4) -> str:
    """A number with fixed decimals; ``-`` for a missing one, ``+inf``/``-inf`` for infinite."""
    x = from_json_number(value)
    if x is None:
        return DASH
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.{places}f}"


def count(value: Any) -> str:
    return DASH if value is None else str(int(value))


def ival(iv: Mapping[str, Any] | None, places: int = 4) -> str:
    """``estimate (low to high)`` from a stored interval."""
    if iv is None:
        return DASH
    return (
        f"{num(iv.get('estimate'), places)} ({num(iv['low'], places)} to {num(iv['high'], places)})"
    )


def rule_of(iv: Mapping[str, Any] | None) -> str:
    return DASH if iv is None else str(iv["method"])


def esc(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    body = [[esc(c) for c in row] for row in rows]
    if not body:
        return ["(no rows)", ""]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    out.append("")
    return out


def _short(sha: Any) -> str:
    return DASH if not sha else f"`{str(sha)[:16]}`"


# ----------------------------------------------------------------------------- sections


def _banner(results: Mapping[str, Any]) -> list[str]:
    if results.get("mode") != "quick":
        return []
    return [
        f"> **{QUICK_BANNER}.** Development units only, re-split by a dev-only hash into fit, "
        "calibration and evaluation sets; no ledger entry and no held-out data. The external "
        "check runs on quick evaluation units cut at a hashed row, not on NASA's test units. "
        "Nothing below is a result; it shows that the pipeline runs.",
        "",
    ]


def _run_lines(results: Mapping[str, Any]) -> list[str]:
    hashes = results.get("hashes", {})
    seeds = results.get("seeds", {})
    key = seeds.get("key", DASH)
    key_text = "the key `quick`" if key == "quick" else "the pre-registration hash"
    labels = ", ".join(f"`{x}`" for x in seeds.get("labels", []))
    return [
        f"Mode: {results.get('mode', DASH)}. Pre-registration hash: "
        f"{_short(hashes.get('prereg_sha256'))}. Code hash: {_short(hashes.get('code_sha256'))}. "
        f"`tuned.json` hash: {_short(hashes.get('tuned_sha256'))}. Data manifest hash: "
        f"{_short(hashes.get('manifest_sha256'))}.",
        "",
        f"Seeds are derived from {key_text} for the labels {labels}; their values are not stored.",
        "",
    ]


def _claim_rule_lines(design: Mapping[str, Any]) -> list[str]:
    band = num(design.get("perc_band"))
    margin = num(design.get("select_margin"))
    floor = num(design.get("select_floor"))
    lo, hi = num(design.get("ext_lo")), num(design.get("ext_hi"))
    return [
        f"- C2-PERC: win if the interval for Gap_B lies wholly above +{band} or wholly below "
        f"-{band}; loss if it lies inside [-{band}, +{band}]; otherwise inconclusive.",
        f"- C2-SELECT: win if the lower end of the paired difference (VW minus V1, closed-loop "
        f"c^lo) is at least {margin} and the lower end of VW's c^lo is at least {floor}; loss if "
        f"the upper end of the difference is below {margin} or the upper end of VW's c^lo is "
        f"below {floor}; otherwise inconclusive. If VW's fixed-point steps alternate between two "
        "pairs (a two-cycle), the rule is applied with each pair and the label is kept only if "
        "both give it; if the steps neither settled nor reached a two-cycle, the row is "
        "inconclusive.",
        f"- C2-EXT: win if the Clopper-Pearson interval for V1's lower-bound coverage on the "
        f"external units lies inside [{lo}, {hi}]; loss if it has no point in common with that "
        "range; otherwise inconclusive.",
        "- An interval too wide to decide is labelled inconclusive, not a loss.",
        "",
    ]


def _claim_interval(claim: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ivs = claim.get("intervals", {})
    for key in ("gap", "diff", "coverage_lo"):
        if key in ivs:
            found: Mapping[str, Any] = ivs[key]
            return found
    return None


def _verdict_section(results: Mapping[str, Any]) -> list[str]:
    design = results.get("design", {})
    claims = results.get("claims", [])
    primary = [c for c in claims if c.get("primary")]
    secondary = [c for c in claims if not c.get("primary")]
    out = ["## Verdicts", ""]
    out += [
        f"Primary rows are judged at the level {num(design.get('primary_alpha'), 5)} (0.05 over "
        "the six primary verdicts: four for C2-PERC, one each for C2-SELECT and C2-EXT). A loss "
        "and an inconclusive result are published with the same prominence as a win. The interval "
        "rule is `t` (Student-t, used only when n is at least 100 and the sample kurtosis is at "
        "most n / 50), `betting` (the fixed-sample betting interval of ADR 0004: valid at any "
        "size for independent values with one mean, fed in a uniformly random order) or "
        "`clopper-pearson` (exact, for a rate). An empty interval is labelled inconclusive. "
        "Two conditions of that validity are not met here, so for the rows they touch the "
        "stated level is nominal, not proven. The betting interval is fed the values in a fixed "
        "order (a canonical order of the values, then one fixed permutation), so the subset "
        "order of the units does not change it, but the order is not random. V2L's and V3L's "
        "per-unit values are coupled through the shared alpha_t that every earlier unit in the "
        "queue has moved, so they are not independent, whichever rule is used.",
        "",
    ]
    rows = []
    for c in primary:
        main = _claim_interval(c)
        rows.append(
            (
                c["claim"],
                c["row"],
                c.get("verdict") or DASH,
                ival(main),
                rule_of(main),
                count(main["n"]) if main else DASH,
            )
        )
    out += table(["Claim", "Row", "Verdict", "Estimate (interval)", "Interval rule", "n"], rows)
    for c in primary:
        vw = c.get("intervals", {}).get("vw")
        if vw is not None:
            out += [
                f"{c['claim']} also needs VW's closed-loop c^lo: {ival(vw)}, interval rule "
                f"{rule_of(vw)}, n {count(vw['n'])}.",
                "",
            ]
        state = c.get("vw_state")
        if state == VW_TWO_CYCLE:
            ivs = c.get("intervals", {})
            diff_alt, vw_alt = ivs.get("diff_alt"), ivs.get("vw_alt")
            out += [
                f"VW's fixed-point steps alternate between two pairs, so {c['claim']} is judged "
                f"with each and keeps a label only if both give it. With the other pair: "
                f"difference {ival(diff_alt)}, interval rule {rule_of(diff_alt)}; VW's c^lo "
                f"{ival(vw_alt)}, interval rule {rule_of(vw_alt)}.",
                "",
            ]
        elif state == VW_UNSETTLED:
            out += [
                f"{c['claim']} is inconclusive by rule: VW's fixed-point steps neither settled "
                "nor reached a two-cycle, so its quantiles depend on where the iteration stopped.",
                "",
            ]
    out += ["Rules:", ""]
    out += _claim_rule_lines(design)
    if secondary:
        out += ["### Secondary rows", "", SECONDARY_NOTE, ""]
        rows = []
        for c in secondary:
            main = _claim_interval(c)
            rows.append(
                (
                    c["claim"],
                    c["row"],
                    c.get("condition", DASH),
                    ival(main),
                    rule_of(main),
                    count(main["n"]) if main else DASH,
                )
            )
        out += table(
            ["Claim", "Row", "Condition", "Estimate (interval)", "Interval rule", "n"], rows
        )
    return out


def _units_section(results: Mapping[str, Any]) -> list[str]:
    units = results.get("units", {})
    out = ["## Units", ""]
    rows = []
    for cond in CONDITIONS:
        u = units.get(cond)
        if not u:
            continue
        for role in ("fit", "calib", "eval"):
            part = u.get(role, {})
            subsets = ", ".join(f"{s} {n}" for s, n in part.get("by_subset", {}).items())
            rows.append((cond, role, count(part.get("total")), subsets or DASH))
    ext = units.get("external", {})
    if ext:
        subsets = ", ".join(f"{s} {n}" for s, n in ext.get("by_subset", {}).items())
        rows.append(("external", ext.get("source", DASH), count(ext.get("total")), subsets))
    out += table(["Condition", "Role", "Units", "By subset"], rows)
    out += [
        "The decision stratum is the rows whose shared point estimate m is at most "
        f"{num(results.get('design', {}).get('stratum_m'), 1)} cycles. A unit with no stratum row "
        "in some order is left out of the stratum for that method and loop; the counts are in the "
        "coverage tables below.",
        "",
    ]
    return out


def _coverage_section(results: Mapping[str, Any]) -> list[str]:
    out = ["## Coverage by method and loop", ""]
    out += [
        "Per-unit coverage averaged over the checkpoints each unit contributes, then over the "
        "orders, then over units with equal weight. Intervals at the descriptive level 0.05.",
        "",
    ]
    for cond in CONDITIONS:
        c = results.get("conditions", {}).get(cond)
        if not c:
            continue
        rows = []
        for method in METHODS:
            for loop in LOOPS:
                cell = c.get("cells", {}).get(method, {}).get(loop)
                if cell is None:
                    continue
                rows.append(
                    (
                        method,
                        loop,
                        ival(cell["coverage"]),
                        ival(cell["coverage_lo"]),
                        rule_of(cell["coverage_lo"]),
                        ival(cell["stratum_coverage_lo"]),
                        f"{count(cell['stratum_units'])} ({count(cell['stratum_excluded'])} out)",
                        count(cell["orders"]),
                    )
                )
        out += [f"### {cond}", ""]
        out += table(
            [
                "Method",
                "Loop",
                "c",
                "c^lo",
                "c^lo rule",
                "Stratum c^lo",
                "Stratum units",
                "Orders",
            ],
            rows,
        )
    out += [
        "V1 and VW keep their quantiles fixed, so their outcome does not depend on the order; one "
        "replay stands for every order (Orders 1).",
        "",
    ]
    return out


def _gap_section(results: Mapping[str, Any]) -> list[str]:
    out = ["## Gap_B: closed-loop minus open-loop c^lo, paired by unit", ""]
    rows = []
    for cond in CONDITIONS:
        c = results.get("conditions", {}).get(cond)
        if not c:
            continue
        for method in METHODS:
            g = c.get("gap_b", {}).get(method)
            if g is None:
                continue
            rows.append(
                (
                    cond,
                    method,
                    ival(g["pooled"]),
                    rule_of(g["pooled"]),
                    ival(g["stratum"]),
                    rule_of(g["stratum"]),
                    f"{count(g['stratum_units'])} ({count(g['stratum_dropped'])} out)",
                )
            )
    out += table(
        ["Condition", "Method", "Pooled", "Rule", "Stratum", "Rule", "Stratum units"], rows
    )
    out += [
        "Intervals at the descriptive level 0.05; the registered C2-PERC rows above use the "
        "primary level.",
        "",
    ]
    return out


def _subset_section(results: Mapping[str, Any]) -> list[str]:
    out = ["## By subset (registered, descriptive)", ""]
    out += [
        "The per-subset breakdown registered in advance: each subset's units only, intervals at "
        "the descriptive level 0.05, no label. Units are the subset's evaluation units; stratum "
        "units are those with stratum rows in every order (for Gap_B, in both loops).",
        "",
    ]
    for cond in CONDITIONS:
        c = results.get("conditions", {}).get(cond)
        if not c:
            continue
        subsets = c.get("by_subset", {})
        rows = []
        gap_rows = []
        for subset, entry in subsets.items():
            for method in METHODS:
                for loop in LOOPS:
                    cell = entry.get("cells", {}).get(method, {}).get(loop)
                    if cell is None:
                        continue
                    rows.append(
                        (
                            subset,
                            method,
                            loop,
                            ival(cell["coverage"]),
                            ival(cell["coverage_lo"]),
                            ival(cell["stratum_coverage_lo"]),
                            count(cell["units"]),
                            count(cell["stratum_units"]),
                        )
                    )
                g = entry.get("gap_b", {}).get(method)
                if g is not None:
                    gap_rows.append(
                        (
                            subset,
                            method,
                            ival(g["pooled"]),
                            rule_of(g["pooled"]),
                            ival(g["stratum"]),
                            rule_of(g["stratum"]),
                            count(g["units"]),
                            count(g["stratum_units"]),
                        )
                    )
        out += [f"### {cond}: coverage by subset", ""]
        out += table(
            ["Subset", "Method", "Loop", "c", "c^lo", "Stratum c^lo", "Units", "Stratum units"],
            rows,
        )
        out += [f"### {cond}: Gap_B by subset (closed minus open c^lo, paired by unit)", ""]
        out += table(
            [
                "Subset",
                "Method",
                "Pooled",
                "Rule",
                "Stratum",
                "Rule",
                "Units",
                "Stratum units",
            ],
            gap_rows,
        )
    ext = results.get("external", {})
    if ext:
        out += ["### External check by subset (V1, lower bound)", ""]
        out += table(
            ["Subset", "Covered (k)", "Units (n)"],
            [
                (subset, count(v.get("k")), count(v.get("n")))
                for subset, v in ext.get("by_subset", {}).items()
            ],
        )
    return out


def _runs_section(results: Mapping[str, Any]) -> list[str]:
    runs = results.get("runs")
    if not runs:
        return []
    out = ["## Runs", ""]
    out += [
        f"This is full run {count(runs.get('number'))}. Its `results.json` is also kept as "
        f"`{runs.get('archive', DASH)}`, a copy that a later run does not replace.",
        "",
    ]
    earlier = runs.get("earlier", [])
    if not earlier:
        out += ["No earlier full run is recorded in the ledger.", ""]
        return out
    out += ["Earlier full runs, as the ledger records them:", ""]
    rows = []
    for e in earlier:
        labels = list(e.get("verdicts", {}).values())
        tally = ", ".join(
            f"{label} {labels.count(label)}" for label in ("win", "loss", "inconclusive")
        )
        rows.append(
            (
                count(e.get("number")),
                e.get("utc") or DASH,
                _short(e.get("results_sha256")),
                f"`{e['archive']}`" if e.get("archive") else DASH,
                tally,
            )
        )
    out += table(
        ["Run", "Recorded (UTC)", "results.json sha256", "Archived copy", "Verdicts"], rows
    )
    return out


def _policy_section(results: Mapping[str, Any]) -> list[str]:
    out = ["## Policy outcomes (descriptive)", ""]
    rows = []
    for cond in CONDITIONS:
        c = results.get("conditions", {}).get(cond)
        if not c:
            continue
        for method in METHODS:
            for loop in LOOPS:
                cell = c.get("cells", {}).get(method, {}).get(loop)
                if cell is None:
                    continue
                p = cell["policy"]
                rows.append(
                    (
                        cond,
                        method,
                        loop,
                        num(p["failed_fraction"]),
                        num(p["replacements"], 1),
                        num(p["mean_rul_at_replacement"], 1),
                        num(p["mean_width"], 1),
                        f"{num(p['infinite_rows'], 1)} of {num(p['rows'], 1)}",
                    )
                )
    out += table(
        [
            "Condition",
            "Method",
            "Loop",
            "Failed unreplaced",
            "Replacements",
            "Remaining life at replacement",
            "Mean finite width",
            "Rows with an infinite side",
        ],
        rows,
    )
    out += [
        "Per order: the fraction of units that failed unreplaced, the replacements, and the rows. "
        "Remaining life at replacement and widths are in cycles.",
        "",
    ]
    return out


def _quantile_section(results: Mapping[str, Any]) -> list[str]:
    out = ["## Quantiles and VW's fixed-point steps", ""]
    rows = []
    hist_lines: list[str] = []
    for cond in CONDITIONS:
        c = results.get("conditions", {}).get(cond)
        if not c:
            continue
        methods = c.get("methods", {})
        for method in METHODS:
            m = methods.get(method)
            if m is None:
                continue
            if m.get("kind") == "split":
                q_lo, q_hi = m["quantiles"]
                rows.append((cond, method, num(q_lo), num(q_hi), DASH))
            else:
                q_lo, q_hi = m["initial_quantiles"]
                finals = m.get("final_alpha_t", {})
                parts = []
                for loop in LOOPS:
                    vals = [from_json_number(v) for v in finals.get(loop, [])]
                    got = [v for v in vals if v is not None]
                    if got:
                        parts.append(f"{loop} {num(min(got))} to {num(max(got))}")
                rows.append(
                    (cond, f"{method} (start)", num(q_lo), num(q_hi), "; ".join(parts) or DASH)
                )
        vw = methods.get("VW")
        if vw is not None and vw.get("history"):
            steps = "; ".join(
                f"step {i}: ({num(a)}, {num(b)})" for i, (a, b) in enumerate(vw["history"])
            )
            ended = {
                VW_SETTLED: "The last step did not move the quantiles (settled).",
                VW_TWO_CYCLE: "The last step returned the quantiles to those two steps earlier: "
                "the steps alternate between two pairs (a two-cycle).",
                VW_UNSETTLED: "The last step moved the quantiles, and the steps had not reached "
                "a two-cycle (unsettled).",
            }[vw_state(vw["history"])]
            hist_lines += [f"VW, {cond}: {steps}. {ended}", ""]
    out += table(["Condition", "Method", "q_lo", "q_hi", "Final alpha_t over orders"], rows)
    out += [
        "Quantiles are in score units, r = (RUL - m) / s; step 0 is V1's. V2L and V3L start from "
        "V1's calibration draw and move with every resolved life.",
        "",
    ]
    out += hist_lines
    return out


def _values_section(results: Mapping[str, Any]) -> list[str]:
    tuned = results.get("tuned", {})
    design = results.get("design", {})
    out = ["## Values used", ""]
    rows = []
    h = tuned.get("h", {})
    rows.append(("h (bound rule, cycles)", num(h.get("value"), 1), h.get("from", DASH)))
    m = tuned.get("stratum_m", {})
    rows.append(("M (decision stratum, cycles)", num(m.get("value"), 1), m.get("from", DASH)))
    for name, entry in tuned.get("estimator", {}).items():
        value = entry.get("value")
        shown = num(value, 4) if isinstance(value, float) else str(value)
        rows.append((f"estimator {name}", shown, entry.get("from", DASH)))
    out += table(["Value", "Used", "From"], rows)
    for key, text in tuned.get("provenance", {}).items():
        out.append(f"- `tuned.json` {key}: {esc(text)}")
    if tuned.get("provenance"):
        out.append("")
    shown_design = [
        ("k (engines at once)", count(design.get("k"))),
        ("R (orders)", count(design.get("r"))),
        ("M used by the run (stratum, cycles)", num(design.get("stratum_m"), 1)),
        ("alpha (two-sided)", num(design.get("alpha"))),
        ("gamma (ACI step)", num(design.get("gamma"))),
        ("ACI pool window", count(design.get("aci_window"))),
        ("VW fixed-point steps", count(design.get("vw_steps"))),
        ("cross-fitting folds", count(design.get("n_folds"))),
    ]
    out += table(["Design constant", "Value"], shown_design)
    return out


def _limits(results: Mapping[str, Any]) -> list[str]:
    """The limits section; its unit counts are read from ``results["units"]``."""
    units = results.get("units", {})
    fd003_calib = units.get("nominal", {}).get("calib", {}).get("by_subset", {}).get("FD003")
    shifted = units.get("shifted", {})
    return [
        "Not field data. C-MAPSS trajectories come from NASA's C-MAPSS engine model; a positive or "
        "negative result here is about C-MAPSS degradation.",
        "No physics feedback. Replacing one engine does not change another in the replay; in a "
        "plant where maintenance changes the load on other equipment, the covariate-shift argument "
        "for VW would not hold as it stands.",
        "One policy family. The bound rule is tested at one threshold h; other policies select "
        "other checkpoints.",
        f"Small strata. FD003 contributes {count(fd003_calib)} calibration units, and the shifted "
        f"condition calibrates on {count(shifted.get('calib', {}).get('total'))} units and "
        f"evaluates on {count(shifted.get('eval', {}).get('total'))}.",
        "V1's statement covers a checkpoint drawn uniformly from a unit's life; it does not cover "
        "checkpoints a policy chose, nor NASA's truncation points. VW's statement is approximate "
        "because its policy depends on the calibration data through the fixed-point steps.",
    ]


# ----------------------------------------------------------------------------- documents


def _truth_lines(results: Mapping[str, Any]) -> list[str]:
    """The capped truth, stated next to the data note (review D, S4), or nothing if not recorded."""
    truth = results.get("truth")
    if not isinstance(truth, Mapping) or truth.get("capped_at") is None:
        return []
    cap = truth["capped_at"]
    return [
        f"Truth: min(remaining life, {cap} cycles). Bounds, scores, coverage, labels and C2-EXT "
        f"are measured against it, and bounds are limited to {cap}. Remaining life at "
        "replacement (the cost) is uncapped.",
        "",
    ]


def render_results_md(results: Mapping[str, Any]) -> str:
    """The text of ``RESULTS.md`` (LF line ends, one final newline), from ``results`` alone."""
    out = ["# Results (SIMULATED data)", ""]
    out += _banner(results)
    out += [f"**{results.get('data', DATA_NOTE)}.**", ""]
    out += _truth_lines(results)
    out += [
        "Generated by `benchmarks/run.py` from `benchmarks/results.json`; every number here is "
        "read from `results.json`, and `python benchmarks/run.py --check` regenerates this file "
        "and compares it.",
        "",
    ]
    out += _run_lines(results)
    out += _runs_section(results)
    out += _verdict_section(results)
    out += _units_section(results)
    out += _coverage_section(results)
    out += _gap_section(results)
    out += _subset_section(results)
    out += _policy_section(results)
    out += _quantile_section(results)
    out += _values_section(results)
    out += ["## What these results cannot say", ""]
    out += [f"- {x}" for x in _limits(results)]
    out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def render_readme_block(results: Mapping[str, Any]) -> str:
    """The README results block, markers included, LF line ends, no trailing newline."""
    out = [
        README_BEGIN,
        "",
        "<!-- Generated by benchmarks/run.py from benchmarks/results.json; "
        "do not edit between the markers. -->",
        "",
    ]
    if results.get("mode") == "quick":
        out += [f"> **{QUICK_BANNER}.** Development units only; nothing below is a result.", ""]
    out += [f"**{results.get('data', DATA_NOTE)}.**", ""]
    out += _truth_lines(results)
    rows = []
    for c in results.get("claims", []):
        if not c.get("primary"):
            continue
        main = _claim_interval(c)
        shown, rule = ival(main), rule_of(main)
        ivs = c.get("intervals", {})
        vw = ivs.get("vw")
        if vw is not None:  # C2-SELECT's rule also reads VW's own c^lo
            shown += f"; VW's c^lo {ival(vw)}"
            rule += f"; VW's c^lo {rule_of(vw)}"
        if c.get("vw_state") == VW_TWO_CYCLE:  # judged with both pairs of VW's two-cycle
            shown += (
                f"; other VW pair: difference {ival(ivs.get('diff_alt'))}, VW's c^lo "
                f"{ival(ivs.get('vw_alt'))}"
            )
            rule += f"; other pair {rule_of(ivs.get('diff_alt'))}, {rule_of(ivs.get('vw_alt'))}"
        elif c.get("vw_state") == VW_UNSETTLED:
            shown += "; VW had not settled"
        rows.append((c["claim"], c["row"], c.get("verdict") or DASH, shown, rule))
    out += table(["Claim", "Row", "Verdict", "Estimate (interval)", "Interval rule"], rows)
    out += [
        f"Level {num(results.get('design', {}).get('primary_alpha'), 5)} for each primary row. "
        "Rules, secondary rows and every table: `benchmarks/RESULTS.md`.",
        "",
        README_END,
    ]
    return "\n".join(out)


# ----------------------------------------------------------------------------- README splice


def _span(text: str) -> tuple[int, int]:
    if text.count(README_BEGIN) != 1 or text.count(README_END) != 1:
        raise MarkerError(f"the README needs exactly one {README_BEGIN} and one {README_END}")
    start, stop = text.index(README_BEGIN), text.index(README_END) + len(README_END)
    if stop <= start:
        raise MarkerError("the END marker comes before the BEGIN marker")
    return start, stop


def readme_block_of(readme: str) -> str:
    """The block between (and including) the markers; ``MarkerError`` if they are not there."""
    start, stop = _span(readme.replace("\r\n", "\n"))
    return readme.replace("\r\n", "\n")[start:stop]


def splice_readme(readme: str, block: str) -> str:
    """``readme`` (LF) with the text between and including the markers replaced by ``block``."""
    text = readme.replace("\r\n", "\n")
    start, stop = _span(text)
    return text[:start] + block + text[stop:]


# ----------------------------------------------------------------------------- the lint


def data_strings(results: Mapping[str, Any]) -> list[str]:
    """Free text in ``results`` that comes from outside the report (``tuned.json`` provenance)."""
    prov = results.get("tuned", {}).get("provenance", {})
    return [str(v) for v in prov.values() if isinstance(v, str) and len(v) >= 4]


def absolutes_in(text: str, data: Sequence[str] = ()) -> list[str]:
    """The absolutes ("never", "always", "guaranteed", "100%") in ``text``, outside ``data``."""
    body = text
    for d in sorted(data, key=len, reverse=True):
        body = body.replace(d, " ").replace(esc(d), " ")
    return [m.group(0) for m in FORBIDDEN.finditer(body)]
