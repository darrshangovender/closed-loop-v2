"""The report: canonical JSON, deterministic rendering from ``results.json`` alone, fixed decimals,
the simulated-data note, the README splice, and the absolutes lint on real rendered output.

A realistic ``results.json`` comes from ``benchmarks/run.py``'s own pipeline, run on made-up
trajectories held in memory (no data files, no ledger) with a small design so that it is quick.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from closed_loop_v2 import report
from closed_loop_v2.estimator import EstimatorParams
from closed_loop_v2.stats import Interval
from closed_loop_v2.types import SUBSETS, Trajectory
from closed_loop_v2.verdicts import vw_state

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "benchmarks" / "run.py"
MODULE = "closed_loop_v2_bench_run"


def load_run() -> ModuleType:
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    spec = importlib.util.spec_from_file_location(MODULE, RUN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def unit(subset: str, number: int, rng: np.random.Generator) -> Trajectory:
    n = int(rng.integers(15, 40))
    t = np.arange(1, n + 1)
    wear = (t / n)[:, None] * rng.uniform(0.5, 2.0, size=21)
    sensors = wear + 0.3 * rng.normal(size=(n, 21))
    return Trajectory(subset, number, "train", t, rng.normal(size=(n, 3)), sensors, 0)


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """A ``results.json`` dict from the real pipeline, parsed back from its canonical text."""
    run = load_run()
    rng = np.random.default_rng(11)
    units = [unit(s, u, rng) for s in SUBSETS for u in range(1, 16)]
    fit, calib, ev = units[0::3], units[1::3], units[2::3]
    sets = run.DevSets(
        fit={"nominal": fit, "shifted": run._in(fit, run.SHIFT_FIT)},
        calib={"nominal": calib, "shifted": run._in(calib, run.SHIFT_FIT)},
    )
    params = EstimatorParams(8, 125.0, 1.0, 10, 0.2, 8, 3)
    names = [*(f for f in params.__dataclass_fields__), "h", "stratum_m"]
    tuned = run.Tuned(
        params, 4.0, 15.0, dict.fromkeys(names, "tuned.json"), {"notes": "never mind"}, True
    )
    design = run.Design(k=4, r=3, stratum_m=15.0, vw_steps=3)  # M as tuned, like run_design
    out = run.run_pipeline(
        sets,
        lambda: run.Heldout(ev, [run.quick_cut(t) for t in ev], "made-up units"),
        mode="full",
        key="report-test",
        hashes={"prereg_sha256": "a" * 64, "code_sha256": "b" * 64},
        tuned=tuned,
        design=design,
        log=lambda _: None,
        headroom=lambda: None,
    )
    parsed: dict[str, Any] = json.loads(report.to_json(out))
    return parsed


# ----------------------------------------------------------------------------- JSON


def test_to_json_keeps_full_precision_and_writes_infinities_as_text() -> None:
    value = 0.1 + 0.2
    text = report.to_json({"x": value, "q": (-math.inf, math.inf), "gone": math.nan, "n": 3})
    back = json.loads(text)
    assert back["x"] == value  # no rounding
    assert back["q"] == ["-inf", "inf"] and back["gone"] is None and back["n"] == 3
    assert text.endswith("\n") and "\r" not in text
    assert report.from_json_number("-inf") == -math.inf and report.from_json_number(None) is None
    with pytest.raises(TypeError):
        report.to_json({"bad": object()})


def test_results_json_holds_every_section(results: dict[str, Any]) -> None:
    assert results["schema"] == report.SCHEMA
    assert results["data"] == report.DATA_NOTE
    assert results["seeds"]["labels"] == ["fit-nominal", "fit-shifted", "calib-draw", "orders"]
    assert "value" not in json.dumps(results["seeds"])  # labels only, no seed values
    nominal = results["conditions"]["nominal"]
    assert set(nominal["cells"]) == {"V1", "VW", "V2L", "V3L"}
    cell = nominal["cells"]["V2L"]["closed"]
    assert cell["orders"] == 3 and cell["coverage_lo"]["method"] in ("t", "betting")
    assert len(cell["per_unit"]["cover_lo"]) == len(nominal["eval_units"])
    assert len(nominal["methods"]["VW"]["history"]) == 4  # step 0 (V1) and three steps
    assert nominal["methods"]["VW"]["history"][0] == nominal["methods"]["V1"]["quantiles"]
    assert results["tuned"]["h"] == {"value": 4.0, "from": "tuned.json"}
    assert results["tuned"]["stratum_m"] == {"value": 15.0, "from": "tuned.json"}
    primary = [c for c in results["claims"] if c["primary"]]
    assert [c["claim"] for c in primary] == ["C2-PERC"] * 4 + ["C2-SELECT", "C2-EXT"]
    assert all(c["verdict"] in ("win", "loss", "inconclusive") for c in primary)
    assert all(c["verdict"] is None for c in results["claims"] if not c["primary"])
    secondary = {c["claim"] for c in results["claims"] if not c["primary"]}
    assert secondary == {"C2-SELECT", "C2-CENSOR", "C2-SHIFT"}
    assert "runs" not in results  # only a full run from benchmarks/run.py --full carries it


def test_the_per_subset_breakdown_is_stored(results: dict[str, Any]) -> None:
    nominal = results["conditions"]["nominal"]
    by = nominal["by_subset"]
    assert list(by) == list(SUBSETS)
    keys = nominal["eval_units"]
    for subset, entry in by.items():
        n = sum(k.split("/")[0] == subset for k in keys)
        assert entry["units"] == n
        for method in ("V1", "VW", "V2L", "V3L"):
            for loop in ("open", "closed"):
                cell = entry["cells"][method][loop]
                assert cell["coverage"]["n"] == cell["coverage_lo"]["n"] == cell["units"] == n
                assert cell["stratum_coverage_lo"]["n"] == cell["stratum_units"] <= n
                assert cell["coverage_lo"]["alpha"] == 0.05
            g = entry["gap_b"][method]
            assert g["pooled"]["n"] == g["units"] == n
            assert g["stratum"]["n"] == g["stratum_units"] <= n
        # the per-subset mean of c^lo is the mean of that subset's per-unit values
        per_unit = nominal["cells"]["V1"]["closed"]["per_unit"]["cover_lo"]
        mine = [v for k, v in zip(keys, per_unit, strict=True) if k.split("/")[0] == subset]
        got = entry["cells"]["V1"]["closed"]["coverage_lo"]["estimate"]
        assert got == pytest.approx(sum(mine) / len(mine))
    assert set(results["conditions"]["shifted"]["by_subset"]) == {"FD003", "FD004"}


def test_the_stratum_row_of_select_pairs_units_in_both_strata(results: dict[str, Any]) -> None:
    (row,) = [c for c in results["claims"] if c["row"] == "VW minus V1, closed-loop stratum c^lo"]
    assert row["claim"] == "C2-SELECT" and not row["primary"] and row["verdict"] is None
    cells = results["conditions"]["nominal"]["cells"]
    vw = cells["VW"]["closed"]["per_unit"]["stratum_cover_lo"]
    v1 = cells["V1"]["closed"]["per_unit"]["stratum_cover_lo"]
    pairs = [(a, b) for a, b in zip(vw, v1, strict=True) if a is not None and b is not None]
    diff = row["intervals"]["diff"]
    assert diff["n"] == len(pairs) and diff["alpha"] == 0.05
    if pairs:
        assert diff["estimate"] == pytest.approx(sum(a - b for a, b in pairs) / len(pairs))


# ----------------------------------------------------------------------------- rendering


def test_rendering_is_deterministic_and_reads_only_results(results: dict[str, Any]) -> None:
    md = report.render_results_md(results)
    assert md == report.render_results_md(copy.deepcopy(results))
    assert md == report.render_results_md(json.loads(report.to_json(results)))
    block = report.render_readme_block(results)
    assert block == report.render_readme_block(copy.deepcopy(results))
    assert md.endswith("\n") and not md.endswith("\n\n") and "\r" not in md


def test_a_changed_number_changes_the_document(results: dict[str, Any]) -> None:
    edited = copy.deepcopy(results)
    edited["claims"][0]["intervals"]["gap"]["estimate"] = 0.123456789
    md = report.render_results_md(edited)
    assert md != report.render_results_md(results)
    assert "0.1235 (" in md  # four decimals, generated from the stored value


def test_cycles_show_one_decimal(results: dict[str, Any]) -> None:
    edited = copy.deepcopy(results)
    edited["conditions"]["nominal"]["cells"]["V1"]["closed"]["policy"]["mean_width"] = 12.345678
    assert "| 12.3 |" in report.render_results_md(edited)


def test_every_document_says_the_data_is_simulated(results: dict[str, Any]) -> None:
    for text in (report.render_results_md(results), report.render_readme_block(results)):
        assert "C-MAPSS: simulated by NASA's C-MAPSS model, not field data" in text
    assert report.render_results_md(results).startswith("# Results (SIMULATED data)")


def test_every_document_states_the_capped_truth(results: dict[str, Any]) -> None:
    """Review D, S4: a reader must see that coverage and C2-EXT are against min(RUL, 125)."""
    assert results["truth"] == {"capped_at": 125, "definition": "min(remaining life, cap)"}
    for text in (report.render_results_md(results), report.render_readme_block(results)):
        assert "Truth: min(remaining life, 125 cycles)." in text
        assert "bounds are limited to 125" in text
        assert "Remaining life at replacement (the cost) is uncapped." in text
    moved = {**results, "truth": {"capped_at": 90, "definition": "min(remaining life, cap)"}}
    shown = report.render_readme_block(moved)
    assert "min(remaining life, 90 cycles)" in shown and "bounds are limited to 90" in shown
    older = {k: v for k, v in results.items() if k != "truth"}
    assert "Truth:" not in report.render_results_md(older)  # nothing recorded, nothing claimed


def test_the_quick_banner_shows_only_in_quick_mode(results: dict[str, Any]) -> None:
    assert report.QUICK_BANNER not in report.render_results_md(results)
    quick = {**results, "mode": "quick"}
    assert report.QUICK_BANNER in report.render_results_md(quick)
    assert report.QUICK_BANNER in report.render_readme_block(quick)


def test_every_primary_verdict_is_shown(results: dict[str, Any]) -> None:
    md, block = report.render_results_md(results), report.render_readme_block(results)
    for c in results["claims"]:
        if c["primary"]:
            line = f"| {c['claim']} | {c['row']} | {c['verdict']} |"
            assert line in md and line in block


def test_the_readme_select_row_shows_vws_interval(results: dict[str, Any]) -> None:
    (sel,) = [c for c in results["claims"] if c["claim"] == "C2-SELECT" and c["primary"]]
    block = report.render_readme_block(results)
    (line,) = [x for x in block.split("\n") if x.startswith("| C2-SELECT |")]
    assert report.ival(sel["intervals"]["diff"]) in line
    assert f"VW's c^lo {report.ival(sel['intervals']['vw'])}" in line


def test_the_subset_tables_and_the_secondary_select_row_are_rendered(
    results: dict[str, Any],
) -> None:
    md = report.render_results_md(results)
    assert "## By subset (registered, descriptive)" in md
    assert "### nominal: coverage by subset" in md and "### shifted: Gap_B by subset" in md
    cell = results["conditions"]["nominal"]["by_subset"]["FD002"]["cells"]["VW"]["closed"]
    assert f"| FD002 | VW | closed | {report.ival(cell['coverage'])} |" in md
    for subset, v in results["external"]["by_subset"].items():
        assert f"| {subset} | {v['k']} | {v['n']} |" in md
    assert "| C2-SELECT | VW minus V1, closed-loop stratum c^lo | nominal |" in md


def test_the_limits_read_their_counts_from_results(results: dict[str, Any]) -> None:
    md = report.render_results_md(results)
    assert "nothing here is typed by hand" not in md
    assert "every number here is read from `results.json`" in md
    units = results["units"]
    fd003 = units["nominal"]["calib"]["by_subset"]["FD003"]
    shifted = units["shifted"]
    want = (
        f"FD003 contributes {fd003} calibration units, and the shifted condition calibrates on "
        f"{shifted['calib']['total']} units and evaluates on {shifted['eval']['total']}."
    )
    assert want in md
    edited = copy.deepcopy(results)
    edited["units"]["shifted"]["eval"]["total"] = 182
    assert "evaluates on 182." in report.render_results_md(edited)


def test_the_validity_wording_is_qualified(results: dict[str, Any]) -> None:
    md = report.render_results_md(results)
    assert "valid at any size for independent values with one mean, fed in a uniformly" in md
    assert "valid at any size and conservative" not in md
    # review D, S2: the two unmet conditions are stated
    assert "the stated level is nominal, not proven" in md
    assert "but the order is not random" in md
    assert "V2L's and V3L's per-unit values are coupled through the shared alpha_t" in md


def test_select_records_how_vw_ended_and_says_what_that_means(results: dict[str, Any]) -> None:
    """Review D, N3: the row carries VW's end state; a two-cycle shows the other pair's intervals
    in both documents and an unsettled VW is called inconclusive by rule."""
    (row,) = [c for c in results["claims"] if c["claim"] == "C2-SELECT" and c["primary"]]
    vw = results["conditions"]["nominal"]["methods"]["VW"]
    assert row["vw_state"] == vw["state"] == vw_state(vw["history"])
    assert (vw["alt_quantiles"] is None) is (vw["state"] != "two-cycle")
    assert (vw["alt_closed"] is None) is (vw["state"] != "two-cycle")
    unsettled_note = "is inconclusive by rule: VW's fixed-point steps neither settled"
    cycle_note = "VW's fixed-point steps alternate between two pairs, so C2-SELECT is judged"
    diff_alt = Interval(0.0123, 0.0456, "t", 300, 0.03, 0.05 / 6).to_dict()
    vw_alt = Interval(0.9012, 0.9345, "betting", 300, 0.92, 0.05 / 6).to_dict()

    def with_state(state: str, **extra: Any) -> dict[str, Any]:
        claims = [
            {**c, "vw_state": state, "intervals": {**c["intervals"], **extra}} if c is row else c
            for c in results["claims"]
        ]
        return {**results, "claims": claims}

    md = report.render_results_md(with_state("settled"))
    assert unsettled_note not in md and cycle_note not in md
    case = with_state("unsettled")
    md, block = report.render_results_md(case), report.render_readme_block(case)
    assert unsettled_note in md and cycle_note not in md
    assert "VW had not settled" in block
    case = with_state("two-cycle", diff_alt=diff_alt, vw_alt=vw_alt)
    md, block = report.render_results_md(case), report.render_readme_block(case)
    assert cycle_note in md and unsettled_note not in md
    shown = (
        f"With the other pair: difference {report.ival(diff_alt)}, interval rule t; VW's c^lo "
        f"{report.ival(vw_alt)}, interval rule betting."
    )
    assert shown in md
    assert (
        f"other VW pair: difference {report.ival(diff_alt)}, VW's c^lo {report.ival(vw_alt)}"
        in block
    )
    assert "VW had not settled" not in block


@pytest.mark.parametrize(
    ("history", "ended"),
    [
        ([[-2.0, 2.0], [-2.0, 2.0]], "The last step did not move the quantiles (settled)."),
        ([[-2.0, 2.0], [-3.0, 1.5], [-2.0, 2.0]], "alternate between two pairs (a two-cycle)."),
        ([[-2.0, 2.0], [-3.0, 1.5]], "had not reached a two-cycle (unsettled)."),
    ],
)
def test_the_quantile_section_says_how_vws_steps_ended(
    results: dict[str, Any], history: list[list[float]], ended: str
) -> None:
    case = copy.deepcopy(results)
    case["conditions"]["nominal"]["methods"]["VW"]["history"] = history
    assert ended in report.render_results_md(case)


def test_the_runs_section_lists_earlier_runs(results: dict[str, Any]) -> None:
    assert "## Runs" not in report.render_results_md(results)
    case = {
        **results,
        "runs": {
            "number": 2,
            "archive": "benchmarks/runs/results-run-2.json",
            "earlier": [
                {
                    "number": 1,
                    "utc": "2026-11-20T10:00:00Z",
                    "results_sha256": "c" * 64,
                    "archive": "benchmarks/runs/results-run-1.json",
                    "verdicts": {"C2-PERC V1, pooled": "loss", "C2-EXT x": "win"},
                }
            ],
        },
    }
    md = report.render_results_md(case)
    assert "This is full run 2. Its `results.json` is also kept as " in md
    assert "`benchmarks/runs/results-run-2.json`" in md
    line = (
        "| 1 | 2026-11-20T10:00:00Z | `cccccccccccccccc` | `benchmarks/runs/results-run-1.json` "
        "| win 1, loss 1, inconclusive 0 |"
    )
    assert line in md
    assert report.absolutes_in(md, report.data_strings(case)) == []
    first = {**results, "runs": {"number": 1, "archive": "x.json", "earlier": []}}
    assert "No earlier full run is recorded in the ledger." in report.render_results_md(first)


def test_infinite_quantiles_are_printed_as_signed_infinity(results: dict[str, Any]) -> None:
    edited = copy.deepcopy(results)
    edited["conditions"]["nominal"]["methods"]["V1"]["quantiles"] = ["-inf", "inf"]
    assert "| nominal | V1 | -inf | +inf |" in report.render_results_md(edited)


# ----------------------------------------------------------------------------- the lint


@pytest.mark.parametrize("mode", ["full", "quick"])
def test_rendered_documents_use_no_absolutes(results: dict[str, Any], mode: str) -> None:
    case = {**results, "mode": mode}
    data = report.data_strings(case)
    assert data == ["never mind"]  # tuned.json free text is data, not the report's words
    for text in (report.render_results_md(case), report.render_readme_block(case)):
        assert report.absolutes_in(text, data) == []
    assert report.absolutes_in(report.render_results_md(case)) == ["never"]  # the data string


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("this is never wrong", ["never"]),
        ("Always", ["Always"]),
        ("coverage is guaranteed", ["guaranteed"]),
        ("it guarantees it", ["guarantees"]),
        ("100% of units", ["100%"]),
        ("nevertheless, always-on", ["always"]),
        ("a 1000% rise", []),
        ("nothing to see", []),
    ],
)
def test_the_lint_finds_absolutes(text: str, found: list[str]) -> None:
    assert report.absolutes_in(text) == found


# ----------------------------------------------------------------------------- README


def test_splice_replaces_only_the_block(results: dict[str, Any]) -> None:
    block = report.render_readme_block(results)
    begin, end = report.README_BEGIN, report.README_END
    readme = f"# Title\r\n\r\nIntro.\r\n\r\n{begin}\r\nold\r\n{end}\r\nTail.\r\n"
    spliced = report.splice_readme(readme, block)
    assert spliced.startswith("# Title\n\nIntro.\n\n") and spliced.endswith("\nTail.\n")
    assert report.readme_block_of(spliced) == block
    assert report.splice_readme(spliced, block) == spliced


@pytest.mark.parametrize(
    "readme",
    [
        "no markers",
        f"{report.README_BEGIN} only",
        f"{report.README_END} then {report.README_BEGIN}",
        f"{report.README_BEGIN}{report.README_BEGIN}{report.README_END}",
    ],
)
def test_bad_markers_raise(readme: str) -> None:
    with pytest.raises(report.MarkerError):
        report.splice_readme(readme, "x")
