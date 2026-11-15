"""``scripts/reachability.py`` on a made-up ``--quick`` result (no data files at all)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reachability.py"
MODULE = "closed_loop_v2_scripts_reachability"
N_QUICK = 60


def load_script() -> ModuleType:
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    spec = importlib.util.spec_from_file_location(MODULE, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reach() -> ModuleType:
    return load_script()


def _cell(cover_lo: np.ndarray, stratum: np.ndarray) -> dict[str, Any]:
    return {
        "per_unit": {
            "cover_lo": [float(x) for x in cover_lo],
            "stratum_cover_lo": [None if np.isnan(x) else float(x) for x in stratum],
        }
    }


def quick_result(gap: float = 0.0) -> dict[str, Any]:
    """Lower-bound coverage near 0.95 with a small spread; the closed loop shifted by ``gap``."""
    rng = np.random.default_rng(3)
    base = np.clip(rng.normal(0.95, 0.02, N_QUICK), 0.0, 1.0)
    noise = rng.normal(0.0, 0.01, N_QUICK)
    closed = np.clip(base + gap + noise, 0.0, 1.0)
    stratum = base.copy()
    stratum[:6] = np.nan  # six units without stratum rows
    cells = {
        m: {
            "open": _cell(base, stratum),
            "closed": _cell(closed, np.where(np.isnan(stratum), np.nan, closed)),
        }
        for m in ("V1", "V2L", "VW", "V3L")
    }
    return {
        "mode": "quick",
        "conditions": {
            "nominal": {"eval_units": [f"FD001/{i}" for i in range(N_QUICK)], "cells": cells}
        },
    }


def test_lower_to_mean_only_lowers_and_hits_the_target(reach: ModuleType) -> None:
    a = np.array([0.9, 0.95, 1.0])
    b = np.array([0.95, 0.95, 0.95])
    for target in (-0.1, 0.0, 0.07):
        a2, b2 = reach.lower_to_mean(a, b, target)
        assert np.all(a2 <= a) and np.all(b2 <= b)
        assert float(np.mean(a2 - b2)) == pytest.approx(target)


def test_no_gap_gives_losses_and_a_band_gap_does_not_win(reach: ModuleType) -> None:
    rep = reach.reachability(quick_result(0.0))
    pooled = rep["perc"][0]
    assert pooled["row"] == "C2-PERC V1, pooled" and pooled["n_target"] == 373
    assert pooled["labels_at_gap_0"] == {"loss": reach.RESAMPLES}
    assert "win" not in pooled["labels_at_gap_minus_band"]
    assert pooled["smallest_gap_all_wins"] is not None
    assert pooled["smallest_gap_all_wins"] < -reach.PERC_BAND
    stratum = rep["perc"][1]
    assert stratum["dev_units"] == N_QUICK - 6
    assert stratum["n_target"] == (373 * (N_QUICK - 6)) // N_QUICK


def test_select_needs_a_floor_vw_cannot_reach_below_it(reach: ModuleType) -> None:
    rep = reach.reachability(quick_result(0.0))
    s = rep["select"]
    # VW's coverage sits near 0.95, above the 0.92 floor, so a large enough difference wins
    assert s["smallest_diff_all_wins"] is not None
    assert s["smallest_diff_all_wins"] >= reach.SELECT_MARGIN
    assert "win" not in s["labels_at_diff_0"]


def test_ext_cases(reach: ModuleType) -> None:
    cases = reach.ext_row()["cases"]
    assert cases["0.95"]["label"] == "win"
    assert cases["0.92"]["label"] == "inconclusive"
    assert cases["0.85"]["label"] == "loss"


def test_main_writes_both_files_and_refuses_a_full_result(
    reach: ModuleType, tmp_path: Path
) -> None:
    src = tmp_path / "quick.json"
    raw = json.dumps({**quick_result(-0.1), "hashes": {"code_sha256": "c" * 64}})
    src.write_text(raw, encoding="utf-8")
    j, m = tmp_path / "r.json", tmp_path / "R.md"
    assert reach.main(["--input", str(src), "--json", str(j), "--md", str(m)]) == 0
    (rep,) = json.loads(j.read_text(encoding="utf-8"))["points"]
    assert rep["perc"][0]["dev_mean"] < -0.05
    # review D, N7: the input is pinned by its bytes and the code that made it
    assert rep["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert rep["input_code_sha256"] == "c" * 64
    text = m.read_text(encoding="utf-8")
    assert "made-up inputs" in text.lower() and rep["input_sha256"] in text
    assert "Concentrated shift" in text
    full = dict(quick_result(), mode="full")
    src.write_text(json.dumps(full), encoding="utf-8")
    assert reach.main(["--input", str(src), "--json", str(j), "--md", str(m)]) == 1


def test_main_reads_both_points_in_order(reach: ModuleType, tmp_path: Path) -> None:
    q, d = tmp_path / "quick.json", tmp_path / "dev-point.json"
    q.write_text(json.dumps(quick_result(0.0)), encoding="utf-8")
    d.write_text(json.dumps({**quick_result(-0.05), "mode": "dev-point"}), encoding="utf-8")
    j, m = tmp_path / "r.json", tmp_path / "R.md"
    assert reach.main(["--input", str(q), "--input", str(d), "--json", str(j), "--md", str(m)]) == 0
    points = json.loads(j.read_text(encoding="utf-8"))["points"]
    assert [p["point"] for p in points] == ["quick", "dev-point"]
    text = m.read_text(encoding="utf-8")
    assert text.index("Operating point: quick") < text.index("Operating point: dev-point")


def test_concentrate_moves_only_the_carriers_and_stops_at_the_floor(reach: ModuleType) -> None:
    a = np.full(10, 0.95)
    b = np.full(10, 0.95)
    carriers = np.array([2], dtype=np.intp)
    a2, b2 = reach.concentrate(a, b, -0.05, carriers)
    assert np.array_equal(b2, b) and np.array_equal(np.delete(a2, 2), np.delete(a, 2))
    assert float(np.mean(a2 - b2)) == pytest.approx(-0.05)  # one unit carries -0.5
    a3, _ = reach.concentrate(a, b, -0.2, carriers)  # would need -2.0 on one unit: floored at 0
    assert a3[2] == 0.0 and float(np.mean(a3 - b)) == pytest.approx(-0.095)
    _, b4 = reach.concentrate(a, b, 0.05, carriers)  # a rise lowers the other side
    assert b4[2] == pytest.approx(0.45) and np.array_equal(np.delete(b4, 2), np.delete(b, 2))
    assert reach.carriers_for(373, np.random.default_rng(1)).size == 38


def test_a_concentrated_gap_is_harder_to_win_than_a_uniform_one(reach: ModuleType) -> None:
    rep = reach.reachability(quick_result(0.0))
    pooled = rep["perc"][0]
    uni, con = pooled["uniform"], pooled["concentrated"]
    assert uni["smallest_gap_all_wins"] == pooled["smallest_gap_all_wins"]
    assert con["smallest_gap_all_wins"] is None or (
        con["smallest_gap_all_wins"] <= uni["smallest_gap_all_wins"]
    )
    assert set(con["at_fixed_gaps"]) == {"-0.055", "-0.08", "-0.1"}
    half_uni = uni["at_fixed_gaps"]["-0.055"]["median_half_width"]
    half_con = con["at_fixed_gaps"]["-0.055"]["median_half_width"]
    assert half_con > half_uni  # the gap on a few units spreads the differences


def test_vw_states_reads_each_fit_of_the_input_and_the_page_says_so(
    reach: ModuleType, tmp_path: Path
) -> None:
    """Review D, N3: the select rows use VW's last pair; the page names how each VW fit ended."""
    cycle = {"history": [[-2.0, 2.0], [-3.0, 1.5], [-2.0, 2.0]]}
    settled = {"history": [[-2.0, 2.0], [-2.0, 2.0]]}
    quick = quick_result(0.0)
    quick["conditions"]["nominal"]["methods"] = {"VW": cycle}
    assert reach.vw_states(quick) == ["two-cycle"]
    dev = {
        "mode": "dev-point",
        "halves": [{"methods": {"VW": cycle}}, {"methods": {"VW": settled}}],
    }
    assert reach.vw_states(dev) == ["two-cycle", "settled"]
    assert reach.vw_states(quick_result(0.0)) == []  # not recorded in the input
    src, j, m = tmp_path / "quick.json", tmp_path / "r.json", tmp_path / "R.md"
    src.write_text(json.dumps(quick), encoding="utf-8")
    assert reach.main(["--input", str(src), "--json", str(j), "--md", str(m)]) == 0
    (rep,) = json.loads(j.read_text(encoding="utf-8"))["points"]
    assert rep["vw_states"] == ["two-cycle"]
    assert "VW's fixed-point steps in this input ended: two-cycle." in m.read_text(encoding="utf-8")
