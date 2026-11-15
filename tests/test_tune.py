"""``scripts/tune.py`` (docs/adr/0003-tuning-protocol.md) on synthetic files only.

The selection rules are tested on hand-made rows; the whole script runs on packet A's synthetic
C-MAPSS writer for all four subsets, with a shrunken grid and boosting settings. Nothing here reads
the real data: an autouse fixture points ``CLOSED_LOOP_V2_DATA`` at an empty folder unless a test
asks for the synthetic files, and the real ``tuned.json`` and ``TUNING.md`` must not change.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from closed_loop_v2.data import load, paths
from closed_loop_v2.data.roles import UNITS_PER_FILE
from closed_loop_v2.estimator import DEFAULT_PARAMS, Estimator
from closed_loop_v2.types import SUBSETS

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from synthetic_cmapss import Synthetic, heldout_names, recording_opens, rows_text  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TUNE_PY = REPO / "scripts" / "tune.py"
MODULE = "closed_loop_v2_scripts_tune"
N_TRAIN, N_TEST = 30, 6
WATCHED = ("benchmarks/tuned.json", "docs/TUNING.md")
SMALL = dataclasses.replace(
    DEFAULT_PARAMS, max_iter=5, learning_rate=0.3, max_leaf_nodes=4, min_samples_leaf=3
)


def load_tune() -> ModuleType:
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    spec = importlib.util.spec_from_file_location(MODULE, TUNE_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tune() -> ModuleType:
    return load_tune()


def _snapshot() -> dict[str, bytes | None]:
    return {n: (REPO / n).read_bytes() if (REPO / n).exists() else None for n in WATCHED}


@pytest.fixture(autouse=True)
def _no_real_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    empty = tmp_path / "no-data-here"
    empty.mkdir()
    monkeypatch.setenv(paths.DATA_ENV, str(empty))
    monkeypatch.delenv("CLOSED_LOOP_V2_LOCK", raising=False)
    before = _snapshot()
    load.clear_cache()
    yield
    load.clear_cache()
    assert _snapshot() == before, "a test changed the real tuned.json or TUNING.md"


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    """Synthetic files for all four subsets, in the real layout, wired in as the data in use."""
    synth = Synthetic(tmp_path / "cmapss")
    synth.directory.mkdir()
    for i, subset in enumerate(SUBSETS):
        rng = np.random.default_rng(70 + i)
        train = "".join(rows_text(u, 12 + u % 9, rng) for u in range(1, N_TRAIN + 1))
        test = "".join(rows_text(u, 5 + u % 4, rng) for u in range(1, N_TEST + 1))
        synth.write(f"train_{subset}.txt", train.encode("ascii"))
        synth.write(f"test_{subset}.txt", test.encode("ascii"))
        synth.write(f"RUL_{subset}.txt", "".join(f"{10 + u}\r\n" for u in range(N_TEST)).encode())
        monkeypatch.setitem(UNITS_PER_FILE["train"], subset, N_TRAIN)
        monkeypatch.setitem(UNITS_PER_FILE["test"], subset, N_TEST)
    synth.write("readme.txt", b"synthetic stand-in\r\n")
    monkeypatch.setattr(paths, "manifest", synth.manifest)
    monkeypatch.setenv(paths.DATA_ENV, str(synth.directory))
    return synth


@pytest.fixture
def small(tune: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A shrunken grid and boosting settings, and a commit charge well under the guard."""
    monkeypatch.setattr(tune, "BASE_PARAMS", SMALL)
    monkeypatch.setattr(tune, "WINDOWS", (3, 5))
    monkeypatch.setattr(tune, "CAPS", (125.0,))
    monkeypatch.setattr(tune, "S_MINS", (1.0, 2.0))
    monkeypatch.setattr(tune, "commit_charge_percent", lambda: 40.0)
    return tune


# ----------------------------------------------------------------------------- selection rules


def _row(window: int, cap: float, s_min: float, width: float) -> dict[str, Any]:
    return {"window": window, "cap": cap, "s_min": s_min, "mean_width": width}


def test_the_narrowest_setting_wins_outside_the_tie_band(tune: ModuleType) -> None:
    rows = [_row(15, 125.0, 2.0, 50.0), _row(50, 150.0, 5.0, 40.0)]
    assert tune.choose_setting(rows)["window"] == 50


def test_a_tie_within_one_percent_goes_to_window_then_cap_then_larger_s_min(
    tune: ModuleType,
) -> None:
    rows = [
        _row(50, 125.0, 2.0, 40.0),
        _row(30, 150.0, 2.0, 40.3),
        _row(30, 125.0, 2.0, 40.39),
        _row(30, 125.0, 5.0, 40.39),
        _row(15, 125.0, 2.0, 40.41),  # just outside 1% of 40.0
    ]
    chosen = tune.choose_setting(rows)
    assert (chosen["window"], chosen["cap"], chosen["s_min"]) == (30, 125.0, 5.0)


def _h(h: float, failed: float, rul: float | None) -> dict[str, Any]:
    return {"h": h, "failed_fraction": failed, "mean_rul_at_replacement": rul}


def test_h_is_the_smallest_remaining_life_within_the_failure_limit(tune: ModuleType) -> None:
    rows = [_h(0, 0.30, 2.0), _h(5, 0.05, 6.0), _h(10, 0.01, 11.0), _h(15, 0.0, 16.0)]
    row, met = tune.choose_h(rows)
    assert (row["h"], met) == (5, True)  # 0.05 itself is within the limit


def test_h_falls_back_to_the_lowest_failure_fraction(tune: ModuleType) -> None:
    rows = [_h(0, 0.40, 2.0), _h(5, 0.20, 6.0), _h(10, 0.20, 9.0), _h(15, 0.30, None)]
    row, met = tune.choose_h(rows)
    assert (row["h"], met) == (5, False)


def test_h_without_replacements_cannot_qualify(tune: ModuleType) -> None:
    row, met = tune.choose_h([_h(0, 0.0, None), _h(5, 0.02, 7.0)])
    assert (row["h"], met) == (5, True)


def test_m_grid_is_five_cycle_steps_up_to_cap(tune: ModuleType) -> None:
    assert tune.m_grid(125.0)[:3] == [5.0, 10.0, 15.0]
    assert tune.m_grid(125.0)[-1] == 125.0
    assert len(tune.m_grid(150.0)) == 30


def test_outer_folds_are_balanced_and_fixed(tune: ModuleType) -> None:
    a = tune.outer_folds(195, 12345)
    assert np.array_equal(a, tune.outer_folds(195, 12345))
    assert sorted(np.bincount(a).tolist()) == [39, 39, 39, 39, 39]
    assert not np.array_equal(a, tune.outer_folds(195, 12346))


# ----------------------------------------------------------------------------- the script


@pytest.mark.usefixtures("dataset")
def test_raising_s_min_at_prediction_equals_fitting_with_it(tune: ModuleType) -> None:
    units = load.development_units("dev")
    low = dataclasses.replace(SMALL, window=3, s_min=1.0)
    high = dataclasses.replace(SMALL, window=3, s_min=4.0)
    est_low = Estimator.fit(units, low, seed=7)
    est_high = Estimator.fit(units, high, seed=7)
    via_floor = tune.predictions(units, [est_low] * len(units), 4.0)
    for t, p in zip(units, via_floor, strict=True):
        direct = est_high.predict(t)
        assert np.array_equal(p.m, direct.m) and np.array_equal(p.s, direct.s)


@pytest.mark.usefixtures("dataset")
def test_tune_writes_both_files_from_dev_units_only(
    small: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles: list[str] = []
    real = small.development_units

    def recording(role: str, *args: Any, **kwargs: Any) -> Any:
        roles.append(role)
        return real(role, *args, **kwargs)

    monkeypatch.setattr(small, "development_units", recording)
    out_json, out_md = tmp_path / "tuned.json", tmp_path / "TUNING.md"
    with recording_opens() as opened:
        code = small.main(["--json", str(out_json), "--md", str(out_md)])
    assert code == 0
    assert roles == ["dev"]
    assert heldout_names(opened) == []
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["estimator"]["window"] in (3, 5)
    assert data["estimator"]["s_min"] in (1.0, 2.0)
    assert data["estimator"]["max_iter"] == SMALL.max_iter
    assert data["h"] in small.H_GRID
    assert data["stratum_m"] % 5 == 0 and data["stratum_m"] > 0
    rec = data["record"]
    assert rec["dev_units"] == len(load.development_units("dev"))
    assert len(rec["estimator_grid"]) == 4
    assert [r["h"] for r in rec["h_grid"]] == list(small.H_GRID)
    assert rec["m_grid"][-1]["fraction"] >= 0.90
    assert all(r["fraction"] < 0.90 for r in rec["m_grid"][:-1])
    md = out_md.read_text(encoding="utf-8")
    assert "simulated by NASA" in md and rec["dev_list_sha256"] in md


@pytest.mark.usefixtures("dataset")
def test_tune_repeats_byte_for_byte(small: ModuleType, tmp_path: Path) -> None:
    outs = []
    for i in range(2):
        j, m = tmp_path / f"t{i}.json", tmp_path / f"t{i}.md"
        assert small.main(["--json", str(j), "--md", str(m)]) == 0
        outs.append((j.read_bytes(), m.read_bytes()))
    assert outs[0] == outs[1]


@pytest.mark.usefixtures("dataset")
def test_tune_refuses_at_high_commit_charge_and_writes_nothing(
    small: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(small, "commit_charge_percent", lambda: 95.0)
    out_json, out_md = tmp_path / "tuned.json", tmp_path / "TUNING.md"
    assert small.main(["--json", str(out_json), "--md", str(out_md)]) == 3
    assert not out_json.exists() and not out_md.exists()


@pytest.mark.usefixtures("dataset")
def test_a_held_lock_refuses(small: ModuleType, tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    lock.write_text("owner=someone-else task=x\n", encoding="utf-8")
    out_json = tmp_path / "tuned.json"
    code = small.main(["--lock", str(lock), "--json", str(out_json), "--md", str(tmp_path / "m")])
    assert code == 3
    assert not out_json.exists()
    assert lock.read_text(encoding="utf-8") == "owner=someone-else task=x\n"
