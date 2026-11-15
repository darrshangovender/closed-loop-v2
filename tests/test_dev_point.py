"""``scripts/dev_point.py`` (review D, B1) on synthetic files only.

The whole script runs on packet A's synthetic C-MAPSS writer for all four subsets, after
``scripts/tune.py`` has written a ``tuned.json`` with shrunken boosting settings. Nothing here reads
the real data, and the real ``benchmarks/.cache/dev-point.json`` must not change.
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
from closed_loop_v2.estimator import DEFAULT_PARAMS
from closed_loop_v2.types import SUBSETS

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from synthetic_cmapss import Synthetic, heldout_names, recording_opens, rows_text  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
N_TRAIN, N_TEST = 60, 6
WATCHED = ("benchmarks/.cache/dev-point.json", "benchmarks/tuned.json")
SMALL = dataclasses.replace(
    DEFAULT_PARAMS, max_iter=5, learning_rate=0.3, max_leaf_nodes=4, min_samples_leaf=3
)


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dev_point() -> ModuleType:
    return _load("closed_loop_v2_scripts_dev_point", SCRIPTS / "dev_point.py")


@pytest.fixture(scope="module")
def reach() -> ModuleType:
    return _load("closed_loop_v2_scripts_reachability", SCRIPTS / "reachability.py")


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
    assert _snapshot() == before, "a test changed the real dev-point.json or tuned.json"


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    synth = Synthetic(tmp_path / "cmapss")
    synth.directory.mkdir()
    for i, subset in enumerate(SUBSETS):
        rng = np.random.default_rng(90 + i)
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
def small(dev_point: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Shrunken boosting settings and grid in the ``tune`` module dev_point uses."""
    tune = dev_point.tune
    monkeypatch.setattr(tune, "BASE_PARAMS", SMALL)
    monkeypatch.setattr(tune, "WINDOWS", (3, 5))
    monkeypatch.setattr(tune, "S_MINS", (1.0, 2.0))
    monkeypatch.setattr(tune, "commit_charge_percent", lambda: 40.0)
    monkeypatch.setattr(dev_point, "commit_charge_percent", lambda: 40.0)
    return dev_point


def test_halves_are_balanced_and_fixed(dev_point: ModuleType) -> None:
    a = dev_point.halves(195, 7)
    assert sorted(np.bincount(a)) == [97, 98]
    assert np.array_equal(a, dev_point.halves(195, 7))
    assert not np.array_equal(a, dev_point.halves(195, 8))


def test_merge_cells_concatenates_the_halves_in_order(dev_point: ModuleType) -> None:
    def part(values: list[float], pol: str) -> dict[str, Any]:
        per = {k: values for k in dev_point.PER_UNIT_KEYS}
        return {"cells": {"V1": {"open": {"per_unit": per, "policy": pol, "stratum_units": 1}}}}

    got = dev_point.merge_cells([part([0.1, 0.2], "a"), part([0.3], "b")])
    cell = got["V1"]["open"]
    assert cell["per_unit"]["cover_lo"] == [0.1, 0.2, 0.3]
    assert cell["policy_by_half"] == ["a", "b"]


@pytest.mark.usefixtures("dataset")
def test_dev_point_reads_dev_units_only_and_feeds_reachability(
    small: ModuleType, reach: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuned_path = tmp_path / "tuned.json"
    assert small.tune.main(["--json", str(tuned_path), "--md", str(tmp_path / "T.md")]) == 0
    roles: list[str] = []
    real = small.development_units

    def recording(role: str, *args: Any, **kwargs: Any) -> Any:
        roles.append(role)
        return real(role, *args, **kwargs)

    monkeypatch.setattr(small, "development_units", recording)
    out = tmp_path / "dev-point.json"
    with recording_opens() as opened:
        code = small.main(["--tuned", str(tuned_path), "--out", str(out)])
    assert code == 0
    assert roles == ["dev"]
    assert heldout_names(opened) == []
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "dev-point"
    assert data["reproduction"]["reproduces_tuned_json"] is True  # tune's predictions, rebuilt
    n_dev = len(load.development_units("dev"))
    names = data["conditions"]["nominal"]["eval_units"]
    assert len(names) == len(set(names)) == n_dev  # every dev unit evaluated once
    assert [h["eval_units"] + h["calib_units"] for h in data["halves"]] == [n_dev, n_dev]
    # each half is calibrated on the other: no unit is calibrated on and evaluated in one half
    first = data["halves"][0]["eval_units"]
    evaluated = [set(names[:first]), set(names[first:])]
    for h, ev in zip(data["halves"], evaluated, strict=True):
        assert not set(h["calib_unit_names"]) & ev
        assert set(h["calib_unit_names"]) | ev == set(names)
    cells = data["conditions"]["nominal"]["cells"]
    assert set(cells) == {"V1", "VW", "V2L", "V3L"}
    assert all(len(cells[m][lp]["per_unit"]["cover_lo"]) == n_dev for m in cells for lp in cells[m])
    tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    assert data["operating_point"]["h"] == tuned["h"]
    assert data["operating_point"]["stratum_m"] == tuned["stratum_m"]
    prov = data["provenance"]
    assert prov["dev_list_sha256"] == tuned["record"]["dev_list_sha256"]
    assert set(prov) >= {"tuned_sha256", "code_sha256", "seed_key", "versions"}

    j, m = tmp_path / "r.json", tmp_path / "R.md"
    assert reach.main(["--input", str(out), "--json", str(j), "--md", str(m)]) == 0
    (point,) = json.loads(j.read_text(encoding="utf-8"))["points"]
    assert point["point"] == "dev-point" and point["quick_eval_units"] == n_dev
    assert point["input_code_sha256"] == prov["code_sha256"]


@pytest.mark.usefixtures("dataset")
def test_dev_point_refuses_boosting_settings_tune_did_not_use(
    small: ModuleType, tmp_path: Path
) -> None:
    tuned_path = tmp_path / "tuned.json"
    assert small.tune.main(["--json", str(tuned_path), "--md", str(tmp_path / "T.md")]) == 0
    data = json.loads(tuned_path.read_text(encoding="utf-8"))
    data["estimator"]["max_iter"] = SMALL.max_iter + 1
    tuned_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="boosting settings"):
        small.main(["--tuned", str(tuned_path), "--out", str(tmp_path / "d.json")])


@pytest.mark.usefixtures("dataset")
def test_dev_point_fails_when_the_predictions_do_not_reproduce_tuned_json(
    small: ModuleType, tmp_path: Path
) -> None:
    tuned_path = tmp_path / "tuned.json"
    assert small.tune.main(["--json", str(tuned_path), "--md", str(tmp_path / "T.md")]) == 0
    data = json.loads(tuned_path.read_text(encoding="utf-8"))
    data["record"]["chosen_quantiles"]["q_lo"] -= 0.5
    tuned_path.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "d.json"
    assert small.main(["--tuned", str(tuned_path), "--out", str(out)]) == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproduction"]["reproduces_tuned_json"] is False
