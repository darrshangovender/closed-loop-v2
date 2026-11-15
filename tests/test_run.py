"""``benchmarks/run.py`` on a temporary root: freeze, full run, check and quick run.

Every test runs against a miniature repository under ``tmp_path`` with its own pre-registration,
``tuned.json``, ledger and README. Tests that need data use packet A's synthetic C-MAPSS writer
(``tests/data/synthetic_cmapss.py``) for all four subsets, swapped in as the manifest in use, so
``HeldoutAccess`` treats the files as synthetic and accepts the temporary ledger. Nothing here reads
the real data: an autouse fixture points ``CLOSED_LOOP_V2_DATA`` at an empty folder unless a test
asks for the synthetic files, and another fails the test if the real repository's ledger, results
or README change.

The miniature pre-registration carries the registered-values table that ``--registered-table``
prints, as ``--freeze`` requires; ``tuned.json`` gives ``stratum_m`` like ``scripts/tune.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import shutil
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from closed_loop.audit.ledger import Ledger
from closed_loop.audit.prereg import code_sha256, file_sha256

from closed_loop_v2 import report
from closed_loop_v2.data import load, paths
from closed_loop_v2.data.roles import UNITS_PER_FILE
from closed_loop_v2.estimator import DEFAULT_PARAMS
from closed_loop_v2.types import SUBSETS, Trajectory

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from synthetic_cmapss import (  # noqa: E402
    Synthetic,
    heldout_names,
    recording_opens,
    rows_text,
    rul_values,
)

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "benchmarks" / "run.py"
TUNE_PY = REPO / "scripts" / "tune.py"
MODULE = "closed_loop_v2_bench_run"
TUNE_MODULE = "closed_loop_v2_scripts_tune"
PREREG = "# Pre-registration (test copy)\n\nClaims C2-PERC, C2-SELECT, C2-EXT.\n"
TUNED = {
    "estimator": {
        "window": 5,
        "cap": 125.0,
        "s_min": 1.0,
        "max_iter": 8,
        "learning_rate": 0.2,
        "max_leaf_nodes": 8,
        "min_samples_leaf": 3,
    },
    "h": 3.0,
    "stratum_m": 10.0,
    "tuned_on": "a test copy",
}
N_TRAIN, N_TEST = 30, 6
README = f"# Test\n\nIntro.\n\n{report.README_BEGIN}\nnothing yet\n{report.README_END}\n\nEnd.\n"
WATCHED = (
    "benchmarks/ledger.jsonl",
    "benchmarks/results.json",
    "benchmarks/RESULTS.md",
    "README.md",
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


def load_run() -> ModuleType:
    return _load(MODULE, RUN_PY)


@pytest.fixture(scope="module")
def run_module() -> ModuleType:
    return load_run()


def _snapshot() -> dict[str, bytes | None]:
    return {n: (REPO / n).read_bytes() if (REPO / n).exists() else None for n in WATCHED}


@pytest.fixture(autouse=True)
def _the_real_repository_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_module: ModuleType
) -> Iterator[None]:
    """No real data, no shared lock, a small design; and the real records must not change."""
    empty = tmp_path / "no-data-here"
    empty.mkdir()
    monkeypatch.setenv(paths.DATA_ENV, str(empty))
    monkeypatch.delenv("CLOSED_LOOP_V2_LOCK", raising=False)
    monkeypatch.setattr(run_module, "DESIGN", run_module.Design(k=4, r=3, stratum_m=10.0))
    before = _snapshot()
    load.clear_cache()
    yield
    load.clear_cache()
    assert _snapshot() == before, "a test changed the real repository's ledger, results or README"


@pytest.fixture
def repo(tmp_path: Path, run_module: ModuleType) -> Path:
    """A miniature repository: a package, a copy of run.py, a pre-registration, tuned.json.

    The pre-registration ends with the table ``--registered-table`` prints for this design and
    ``tuned.json``, pasted as a person would paste it.
    """
    root = tmp_path / "repo"
    (root / "closed_loop_v2").mkdir(parents=True)
    (root / "closed_loop_v2" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (root / "benchmarks").mkdir()
    shutil.copy(RUN_PY, root / "benchmarks" / "run.py")
    (root / "docs").mkdir()
    (root / "benchmarks" / "tuned.json").write_text(json.dumps(TUNED), encoding="utf-8")
    code, table = invoke(run_module, root, "--registered-table")
    assert code == 0, table
    (root / "docs" / "PREREGISTRATION.md").write_bytes((PREREG + "\n" + table).encode())
    (root / "README.md").write_bytes(README.encode())
    return root


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    """Synthetic files for all four subsets, in the real layout, wired in as the data in use."""
    synth = Synthetic(tmp_path / "cmapss")
    synth.directory.mkdir()
    for i, subset in enumerate(SUBSETS):
        rng = np.random.default_rng(40 + i)
        train = "".join(rows_text(u, 12 + u % 9, rng) for u in range(1, N_TRAIN + 1))
        test = "".join(rows_text(u, 5 + u % 4, rng) for u in range(1, N_TEST + 1))
        synth.write(f"train_{subset}.txt", train.encode("ascii"))
        synth.write(f"test_{subset}.txt", test.encode("ascii"))
        rul = "".join(f"{v}\r\n" for v in rul_values(N_TEST))
        synth.write(f"RUL_{subset}.txt", rul.encode("ascii"))
        monkeypatch.setitem(UNITS_PER_FILE["train"], subset, N_TRAIN)
        monkeypatch.setitem(UNITS_PER_FILE["test"], subset, N_TEST)
    synth.write("readme.txt", b"synthetic stand-in\r\n")
    monkeypatch.setattr(paths, "manifest", synth.manifest)
    monkeypatch.setenv(paths.DATA_ENV, str(synth.directory))
    return synth


def invoke(module: ModuleType, repo: Path, *argv: str, reader: Any = None) -> tuple[int, str]:
    out = io.StringIO()
    code = module.main(
        ["--root", str(repo), *argv],
        out=out,
        commit_reader=reader if reader is not None else (lambda: 40.0),
    )
    return code, out.getvalue()


def kinds(repo: Path) -> list[str]:
    return [e["kind"] for e in Ledger(repo / "benchmarks" / "ledger.jsonl").entries()]


def prereg_text(repo: Path) -> str:
    return (repo / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- freeze


def test_freeze_records_the_hashes_and_refuses_twice(run_module: ModuleType, repo: Path) -> None:
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 0, text
    (entry,) = Ledger(repo / "benchmarks" / "ledger.jsonl").entries()
    assert entry["kind"] == "freeze"
    assert entry["manifest_sha256"] == file_sha256(paths.MANIFEST_PATH)  # the package's own
    assert entry["tuned_sha256"] == file_sha256(repo / "benchmarks" / "tuned.json")
    assert entry["seed_labels"] == ["fit-nominal", "fit-shifted", "calib-draw", "orders"]
    assert entry["design"]["stratum_m"] == TUNED["stratum_m"]  # the tuned M, not the fallback
    hashes = run_module.current_hashes(run_module.Paths(repo))
    assert all(entry[k] == v for k, v in hashes.items())
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 1 and "refusing to freeze" in text
    assert kinds(repo) == ["freeze"]


@pytest.mark.parametrize(
    "broken", ["no prereg", "no tuned", "tuned without h", "tuned without M", "tuned typo"]
)
def test_freeze_refuses_without_its_files(run_module: ModuleType, repo: Path, broken: str) -> None:
    tuned = repo / "benchmarks" / "tuned.json"
    if broken == "no prereg":
        (repo / "docs" / "PREREGISTRATION.md").unlink()
    elif broken == "no tuned":
        tuned.unlink()
    elif broken == "tuned without h":
        tuned.write_text(json.dumps({k: v for k, v in TUNED.items() if k != "h"}), "utf-8")
    elif broken == "tuned without M":
        tuned.write_text(json.dumps({k: v for k, v in TUNED.items() if k != "stratum_m"}), "utf-8")
    else:
        tuned.write_text(json.dumps({**TUNED, "hh": 1.0}), "utf-8")
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 1 and "refusing to freeze" in text
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()


# ----------------------------------------------------------------------------- full: refusals


def test_full_refuses_without_a_freeze(run_module: ModuleType, repo: Path) -> None:
    code, text = invoke(run_module, repo, "--full")
    assert code == 1 and "no freeze entry" in text
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()


@pytest.mark.parametrize("what", ["prereg", "tuned"])
def test_full_refuses_on_changed_bytes(run_module: ModuleType, repo: Path, what: str) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    if what == "prereg":
        (repo / "docs" / "PREREGISTRATION.md").write_bytes(
            PREREG.replace("C2-EXT", "C2-X").encode()
        )
    else:
        (repo / "benchmarks" / "tuned.json").write_text(json.dumps({**TUNED, "h": 4.0}), "utf-8")
    code, text = invoke(run_module, repo, "--full")
    assert code == 1 and "differs from the freeze entry" in text
    assert kinds(repo) == ["freeze"]  # no held-out read, no run entry


def test_full_refuses_at_high_commit_charge(run_module: ModuleType, repo: Path) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    code, text = invoke(run_module, repo, "--full", reader=lambda: 95.0)
    assert code == 3 and "refused" in text
    assert kinds(repo) == ["freeze"]


def test_a_held_lock_refuses(run_module: ModuleType, repo: Path, tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    lock.write_text("owner=someone else\n", encoding="utf-8")
    code, text = invoke(run_module, repo, "--quick", "--lock", str(lock))
    assert code == 3 and "held" in text
    assert lock.read_text(encoding="utf-8") == "owner=someone else\n"  # left alone


# ----------------------------------------------------------------------------- full and check


def test_full_run_then_check_catches_every_edit(
    run_module: ModuleType, repo: Path, dataset: Synthetic
) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    code, text = invoke(run_module, repo, "--full")
    assert code == 0, text
    ledger = Ledger(repo / "benchmarks" / "ledger.jsonl")
    entries = ledger.entries()
    assert [e["kind"] for e in entries] == ["freeze", "heldout-use", "heldout-use", "full"]
    assert [e["what"] for e in entries[1:3]] == ["eval", "test"]
    results_json = repo / "benchmarks" / "results.json"
    assert entries[-1]["results_sha256"] == file_sha256(results_json)
    results = json.loads(results_json.read_text(encoding="utf-8"))
    assert results["mode"] == "full"
    assert results["units"]["external"]["total"] == N_TEST * len(SUBSETS)
    assert results["tuned"]["stratum_m"] == {"value": TUNED["stratum_m"], "from": "tuned.json"}
    assert results["design"]["stratum_m"] == TUNED["stratum_m"]
    nominal = results["conditions"]["nominal"]
    assert set(nominal["by_subset"]) == {k.split("/")[0] for k in nominal["eval_units"]}
    assert set(results["conditions"]["shifted"]["by_subset"]) <= {"FD003", "FD004"}
    secondary = [(c["claim"], c["row"]) for c in results["claims"] if not c["primary"]]
    assert ("C2-SELECT", "VW minus V1, closed-loop stratum c^lo") in secondary
    assert results["runs"] == {
        "number": 1,
        "archive": "benchmarks/runs/results-run-1.json",
        "earlier": [],
    }
    archived = repo / "benchmarks" / "runs" / "results-run-1.json"
    assert archived.read_bytes() == results_json.read_bytes()
    assert entries[-1]["run_number"] == 1
    assert entries[-1]["archive"] == "benchmarks/runs/results-run-1.json"
    assert results["hashes"]["prereg_sha256"] == entries[0]["prereg_sha256"]
    assert set(entries[-1]["verdicts"]) == {
        f"{c['claim']} {c['row']}" for c in results["claims"] if c["primary"]
    }
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Test\n\nIntro.\n\n") and readme.endswith("\n\nEnd.\n")
    assert report.DATA_NOTE in readme and "nothing yet" not in readme

    code, text = invoke(run_module, repo, "--check")
    assert code == 0 and "results: OK" in text, text

    def edited(path: Path, change: Any, expect: str) -> None:
        original = path.read_bytes()
        path.write_bytes(change(original))
        try:
            code, text = invoke(run_module, repo, "--check")
            assert code == 1 and expect in text, text
        finally:
            path.write_bytes(original)

    md = repo / "benchmarks" / "RESULTS.md"
    edited(md, lambda b: b.replace(b"inconclusive", b"win", 1), "RESULTS.md is not what")
    edited(md, lambda b: b + b"extra\n", "RESULTS.md is not what")
    edited(repo / "README.md", lambda b: b.replace(b"| C2-EXT |", b"| C2-EXT  |"), "README")
    claim = results["claims"][0]["verdict"]
    other = b'"verdict": "loss"' if claim != "loss" else b'"verdict": "win"'
    edited(
        results_json,
        lambda b: b.replace(f'"verdict": "{claim}"'.encode(), other, 1),
        "differs from the digest",
    )
    # review D, N3: a VW state that is not what the step history gives, and a two-cycle row
    # without the other pair's intervals, are both reported (on top of the digest)
    state = next(c["vw_state"] for c in results["claims"] if "vw_state" in c)
    wrongs = {"unsettled": "stored VW state", "settled": "stored VW state"}
    wrongs["two-cycle"] = "cannot be rebuilt"
    for wrong, expect in wrongs.items():
        if wrong == state:
            continue
        old, new = f'"vw_state": "{state}"'.encode(), f'"vw_state": "{wrong}"'.encode()
        edited(results_json, lambda b, o=old, n=new: b.replace(o, n), expect)
    edited(repo / "docs" / "PREREGISTRATION.md", lambda b: b + b"more\n", "prereg: CHANGED")
    assert invoke(run_module, repo, "--check")[0] == 0  # every file restored

    (repo / "closed_loop_v2" / "fix.py").write_text("y = 2\n", encoding="utf-8")
    code, text = invoke(run_module, repo, "--check")
    assert code == 0 and "code: CHANGED since the freeze (reported, not a failure)" in text


@pytest.mark.parametrize("error", [RuntimeError, OSError])
def test_a_failure_after_the_held_out_read_is_recorded_and_raised(
    run_module: ModuleType,
    repo: Path,
    dataset: Synthetic,
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    """After the read, even an ``OSError`` ('not run' before it) is ``full-failed``."""
    assert invoke(run_module, repo, "--freeze")[0] == 0

    def boom(*args: object, **kwargs: object) -> None:
        raise error("boom in the replay")

    monkeypatch.setattr(run_module, "evaluate_condition", boom)
    with pytest.raises(error, match="boom"):
        invoke(run_module, repo, "--full")
    entries = Ledger(repo / "benchmarks" / "ledger.jsonl").entries()
    assert [e["kind"] for e in entries] == ["freeze", "heldout-use", "heldout-use", "full-failed"]
    assert entries[-1]["heldout_read"] is True and "boom" in entries[-1]["error"]
    assert not (repo / "benchmarks" / "results.json").exists()


def test_full_without_data_is_not_run_and_writes_no_ledger_line(
    run_module: ModuleType, repo: Path
) -> None:
    """A ``DataError`` before the held-out read is an operational slip: exit 4, nothing recorded."""
    assert invoke(run_module, repo, "--freeze")[0] == 0
    code, text = invoke(run_module, repo, "--full")
    assert code == 4 and "not run: DataError" in text, text
    assert kinds(repo) == ["freeze"]


def test_a_pause_while_fitting_writes_no_ledger_line(
    run_module: ModuleType, repo: Path, dataset: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit charge passes the start, then crosses the guard while the models are fitted,
    before any held-out read: exit 3 and nothing recorded, so the one-shot ledger keeps no slip."""
    assert invoke(run_module, repo, "--freeze")[0] == 0
    state = {"sets": False}
    real = run_module.full_sets

    def sets_then_high(*args: Any, **kwargs: Any) -> Any:
        state["sets"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "full_sets", sets_then_high)
    with recording_opens() as opened:
        code, text = invoke(
            run_module, repo, "--full", reader=lambda: 95.0 if state["sets"] else 40.0
        )
    assert code == 3 and "nothing recorded" in text, text
    assert kinds(repo) == ["freeze"]
    assert heldout_names(opened) == []


def test_full_with_broken_readme_markers_is_not_run(
    run_module: ModuleType, repo: Path, dataset: Synthetic
) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    (repo / "README.md").write_bytes((README + report.README_BEGIN + "\n").encode())
    with recording_opens() as opened:
        code, text = invoke(run_module, repo, "--full")
    assert code == 4 and "not run" in text and "README" in text, text
    assert kinds(repo) == ["freeze"]  # no held-out read, no full-failed
    assert heldout_names(opened) == []


def test_a_failure_at_the_full_entry_is_recorded_and_the_results_are_kept(
    run_module: ModuleType, repo: Path, dataset: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0

    def boom() -> dict[str, str]:
        raise RuntimeError("boom at the full entry")

    monkeypatch.setattr(run_module, "library_versions", boom)
    with pytest.raises(RuntimeError, match="boom"):
        invoke(run_module, repo, "--full")
    assert kinds(repo) == ["freeze", "heldout-use", "heldout-use", "full-failed"]
    kept = repo / "benchmarks" / ".cache" / "full-unpublished.json"
    assert kept.read_bytes() == (repo / "benchmarks" / "results.json").read_bytes()


def test_each_full_run_is_archived_and_a_rerun_lists_the_first(
    run_module: ModuleType, repo: Path, dataset: Synthetic
) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    runs = repo / "benchmarks" / "runs"
    runs.mkdir()
    occupied = runs / "results-run-2.json"
    occupied.write_bytes(b"left here by hand\n")
    results_json = repo / "benchmarks" / "results.json"
    assert invoke(run_module, repo, "--full")[0] == 0
    first = (runs / "results-run-1.json").read_bytes()
    assert first == results_json.read_bytes()
    code, text = invoke(run_module, repo, "--full")
    assert code == 0, text
    assert (runs / "results-run-1.json").read_bytes() == first  # not replaced by the rerun
    assert occupied.read_bytes() == b"left here by hand\n"  # an existing file is not replaced
    assert (runs / "results-run-2-again.json").read_bytes() == results_json.read_bytes()
    fulls = [
        e for e in Ledger(repo / "benchmarks" / "ledger.jsonl").entries() if e["kind"] == "full"
    ]
    assert [e["run_number"] for e in fulls] == [1, 2]
    assert [e["archive"] for e in fulls] == [
        "benchmarks/runs/results-run-1.json",
        "benchmarks/runs/results-run-2-again.json",
    ]
    results = json.loads(results_json.read_text(encoding="utf-8"))
    assert results["runs"]["number"] == 2
    (earlier,) = results["runs"]["earlier"]
    assert earlier["number"] == 1 and earlier["results_sha256"] == fulls[0]["results_sha256"]
    md = (repo / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    assert "This is full run 2." in md and "| 1 |" in md
    assert "`benchmarks/runs/results-run-1.json`" in md
    code, text = invoke(run_module, repo, "--check")
    assert code == 0 and "2 full runs recorded" in text, text
    (runs / "results-run-1.json").write_bytes(first.replace(b'"mode": "full"', b'"mode": "edit"'))
    code, text = invoke(run_module, repo, "--check")
    assert code == 1 and "archived copy benchmarks/runs/results-run-1.json differs" in text
    (runs / "results-run-1.json").unlink()
    code, text = invoke(run_module, repo, "--check")
    assert code == 1 and "archived copy benchmarks/runs/results-run-1.json is missing" in text


def test_full_refuses_when_a_registered_value_in_the_code_changed_after_the_freeze(
    run_module: ModuleType, repo: Path, dataset: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review D, S3: the document and tuned.json are fixed by their hashes, but a constant in the
    code (here the betting truncation) is not; the run refuses before any held-out read."""
    assert invoke(run_module, repo, "--freeze")[0] == 0
    monkeypatch.setattr(run_module, "BET_TRUNCATION", 0.5)
    code, text = invoke(run_module, repo, "--full")
    assert code == 1 and "refusing a full run" in text and "registered-values table" in text
    assert kinds(repo) == ["freeze"]  # no held-out read, no run entry
    assert not (repo / "benchmarks" / "results.json").exists()


def test_check_fails_when_a_registered_value_in_the_code_changed_after_the_freeze(
    run_module: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    code, text = invoke(run_module, repo, "--check")
    assert "registered values: MATCH the pre-registration" in text, text
    assert code == 0, text
    monkeypatch.setattr(run_module, "SPREAD_KURTOSIS_DIVISOR", 40.0)  # as if v0.1 had changed
    code, text = invoke(run_module, repo, "--check")
    assert code == 1 and "registered values: DIFFER" in text, text
    assert "prereg: MATCH" in text  # the document itself is unchanged


def test_freeze_records_the_versions_with_closed_loop_and_its_code_hash(
    run_module: ModuleType, repo: Path
) -> None:
    import closed_loop

    assert invoke(run_module, repo, "--freeze")[0] == 0
    (entry,) = Ledger(repo / "benchmarks" / "ledger.jsonl").entries()
    versions = entry["versions"]
    assert set(versions) >= {"python", "numpy", "scipy", "scikit-learn", "closed-loop"}
    assert versions["closed-loop"] == metadata.version("closed-loop")
    v01 = Path(closed_loop.__file__).resolve().parent
    assert versions["closed-loop code_sha256"] == code_sha256([v01])


def test_check_on_an_unfrozen_repository(run_module: ModuleType, repo: Path) -> None:
    code, text = invoke(run_module, repo, "--check")
    assert code == 0 and "NOT FROZEN" in text and "none published yet" in text
    (repo / "benchmarks" / "results.json").write_text("{}", encoding="utf-8")
    code, text = invoke(run_module, repo, "--check")
    assert code == 1 and "no full run" in text


def test_check_finds_a_broken_ledger(run_module: ModuleType, repo: Path) -> None:
    assert invoke(run_module, repo, "--freeze")[0] == 0
    ledger = repo / "benchmarks" / "ledger.jsonl"
    ledger.write_bytes(ledger.read_bytes().replace(b'"kind":"freeze"', b'"kind":"full"'))
    code, text = invoke(run_module, repo, "--check")
    assert code == 1 and "BROKEN" in text


# ----------------------------------------------------------------------------- quick


def test_quick_writes_no_ledger_line_and_opens_no_held_out_file(
    run_module: ModuleType, repo: Path, dataset: Synthetic
) -> None:
    with recording_opens() as opened:
        code, text = invoke(run_module, repo, "--quick")
    assert code == 0, text
    assert text.count(report.QUICK_BANNER) == 2
    assert heldout_names(opened) == []
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()
    assert not (repo / "benchmarks" / "results.json").exists()
    quick = json.loads((repo / "benchmarks" / ".cache" / "quick.json").read_text("utf-8"))
    assert quick["mode"] == "quick" and quick["seeds"]["key"] == "quick"
    assert quick["units"]["external"]["source"].startswith("quick evaluation units")
    md = (repo / "benchmarks" / ".cache" / "quick.md").read_text("utf-8")
    assert report.QUICK_BANNER in md
    assert (repo / "README.md").read_bytes() == README.encode()


def test_quick_refuses_at_high_commit_charge(run_module: ModuleType, repo: Path) -> None:
    code, text = invoke(run_module, repo, "--quick", reader=lambda: 88.0)
    assert code == 3 and "refused" in text
    assert not (repo / "benchmarks" / ".cache").exists()


def test_quick_without_data_is_not_run(run_module: ModuleType, repo: Path) -> None:
    code, text = invoke(run_module, repo, "--quick")
    assert code == 4 and "not run" in text


# ----------------------------------------------------------------------------- pieces


def test_tuned_values_and_their_sources(run_module: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "tuned.json"
    defaults = run_module.load_tuned(path)
    assert not defaults.from_file and set(defaults.sources.values()) == {"default"}
    path.write_text(json.dumps({"h": 7, "estimator": {"window": 12}}), encoding="utf-8")
    partial = run_module.load_tuned(path)
    assert partial.h == 7.0 and partial.params.window == 12
    assert partial.sources["window"] == "tuned.json" and partial.sources["cap"] == "default"
    assert run_module.tuned_problems(path) != []  # incomplete: not freezable
    for bad in (
        {"unknown": 1},
        {"estimator": {"windw": 3}},
        {"estimator": {"window": 2.5}},
        {"estimator": {"cap": "big"}},
        {"h": float("inf")},
        {"notes": 3},
        {"estimator": {"window": 0}},
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(run_module.TunedError):
            run_module.load_tuned(path)


def test_quick_split_and_cut_are_fixed_by_unit_number(run_module: ModuleType) -> None:
    roles = [run_module.quick_role("FD002", u) for u in range(1, 261)]
    assert roles == [run_module.quick_role("FD002", u) for u in range(1, 261)]
    share = {r: roles.count(r) / len(roles) for r in ("fit", "calib", "eval")}
    assert 0.4 < share["fit"] < 0.6 and 0.15 < share["calib"] < 0.35
    n = 40
    t = Trajectory("FD003", 7, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.ones((n, 21)), 0)
    cut = run_module.quick_cut(t)
    assert 1 <= cut.n <= n and cut.rul_at_end == n - cut.n and cut.source == "test"
    assert cut.failure_cycle == t.failure_cycle


# ----------------------------------------------------------------------------- tuned M (B1)


def test_tuned_values_accept_tune_py_keys_and_refuse_others(
    run_module: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"h": 7}), encoding="utf-8")
    partial = run_module.load_tuned(path)
    assert partial.sources["stratum_m"] == "default"
    assert partial.stratum_m == run_module.DESIGN.stratum_m
    path.write_text(json.dumps({**TUNED, "record": {"grid": [1, 2]}}), encoding="utf-8")
    full = run_module.load_tuned(path)
    assert full.stratum_m == TUNED["stratum_m"] and full.sources["stratum_m"] == "tuned.json"
    assert run_module.tuned_problems(path) == []
    assert full.to_dict()["stratum_m"] == {"value": TUNED["stratum_m"], "from": "tuned.json"}
    bad_files: tuple[dict[str, Any], ...] = (
        {"record": [1, 2]},
        {"record": "grids"},
        {"stratum_m": 0},
        {"stratum_m": "fifty"},
        {"stratum_m": float("nan")},
        {"stratum_m": True},
        {"recordd": {}},
    )
    for bad in bad_files:
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(run_module.TunedError):
            run_module.load_tuned(path)


@pytest.fixture
def small_tune(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """``scripts/tune.py`` with a shrunken grid and boosting settings, as in ``test_tune.py``."""
    tune = _load(TUNE_MODULE, TUNE_PY)
    small = dataclasses.replace(
        DEFAULT_PARAMS, max_iter=5, learning_rate=0.3, max_leaf_nodes=4, min_samples_leaf=3
    )
    monkeypatch.setattr(tune, "BASE_PARAMS", small)
    monkeypatch.setattr(tune, "WINDOWS", (3, 5))
    monkeypatch.setattr(tune, "CAPS", (125.0,))
    monkeypatch.setattr(tune, "S_MINS", (1.0, 2.0))
    monkeypatch.setattr(tune, "commit_charge_percent", lambda: 40.0)
    return tune


def test_a_tuned_json_written_by_tune_py_loads_and_sets_the_runs_m(
    run_module: ModuleType,
    repo: Path,
    dataset: Synthetic,
    small_tune: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 7.5 is not on tune.py's 5-cycle grid, so the run's M cannot match it by chance
    monkeypatch.setattr(run_module, "DESIGN", run_module.Design(k=4, r=3, stratum_m=7.5))
    tuned_path = repo / "benchmarks" / "tuned.json"
    assert small_tune.main(["--json", str(tuned_path), "--md", str(tmp_path / "TUNING.md")]) == 0
    written = json.loads(tuned_path.read_text(encoding="utf-8"))
    assert {"estimator", "h", "stratum_m", "record"} <= set(written)
    tuned = run_module.load_tuned(tuned_path)
    assert tuned.stratum_m == written["stratum_m"] and tuned.stratum_m != 7.5
    assert set(tuned.sources.values()) == {"tuned.json"}
    assert run_module.tuned_problems(tuned_path) == []
    assert run_module.run_design(tuned).stratum_m == written["stratum_m"]
    code, text = invoke(run_module, repo, "--quick")
    assert code == 0, text
    quick = json.loads((repo / "benchmarks" / ".cache" / "quick.json").read_text("utf-8"))
    assert quick["design"]["stratum_m"] == written["stratum_m"]
    assert quick["tuned"]["stratum_m"] == {"value": written["stratum_m"], "from": "tuned.json"}


def test_the_pipeline_refuses_a_design_whose_m_is_not_the_tuned_m(run_module: ModuleType) -> None:
    tuned = run_module.Tuned(
        DEFAULT_PARAMS, 3.0, 10.0, dict.fromkeys(["h", "stratum_m"], "tuned.json"), {}, True
    )
    with pytest.raises(ValueError, match="tuned M"):
        run_module.run_pipeline(
            None,
            lambda: None,
            mode="quick",
            key="quick",
            hashes={},
            tuned=tuned,
            design=run_module.Design(stratum_m=50.0),
            log=lambda _: None,
            headroom=lambda: None,
        )


# ----------------------------------------------------------------------------- freeze guard (S4)


def test_the_registered_table_prints_every_value_exactly(
    run_module: ModuleType, repo: Path
) -> None:
    code, text = invoke(run_module, repo, "--registered-table")
    assert code == 0
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()  # reads no ledger, writes none
    lines = text.rstrip("\n").split("\n")
    assert lines[:2] == ["| Value | Registered | From |", "|---|---|---|"]
    names = [line.split(" | ")[0].removeprefix("| ").strip("`") for line in lines[2:]]
    estimator = [f"estimator.{f.name}" for f in dataclasses.fields(DEFAULT_PARAMS)]
    assert names == [
        "k",
        "r",
        "stratum_m",
        "alpha",
        "gamma",
        "aci_window",
        "vw_steps",
        "n_folds",
        "h",
        *estimator,
        "PERC_BAND",
        "SELECT_MARGIN",
        "SELECT_FLOOR",
        "EXT_LO",
        "EXT_HI",
        "PRIMARY_ALPHA",
        "SECONDARY_ALPHA",
        "SPREAD_MIN_PAIRS",
        "SPREAD_KURTOSIS_DIVISOR",
        "BET_TRUNCATION",
        "ORDER_KEY.0",
        "ORDER_KEY.1",
    ]
    assert f"| `PRIMARY_ALPHA` | {0.05 / 6!r} |" in text  # every digit, not rounded
    assert "| `stratum_m` | 10.0 | `benchmarks/tuned.json` |" in text  # from tuned.json
    assert "| `k` | 4 |" in text  # the design in use
    assert "| `BET_TRUNCATION` | 0.75 | `closed_loop_v2/stats.py` |" in text
    assert "| `SPREAD_KURTOSIS_DIVISOR` | 50.0 | `closed_loop.stats.confseq` (v0.1) |" in text
    assert "| `estimator.learning_rate` | 0.2 | `benchmarks/tuned.json` |" in text


@pytest.mark.parametrize("marker", ["[T]", "[R]", "DRAFT", "Placeholders:"])
def test_freeze_refuses_a_placeholder(run_module: ModuleType, repo: Path, marker: str) -> None:
    path = repo / "docs" / "PREREGISTRATION.md"
    path.write_bytes((prereg_text(repo) + f"\nSomething = {marker}\n").encode())
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 1 and f"still contains {marker!r}" in text, text
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()


def test_freeze_refuses_without_the_registered_table_and_prints_it(
    run_module: ModuleType, repo: Path
) -> None:
    path = repo / "docs" / "PREREGISTRATION.md"
    _, table = invoke(run_module, repo, "--registered-table")
    path.write_bytes(PREREG.encode())
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 1 and "registered-values table" in text
    assert table.strip("\n") in text  # printed, ready to paste
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()
    edited = table.replace("| `k` | 4 |", "| `k` | 5 |")
    assert edited != table
    path.write_bytes((PREREG + "\n" + edited).encode())  # a number that differs from the code
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 1 and "registered-values table" in text
    assert not (repo / "benchmarks" / "ledger.jsonl").exists()


def test_freeze_accepts_the_table_with_crlf_line_ends(run_module: ModuleType, repo: Path) -> None:
    path = repo / "docs" / "PREREGISTRATION.md"
    path.write_bytes(prereg_text(repo).replace("\n", "\r\n").encode())
    code, text = invoke(run_module, repo, "--freeze")
    assert code == 0, text
    assert kinds(repo) == ["freeze"]


def test_calibration_and_evaluation_refuse_a_truth_not_capped_at_the_estimators_cap(
    run_module: ModuleType, tmp_path: Path
) -> None:
    """Review D, N5: the estimator caps its own targets and accepts uncapped trajectories, so the
    guard sits where calibration and the replay take the truth from the trajectories."""
    rng = np.random.default_rng(5)
    n = 30
    units = [
        Trajectory(
            "FD001",
            u,
            "train",
            np.arange(1, n + 1),
            rng.normal(size=(n, 3)),
            rng.normal(size=(n, 21)),
            0,
        )
        for u in range(1, 5)
    ]
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps(TUNED), encoding="utf-8")
    tuned = run_module.load_tuned(path)
    cap = run_module.truth_cap(tuned)
    design = run_module.run_design(tuned)
    capped = [u.capped(cap) for u in units]
    sets = run_module.DevSets(fit={"nominal": capped}, calib={"nominal": units})
    with pytest.raises(ValueError, match="nominal calibration: trajectories capped at None"):
        run_module.fit_condition("nominal", sets, tuned, design, "k", lambda _: None)
    with pytest.raises(ValueError, match="nominal evaluation: trajectories capped at None"):
        run_module.evaluate_condition("nominal", None, units, tuned, design, "k", lambda _: None)
    run_module.require_cap(capped, cap, "x")  # capped at the cap: accepted
    with pytest.raises(ValueError, match=f"capped at 100, not at {cap}"):
        run_module.require_cap([capped[0], units[1].capped(100)], cap, "x")


def test_the_external_check_limits_the_lower_bound_to_the_cap(run_module: ModuleType) -> None:
    """A NASA test unit with 200 cycles left, capped at 125: a raw lower bound of 130 is limited to
    125 (review D, S1), so it covers the capped truth (125), as it covers the remaining life (200).
    Before the limit it counted as a miss of the capped truth."""
    from types import SimpleNamespace

    from closed_loop_v2.conformal import Quantiles
    from closed_loop_v2.estimator import Prediction

    n = 4
    t = Trajectory(
        "FD001", 1, "test", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 200
    )
    pred = Prediction(np.full(n, 130.0), np.ones(n))
    fitted = SimpleNamespace(
        estimator=SimpleNamespace(predict=lambda _t: pred, params=SimpleNamespace(cap=125.0)),
        v1=SimpleNamespace(quantiles=lambda: Quantiles(0.0, 1.0)),
    )
    assert run_module.external_check(fitted, [t.capped(125)], "test")["k"] == 1
    with pytest.raises(ValueError, match="external: trajectories capped at None, not at 125"):
        run_module.external_check(fitted, [t], "test")  # review D, N5: uncapped is refused
    with pytest.raises(ValueError, match="capped at 90, not at 125"):
        run_module.external_check(fitted, [t.capped(90)], "test")
    short = Trajectory(
        "FD001", 2, "test", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 100
    )  # 100 cycles left: a bound of 125 (130 limited) misses, capped or not
    assert run_module.external_check(fitted, [short.capped(125)], "test")["k"] == 0


# ----------------------------------------------------------------------------- VW's end state (N3)


def _wearing_unit(subset: str, number: int, rng: np.random.Generator) -> Trajectory:
    n = int(rng.integers(15, 40))
    t = np.arange(1, n + 1)
    wear = (t / n)[:, None] * rng.uniform(0.5, 2.0, size=21)
    sensors = wear + 0.3 * rng.normal(size=(n, 21))
    return Trajectory(subset, number, "train", t, rng.normal(size=(n, 3)), sensors, 0)


@pytest.fixture(scope="module")
def vw_cases(run_module: ModuleType) -> dict[str, Any]:
    """One nominal condition fitted on made-up units and evaluated three ways: VW in a two-cycle
    between V1's pair ``a`` (its last) and a pair ``b`` with a higher lower quantile, VW settled at
    ``b``, and VW unsettled (``a``, ``b``, then a third pair)."""
    from closed_loop_v2.conformal import Quantiles, SplitConformal
    from closed_loop_v2.estimator import EstimatorParams

    rng = np.random.default_rng(5)
    # 20 calibration units, enough for finite quantiles at alpha 0.10
    units = [_wearing_unit(s, u, rng) for s in SUBSETS for u in range(1, 16)]
    params = EstimatorParams(8, 125.0, 1.0, 10, 0.2, 8, 3)
    names = [*(f.name for f in dataclasses.fields(params)), "h", "stratum_m"]
    tuned = run_module.Tuned(params, 4.0, 15.0, dict.fromkeys(names, "tuned.json"), {}, True)
    design = run_module.Design(k=4, r=3, stratum_m=15.0, vw_steps=3)
    capped = [u.capped(run_module.truth_cap(tuned)) for u in units]
    fit, calib, ev = capped[0::3], capped[1::3], capped[2::3]
    sets = run_module.DevSets(fit={"nominal": fit}, calib={"nominal": calib})
    fitted = run_module.fit_condition("nominal", sets, tuned, design, "n3", lambda _: None)
    a = fitted.v1.quantiles()
    # a higher q_lo raises the lower bounds, so b misses where a covers: the pairs can be told apart
    b = Quantiles(a.q_lo + 2.0, a.q_hi)
    c = Quantiles(a.q_lo + 1.0, a.q_hi)
    assert len({a, b, c}) == 3  # three different pairs, or the states below would collapse
    vws = {
        "two-cycle": SplitConformal(a, history=(a, b, a), moved_last_step=True),
        "settled": SplitConformal(b, history=(a, b, b), moved_last_step=False),
        "unsettled": SplitConformal(c, history=(a, b, c), moved_last_step=True),
    }
    out: dict[str, Any] = {"a": a, "b": b}
    for name, vw in vws.items():
        case = dataclasses.replace(fitted, vw=vw)
        out[name] = run_module.evaluate_condition(
            "nominal", case, ev, tuned, design, "n3", lambda _: None
        )
    return out


def test_a_two_cycling_vw_replays_its_other_pair_in_the_closed_loop(
    run_module: ModuleType, vw_cases: dict[str, Any]
) -> None:
    cyc, cyc_table = vw_cases["two-cycle"]
    settled, settled_table = vw_cases["settled"]
    b = vw_cases["b"]
    vw = cyc["methods"]["VW"]
    assert vw["state"] == "two-cycle" and vw["alt_quantiles"] == [b.q_lo, b.q_hi]
    # the other pair's closed loop is VW's closed loop when VW settles at that pair
    alt = cyc_table[(run_module.VW_ALT, "closed")].cover_lo
    np.testing.assert_array_equal(alt, settled_table[("VW", "closed")].cover_lo)
    assert alt.mean() < cyc_table[("VW", "closed")].cover_lo.mean()  # the two pairs differ
    at_b = settled["cells"]["VW"]["closed"]
    assert vw["alt_closed"]["per_unit"]["cover_lo"] == at_b["per_unit"]["cover_lo"]
    assert vw["alt_closed"]["policy"] == at_b["policy"]
    assert vw["alt_closed"]["stratum_units"] == at_b["stratum_units"]
    assert set(cyc["cells"]) == set(run_module.METHODS)  # not a reported method
    for name in ("settled", "unsettled"):
        out, table = vw_cases[name]
        m = out["methods"]["VW"]
        assert m["state"] == name and m["alt_quantiles"] is None and m["alt_closed"] is None
        with pytest.raises(KeyError):
            table[(run_module.VW_ALT, "closed")]


def test_select_is_judged_with_both_pairs_of_a_two_cycle(
    run_module: ModuleType, vw_cases: dict[str, Any]
) -> None:
    from closed_loop_v2.stats import Interval
    from closed_loop_v2.verdicts import select_claim_verdict, verdict_of_claim

    external = {"coverage_lo": Interval(0.93, 0.97, "clopper-pearson", 100, 0.95, 0.01).to_dict()}

    def select(table: Any, state: str) -> dict[str, Any]:
        claims = run_module.build_claims(table, table, external, vw_end=state)
        (row,) = [c for c in claims if c["claim"] == "C2-SELECT" and c["primary"]]
        return dict(row)

    cyc = select(vw_cases["two-cycle"][1], "two-cycle")
    at_b = select(vw_cases["settled"][1], "settled")
    ivs = cyc["intervals"]
    assert cyc["vw_state"] == "two-cycle" and at_b["vw_state"] == "settled"
    # the other pair's intervals are the ones a VW settled at that pair would have
    assert (ivs["diff_alt"], ivs["vw_alt"]) == (at_b["intervals"]["diff"], at_b["intervals"]["vw"])
    rows = [
        (Interval.from_dict(ivs["diff"]), Interval.from_dict(ivs["vw"])),
        (Interval.from_dict(ivs["diff_alt"]), Interval.from_dict(ivs["vw_alt"])),
    ]
    assert cyc["verdict"] == select_claim_verdict("two-cycle", rows) == verdict_of_claim(cyc)
    unsettled = select(vw_cases["unsettled"][1], "unsettled")
    assert unsettled["verdict"] == "inconclusive" and "diff_alt" not in unsettled["intervals"]
    with pytest.raises(KeyError):  # a two-cycle needs the other pair's replay
        select(vw_cases["settled"][1], "two-cycle")
