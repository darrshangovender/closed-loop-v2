"""The development loader: dev and calib only, checked first, never touching held-out files."""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from synthetic_cmapss import SYNTH_UNITS, Synthetic, heldout_names, recording_opens

from closed_loop_v2.data import load, paths
from closed_loop_v2.data.errors import DataError, HeldoutError
from closed_loop_v2.data.load import development_units
from closed_loop_v2.data.roles import ROLE_COUNTS, UNITS_PER_FILE, role_of
from closed_loop_v2.types import SUBSETS


def _expected(subset: str, role: str) -> list[int]:
    n = SYNTH_UNITS[subset][0]
    return [u for u in range(1, n + 1) if role_of(subset, u) == role]


# --------------------------------------------------------------------------- synthetic files


@pytest.mark.parametrize("role", ["dev", "calib"])
def test_returns_exactly_the_units_of_that_role_in_order(synth: Synthetic, role: str) -> None:
    units = development_units(role, ("FD001", "FD002"))
    assert [(t.subset, t.unit) for t in units] == [
        (s, u) for s in ("FD001", "FD002") for u in _expected(s, role)
    ]
    assert all(role_of(t.subset, t.unit) == role for t in units)
    assert all(t.source == "train" and t.rul()[-1] == 0 for t in units)


def test_dev_and_calib_are_disjoint_and_leave_out_eval(synth: Synthetic) -> None:
    dev = {t.key for t in development_units("dev", ("FD001", "FD002"))}
    calib = {t.key for t in development_units("calib", ("FD001", "FD002"))}
    assert dev and calib and not dev & calib
    loaded = dev | calib
    for subset, (n, _) in SYNTH_UNITS.items():
        for unit in range(1, n + 1):
            assert (("train", subset, unit) in loaded) == (role_of(subset, unit) != "eval")


def test_subsets_argument_selects_files_and_order(synth: Synthetic) -> None:
    units = development_units("dev", ("FD002",))
    assert {t.subset for t in units} == {"FD002"}
    both = development_units("dev", ("FD002", "FD001"))
    assert [t.subset for t in both][0] == "FD002" and [t.subset for t in both][-1] == "FD001"


def test_eval_is_refused_with_heldout_error(synth: Synthetic) -> None:
    with pytest.raises(HeldoutError, match="eval"):
        development_units("eval", ("FD001",))


def test_eval_is_refused_before_any_file_is_touched(synth: Synthetic) -> None:
    with recording_opens() as opened, pytest.raises(HeldoutError):
        development_units("eval", ("FD001",))
    assert opened == []


@pytest.mark.parametrize("role", ["test", "all", "", "DEV"])
def test_other_roles_are_refused(synth: Synthetic, role: str) -> None:
    with pytest.raises(ValueError, match="role must be"):
        development_units(role, ("FD001",))


def test_unknown_or_repeated_subsets_are_refused(synth: Synthetic) -> None:
    with pytest.raises(ValueError, match="FD005"):
        development_units("dev", ("FD005",))
    with pytest.raises(ValueError, match="repeated"):
        development_units("dev", ("FD001", "FD001"))


def test_load_never_opens_heldout_files_synthetic(synth: Synthetic) -> None:
    # The synthetic directory has test_ and RUL_ files, so the loader could open them.
    assert (synth.directory / "test_FD001.txt").exists()
    assert (synth.directory / "RUL_FD001.txt").exists()
    with recording_opens() as opened:
        development_units("dev", ("FD001", "FD002"))
        development_units("calib", ("FD001", "FD002"))
    names = [p.replace("\\", "/").rsplit("/", 1)[-1] for p in opened]
    assert "train_FD001.txt" in names and "train_FD002.txt" in names  # the check can fail
    assert heldout_names(opened) == []


@pytest.mark.parametrize(
    "reader",
    [
        lambda p: np.loadtxt(p),
        lambda p: np.genfromtxt(p),
        lambda p: io.FileIO(p).close(),
        lambda p: os.close(os.open(p, os.O_RDONLY)),
        lambda p: p.read_text(),
        lambda p: open(p, "rb").close(),
    ],
    ids=["loadtxt", "genfromtxt", "FileIO", "os.open", "read_text", "open"],
)
def test_the_recorder_sees_every_way_python_reads_a_file(
    synth: Synthetic, reader: Callable[[Path], object]
) -> None:
    # Review of 10 November, finding 3: loadtxt, genfromtxt, FileIO and os.open were missed before.
    with recording_opens() as opened:
        reader(synth.directory / "RUL_FD001.txt")
    assert heldout_names(opened) == ["RUL_FD001.txt"]


def test_the_recorder_sees_a_heldout_open(synth: Synthetic) -> None:
    """A guard on the guard: the same recorder does flag a read of a held-out file."""
    with recording_opens() as opened:
        (synth.directory / "test_FD001.txt").read_bytes()
    assert heldout_names(opened) == ["test_FD001.txt"]
    with recording_opens() as opened:
        with (synth.directory / "RUL_FD001.txt").open("rb"):
            pass
    assert heldout_names(opened) == ["RUL_FD001.txt"]


def test_the_loader_verifies_the_training_file_before_parsing(synth: Synthetic) -> None:
    data = bytearray(synth.read("train_FD001.txt"))
    data[50] ^= 0x01
    synth.write("train_FD001.txt", bytes(data), rehash=False)
    with pytest.raises(DataError, match=r"train_FD001\.txt.*sha256"):
        development_units("dev", ("FD001",))


def test_a_changed_heldout_file_does_not_stop_the_dev_loader(synth: Synthetic) -> None:
    # The dev loader checks the files it reads. It does not open the others, even to hash them.
    synth.write("test_FD001.txt", b"changed\r\n", rehash=False)
    synth.write("RUL_FD001.txt", b"changed\r\n", rehash=False)
    assert development_units("dev", ("FD001",))


def test_the_loader_does_not_parse_the_rows_of_eval_units(synth: Synthetic) -> None:
    """Only the unit column of an eval row is read: the rest of the row may be anything."""
    eval_units = {u for u in range(1, SYNTH_UNITS["FD001"][0] + 1) if role_of("FD001", u) == "eval"}
    assert eval_units
    lines = synth.read("train_FD001.txt").decode("ascii").splitlines(keepends=True)
    damaged = [
        f"{ln.split()[0]}  2  not-a-number\r\n" if int(ln.split()[0]) in eval_units else ln
        for ln in lines
    ]
    synth.write("train_FD001.txt", "".join(damaged).encode("ascii"))
    units = development_units("dev", ("FD001",)) + development_units("calib", ("FD001",))
    assert {t.unit for t in units}.isdisjoint(eval_units) and units


def test_a_training_file_without_unit_n_is_refused(synth: Synthetic) -> None:
    text = synth.read("train_FD001.txt").decode("ascii")
    last = SYNTH_UNITS["FD001"][0]
    kept = "".join(ln for ln in text.splitlines(keepends=True) if int(ln.split()[0]) != last)
    synth.write("train_FD001.txt", kept.encode("ascii"))
    with pytest.raises(DataError, match=rf"train_FD001\.txt.*missing \[{last}\]"):
        development_units("dev", ("FD001",))


def test_results_are_cached_per_process(synth: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[str] = []
    real = paths.read_verified

    def counting(directory: Path, name: str) -> bytes:
        reads.append(name)
        return real(directory, name)

    monkeypatch.setattr(load, "read_verified", counting)
    first = development_units("dev", ("FD001",))
    second = development_units("calib", ("FD001",))
    third = development_units("dev", ("FD001",))
    assert reads == ["train_FD001.txt"]  # one read serves dev and calib, and a repeat
    assert third == first and third is not first  # a new list each time, the same units
    assert all(a is b for a, b in zip(first, third, strict=True))
    assert second


def test_the_cache_is_keyed_by_data_directory(
    synth: Synthetic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = development_units("dev", ("FD001",))
    other = tmp_path / "other"
    other.mkdir()
    for name in synth.hashes:
        (other / name).write_bytes(synth.read(name))
    monkeypatch.setenv(paths.DATA_ENV, str(other))
    second = development_units("dev", ("FD001",))
    assert [t.unit for t in first] == [t.unit for t in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_arrays_are_read_only_so_callers_cannot_alter_the_cache(synth: Synthetic) -> None:
    traj = development_units("dev", ("FD001",))[0]
    with pytest.raises(ValueError, match="read-only"):
        traj.sensors[0, 0] = 1.0
    assert not traj.cycles.flags.writeable and not traj.settings.flags.writeable


# --------------------------------------------------------------------------- the real files


@pytest.mark.data
@pytest.mark.parametrize("subset", SUBSETS)
def test_real_dev_and_calib_counts_per_subset(subset: str) -> None:
    for role in ("dev", "calib"):
        units = development_units(role, (subset,))
        assert len(units) == ROLE_COUNTS[subset][role]


@pytest.mark.data
def test_real_dev_and_calib_totals() -> None:
    assert len(development_units("dev")) == 195
    assert len(development_units("calib")) == 141


@pytest.mark.data
@pytest.mark.parametrize("role", ["dev", "calib"])
def test_real_units_have_the_right_role_shape_and_rul(role: str) -> None:
    units = development_units(role)
    seen = set()
    for traj in units:
        assert role_of(traj.subset, traj.unit) == role
        assert 1 <= traj.unit <= UNITS_PER_FILE["train"][traj.subset]
        assert traj.key not in seen
        seen.add(traj.key)
        assert traj.source == "train" and traj.rul_at_end == 0
        assert traj.settings.shape == (traj.n, 3) and traj.sensors.shape == (traj.n, 21)
        assert traj.settings.dtype == np.float64 and traj.sensors.dtype == np.float64
        assert np.isfinite(traj.settings).all() and np.isfinite(traj.sensors).all()
        assert traj.rul()[-1] == 0 and traj.rul()[0] == traj.n - 1
        assert traj.failure_cycle == traj.n


@pytest.mark.data
def test_real_units_are_in_subset_then_unit_order() -> None:
    units = development_units("dev")
    keys = [(SUBSETS.index(t.subset), t.unit) for t in units]
    assert keys == sorted(keys)


@pytest.mark.data
def test_real_load_never_opens_heldout_files() -> None:
    """Patch ``open`` and the Path readers, load every dev and calib unit, list what was read."""
    load.clear_cache()  # a cached parse would read nothing and prove nothing
    with recording_opens() as opened:
        development_units("dev")
        development_units("calib")
    names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in opened}
    assert {f"train_{s}.txt" for s in SUBSETS} <= names  # it did read the training files
    assert heldout_names(opened) == []
    assert not any("RUL_" in n or n.startswith("test_") for n in names)
