"""HeldoutAccess on synthetic files, with a temporary ledger and pre-registration.

The synthetic eval units, test files and RUL files are written to a pytest ``tmp_path``. No real
held-out file is opened by any test in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from closed_loop.audit.ledger import Ledger
from closed_loop.audit.prereg import prereg_sha256
from synthetic_cmapss import SYNTH_UNITS, Synthetic, cycles_of, rul_values

from closed_loop_v2 import audit
from closed_loop_v2.data import HeldoutAccess
from closed_loop_v2.data.errors import DataError, HeldoutError
from closed_loop_v2.data.roles import role_of

ALL = tuple(SYNTH_UNITS)


@dataclass
class Setup:
    ledger: Ledger
    prereg: Path
    access: HeldoutAccess

    def freeze(self) -> dict[str, Any]:
        return self.ledger.append("freeze", prereg_sha256=prereg_sha256(self.prereg))

    def uses(self) -> list[dict[str, Any]]:
        return [e for e in self.ledger.entries() if e["kind"] == "heldout-use"]


@pytest.fixture
def setup(synth: Synthetic, tmp_path: Path) -> Setup:
    prereg = tmp_path / "PREREGISTRATION.md"
    prereg.write_bytes(b"# Pre-registration, version one\n")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    return Setup(ledger, prereg, HeldoutAccess(ledger, prereg))


def _eval_units(subset: str) -> list[int]:
    return [u for u in range(1, SYNTH_UNITS[subset][0] + 1) if role_of(subset, u) == "eval"]


# --------------------------------------------------------------------------- before a freeze


def test_refuses_before_any_freeze(setup: Setup) -> None:
    with pytest.raises(HeldoutError, match="held out"):
        setup.access.eval_units(ALL)
    with pytest.raises(HeldoutError, match="held out"):
        setup.access.test_units(ALL)
    assert not setup.access.is_frozen()
    assert setup.ledger.entries() == []  # a refusal is not a use


def test_the_gate_comes_before_any_file_is_read(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOSED_LOOP_V2_DATA", str(tmp_path / "no-such-directory"))
    with pytest.raises(HeldoutError):  # not DataError: the data was never looked for
        setup.access.eval_units(ALL)
    with pytest.raises(HeldoutError):
        setup.access.test_units(ALL)


def test_a_freeze_for_another_document_does_not_open_it(setup: Setup) -> None:
    setup.ledger.append("freeze", prereg_sha256="0" * 64)
    with pytest.raises(HeldoutError):
        setup.access.eval_units(ALL)
    assert not setup.access.is_frozen()


def test_only_a_freeze_entry_counts(setup: Setup) -> None:
    # Another kind of entry that carries the same hash is not a freeze.
    digest = prereg_sha256(setup.prereg)
    setup.ledger.append("full", prereg_sha256=digest)
    setup.ledger.append("heldout-use", prereg_sha256=digest, n=1)
    setup.ledger.append("full-failed", prereg_sha256=digest)
    assert not setup.access.is_frozen()
    with pytest.raises(HeldoutError):
        setup.access.test_units(ALL)


def test_a_freeze_without_a_hash_field_does_not_open_it(setup: Setup) -> None:
    setup.ledger.append("freeze")
    assert not setup.access.is_frozen()


def test_refuses_a_ledger_that_does_not_verify(setup: Setup) -> None:
    setup.freeze()
    raw = setup.ledger.path.read_bytes()
    setup.ledger.path.write_bytes(raw.replace(b'"freeze"', b'"Freeze"'))
    with pytest.raises(HeldoutError, match="does not verify"):
        setup.access.eval_units(ALL)
    with pytest.raises(HeldoutError, match="does not verify"):
        setup.access.is_frozen()


def test_refuses_when_the_preregistration_cannot_be_read(setup: Setup) -> None:
    setup.freeze()
    setup.prereg.unlink()
    with pytest.raises(HeldoutError, match=r"PREREGISTRATION\.md"):
        setup.access.eval_units(ALL)


# --------------------------------------------------------------------------- after a freeze


def test_is_frozen_after_a_matching_freeze(setup: Setup) -> None:
    assert not setup.access.is_frozen()
    setup.freeze()
    assert setup.access.is_frozen()


def test_eval_units_after_a_freeze(setup: Setup) -> None:
    setup.freeze()
    units = setup.access.eval_units(ALL)
    assert [(t.subset, t.unit) for t in units] == [(s, u) for s in ALL for u in _eval_units(s)]
    assert all(role_of(t.subset, t.unit) == "eval" for t in units)
    assert all(t.source == "train" and t.rul_at_end == 0 and t.rul()[-1] == 0 for t in units)
    assert all(t.n == cycles_of(t.unit) for t in units)


def test_eval_units_respects_the_subsets_argument(setup: Setup) -> None:
    setup.freeze()
    units = setup.access.eval_units(("FD002",))
    assert {t.subset for t in units} == {"FD002"}
    assert [t.unit for t in units] == _eval_units("FD002")


def test_test_units_after_a_freeze_take_rul_from_the_rul_file_in_unit_order(setup: Setup) -> None:
    setup.freeze()
    units = setup.access.test_units(ALL)
    assert [(t.subset, t.unit) for t in units] == [
        (s, u) for s in ALL for u in range(1, SYNTH_UNITS[s][1] + 1)
    ]
    for t in units:
        assert t.source == "test"
        assert t.rul_at_end == rul_values(SYNTH_UNITS[t.subset][1])[t.unit - 1]
        assert t.failure_cycle == t.n + t.rul_at_end
        assert t.rul()[-1] == t.rul_at_end


def test_every_read_writes_a_ledger_line_before_returning(setup: Setup) -> None:
    setup.freeze()
    assert setup.ledger.heldout_evaluations() == 0
    eval_units = setup.access.eval_units(ALL)
    test_units = setup.access.test_units(("FD001",))
    first, second = setup.uses()
    assert (first["what"], first["n"], first["subsets"], first["units"]) == (
        "eval",
        1,
        ["FD001", "FD002"],
        len(eval_units),
    )
    assert (second["what"], second["n"], second["subsets"], second["units"]) == (
        "test",
        1,
        ["FD001"],
        len(test_units),
    )
    assert first["prereg_sha256"] == second["prereg_sha256"] == prereg_sha256(setup.prereg)
    assert (first["heldout_evaluations"], second["heldout_evaluations"]) == (1, 2)
    assert setup.ledger.heldout_evaluations() == 2


def test_a_repeated_read_counts_again(setup: Setup) -> None:
    setup.freeze()
    setup.access.eval_units(("FD001",))
    setup.access.eval_units(("FD001",))
    assert setup.ledger.heldout_evaluations() == 2


def test_the_ledger_line_is_written_even_if_the_read_then_fails(
    setup: Setup, synth: Synthetic
) -> None:
    setup.freeze()
    changed = bytearray(synth.read("train_FD001.txt"))
    changed[40] ^= 0x01
    synth.write("train_FD001.txt", bytes(changed), rehash=False)
    with pytest.raises(DataError, match=r"train_FD001\.txt"):
        setup.access.eval_units(("FD001",))
    assert len(setup.uses()) == 1
    assert setup.ledger.heldout_evaluations() == 1


def test_a_bad_rul_file_fails_after_the_ledger_line(setup: Setup, synth: Synthetic) -> None:
    setup.freeze()
    synth.write("RUL_FD001.txt", b"5\r\n6\r\n")
    with pytest.raises(DataError, match=r"RUL_FD001\.txt"):
        setup.access.test_units(("FD001",))
    assert len(setup.uses()) == 1


def test_an_unknown_subset_is_refused_before_the_ledger_is_touched(setup: Setup) -> None:
    setup.freeze()
    with pytest.raises(ValueError, match="FD009"):
        setup.access.eval_units(("FD009",))
    with pytest.raises(ValueError, match="no subsets"):
        setup.access.test_units(())
    assert setup.uses() == []


# --------------------------------------------------------------------------- a changed document


def test_refuses_when_one_byte_of_the_preregistration_changes(setup: Setup) -> None:
    setup.freeze()
    setup.access.eval_units(("FD001",))
    original = setup.prereg.read_bytes()
    setup.prereg.write_bytes(original.replace(b"one", b"One"))
    assert not setup.access.is_frozen()
    with pytest.raises(HeldoutError, match="held out"):
        setup.access.eval_units(("FD001",))
    with pytest.raises(HeldoutError, match="held out"):
        setup.access.test_units(("FD001",))
    assert len(setup.uses()) == 1  # nothing was recorded for the refused reads


def test_appending_to_the_preregistration_closes_the_gate_and_restoring_it_reopens(
    setup: Setup,
) -> None:
    setup.freeze()
    original = setup.prereg.read_bytes()
    setup.prereg.write_bytes(original + b"One more line.\n")
    with pytest.raises(HeldoutError):
        setup.access.test_units(("FD001",))
    setup.prereg.write_bytes(original)
    assert setup.access.test_units(("FD001",))


def test_a_second_freeze_does_not_reopen_an_edited_document(setup: Setup) -> None:
    # Review of 10 November, finding 2: only the first freeze counts.
    setup.freeze()
    assert setup.access.eval_units(("FD001",))
    setup.prereg.write_bytes(b"# Pre-registration, version two\n")
    setup.freeze()
    with pytest.raises(HeldoutError, match="first freeze"):
        setup.access.eval_units(("FD001",))
    assert not setup.access.is_frozen()
    assert len(setup.uses()) == 1


def test_a_first_freeze_for_another_document_keeps_it_shut_after_a_matching_one(
    setup: Setup,
) -> None:
    setup.ledger.append("freeze", prereg_sha256="0" * 64)
    setup.freeze()
    assert not setup.access.is_frozen()
    with pytest.raises(HeldoutError):
        setup.access.test_units(ALL)


# ------------------------------------------- real manifest in use: bound to the repository's files


def test_with_the_real_manifest_a_scratch_ledger_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review of 10 November, finding 1. No synthetic manifest here, so the package's own manifest is
    # in use and the files would be the real ones. The refusal comes before any file is looked for.
    monkeypatch.setenv("CLOSED_LOOP_V2_DATA", str(tmp_path / "no-such-directory"))
    prereg = tmp_path / "PREREGISTRATION.md"
    prereg.write_bytes(b"# Pre-registration\n")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("freeze", prereg_sha256=prereg_sha256(prereg))
    access = HeldoutAccess(ledger, prereg)
    for read in (access.eval_units, access.test_units):
        with pytest.raises(HeldoutError, match="repository's own ledger"):
            read(ALL)
    with pytest.raises(HeldoutError, match="repository's own ledger"):
        access.is_frozen()
    assert [e["kind"] for e in ledger.entries()] == ["freeze"]


def _bound_to(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Ledger, Path]:
    """Point the repository's ledger and pre-registration at temporary files (the opt-in)."""
    prereg = tmp_path / "PREREGISTRATION.md"
    prereg.write_bytes(b"# Pre-registration\n")
    monkeypatch.setattr(audit, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(audit, "PREREG", prereg)
    monkeypatch.setenv("CLOSED_LOOP_V2_DATA", str(tmp_path / "no-such-directory"))
    return Ledger(tmp_path / "ledger.jsonl"), prereg


def test_with_the_real_manifest_the_repository_files_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, prereg = _bound_to(tmp_path, monkeypatch)
    access = HeldoutAccess(ledger, prereg)
    assert not access.is_frozen()
    ledger.append("freeze", prereg_sha256=prereg_sha256(prereg))
    assert access.is_frozen()  # no data is read: is_frozen only checks the ledger


def test_a_manifest_changed_since_the_freeze_keeps_the_gate_shut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, prereg = _bound_to(tmp_path, monkeypatch)
    ledger.append("freeze", prereg_sha256=prereg_sha256(prereg), manifest_sha256="0" * 64)
    access = HeldoutAccess(ledger, prereg)
    with pytest.raises(HeldoutError, match="changed since the freeze"):
        access.eval_units(ALL)
    assert [e["kind"] for e in ledger.entries()] == ["freeze"]
