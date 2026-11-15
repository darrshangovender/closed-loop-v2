"""Paths and helpers for the audit trail, built on closed-loop v0.1's ``closed_loop.audit``.

v0.1 supplies the hash-chained ``Ledger``, the pre-registration and code hashes, the heavy-run
lock and the commit-charge guard. This module fixes where v0.2 keeps its own records and which
files its code hash covers (docs/adr/0001-design.md, section 12).

The shared heavy-run lock is a file whose path comes from the environment variable
``CLOSED_LOOP_V2_LOCK``. With it unset there is no shared lock, and the commit-charge guard still
applies. ``REPO`` is found from this file, so the package needs a source checkout (an editable
install).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from closed_loop.audit.guard import (
    GuardError,
    HeavyRunLock,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.prereg import code_sha256

__all__ = [
    "CACHE",
    "LEDGER",
    "LOCK_ENV",
    "PREREG",
    "REPO",
    "RESULTS",
    "TUNED",
    "GuardError",
    "code_sha256_v2",
    "heavy_lock_path",
    "heavy_run",
]

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "PREREGISTRATION.md"
TUNED = REPO / "benchmarks" / "tuned.json"
LEDGER = REPO / "benchmarks" / "ledger.jsonl"
RESULTS = REPO / "benchmarks" / "results.json"
CACHE = REPO / "benchmarks" / ".cache"
LOCK_ENV = "CLOSED_LOOP_V2_LOCK"


def code_sha256_v2() -> str:
    """The code hash the ledger records: all of ``closed_loop_v2/`` plus ``benchmarks/run.py``.

    ``FileNotFoundError`` if ``benchmarks/run.py`` does not exist yet.
    """
    return code_sha256([REPO / "closed_loop_v2", REPO / "benchmarks" / "run.py"])


def heavy_lock_path() -> Path | None:
    """The shared heavy-run lock file named by ``CLOSED_LOOP_V2_LOCK``, or ``None`` if unset."""
    value = os.environ.get(LOCK_ENV, "").strip()
    return Path(value) if value else None


def heavy_run(
    owner: str,
    task: str,
    *,
    reader: Callable[[], float | None] = commit_charge_percent,
) -> AbstractContextManager[object]:
    """Context for a heavy run: the commit-charge guard, then the shared lock if one is set.

    Raises ``GuardError`` (before taking any lock) when the commit charge is at or above the 88%
    limit, and again when the lock file is held or stale. The lock file is removed on exit if this
    call created it. ``reader`` is the commit-charge source; tests replace it.
    """
    require_headroom(reader)
    path = heavy_lock_path()
    if path is None:
        return nullcontext()
    return HeavyRunLock(path, owner, task)
