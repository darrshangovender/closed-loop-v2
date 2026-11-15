"""Entry point for closed-loop v0.2: quick run, freeze, full run and check (0002, section 4).

C-MAPSS is simulated by NASA's C-MAPSS engine model; it is not field data.

    python benchmarks/run.py            the full run (same as --full)
    python benchmarks/run.py --quick    dev units only, re-split; no ledger, no held-out data
    python benchmarks/run.py --freeze   record the pre-registration, tuned.json, code and manifest
    python benchmarks/run.py --check    verify the ledger, the hashes and the published documents

``--quick`` re-splits the dev units by ``sha256("closed-loop-v2/quick/{subset}/{unit}")`` into fit
(< 0.5), calibration (< 0.75) and evaluation, runs the whole pipeline (both conditions, every
method, both loops, every verdict), and writes ``benchmarks/.cache/quick.json`` and
``benchmarks/.cache/quick.md``. It never opens the ledger or a held-out file. Its external check
stands in for NASA's test units with the quick evaluation units cut at a hashed row
(``sha256("closed-loop-v2/quick-cut/{subset}/{unit}")``), so that the code path runs; that is not
NASA's truncation. Seeds use the key ``"quick"``.

``--freeze`` refuses when the ledger has any entry (one freeze per ledger, and ``HeldoutAccess``
counts only the first freeze), when ``docs/PREREGISTRATION.md`` or a complete
``benchmarks/tuned.json`` is missing, when the pre-registration still carries ``[T]``, ``[R]``,
``DRAFT`` or ``Placeholders:``, or when it does not contain, verbatim, the registered-values table
that ``registered_table`` generates from ``DESIGN`` and ``tuned.json`` (``--registered-table``, a
hidden flag, prints that table and reads no data and no ledger; a refusal prints it too). It
appends ``freeze`` with ``prereg_sha256``, ``tuned_sha256``, ``code_sha256``,
``manifest_sha256`` and the library versions (v0.1's version and the hash of its code among them,
since v0.1 supplies the t guard, Clopper-Pearson, the ledger and the guard). The manifest hash is
``file_sha256`` of the package's own ``data/MANIFEST.json``
(``closed_loop_v2.data.paths.MANIFEST_PATH``), the file the loader verifies the data against and
the value ``HeldoutAccess`` compares, whatever ``--root`` says.

``--full`` refuses unless the pre-registration, ``tuned.json`` and manifest hashes equal the freeze
entry's (a changed code hash is a warning) and the document still holds the registered-values table
the code gives now (a registered constant changed in the code, or in the installed v0.1, refuses
the run before any held-out read), and stops as "not run" (exit 4, no ledger line) if the
README's result markers are broken. Under the commit-charge guard and the heavy-run lock it fits
the estimator on dev units (nominal) and on FD001 and FD002 dev units (shifted), calibrates on
calib units, then reads the eval units and NASA's test units through ``HeldoutAccess`` (each read
writes a ledger line), replays both conditions with ``R`` orders in each loop, computes metrics,
intervals and verdicts, writes the results text to ``benchmarks/.cache/full-unpublished.json``,
then writes ``benchmarks/results.json``, ``benchmarks/RESULTS.md``, the README block between its
markers and an archive copy ``benchmarks/runs/results-run-{n}.json`` (``n`` counts the ``full``
ledger entries, this run included; an existing file is not replaced, the copy takes an ``-again``
suffix instead), and appends ``full`` with the results' sha256, the run number and the archive path.
A ``DataError``, ``TunedError`` or ``OSError`` before the first held-out read is "not run" (exit 4,
no ledger line). Any other exception after the pre-flight, and any exception once held-out data has
been read, appends ``full-failed`` with the error (and whether held-out data was read), then
re-raises; a commit-charge stop between stages is recorded the same way and exits 3.

``--check`` needs no data: it verifies the chain, compares the hashes with the freeze (a changed
code hash is reported, not a failure, as in v0.1), fails if the document no longer holds the
registered-values table the code gives now, checks the results' sha256 against the last
``full`` entry, regenerates ``RESULTS.md`` and the README block from ``results.json`` and compares
them byte for byte, runs the absolutes lint on them, and rebuilds every verdict from its stored
intervals.

The heavy-run lock comes from ``--lock`` or ``CLOSED_LOOP_V2_LOCK`` (``closed_loop_v2.audit``); with
neither there is no lock, and the 88% commit-charge guard still applies. ``--root`` (hidden) points
at another repository root, for tests; the data directory comes from ``CLOSED_LOOP_V2_DATA``. With
the real manifest in use, ``HeldoutAccess`` opens only against the repository's own ledger and
pre-registration, so a full run under another root works only on a synthetic dataset (tests swap
``closed_loop_v2.data.paths.manifest``).

Exit codes: 0 done; 1 a check failed or a run was refused (broken ledger, no freeze, changed
pre-registration, results that do not match their record); 2 usage; 3 refused or paused (commit
charge, lock held); 4 not run (an operational error, such as missing data).

**Design constants.** ``DESIGN`` holds the numbers the pre-registration fixes (``k``, ``R``, the
VW step count and the rest). They are in the code hash and every ``results.json`` records them.
``tuned.json`` holds the tuned values: the estimator's parameters, ``h`` and the decision
stratum ``M`` (``stratum_m``), plus free-text provenance and a ``record`` object (the grids, kept as
data and not read). A missing value falls back to its default (``DESIGN.stratum_m`` for ``M``), and
``results.json`` records where each value came from; ``--freeze`` needs them all. A run uses
``dataclasses.replace(DESIGN, stratum_m=tuned.stratum_m)`` (``run_design``), so the ``M`` the run
uses is the tuned one (review of packet C, finding B1).

**``results.json`` schema** (``closed-loop-v2/results/1``), every number at full precision:
``mode``; ``data`` (the simulated-data note); ``hashes``; ``seeds`` (the key kind and the labels,
not the values); ``design`` (with the ``M`` the run used); ``tuned`` (each value with its source,
``stratum_m`` included, and free-text provenance); ``units`` (per condition and role, totals and per
subset, and the external set); ``conditions.<nominal|shifted>`` with ``eval_units`` (unit keys, the
order of every per-unit list), ``estimator`` (kept and dropped sensors), ``stratum_rows``,
``methods`` (V1 and VW quantiles; VW's step history, whether its last step moved, how the steps
ended (``state``: settled, two-cycle or unsettled) and, for a two-cycle, the other pair with its
closed-loop ``c^lo`` interval, policy outcomes and per-unit values; V2L and V3L starting quantiles
and final ``alpha_t`` per order and loop), ``cells.<method>.<loop>`` (intervals
for ``c``, ``c^lo`` and their stratum versions with the rule used, stratum counts and exclusions,
policy outcomes, per-unit values), ``gap_b.<method>`` (pooled and stratum intervals, units kept and
dropped) and ``by_subset.<subset>`` (per method and loop: intervals at ``SECONDARY_ALPHA`` for
``c``, ``c^lo`` and stratum ``c^lo`` with unit counts; per method: ``Gap_B`` pooled and stratum,
paired, with unit counts); ``external`` (``k``, ``n``, per subset, the Clopper-Pearson interval);
``claims`` (claim, row, condition, primary or not, level, label or ``null``, the intervals the rule
reads); and, for a full run only, ``runs`` (this run's number and archive path, and the earlier
``full`` entries of the ledger). Infinite numbers are the strings ``"inf"`` and ``"-inf"``; a
missing number is ``null``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closed_loop.audit.guard import (  # noqa: E402
    COMMIT_LIMIT_PCT,
    GuardError,
    HeavyRunLock,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.ledger import Ledger, LedgerError  # noqa: E402
from closed_loop.audit.prereg import code_sha256, file_sha256, prereg_sha256  # noqa: E402
from closed_loop.stats.confseq import SPREAD_KURTOSIS_DIVISOR, SPREAD_MIN_PAIRS  # noqa: E402

from closed_loop_v2 import report  # noqa: E402
from closed_loop_v2.audit import heavy_run  # noqa: E402
from closed_loop_v2.conformal import (  # noqa: E402
    CalibPairs,
    SplitConformal,
    bounds,
    cap_of,
    fit_v1,
    fit_vw,
    make_v2l,
    make_v3l,
)
from closed_loop_v2.data import (  # noqa: E402
    DataError,
    HeldoutAccess,
    HeldoutError,
    development_units,
)
from closed_loop_v2.data.paths import MANIFEST_PATH  # noqa: E402
from closed_loop_v2.estimator import (  # noqa: E402
    DEFAULT_PARAMS,
    Estimator,
    EstimatorParams,
    Prediction,
)
from closed_loop_v2.metrics import (  # noqa: E402
    LifeMetrics,
    UnitTable,
    gap_b,
    life_metrics,
    policy_outcomes,
)
from closed_loop_v2.policy import BoundRule  # noqa: E402
from closed_loop_v2.replay import orders, replay, replay_static  # noqa: E402
from closed_loop_v2.seeds import QUICK_KEY, SEED_LABELS, run_seed  # noqa: E402
from closed_loop_v2.stats import (  # noqa: E402
    BET_TRUNCATION,
    ORDER_KEY,
    Interval,
    binomial_interval,
    mean_interval,
    paired_interval,
)
from closed_loop_v2.types import N_SENSORS, SUBSETS, Trajectory  # noqa: E402
from closed_loop_v2.verdicts import (  # noqa: E402
    EXT_HI,
    EXT_LO,
    PERC_BAND,
    PRIMARY_ALPHA,
    SECONDARY_ALPHA,
    SELECT_FLOOR,
    SELECT_MARGIN,
    VW_TWO_CYCLE,
    ext_verdict,
    perc_verdict,
    select_claim_verdict,
    verdict_of_claim,
    vw_state,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED, EXIT_NOT_RUN = 0, 1, 2, 3, 4
OWNER = "closed-loop-v2"
SHIFT_FIT: tuple[str, ...] = ("FD001", "FD002")  # single fault mode: fit and calibrate here
SHIFT_EVAL: tuple[str, ...] = ("FD003", "FD004")  # two fault modes: evaluate here
QUICK_FIT_BELOW = 0.5
QUICK_CALIB_BELOW = 0.75
METHODS: tuple[str, ...] = ("V1", "VW", "V2L", "V3L")
# the other pair of a two-cycling VW, replayed in the closed loop only for C2-SELECT
VW_ALT = "VW-alt"
CONDITIONS: tuple[str, ...] = ("nominal", "shifted")
PROVENANCE_KEYS: tuple[str, ...] = ("tuned_on", "criterion", "script", "notes")
RECORD_KEY = "record"  # tuned.json's grids (ADR 0003 section 6): validated as an object, not read
# --freeze refuses a pre-registration that still carries any of these (review of packet C, S4)
PREREG_PLACEHOLDERS: tuple[str, ...] = ("[T]", "[R]", "DRAFT", "Placeholders:")


@dataclass(frozen=True)
class Design:
    """The numbers the pre-registration fixes; ``--registered-table`` prints them for it.

    ``stratum_m`` here is only the fallback: a run takes ``M`` from ``tuned.json`` (``run_design``).
    """

    k: int = 20  # engines running at once
    r: int = 20  # registered random orders per loop
    stratum_m: float = 50.0  # decision stratum (rows with m <= M cycles): fallback for tuned.json
    alpha: float = 0.10  # two-sided level; 0.05 on each side
    gamma: float = 0.02  # ACI step
    aci_window: int = 500  # ACI rolling pool
    vw_steps: int = 10  # VW fixed-point steps from V1's quantiles
    n_folds: int = 5  # unit-grouped cross-fitting folds of the scale model


DESIGN = Design()
DEFAULT_H = 20.0  # cycles; provisional until tuned on dev


# ----------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class Paths:
    """Where everything lives, relative to a repository root."""

    root: Path

    @property
    def prereg(self) -> Path:
        return self.root / "docs" / "PREREGISTRATION.md"

    @property
    def tuned(self) -> Path:
        return self.root / "benchmarks" / "tuned.json"

    @property
    def ledger(self) -> Path:
        return self.root / "benchmarks" / "ledger.jsonl"

    @property
    def results_json(self) -> Path:
        return self.root / "benchmarks" / "results.json"

    @property
    def results_md(self) -> Path:
        return self.root / "benchmarks" / "RESULTS.md"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def cache(self) -> Path:
        return self.root / "benchmarks" / ".cache"

    @property
    def unpublished(self) -> Path:
        return self.cache / "full-unpublished.json"

    @property
    def runs(self) -> Path:
        return self.root / "benchmarks" / "runs"

    @property
    def code_roots(self) -> tuple[Path, Path]:
        return (self.root / "closed_loop_v2", self.root / "benchmarks" / "run.py")


def current_hashes(paths: Paths) -> dict[str, str | None]:
    """The four hashes a freeze records, as the files are now (``None`` for a missing file).

    The manifest is the package's own ``data/MANIFEST.json``, not one under ``paths.root``.
    """
    return {
        "prereg_sha256": prereg_sha256(paths.prereg) if paths.prereg.is_file() else None,
        "tuned_sha256": file_sha256(paths.tuned) if paths.tuned.is_file() else None,
        "code_sha256": code_sha256(list(paths.code_roots)),
        "manifest_sha256": file_sha256(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None,
    }


# ----------------------------------------------------------------------------- tuned values


class TunedError(ValueError):
    """``tuned.json`` does not parse or holds an unknown or invalid value."""


@dataclass(frozen=True)
class Tuned:
    """The tuned values a run uses, where each came from, and free-text provenance.

    ``sources`` maps every estimator field, ``h`` and ``stratum_m`` to ``"tuned.json"`` or
    ``"default"``.
    """

    params: EstimatorParams
    h: float
    stratum_m: float
    sources: dict[str, str]
    provenance: dict[str, str]
    from_file: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "tuned.json" if self.from_file else "defaults",
            "h": {"value": self.h, "from": self.sources["h"]},
            "stratum_m": {"value": self.stratum_m, "from": self.sources["stratum_m"]},
            "estimator": {
                f.name: {"value": getattr(self.params, f.name), "from": self.sources[f.name]}
                for f in fields(EstimatorParams)
            },
            "provenance": dict(self.provenance),
        }


_INT_FIELDS = {f.name for f in fields(EstimatorParams) if f.type in ("int", int)}


def _tuned_number(
    path: Path, data: Mapping[str, Any], key: str, default: float
) -> tuple[float, str]:
    """``data[key]`` as a finite positive float with source ``"tuned.json"``, else the default."""
    if key not in data:
        return default, "default"
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise TunedError(f"{path.name}: {key} must be a finite number")
    if key == "stratum_m" and v <= 0:
        raise TunedError(f"{path.name}: stratum_m must be positive")
    return float(v), "tuned.json"


def load_tuned(path: Path) -> Tuned:
    """Read ``tuned.json`` if it exists; every value it lacks takes its default.

    The keys it accepts are ``estimator``, ``h``, ``stratum_m``, the provenance texts and
    ``record``, the exact shape ``scripts/tune.py`` writes (ADR 0003 section 6). ``record`` must be
    a JSON object and is stored nowhere: the file's hash in the freeze covers it. Any other key is
    refused.
    """
    names = [f.name for f in fields(EstimatorParams)]
    if not path.is_file():
        return Tuned(
            DEFAULT_PARAMS,
            DEFAULT_H,
            DESIGN.stratum_m,
            dict.fromkeys([*names, "h", "stratum_m"], "default"),
            {},
            False,
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise TunedError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TunedError(f"{path.name} must hold a JSON object")
    unknown = sorted(set(data) - {"estimator", "h", "stratum_m", RECORD_KEY, *PROVENANCE_KEYS})
    if unknown:
        raise TunedError(f"{path.name}: unknown keys {unknown}")
    if RECORD_KEY in data and not isinstance(data[RECORD_KEY], dict):
        raise TunedError(f"{path.name}: '{RECORD_KEY}' must be an object")
    est = data.get("estimator", {})
    if not isinstance(est, dict):
        raise TunedError(f"{path.name}: 'estimator' must be an object")
    bad = sorted(set(est) - set(names))
    if bad:
        raise TunedError(f"{path.name}: unknown estimator keys {bad}")
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in names:
        if name in est:
            v = est[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TunedError(f"{path.name}: estimator {name} must be a number")
            if name in _INT_FIELDS and not (isinstance(v, int) or float(v).is_integer()):
                raise TunedError(f"{path.name}: estimator {name} must be a whole number")
            values[name] = int(v) if name in _INT_FIELDS else float(v)
            sources[name] = "tuned.json"
        else:
            values[name] = getattr(DEFAULT_PARAMS, name)
            sources[name] = "default"
    try:
        params = EstimatorParams(**values)
    except ValueError as exc:
        raise TunedError(f"{path.name}: {exc}") from exc
    h_value, sources["h"] = _tuned_number(path, data, "h", DEFAULT_H)
    m_value, sources["stratum_m"] = _tuned_number(path, data, "stratum_m", DESIGN.stratum_m)
    provenance = {}
    for key in PROVENANCE_KEYS:
        if key in data:
            if not isinstance(data[key], str):
                raise TunedError(f"{path.name}: {key} must be text")
            provenance[key] = data[key]
    return Tuned(params, h_value, m_value, sources, provenance, True)


def tuned_problems(path: Path) -> list[str]:
    """Why ``tuned.json`` cannot be frozen: it must exist, validate and give every value.

    Every value means every estimator field, ``h`` and ``stratum_m``.
    """
    if not path.is_file():
        return [f"{path.name} does not exist: tune on dev units first"]
    try:
        tuned = load_tuned(path)
    except TunedError as exc:
        return [str(exc)]
    missing = sorted(k for k, v in tuned.sources.items() if v != "tuned.json")
    return [f"{path.name} does not give {', '.join(missing)}"] if missing else []


def run_design(tuned: Tuned) -> Design:
    """The design a run uses: ``DESIGN`` with the tuned ``M`` (``tuned.json``, or its fallback)."""
    return replace(DESIGN, stratum_m=tuned.stratum_m)


# ----------------------------------------------------------------------------- registered values


def _exact(value: object) -> str:
    """A number as the registered table prints it: ``repr``, the shortest exact round trip."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a number: {value!r}")
    return repr(value)


def registered_table(design: Design, tuned: Tuned) -> str:
    """The registered-values table the pre-registration must contain verbatim (no final newline).

    One row per value: the design constants, ``M`` and ``h`` from ``tuned``, every estimator
    parameter, the verdict constants and levels, and the interval rule's constants (the t guard's
    two from v0.1, the betting truncation and the canonical order's key, ADR 0004). Numbers are
    printed with ``repr``, so nothing is rounded. ``design.stratum_m`` is not printed: ``M`` is
    the tuned value.
    """
    in_run = "`DESIGN` in `benchmarks/run.py`"
    in_verdicts = "`closed_loop_v2/verdicts.py`"
    in_stats = "`closed_loop_v2/stats.py`"
    in_v01 = "`closed_loop.stats.confseq` (v0.1)"

    def tuned_from(name: str) -> str:
        return "`benchmarks/tuned.json`" if tuned.sources[name] == "tuned.json" else "default"

    rows: list[tuple[str, object, str]] = [
        ("k", design.k, in_run),
        ("r", design.r, in_run),
        ("stratum_m", tuned.stratum_m, tuned_from("stratum_m")),
        ("alpha", design.alpha, in_run),
        ("gamma", design.gamma, in_run),
        ("aci_window", design.aci_window, in_run),
        ("vw_steps", design.vw_steps, in_run),
        ("n_folds", design.n_folds, in_run),
        ("h", tuned.h, tuned_from("h")),
    ]
    rows += [
        (f"estimator.{f.name}", getattr(tuned.params, f.name), tuned_from(f.name))
        for f in fields(EstimatorParams)
    ]
    rows += [
        ("PERC_BAND", PERC_BAND, in_verdicts),
        ("SELECT_MARGIN", SELECT_MARGIN, in_verdicts),
        ("SELECT_FLOOR", SELECT_FLOOR, in_verdicts),
        ("EXT_LO", EXT_LO, in_verdicts),
        ("EXT_HI", EXT_HI, in_verdicts),
        ("PRIMARY_ALPHA", PRIMARY_ALPHA, in_verdicts),
        ("SECONDARY_ALPHA", SECONDARY_ALPHA, in_verdicts),
        ("SPREAD_MIN_PAIRS", SPREAD_MIN_PAIRS, in_v01),
        ("SPREAD_KURTOSIS_DIVISOR", SPREAD_KURTOSIS_DIVISOR, in_v01),
        ("BET_TRUNCATION", BET_TRUNCATION, in_stats),
        ("ORDER_KEY.0", ORDER_KEY[0], in_stats),
        ("ORDER_KEY.1", ORDER_KEY[1], in_stats),
    ]
    lines = ["| Value | Registered | From |", "|---|---|---|"]
    lines += [f"| `{name}` | {_exact(value)} | {source} |" for name, value, source in rows]
    return "\n".join(lines)


def _normalised(text: str) -> str:
    """LF line ends and no trailing spaces, as ``prereg_sha256`` normalises before hashing."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))


def has_table(text: str, table: str) -> bool:
    """True if ``text`` contains ``table`` verbatim, after the hash's line normalisation."""
    return _normalised(table) in _normalised(text)


def registered_table_problem(paths: Paths) -> str | None:
    """Why the pre-registration does not hold the table the code would register now, or None.

    Run at ``--full`` before any held-out read and at ``--check`` (review D, S3): after the freeze
    the document and ``tuned.json`` are fixed by their hashes, so a mismatch here means that a
    registered value in the code (``DESIGN``, ``verdicts.py``, ``stats.py``) or in the installed
    v0.1 package (the t guard) has changed since the freeze.
    """
    problems = tuned_problems(paths.tuned)
    if problems:
        return f"tuned.json is not usable ({'; '.join(problems)})"
    if not paths.prereg.is_file():
        return f"{paths.prereg.name} does not exist"
    table = registered_table(DESIGN, load_tuned(paths.tuned))
    if not has_table(paths.prereg.read_text(encoding="utf-8"), table):
        return (
            f"{paths.prereg.name} does not hold the registered-values table the code gives now "
            "(run --registered-table to see it)"
        )
    return None


def prereg_problems(text: str, table: str | None) -> list[str]:
    """Why the pre-registration ``text`` cannot be frozen against the registered ``table``.

    With ``table`` ``None`` (``tuned.json`` is not usable) only the placeholders are checked.
    """
    body = _normalised(text)
    problems = [f"it still contains {p!r}" for p in PREREG_PLACEHOLDERS if p in body]
    if table is not None and not has_table(text, table):
        problems.append(
            "it does not contain the registered-values table verbatim "
            "(the table follows; paste it into the document)"
        )
    return problems


# ----------------------------------------------------------------------------- units


def _draw(text: str) -> float:
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def quick_role(subset: str, unit: int) -> str:
    """``fit``, ``calib`` or ``eval`` for a dev unit in ``--quick``, from its subset and number."""
    u = _draw(f"closed-loop-v2/quick/{subset}/{unit}")
    return "fit" if u < QUICK_FIT_BELOW else "calib" if u < QUICK_CALIB_BELOW else "eval"


def quick_cut(traj: Trajectory) -> Trajectory:
    """A quick evaluation unit cut at a hashed row, standing in for a NASA test unit.

    It keeps rows ``1..c`` with ``c = 1 + floor(u n)`` (at most ``n``), and its remaining life after
    the last kept row is ``n - c``.
    """
    u = _draw(f"closed-loop-v2/quick-cut/{traj.subset}/{traj.unit}")
    c = min(traj.n, 1 + int(u * traj.n))
    return Trajectory(
        traj.subset,
        traj.unit,
        "test",
        traj.cycles[:c].copy(),
        traj.settings[:c].copy(),
        traj.sensors[:c].copy(),
        traj.failure_cycle - c,
    )


@dataclass(frozen=True)
class DevSets:
    """What a run may read before any held-out data: fit and calibration units per condition."""

    fit: dict[str, list[Trajectory]]
    calib: dict[str, list[Trajectory]]


@dataclass(frozen=True)
class Heldout:
    """The evaluation units (all subsets) and the external units, with where they came from."""

    eval_units: list[Trajectory]
    external: list[Trajectory]
    external_source: str


def truth_cap(tuned: Tuned) -> int:
    """The cap on the truth: the estimator's ``cap``, a whole number of cycles (ADR 0001,
    amendment of 10 November 2026). Every trajectory a run reads is capped at it."""
    cap = float(tuned.params.cap)
    if not cap.is_integer():
        raise TunedError(f"estimator cap must be a whole number of cycles, got {cap!r}")
    return int(cap)


def _in(trajs: Sequence[Trajectory], subsets: Sequence[str]) -> list[Trajectory]:
    return [t for t in trajs if t.subset in subsets]


def quick_sets() -> tuple[DevSets, Callable[[], Heldout]]:
    """``--quick``: dev units only, re-split by the quick hash."""
    dev = development_units("dev")
    split: dict[str, list[Trajectory]] = {"fit": [], "calib": [], "eval": []}
    for t in dev:
        split[quick_role(t.subset, t.unit)].append(t)
    sets = DevSets(
        fit={"nominal": split["fit"], "shifted": _in(split["fit"], SHIFT_FIT)},
        calib={"nominal": split["calib"], "shifted": _in(split["calib"], SHIFT_FIT)},
    )

    def heldout() -> Heldout:
        return Heldout(
            split["eval"],
            [quick_cut(t) for t in split["eval"]],
            "quick evaluation units cut at a hashed row",
        )

    return sets, heldout


def full_sets(
    access: HeldoutAccess, on_read: Callable[[], None]
) -> tuple[DevSets, Callable[[], Heldout]]:
    """``--full``: dev and calib units now; eval and NASA's test units only when asked for."""
    dev = development_units("dev")
    calib = development_units("calib")
    sets = DevSets(
        fit={"nominal": dev, "shifted": _in(dev, SHIFT_FIT)},
        calib={"nominal": calib, "shifted": _in(calib, SHIFT_FIT)},
    )

    def heldout() -> Heldout:
        on_read()  # before the read: the ledger line is written before any row is parsed
        eval_units = access.eval_units()
        test_units = access.test_units()
        return Heldout(eval_units, test_units, "NASA test units at their last observed cycle")

    return sets, heldout


def _counts(trajs: Sequence[Trajectory]) -> dict[str, Any]:
    by = {s: sum(t.subset == s for t in trajs) for s in SUBSETS}
    return {"total": len(trajs), "by_subset": {s: n for s, n in by.items() if n}}


def unit_key(t: Trajectory) -> str:
    return f"{t.subset}/{t.unit}"


# ----------------------------------------------------------------------------- the pipeline


Log = Callable[[str], None]


@dataclass
class Fitted:
    """One condition after fitting and calibration, before any evaluation unit is read."""

    estimator: Estimator
    calib: CalibPairs
    v1: SplitConformal
    vw: SplitConformal


def require_cap(units: Sequence[Trajectory], cap: int, what: str) -> None:
    """Refuse trajectories whose truth is not capped at ``cap`` (review D, N5).

    The estimator caps its own targets, so it accepts uncapped trajectories; calibration, VW, the
    replay and the external check take the truth from the trajectories, so an uncapped one there
    would silently score the uncapped remaining life.
    """
    wrong = sorted({str(t.rul_cap) for t in units if t.rul_cap != cap})
    if wrong:
        raise ValueError(f"{what}: trajectories capped at {', '.join(wrong)}, not at {cap}")


def fit_condition(
    cond: str, sets: DevSets, tuned: Tuned, design: Design, key: str, log: Log
) -> Fitted:
    t0 = time.perf_counter()
    fit_units, calib_units = sets.fit[cond], sets.calib[cond]
    require_cap(calib_units, truth_cap(tuned), f"{cond} calibration")
    est = Estimator.fit(
        fit_units, tuned.params, seed=run_seed(key, f"fit-{cond}"), n_folds=design.n_folds
    )
    calib = [(t, est.predict(t)) for t in calib_units]
    calib_seed = run_seed(key, "calib-draw")
    v1 = fit_v1(calib, alpha=design.alpha, seed=calib_seed)
    # the same seed: VW's step 0 is V1, with the same u_j draws
    vw = fit_vw(
        calib, BoundRule(tuned.h), alpha=design.alpha, seed=calib_seed, steps=design.vw_steps
    )
    log(
        f"{cond}: fitted on {len(fit_units)} units, calibrated on {len(calib_units)} "
        f"({time.perf_counter() - t0:.1f} s)"
    )
    return Fitted(est, calib, v1, vw)


def _iv(x: Interval) -> dict[str, Any]:
    return x.to_dict()


def _list(a: NDArray[np.float64]) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in a]


def _qpair(q: Any) -> list[float]:
    return [float(q.q_lo), float(q.q_hi)]


def evaluate_condition(
    cond: str,
    fitted: Fitted,
    eval_units: Sequence[Trajectory],
    tuned: Tuned,
    design: Design,
    key: str,
    log: Log,
) -> tuple[dict[str, Any], UnitTable]:
    """Replay every method in both loops on ``eval_units`` and summarise one condition."""
    t0 = time.perf_counter()
    require_cap(eval_units, truth_cap(tuned), f"{cond} evaluation")
    lives = [(t, fitted.estimator.predict(t)) for t in eval_units]
    rule = BoundRule(tuned.h)
    loops: tuple[tuple[str, BoundRule | None], ...] = (("open", None), ("closed", rule))
    perms = orders(len(lives), design.r, run_seed(key, "orders"))
    calib_seed = run_seed(key, "calib-draw")
    table = UnitTable([unit_key(t) for t in eval_units])
    runs: dict[tuple[str, str], list[LifeMetrics]] = {}
    finals: dict[str, dict[str, list[float]]] = {"V2L": {}, "V3L": {}}
    for name, method in (("V1", fitted.v1), ("VW", fitted.vw)):
        for loop, rl in loops:
            run = replay_static(lives, method.quantiles(), rl)
            runs[(name, loop)] = [life_metrics(lives, run, stratum_m=design.stratum_m)]
    vw_end = vw_state(fitted.vw.history)
    # a two-cycle's other pair, closed loop only: C2-SELECT is judged with both (review D, N3)
    vw_alt = fitted.vw.history[-2] if vw_end == VW_TWO_CYCLE else None
    if vw_alt is not None:
        run = replay_static(lives, vw_alt, rule)
        runs[(VW_ALT, "closed")] = [life_metrics(lives, run, stratum_m=design.stratum_m)]
    for name, make in (("V2L", make_v2l), ("V3L", make_v3l)):
        for loop, rl in loops:
            got = []
            ends = []
            for j in range(design.r):
                # a fresh method per order; its pool starts from V1's calibration draw
                aci = make(
                    fitted.calib,
                    seed=calib_seed,
                    alpha=design.alpha,
                    gamma=design.gamma,
                    window=design.aci_window,
                )
                run = replay(lives, aci, rl, k=design.k, order=perms[j])
                got.append(life_metrics(lives, run, stratum_m=design.stratum_m))
                ends.append(aci.alpha_t)
            runs[(name, loop)] = got
            finals[name][loop] = ends
    cells: dict[str, dict[str, Any]] = {}
    for name in METHODS:
        cells[name] = {}
        for loop, _ in loops:
            cov = table.add(name, loop, runs[(name, loop)])
            pol = policy_outcomes(runs[(name, loop)])
            ins = cov.in_stratum
            a = SECONDARY_ALPHA
            cells[name][loop] = {
                "orders": cov.n_orders,
                "coverage": _iv(mean_interval(cov.cover, lo=0.0, hi=1.0, alpha=a)),
                "coverage_lo": _iv(mean_interval(cov.cover_lo, lo=0.0, hi=1.0, alpha=a)),
                "stratum_coverage": _iv(
                    mean_interval(cov.stratum_cover[ins], lo=0.0, hi=1.0, alpha=a)
                ),
                "stratum_coverage_lo": _iv(
                    mean_interval(cov.stratum_cover_lo[ins], lo=0.0, hi=1.0, alpha=a)
                ),
                "stratum_units": int(ins.sum()),
                "stratum_excluded": cov.stratum_excluded,
                "policy": asdict(pol),
                "per_unit": {
                    "cover": _list(cov.cover),
                    "cover_lo": _list(cov.cover_lo),
                    "stratum_cover": _list(cov.stratum_cover),
                    "stratum_cover_lo": _list(cov.stratum_cover_lo),
                },
            }
    gaps: dict[str, Any] = {}
    for name in METHODS:
        g = gap_b(table, name)
        gaps[name] = {
            "pooled": _iv(
                paired_interval(g.pooled_closed, g.pooled_open, lo=0, hi=1, alpha=SECONDARY_ALPHA)
            ),
            "stratum": _iv(
                paired_interval(g.stratum_closed, g.stratum_open, lo=0, hi=1, alpha=SECONDARY_ALPHA)
            ),
            "stratum_units": int(g.stratum_units.size),
            "stratum_dropped": g.stratum_dropped,
        }
    alt_closed = None
    if vw_alt is not None:
        alt = table.add(VW_ALT, "closed", runs[(VW_ALT, "closed")])
        alt_closed = {
            "coverage_lo": _iv(mean_interval(alt.cover_lo, lo=0.0, hi=1.0, alpha=SECONDARY_ALPHA)),
            "stratum_units": int(alt.in_stratum.sum()),
            "policy": asdict(policy_outcomes(runs[(VW_ALT, "closed")])),
            "per_unit": {"cover_lo": _list(alt.cover_lo)},
        }
    by_subset = subset_breakdown(table, [name for name, _ in loops])
    cm = fitted.estimator.condition_model
    vw = fitted.vw
    out = {
        "eval_units": list(table.units),
        "estimator": {
            "seed_label": f"fit-{cond}",
            "kept_sensors": [int(i) + 1 for i in cm.kept],
            "dropped_sensors": [i + 1 for i in range(N_SENSORS) if i not in cm.kept],
        },
        "stratum_rows": int(sum(int((p.m <= design.stratum_m).sum()) for _, p in lives)),
        "rows": int(sum(t.n for t in eval_units)),
        "methods": {
            "V1": {
                "kind": "split",
                "quantiles": _qpair(fitted.v1.quantiles()),
                "order_independent": True,
            },
            "VW": {
                "kind": "split",
                "quantiles": _qpair(vw.quantiles()),
                "history": [_qpair(q) for q in vw.history],
                "moved_last_step": bool(vw.moved_last_step),
                "state": vw_end,
                "alt_quantiles": None if vw_alt is None else _qpair(vw_alt),
                "alt_closed": alt_closed,
                "order_independent": True,
            },
            **{
                name: {
                    "kind": "adaptive",
                    "rule": "naive" if name == "V2L" else "skip",
                    "initial_quantiles": _qpair(fitted.v1.quantiles()),
                    "final_alpha_t": finals[name],
                    "order_independent": False,
                }
                for name in ("V2L", "V3L")
            },
        },
        "cells": cells,
        "gap_b": gaps,
        "by_subset": by_subset,
    }
    log(f"{cond}: replayed {len(lives)} lives ({time.perf_counter() - t0:.1f} s)")
    return out, table


def subset_of_key(key: str) -> str:
    """The subset of a unit key ``"FD00x/unit"``."""
    return key.split("/", 1)[0]


def subset_breakdown(table: UnitTable, loops: Sequence[str]) -> dict[str, Any]:
    """The registered per-subset breakdown (pre-registration sections 6 and 9), descriptive.

    For each subset present in ``table`` (taken from the unit key), each method and each loop:
    intervals at ``SECONDARY_ALPHA`` for the mean of ``c``, ``c^lo`` and stratum ``c^lo``, each with
    its unit count. For each method: ``Gap_B`` pooled and in the stratum, paired by unit, with unit
    counts; the stratum keeps the subset's units that are in the stratum in both loops.
    """
    a = SECONDARY_ALPHA
    subset = np.array([subset_of_key(u) for u in table.units], dtype=object)
    out: dict[str, Any] = {}
    for s in SUBSETS:
        mask = subset == s
        if not mask.any():
            continue
        cells: dict[str, dict[str, Any]] = {}
        gaps: dict[str, Any] = {}
        for name in METHODS:
            cells[name] = {}
            for loop in loops:
                cov = table[(name, loop)]
                ins = cov.in_stratum & mask
                cells[name][loop] = {
                    "coverage": _iv(mean_interval(cov.cover[mask], lo=0.0, hi=1.0, alpha=a)),
                    "coverage_lo": _iv(mean_interval(cov.cover_lo[mask], lo=0.0, hi=1.0, alpha=a)),
                    "stratum_coverage_lo": _iv(
                        mean_interval(cov.stratum_cover_lo[ins], lo=0.0, hi=1.0, alpha=a)
                    ),
                    "units": int(mask.sum()),
                    "stratum_units": int(ins.sum()),
                }
            g = gap_b(table, name)
            keep = subset[g.stratum_units] == s
            gaps[name] = {
                "pooled": _iv(
                    paired_interval(
                        g.pooled_closed[mask], g.pooled_open[mask], lo=0.0, hi=1.0, alpha=a
                    )
                ),
                "stratum": _iv(
                    paired_interval(
                        g.stratum_closed[keep], g.stratum_open[keep], lo=0.0, hi=1.0, alpha=a
                    )
                ),
                "units": int(mask.sum()),
                "stratum_units": int(keep.sum()),
            }
        out[s] = {"units": int(mask.sum()), "cells": cells, "gap_b": gaps}
    return out


def external_check(fitted: Fitted, units: Sequence[Trajectory], source: str) -> dict[str, Any]:
    """V1's lower-bound coverage on the external units at their last observed row."""
    require_cap(units, int(fitted.estimator.params.cap), "external")
    q = fitted.v1.quantiles()
    hits: dict[str, list[int]] = {}
    for t in units:
        p: Prediction = fitted.estimator.predict(t)
        # the lower bound limited to the cap, and the capped truth at the last observed row, as
        # everywhere else: min(rul_at_end, cap)
        lo_last = float(bounds(p, q, cap_of(t))[0][-1])
        hits.setdefault(t.subset, []).append(int(int(t.rul()[-1]) >= lo_last))
    k = sum(sum(v) for v in hits.values())
    n = sum(len(v) for v in hits.values())
    ci = binomial_interval(k, n, PRIMARY_ALPHA)
    return {
        "method": "V1",
        "source": source,
        "k": k,
        "n": n,
        "by_subset": {s: {"k": sum(v), "n": len(v)} for s, v in hits.items()},
        "coverage_lo": _iv(ci),
    }


def _claim(
    name: str,
    row: str,
    cond: str,
    primary: bool,
    verdict: str | None,
    intervals: Mapping[str, Interval],
) -> dict[str, Any]:
    return {
        "claim": name,
        "row": row,
        "condition": cond,
        "primary": primary,
        "alpha": PRIMARY_ALPHA if primary else SECONDARY_ALPHA,
        "verdict": verdict,
        "intervals": {k: _iv(v) for k, v in intervals.items()},
    }


def build_claims(
    nominal: UnitTable,
    shifted: UnitTable,
    external: Mapping[str, Any],
    *,
    vw_end: str,
) -> list[dict[str, Any]]:
    """Every claim row of 0001 section 9: six primary verdicts and the secondary intervals.

    The secondary rows are C2-SELECT's stratum version, C2-CENSOR and C2-SHIFT; none has a label.
    ``vw_end`` is how the nominal VW's fixed-point steps ended (``verdicts.vw_state``). The primary
    C2-SELECT row records it; for a two-cycle it also holds the intervals for the other pair (the
    ``VW_ALT`` cells of ``nominal``) and keeps a label only if both pairs give it; for an unsettled
    VW it is inconclusive (review D, N3).
    """
    claims = []
    for method in ("V1", "V2L"):
        g = gap_b(nominal, method)
        for row, a, b in (
            ("pooled", g.pooled_closed, g.pooled_open),
            ("stratum", g.stratum_closed, g.stratum_open),
        ):
            iv = paired_interval(a, b, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
            claims.append(
                _claim(
                    "C2-PERC", f"{method}, {row}", "nominal", True, perc_verdict(iv), {"gap": iv}
                )
            )
    vw = nominal[("VW", "closed")].cover_lo
    v1 = nominal[("V1", "closed")].cover_lo
    diff = paired_interval(vw, v1, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
    vw_iv = mean_interval(vw, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
    select_rows = [(diff, vw_iv)]
    select_ivs = {"diff": diff, "vw": vw_iv}
    if vw_end == VW_TWO_CYCLE:
        alt = nominal[(VW_ALT, "closed")].cover_lo
        diff_alt = paired_interval(alt, v1, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
        vw_alt = mean_interval(alt, lo=0.0, hi=1.0, alpha=PRIMARY_ALPHA)
        select_rows.append((diff_alt, vw_alt))
        select_ivs |= {"diff_alt": diff_alt, "vw_alt": vw_alt}
    select_row = _claim(
        "C2-SELECT",
        "VW minus V1, closed-loop c^lo",
        "nominal",
        True,
        select_claim_verdict(vw_end, select_rows),
        select_ivs,
    )
    claims.append({**select_row, "vw_state": vw_end})
    ext = Interval.from_dict(external["coverage_lo"])
    claims.append(
        _claim(
            "C2-EXT",
            "V1 lower-bound coverage, external units",
            "nominal",
            True,
            ext_verdict(ext),
            {"coverage_lo": ext},
        )
    )
    # C2-SELECT's stratum version (pre-registration sections 8 and 9): reported, without a label,
    # paired over the units in the stratum for both VW and V1 in the closed loop
    vw_closed, v1_closed = nominal[("VW", "closed")], nominal[("V1", "closed")]
    both = vw_closed.in_stratum & v1_closed.in_stratum
    select_stratum = paired_interval(
        vw_closed.stratum_cover_lo[both],
        v1_closed.stratum_cover_lo[both],
        lo=0.0,
        hi=1.0,
        alpha=SECONDARY_ALPHA,
    )
    claims.append(
        _claim(
            "C2-SELECT",
            "VW minus V1, closed-loop stratum c^lo",
            "nominal",
            False,
            None,
            {"diff": select_stratum},
        )
    )
    g1 = gap_b(nominal, "V1")
    for method in ("V2L", "V3L"):
        gm = gap_b(nominal, method)
        pooled = paired_interval(gm.pooled, g1.pooled, lo=-1.0, hi=1.0, alpha=SECONDARY_ALPHA)
        claims.append(
            _claim(
                "C2-CENSOR",
                f"Gap_B({method}) minus Gap_B(V1), pooled",
                "nominal",
                False,
                None,
                {"diff": pooled},
            )
        )
        _, im, i1 = np.intersect1d(gm.stratum_units, g1.stratum_units, return_indices=True)
        strat = paired_interval(
            gm.stratum[im], g1.stratum[i1], lo=-1.0, hi=1.0, alpha=SECONDARY_ALPHA
        )
        claims.append(
            _claim(
                "C2-CENSOR",
                f"Gap_B({method}) minus Gap_B(V1), stratum",
                "nominal",
                False,
                None,
                {"diff": strat},
            )
        )
    s1 = shifted[("V1", "closed")].cover_lo
    for method in ("V2L", "V3L"):
        sm = shifted[(method, "closed")].cover_lo
        iv = paired_interval(sm, s1, lo=0.0, hi=1.0, alpha=SECONDARY_ALPHA)
        claims.append(
            _claim(
                "C2-SHIFT",
                f"{method} minus V1, closed-loop c^lo",
                "shifted",
                False,
                None,
                {"diff": iv},
            )
        )
    return claims


def design_dict(design: Design) -> dict[str, Any]:
    return {
        **asdict(design),
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "perc_band": PERC_BAND,
        "select_margin": SELECT_MARGIN,
        "select_floor": SELECT_FLOOR,
        "ext_lo": EXT_LO,
        "ext_hi": EXT_HI,
        "shift_fit_subsets": list(SHIFT_FIT),
        "shift_eval_subsets": list(SHIFT_EVAL),
    }


def run_pipeline(
    sets: DevSets,
    heldout: Callable[[], Heldout],
    *,
    mode: str,
    key: str,
    hashes: Mapping[str, str | None],
    tuned: Tuned,
    design: Design,
    log: Log,
    headroom: Callable[[], object],
    runs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit, calibrate, read the evaluation sets, replay, summarise: the ``results.json`` dict.

    ``design.stratum_m`` must be ``tuned.stratum_m`` (build it with ``run_design``), so that the
    ``M`` a run uses is the tuned one; ``ValueError`` otherwise, before anything is fitted or read.
    ``runs`` (a full run only) is stored as ``results["runs"]``.
    """
    if design.stratum_m != tuned.stratum_m:
        raise ValueError(
            f"the design's M ({design.stratum_m!r}) is not the tuned M ({tuned.stratum_m!r}); "
            "build the design with run_design(tuned)"
        )
    cap = truth_cap(tuned)
    sets = DevSets(
        fit={c: [t.capped(cap) for t in v] for c, v in sets.fit.items()},
        calib={c: [t.capped(cap) for t in v] for c, v in sets.calib.items()},
    )
    fitted = {}
    for cond in CONDITIONS:
        fitted[cond] = fit_condition(cond, sets, tuned, design, key, log)
        headroom()
    raw = heldout()
    held = Heldout(
        [t.capped(cap) for t in raw.eval_units],
        [t.capped(cap) for t in raw.external],
        raw.external_source,
    )
    log(f"evaluation units: {len(held.eval_units)}; external units: {len(held.external)}")
    eval_sets = {"nominal": held.eval_units, "shifted": _in(held.eval_units, SHIFT_EVAL)}
    conditions: dict[str, Any] = {}
    tables: dict[str, UnitTable] = {}
    for cond in CONDITIONS:
        conditions[cond], tables[cond] = evaluate_condition(
            cond, fitted[cond], eval_sets[cond], tuned, design, key, log
        )
        headroom()
    external = external_check(fitted["nominal"], held.external, held.external_source)
    units: dict[str, dict[str, Any]] = {
        cond: {
            "fit": _counts(sets.fit[cond]),
            "calib": _counts(sets.calib[cond]),
            "eval": _counts(eval_sets[cond]),
        }
        for cond in CONDITIONS
    }
    units["external"] = {**_counts(held.external), "source": held.external_source}
    extra = {} if runs is None else {"runs": dict(runs)}
    return {
        "schema": report.SCHEMA,
        "mode": mode,
        "data": report.DATA_NOTE,
        "hashes": dict(hashes),
        "seeds": {
            "key": QUICK_KEY if key == QUICK_KEY else "prereg_sha256",
            "labels": list(SEED_LABELS),
        },
        "design": design_dict(design),
        "tuned": tuned.to_dict(),
        "truth": {"capped_at": cap, "definition": "min(remaining life, cap)"},
        "units": units,
        "conditions": conditions,
        "external": external,
        "claims": build_claims(
            tables["nominal"],
            tables["shifted"],
            external,
            vw_end=conditions["nominal"]["methods"]["VW"]["state"],
        ),
        **extra,
    }


# ----------------------------------------------------------------------------- outputs


def render_documents(text: str) -> tuple[dict[str, Any], str, str]:
    """``results.json`` text to its parsed form, ``RESULTS.md`` and the README block, linted.

    Both documents are rendered from the parsed JSON, exactly as ``--check`` renders them.
    ``ValueError`` if either would use an absolute.
    """
    parsed = json.loads(text)
    md = report.render_results_md(parsed)
    block = report.render_readme_block(parsed)
    data = report.data_strings(parsed)
    for name, body in (("RESULTS.md", md), ("the README block", block)):
        found = report.absolutes_in(body, data)
        if found:
            raise ValueError(f"{name} would contain the absolute {found[0]!r}")
    return parsed, md, block


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def peak_memory_mb() -> tuple[float, float] | None:
    """Peak working set and peak private commit of this process, in MB (Windows), else None."""
    if sys.platform != "win32":
        return None

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    windll: Any = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    handle = windll.kernel32.GetCurrentProcess()
    if not windll.kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    mb = 1024.0 * 1024.0
    return counters.PeakWorkingSetSize / mb, counters.PeakPagefileUsage / mb


def library_versions() -> dict[str, str]:
    from importlib.metadata import version

    import closed_loop

    v01 = Path(closed_loop.__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        # v0.1 supplies the t guard's constants, Clopper-Pearson, the ledger and the guard; its
        # code is outside this repository's code hash, so its version and hash are kept (S3)
        "closed-loop": version("closed-loop"),
        "closed-loop code_sha256": code_sha256([v01]),
    }


def _logger(out: TextIO) -> Log:
    start = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - start:7.1f} s] {message}", file=out, flush=True)

    return log


def _headroom(reader: Callable[[], float | None], limit: float) -> Callable[[], object]:
    return lambda: require_headroom(reader, limit)


def _report_memory(log: Log) -> None:
    peak = peak_memory_mb()
    if peak is not None:
        log(f"peak working set {peak[0]:.0f} MB, peak private commit {peak[1]:.0f} MB")


# ----------------------------------------------------------------------------- commands


def _freeze_entry(ledger: Ledger) -> dict[str, Any] | None:
    for entry in ledger.entries():
        if entry["kind"] == "freeze":
            return entry
    return None


def cmd_freeze(paths: Paths, out: TextIO) -> int:
    ledger = Ledger(paths.ledger)
    try:
        existing = ledger.entries()
    except LedgerError as exc:
        print(f"ledger: BROKEN ({exc})", file=out)
        return EXIT_FAILED
    if existing:
        kinds = sorted({e["kind"] for e in existing})
        print(
            f"refusing to freeze: the ledger already has entries ({', '.join(kinds)}); "
            "one freeze per ledger",
            file=out,
        )
        return EXIT_FAILED
    problems = tuned_problems(paths.tuned)
    tuned = None if problems else load_tuned(paths.tuned)
    table = None if tuned is None else registered_table(DESIGN, tuned)
    show_table = False
    if not paths.prereg.is_file():
        problems.append(f"{paths.prereg.name} does not exist")
    else:
        text = paths.prereg.read_text(encoding="utf-8")
        problems += [f"{paths.prereg.name}: {p}" for p in prereg_problems(text, table)]
        show_table = table is not None and not has_table(text, table)
    if not MANIFEST_PATH.is_file():
        problems.append(f"the package's {MANIFEST_PATH.name} does not exist")
    if problems or tuned is None:
        for line in problems:
            print(f"refusing to freeze: {line}", file=out)
        if show_table:
            print("\nthe registered-values table, to paste verbatim:\n", file=out)
            print(table, file=out)
        return EXIT_FAILED
    hashes = current_hashes(paths)
    entry = ledger.append(
        "freeze",
        **hashes,
        seed_labels=list(SEED_LABELS),
        design=asdict(run_design(tuned)),
        versions=library_versions(),  # v0.1's version and code hash among them (review D, S3)
    )
    print(f"froze pre-registration {hashes['prereg_sha256']}", file=out)
    print(f"ledger head {entry['hash']} (record it outside this repository)", file=out)
    return EXIT_OK


def cmd_registered_table(paths: Paths, out: TextIO) -> int:
    """Print the registered-values table from ``DESIGN`` and ``tuned.json``. No data, no ledger."""
    print(registered_table(DESIGN, load_tuned(paths.tuned)), file=out)
    return EXIT_OK


def cmd_quick(
    paths: Paths, out: TextIO, *, reader: Callable[[], float | None], limit: float
) -> int:
    print(report.QUICK_BANNER, file=out)
    log = _logger(out)
    try:
        tuned = load_tuned(paths.tuned)
        sets, heldout = quick_sets()
        hashes = current_hashes(paths)
        results = run_pipeline(
            sets,
            heldout,
            mode="quick",
            key=QUICK_KEY,
            hashes=hashes,
            tuned=tuned,
            design=run_design(tuned),
            log=log,
            headroom=_headroom(reader, limit),
        )
    except (DataError, TunedError, OSError) as exc:
        print(f"not run: {type(exc).__name__}: {exc}", file=out)
        return EXIT_NOT_RUN
    except GuardError as exc:
        print(f"paused: {exc}", file=out)
        return EXIT_REFUSED
    text = report.to_json(results)
    _, md, _ = render_documents(text)
    _write(paths.cache / "quick.json", text)
    _write(paths.cache / "quick.md", md)
    shown = [_shown(paths.cache / name, paths) for name in ("quick.json", "quick.md")]
    log(f"wrote {shown[0]} and {shown[1]}")
    _report_memory(log)
    print(report.QUICK_BANNER, file=out)
    return EXIT_OK


def _shown(path: Path, paths: Paths) -> str:
    try:
        return path.relative_to(paths.root).as_posix()
    except ValueError:
        return path.name


def _readme_has_markers(paths: Paths) -> bool:
    if not paths.readme.is_file():
        return False
    text = paths.readme.read_text(encoding="utf-8")
    return report.README_BEGIN in text or report.README_END in text


def check_readme_markers(paths: Paths) -> None:
    """``report.MarkerError`` if the README carries result markers that cannot be spliced.

    A README with neither marker is accepted (its block is not written, as in ``publish``).
    """
    if _readme_has_markers(paths):
        report.readme_block_of(paths.readme.read_text(encoding="utf-8"))


def archive_path(paths: Paths, number: int) -> Path:
    """Where run ``number``'s ``results.json`` is kept: ``benchmarks/runs/results-run-{n}.json``.

    An existing file is not replaced: the name takes an ``-again`` suffix instead, as in v0.1.
    """
    path = paths.runs / f"results-run-{number}.json"
    while path.exists():
        path = path.with_name(path.stem + "-again.json")
    return path


def runs_block(paths: Paths, fulls: Sequence[Mapping[str, Any]], archive: Path) -> dict[str, Any]:
    """``results["runs"]``: this run's number and archive path, and the earlier ``full`` entries."""
    earlier = []
    for i, e in enumerate(fulls, start=1):
        verdicts = e.get("verdicts") or {}
        earlier.append(
            {
                "number": int(e.get("run_number", i)),
                "utc": e.get("utc"),
                "results_sha256": e.get("results_sha256"),
                "archive": e.get("archive"),
                "verdicts": {str(k): v for k, v in verdicts.items()},
            }
        )
    return {"number": len(fulls) + 1, "archive": _shown(archive, paths), "earlier": earlier}


def publish(paths: Paths, text: str, out: TextIO, archive: Path | None = None) -> str:
    """Write ``results.json``, ``RESULTS.md``, the README block and the archive copy.

    Everything is rendered and linted before the first file is written. ``archive`` (a full run)
    receives the same text as ``results.json``; it is created only if it does not exist, so an
    earlier run's copy cannot be replaced. Returns the results' sha256.
    """
    _, md, block = render_documents(text)
    readme_text = None
    if _readme_has_markers(paths):
        readme_text = report.splice_readme(paths.readme.read_text(encoding="utf-8"), block)
    _write(paths.results_json, text)
    _write(paths.results_md, md)
    if readme_text is not None:
        _write(paths.readme, readme_text)
        print(f"wrote the results block of {_shown(paths.readme, paths)}", file=out)
    else:
        print(
            f"{_shown(paths.readme, paths)} has no results block markers "
            f"({report.README_BEGIN} and {report.README_END}); its block was not written",
            file=out,
        )
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("xb") as handle:  # "x": fails rather than replace an existing file
            handle.write(text.encode("utf-8"))
        print(f"archived {_shown(archive, paths)}", file=out)
    return file_sha256(paths.results_json)


def cmd_full(
    paths: Paths,
    head: str | None,
    out: TextIO,
    *,
    reader: Callable[[], float | None],
    limit: float,
) -> int:
    ledger = Ledger(paths.ledger)
    try:
        ledger.verify(head)
        freeze = _freeze_entry(ledger)
    except LedgerError as exc:
        print(f"refusing a full run: the ledger is broken ({exc})", file=out)
        return EXIT_FAILED
    if freeze is None:
        print("refusing a full run: no freeze entry in the ledger (run --freeze first)", file=out)
        return EXIT_FAILED
    if not paths.prereg.is_file():
        print(f"refusing a full run: {paths.prereg.name} does not exist", file=out)
        return EXIT_FAILED
    try:
        before = current_hashes(paths)
    except (OSError, ValueError) as exc:
        print(f"not run: {type(exc).__name__}: {exc}", file=out)
        return EXIT_NOT_RUN
    for name, what in (
        ("prereg_sha256", "the pre-registration"),
        ("tuned_sha256", "tuned.json"),
        ("manifest_sha256", "the data manifest"),
    ):
        if before[name] != freeze.get(name):
            print(f"refusing a full run: {what} differs from the freeze entry", file=out)
            return EXIT_FAILED
    if before["code_sha256"] != freeze.get("code_sha256"):
        print("warning: the code has changed since the freeze; both hashes are recorded", file=out)
    try:
        table_problem = registered_table_problem(paths)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        table_problem = f"{type(exc).__name__}: {exc}"
    if table_problem is not None:  # checked before the held-out read (review D, S3)
        print(f"refusing a full run: {table_problem}", file=out)
        return EXIT_FAILED
    try:  # checked before the held-out read: a broken marker after it would lose the results
        check_readme_markers(paths)
    except (report.MarkerError, OSError, UnicodeDecodeError) as exc:
        print(f"not run: the README results block cannot be written ({exc})", file=out)
        return EXIT_NOT_RUN
    fulls = [e for e in ledger.entries() if e["kind"] == "full"]
    number = len(fulls) + 1
    archive = archive_path(paths, number)
    runs = runs_block(paths, fulls, archive)
    prereg_sha = str(before["prereg_sha256"])
    log = _logger(out)
    read = {"heldout": False}

    def mark_read() -> None:
        read["heldout"] = True

    def record_failure(error: str) -> None:
        ledger.append("full-failed", **before, error=error, heldout_read=read["heldout"])

    try:
        tuned = load_tuned(paths.tuned)
        access = HeldoutAccess(ledger, paths.prereg)
        sets, heldout = full_sets(access, mark_read)
        results = run_pipeline(
            sets,
            heldout,
            mode="full",
            key=prereg_sha,
            hashes=before,
            tuned=tuned,
            design=run_design(tuned),
            log=log,
            headroom=_headroom(reader, limit),
            runs=runs,
        )
        text = report.to_json(results)
        _write(paths.unpublished, text)  # kept even if publishing fails: no second read needed
        log(f"wrote {_shown(paths.unpublished, paths)}")
        digest = publish(paths, text, out, archive=archive)
        after = current_hashes(paths)
        verdict_map = {
            f"{c['claim']} {c['row']}": c["verdict"] for c in results["claims"] if c["primary"]
        }
        ledger.append(
            "full",
            **before,
            code_sha256_after=after["code_sha256"],
            results_sha256=digest,
            verdicts=verdict_map,
            versions=library_versions(),
            run_number=number,
            archive=_shown(archive, paths),
        )
    except GuardError as exc:
        if not read["heldout"]:  # a pause while fitting: nothing held out was read, so no record
            print(f"full run paused before any held-out read; nothing recorded: {exc}", file=out)
            return EXIT_REFUSED
        record_failure(f"paused: {exc}")
        print(f"full run paused and recorded: {exc}", file=out)
        return EXIT_REFUSED
    except (DataError, HeldoutError, TunedError, OSError) as exc:
        if not read["heldout"]:  # nothing held out was read: an operational error, not a run
            print(f"not run: {type(exc).__name__}: {exc}", file=out)
            return EXIT_NOT_RUN
        error = f"{type(exc).__name__}: {exc}"
        record_failure(error)
        print(f"full run failed and was recorded: {error}", file=out)
        raise
    except BaseException as exc:
        error = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else f"{type(exc).__name__}: {exc}"
        )
        record_failure(error)
        print(f"full run failed and was recorded: {error}", file=out)
        raise
    ledger.verify()
    if after["code_sha256"] != freeze.get("code_sha256"):
        print("warning: the code that ran differs from the frozen code", file=out)
    print(f"full run recorded; ledger head {ledger.head()}", file=out)
    _report_memory(log)
    return EXIT_OK


def check_published(paths: Paths, ledger: Ledger, out: TextIO) -> int:
    """The published documents against ``results.json`` and the ledger. Needs no data."""
    fulls = [e for e in ledger.entries() if e["kind"] == "full"]
    if not paths.results_json.exists() and not fulls:
        print("results: none published yet", file=out)
        return EXIT_OK
    problems: list[str] = []
    if not fulls:
        problems.append("results.json is there but the ledger has no full run")
    for e in fulls:  # each run's archived copy must still be the results it recorded
        if not e.get("archive"):
            continue
        copy = paths.root / str(e["archive"])
        if not copy.is_file():
            problems.append(f"the archived copy {e['archive']} is missing")
        elif file_sha256(copy) != e.get("results_sha256"):
            problems.append(f"the archived copy {e['archive']} differs from its ledger digest")
    if not paths.results_json.exists():
        problems.append("results.json is missing")
    else:
        digest = file_sha256(paths.results_json)
        if fulls and fulls[-1].get("results_sha256") != digest:
            problems.append(
                "results.json differs from the digest the ledger recorded for the last full run"
            )
        try:
            parsed, md, block = render_documents(paths.results_json.read_bytes().decode("utf-8"))
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"results.json cannot be rendered: {exc}")
        else:
            if not paths.results_md.exists():
                problems.append("RESULTS.md is missing")
            elif paths.results_md.read_bytes().decode("utf-8").replace("\r\n", "\n") != md:
                problems.append("RESULTS.md is not what results.json generates")
            if _readme_has_markers(paths):
                try:
                    current = report.readme_block_of(paths.readme.read_text(encoding="utf-8"))
                except report.MarkerError as exc:
                    problems.append(str(exc))
                else:
                    if current != block:
                        problems.append(
                            "the README results block is not what results.json generates"
                        )
            vw = parsed.get("conditions", {}).get("nominal", {}).get("methods", {}).get("VW", {})
            for c in parsed.get("claims", []):
                if "vw_state" in c and c["vw_state"] != vw_state(vw.get("history", [])):
                    problems.append(
                        f"{c['claim']} {c['row']}: stored VW state {c['vw_state']!r} is not what "
                        "VW's step history gives"
                    )
                try:
                    rebuilt = verdict_of_claim(c)
                except (KeyError, ValueError) as exc:
                    problems.append(f"{c['claim']} {c['row']}: its label cannot be rebuilt ({exc})")
                    continue
                if c.get("primary") and rebuilt != c.get("verdict"):
                    problems.append(
                        f"{c['claim']} {c['row']}: stored label {c.get('verdict')!r} is not what "
                        "its intervals give"
                    )
    if problems:
        for line in problems:
            print(f"results: FAILED ({line})", file=out)
        return EXIT_FAILED
    print(f"results: OK ({len(fulls)} full runs recorded; documents match results.json)", file=out)
    return EXIT_OK


def cmd_check(paths: Paths, head: str | None, out: TextIO) -> int:
    ledger = Ledger(paths.ledger)
    try:
        ledger.verify(head)
    except LedgerError as exc:
        print(f"ledger: BROKEN ({exc})", file=out)
        return EXIT_FAILED
    entries = ledger.entries()
    print(f"ledger: OK ({len(entries)} entries, head {ledger.head()[:16]}...)", file=out)
    print(f"held-out reads recorded: {ledger.heldout_evaluations()}", file=out)
    now = current_hashes(paths)
    freeze = _freeze_entry(ledger)
    if freeze is None:
        shown = (now["prereg_sha256"] or "no file")[:16]
        print(f"prereg: NOT FROZEN (no freeze entry; the document hashes to {shown})", file=out)
        return check_published(paths, ledger, out)
    status = EXIT_OK
    if now["prereg_sha256"] != freeze.get("prereg_sha256"):
        print("prereg: CHANGED since the freeze", file=out)
        status = EXIT_FAILED
    else:
        print(f"prereg: MATCH ({str(now['prereg_sha256'])[:16]}...)", file=out)
    if now["code_sha256"] == freeze.get("code_sha256"):
        print("code: UNCHANGED since the freeze", file=out)
    else:
        print("code: CHANGED since the freeze (reported, not a failure)", file=out)
    for name, label in (("tuned_sha256", "tuned.json"), ("manifest_sha256", "manifest")):
        if now[name] != freeze.get(name):
            print(f"{label}: CHANGED since the freeze", file=out)
            status = EXIT_FAILED
    try:
        table_problem = registered_table_problem(paths)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        table_problem = f"{type(exc).__name__}: {exc}"
    if table_problem is None:
        print("registered values: MATCH the pre-registration", file=out)
    else:
        print(f"registered values: DIFFER ({table_problem})", file=out)
        status = EXIT_FAILED
    published = check_published(paths, ledger, out)
    return status if status != EXIT_OK else published


# ----------------------------------------------------------------------------- main


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="run.py",
        description="closed-loop v0.2 on C-MAPSS (SIMULATED data: NASA's C-MAPSS model, not "
        "field data). With no flag: the full run.",
        epilog="exit codes: 0 done, 1 check failed or refused, 2 usage, 3 refused or paused, "
        "4 not run",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="the full run (also the default)")
    mode.add_argument("--quick", action="store_true", help="dev units only; no ledger")
    mode.add_argument("--check", action="store_true", help="verify the ledger and the results")
    mode.add_argument("--freeze", action="store_true", help="record the freeze entry")
    # prints the registered-values table for the pre-registration; reads no data and no ledger
    mode.add_argument("--registered-table", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lock", metavar="PATH", help="heavy-run lock file")
    parser.add_argument("--head", metavar="HASH", help="expected ledger head, to detect truncation")
    parser.add_argument(
        "--limit", type=float, default=COMMIT_LIMIT_PCT, help="commit-charge limit in percent"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return parser


def _heavy(
    lock: str | None, mode: str, reader: Callable[[], float | None], limit: float
) -> AbstractContextManager[object]:
    """The commit-charge guard at ``limit``, then the heavy-run lock.

    With ``--lock`` that file is the lock; otherwise ``closed_loop_v2.audit.heavy_run`` takes the
    lock named by ``CLOSED_LOOP_V2_LOCK`` (none if unset) and applies the 88% guard again, so a
    ``--limit`` above 88 does not raise the ceiling.
    """
    task = f"benchmarks/run.py --{mode}"
    require_headroom(reader, limit)
    if lock:
        return HeavyRunLock(Path(lock), OWNER, task)
    return heavy_run(OWNER, task, reader=reader)


def main(
    argv: Sequence[str] | None = None,
    *,
    commit_reader: Callable[[], float | None] = commit_charge_percent,
    out: TextIO | None = None,
) -> int:
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(argv)
    paths = Paths(Path(args.root).resolve())
    try:
        if args.registered_table:
            return cmd_registered_table(paths, stream)
        if args.check:
            return cmd_check(paths, args.head, stream)
        if args.freeze:
            return cmd_freeze(paths, stream)
    except (OSError, ValueError, LedgerError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=stream)
        return EXIT_NOT_RUN
    mode = "quick" if args.quick else "full"
    try:
        guard = _heavy(args.lock, mode, commit_reader, args.limit)
        with guard:
            if mode == "quick":
                return cmd_quick(paths, stream, reader=commit_reader, limit=args.limit)
            return cmd_full(paths, args.head, stream, reader=commit_reader, limit=args.limit)
    except GuardError as exc:
        print(f"refused: {exc}", file=stream)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
