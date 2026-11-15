"""Parse C-MAPSS text files into ``Trajectory`` objects. Private to ``closed_loop_v2.data``.

A file has one row per engine cycle: 26 whitespace-separated numbers, namely the unit number,
the cycle, three operational settings and 21 sensors (the dataset's readme.txt).

Training files mix units of every role. ``parse_training`` therefore works in two passes:

1. It reads the unit column of every row and nothing else, and checks that the units are
   ``1..N`` with ``N`` from the readme and that each unit's rows are together.
2. It splits the remaining columns of the rows of units whose role was asked for, and no others.
   A row of any other unit is not split and no array is built for it.

Messages name the file and the line, and never repeat a row's content.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Collection

import numpy as np

from closed_loop_v2.data.errors import DataError
from closed_loop_v2.data.roles import role_of
from closed_loop_v2.types import N_SENSORS, N_SETTINGS, Role, Source, Trajectory

N_COLUMNS = 2 + N_SETTINGS + N_SENSORS
_UINT = re.compile(r"[0-9]+")

__all__ = ["N_COLUMNS", "parse_rul", "parse_test", "parse_training"]

_Lines = list[tuple[int, str]]


def _lines(data: bytes, name: str) -> _Lines:
    """The non-blank lines as ``(1-based line number, text)``."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DataError(f"{name}: byte {exc.start} is not ASCII") from None
    return [(i, line) for i, line in enumerate(text.splitlines(), start=1) if line.strip()]


def _uint(token: str, name: str, lineno: int, column: int) -> int:
    if not _UINT.fullmatch(token):
        raise DataError(f"{name} line {lineno}: column {column} is not a whole number")
    return int(token)


def _unit_column(lines: _Lines, name: str) -> list[int]:
    """The first column of every line, as integers. Nothing else of the line is read."""
    units = []
    for lineno, line in lines:
        head = line.split(None, 1)
        units.append(_uint(head[0], name, lineno, 1))
    return units


def _check_units(units: list[int], name: str, expected_n: int) -> None:
    """Units must be ``1..expected_n`` and each unit's rows must be one block, in order."""
    if not units:
        raise DataError(f"{name}: no rows")
    for previous, current in itertools.pairwise(units):
        if current < previous:
            raise DataError(f"{name}: rows of unit {current} are not together")
    found = set(units)
    wanted = set(range(1, expected_n + 1))
    if found != wanted:
        missing = sorted(wanted - found)[:5]
        extra = sorted(found - wanted)[:5]
        raise DataError(
            f"{name}: expected units 1..{expected_n} (readme.txt); "
            f"missing {missing or 'none'}, unexpected {extra or 'none'}"
        )


def _block(
    lines: _Lines,
    name: str,
    subset: str,
    unit: int,
    source: Source,
    rul_at_end: int,
) -> Trajectory:
    """Build one trajectory from its rows."""
    cycles: list[int] = []
    rows: list[list[str]] = []
    for lineno, line in lines:
        tokens = line.split()
        if len(tokens) != N_COLUMNS:
            raise DataError(f"{name} line {lineno}: {len(tokens)} columns, expected {N_COLUMNS}")
        cycles.append(_uint(tokens[1], name, lineno, 2))
        rows.append(tokens[2:])
    try:
        values = np.array(rows, dtype=np.float64)
    except ValueError:
        raise DataError(f"{name} unit {unit}: a settings or sensor value is not a number") from None
    if not np.isfinite(values).all():
        raise DataError(f"{name} unit {unit}: a settings or sensor value is not finite")
    arrays = (
        np.array(cycles, dtype=np.int64),
        np.ascontiguousarray(values[:, :N_SETTINGS]),
        np.ascontiguousarray(values[:, N_SETTINGS:]),
    )
    for array in arrays:
        array.flags.writeable = False
    try:
        return Trajectory(
            subset=subset,
            unit=unit,
            source=source,
            cycles=arrays[0],
            settings=arrays[1],
            sensors=arrays[2],
            rul_at_end=rul_at_end,
        )
    except ValueError as exc:
        raise DataError(f"{name} unit {unit}: {exc}") from None


def _blocks(lines: _Lines, units: list[int]) -> list[tuple[int, _Lines]]:
    """Group the lines by unit. The units are non-decreasing (``_check_units``)."""
    grouped = []
    start = 0
    for unit, run in itertools.groupby(units):
        count = len(list(run))
        grouped.append((unit, lines[start : start + count]))
        start += count
    return grouped


def parse_training(
    data: bytes, name: str, subset: str, expected_n: int, roles: Collection[Role]
) -> dict[Role, list[Trajectory]]:
    """Trajectories of the training units whose role is in ``roles``, by role, in unit order."""
    wanted = frozenset(roles)
    lines = _lines(data, name)
    units = _unit_column(lines, name)
    _check_units(units, name, expected_n)
    out: dict[Role, list[Trajectory]] = {role: [] for role in sorted(wanted)}
    for unit, block in _blocks(lines, units):
        role = role_of(subset, unit)
        if role in wanted:
            out[role].append(_block(block, name, subset, unit, "train", 0))
    return out


def parse_rul(data: bytes, name: str, expected_n: int) -> list[int]:
    """One whole number per line, one line per test unit, in unit order."""
    lines = _lines(data, name)
    values = []
    for lineno, line in lines:
        tokens = line.split()
        if len(tokens) != 1:
            raise DataError(f"{name} line {lineno}: {len(tokens)} columns, expected 1")
        values.append(_uint(tokens[0], name, lineno, 1))
    if len(values) != expected_n:
        raise DataError(f"{name}: {len(values)} values, expected {expected_n} (readme.txt)")
    return values


def parse_test(
    data: bytes, name: str, subset: str, expected_n: int, rul: list[int]
) -> list[Trajectory]:
    """Every test unit, with ``rul_at_end`` taken from ``rul`` (index ``unit - 1``)."""
    lines = _lines(data, name)
    units = _unit_column(lines, name)
    _check_units(units, name, expected_n)
    return [
        _block(block, name, subset, unit, "test", rul[unit - 1])
        for unit, block in _blocks(lines, units)
    ]
