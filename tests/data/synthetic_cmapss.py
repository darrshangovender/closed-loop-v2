"""Synthetic stand-ins for the C-MAPSS files, laid out like the real ones, for the data tests.

Rows are 26 numbers separated by two spaces with a trailing space and CRLF line endings, as in
NASA's files. The numbers are made up. Two small subsets are built (FD001 and FD002) so that tests
can pick subsets. ``Synthetic.manifest`` is what ``closed_loop_v2.data.paths.manifest`` returns
while a test runs, so the loader's checksum step runs on these files too.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# subset -> (training units, test units)
SYNTH_UNITS: dict[str, tuple[int, int]] = {"FD001": (24, 5), "FD002": (20, 4)}


def rows_text(unit: int, n_cycles: int, rng: np.random.Generator) -> str:
    """The rows of one unit as text, one line per cycle."""
    lines = []
    for cycle in range(1, n_cycles + 1):
        values = rng.normal(size=24)
        numbers = "  ".join(f"{v:.4f}" for v in values)
        lines.append(f"{unit}  {cycle}  {numbers}  \r\n")
    return "".join(lines)


def cycles_of(unit: int) -> int:
    """How many rows the synthetic unit has."""
    return 5 + unit % 4


def train_text(n_units: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rows_text(u, cycles_of(u), rng) for u in range(1, n_units + 1))


def rul_values(n_units: int) -> list[int]:
    return [10 + 7 * u for u in range(1, n_units + 1)]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Synthetic:
    """A directory of synthetic files and the manifest that describes them."""

    directory: Path
    hashes: dict[str, str] = field(default_factory=dict)

    def write(self, name: str, data: bytes, *, rehash: bool = True) -> None:
        """Write a file. With ``rehash`` the manifest is updated to match it."""
        (self.directory / name).write_bytes(data)
        if rehash:
            self.hashes[name] = sha256_hex(data)

    def read(self, name: str) -> bytes:
        return (self.directory / name).read_bytes()

    def manifest(self) -> dict[str, Any]:
        return {"files": dict(self.hashes)}


def make_dataset(directory: Path) -> Synthetic:
    directory.mkdir(parents=True, exist_ok=True)
    synth = Synthetic(directory)
    for i, (subset, (n_train, n_test)) in enumerate(SYNTH_UNITS.items()):
        synth.write(f"train_{subset}.txt", train_text(n_train, seed=10 + i).encode("ascii"))
        synth.write(f"test_{subset}.txt", train_text(n_test, seed=20 + i).encode("ascii"))
        rul = "".join(f"{v}\r\n" for v in rul_values(n_test))
        synth.write(f"RUL_{subset}.txt", rul.encode("ascii"))
    synth.write("readme.txt", b"synthetic stand-in\r\n")
    return synth


_RECORDERS: list[list[str]] = []


def _audit_open(event: str, args: tuple[Any, ...]) -> None:
    """Python's ``open`` audit event, raised by ``open``, ``io.open``, ``io.FileIO``, ``os.open``,
    ``io.open_code`` and so by every ``Path`` reader and ``numpy.loadtxt``/``genfromtxt``."""
    if event != "open" or not _RECORDERS:
        return
    path = args[0] if args else None
    if isinstance(path, (str, bytes, os.PathLike)):
        name = os.fsdecode(path)
        for seen in _RECORDERS:
            seen.append(name)


# An audit hook cannot be removed once added, so it is added once, here, and records only while a
# ``recording_opens`` block is active. Review of 10 November, finding 3: the earlier recorder
# patched six entry points by hand and did not see ``numpy.loadtxt``, ``io.FileIO`` or ``nt.open``.
sys.addaudithook(_audit_open)


@contextmanager
def recording_opens() -> Iterator[list[str]]:
    """Record the path of every file Python opens while the block runs.

    It uses the interpreter's ``open`` audit event (PEP 578), not patches, so every route through
    Python's file layer is seen: ``open``, ``io.open``, ``io.FileIO``, ``os.open``, the ``Path``
    readers, ``numpy.loadtxt``. A C extension that opens files with the C library directly is not
    seen; none is used to read data here.
    """
    seen: list[str] = []
    _RECORDERS.append(seen)
    try:
        yield seen
    finally:
        _RECORDERS.remove(seen)


def heldout_names(opened: list[str]) -> list[str]:
    """Names of held-out files that were opened, without repeats.

    Only the file name is looked at, not the directories: pytest names a test's directory after the
    test, and that name can start with ``test_``.
    """
    names = {Path(p).name for p in opened}
    return sorted(n for n in names if n.startswith(("test_", "RUL_")))
