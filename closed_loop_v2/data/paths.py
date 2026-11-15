"""Where the C-MAPSS files are, and checks that they are the ones the manifest names.

``data/MANIFEST.json`` holds the sha256 of each of the 14 files NASA publishes. A file is read
through ``read_verified``, which hashes the bytes it returns, so what is parsed is what was
checked: there is no gap between a check and a later read.

The manifest is found relative to this source file, so the package needs a source checkout (an
editable install), not a copy in ``site-packages``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from closed_loop_v2.data.errors import DataError

__all__ = [
    "DATA_ENV",
    "MANIFEST_PATH",
    "data_available",
    "data_dir",
    "expected_sha256",
    "manifest",
    "member_names",
    "read_verified",
    "verify",
]

DATA_ENV = "CLOSED_LOOP_V2_DATA"
_REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = _REPO / "data" / "MANIFEST.json"
_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")


@functools.cache
def manifest() -> dict[str, Any]:
    """The parsed ``data/MANIFEST.json``, checked for the fields this package uses."""
    try:
        parsed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read the manifest {MANIFEST_PATH.name}: {exc}") from exc
    files = parsed.get("files") if isinstance(parsed, dict) else None
    if not isinstance(files, dict) or not files:
        raise DataError(f"{MANIFEST_PATH.name}: no 'files' mapping")
    for name, digest in files.items():
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise DataError(f"{MANIFEST_PATH.name}: the sha256 for {name!r} is not 64 hex digits")
    return parsed


def member_names() -> tuple[str, ...]:
    """The file names the manifest lists, in manifest order."""
    return tuple(manifest()["files"])


def expected_sha256(name: str) -> str:
    """The manifest's sha256 for ``name``; ``DataError`` if the manifest does not list it."""
    files = manifest()["files"]
    if name not in files:
        raise DataError(f"{name}: not a file the manifest lists")
    return str(files[name])


def data_dir() -> Path:
    """``CLOSED_LOOP_V2_DATA`` if set and not empty, otherwise ``<repository>/data/cmapss``."""
    value = os.environ.get(DATA_ENV, "").strip()
    return Path(value) if value else _REPO / "data" / "cmapss"


def data_available() -> bool:
    """True if every file the manifest lists exists in ``data_dir()``. Looks at names only."""
    directory = data_dir()
    try:
        return directory.is_dir() and all((directory / n).is_file() for n in member_names())
    except DataError:
        return False


def read_verified(directory: Path, name: str) -> bytes:
    """The bytes of ``directory/name``, after checking their sha256 against the manifest."""
    expected = expected_sha256(name)
    path = Path(directory) / name
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise DataError(f"{name}: missing from {directory}") from None
    except OSError as exc:
        raise DataError(f"{name}: cannot be read from {directory}: {exc}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DataError(
            f"{name}: sha256 is {actual[:16]}... but the manifest says {expected[:16]}...: "
            "the file was changed, or it is not NASA's download"
        )
    return data


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path, names: Iterable[str] | None = None) -> None:
    """Check each file against the manifest; ``DataError`` names every missing or changed file.

    With no ``names`` all the manifest's files are checked. The files are only hashed, not parsed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"data directory {directory} does not exist")
    problems: list[str] = []
    for name in member_names() if names is None else tuple(names):
        expected = expected_sha256(name)
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        try:
            actual = _file_sha256(path)
        except OSError as exc:
            problems.append(f"{name}: cannot be read ({exc})")
            continue
        if actual != expected:
            problems.append(f"{name}: sha256 differs from the manifest")
    if problems:
        raise DataError(f"data in {directory} does not match the manifest: " + "; ".join(problems))
