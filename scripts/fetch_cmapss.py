#!/usr/bin/env python3
"""Fetch NASA's C-MAPSS turbofan data and check it against data/MANIFEST.json.

    python scripts/fetch_cmapss.py [--dest DIR] [--from-zip PATH]

C-MAPSS is simulated by NASA's C-MAPSS engine model, not field data. The repository does not
redistribute it. This script downloads the zip named in the manifest (or reads a local copy),
checks the zip's size and sha256 before it opens it, reads the zip inside it in memory, checks
every member's sha256, and only then writes the 14 files into ``DIR``, each by its bare name.

It does not execute or import anything from the archive. A member whose name has a directory
part, ``..``, a drive or a leading dot is refused, as is any name the manifest does not list.
Nothing is written unless every check passes. Standard library only.

``DIR`` defaults to ``CLOSED_LOOP_V2_DATA`` if set, otherwise ``data/cmapss`` in this repository.
The data repository asks users to acknowledge it and the data donors; see the manifest for the
citations and the terms.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "MANIFEST.json"
DATA_ENV = "CLOSED_LOOP_V2_DATA"
MAX_MEMBER_BYTES = 64 * 1024 * 1024  # the largest real member is about 10 MB
MAX_INNER_ZIP_BYTES = 256 * 1024 * 1024
TIMEOUT_S = 60

# A bare file name: starts with a letter or digit, then letters, digits, space, dot, dash or
# underscore. No slash, backslash or colon, so no directory part and no drive.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")
_HEX64 = re.compile(r"[0-9a-f]{64}")

Opener = Callable[[str], AbstractContextManager[Any]]

# What the zipfile module raises for a corrupt, encrypted or unsupported archive.
_ARCHIVE_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    RuntimeError,
    NotImplementedError,
    EOFError,
    zlib.error,
)


class FetchError(Exception):
    """A download, archive or checksum problem. Nothing was written."""


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = MANIFEST_PATH if path is None else path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        files = manifest["files"]
        zip_info = manifest["zip"]
        inner = manifest["inner_zip"]
        url = manifest["source_url"]
        ok = (
            isinstance(files, dict)
            and files
            and all(isinstance(v, str) and _HEX64.fullmatch(v) for v in files.values())
            and isinstance(zip_info["bytes"], int)
            and _HEX64.fullmatch(zip_info["sha256"])
            and isinstance(inner, str)
            and isinstance(url, str)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"cannot use the manifest {path.name}: {exc!r}") from exc
    if not ok:
        raise FetchError(f"the manifest {path.name} is incomplete or malformed")
    if not url.startswith("https://"):
        raise FetchError("the manifest's source_url is not https")
    return manifest


def check_member_name(name: str) -> None:
    """Refuse a name that is not a plain file name (zip-slip, drive, hidden, directory)."""
    if not _SAFE_NAME.fullmatch(name) or ".." in name or name.endswith((".", " ")):
        raise FetchError(f"refusing archive member with an unsafe name: {name!r}")


def _default_opener(url: str) -> AbstractContextManager[Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "closed-loop-v2-fetch"})
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)  # the manifest URL is https


def _get_zip_bytes(manifest: dict[str, Any], from_zip: Path | None, opener: Opener) -> bytes:
    want = int(manifest["zip"]["bytes"])
    if from_zip is not None:
        try:
            size = from_zip.stat().st_size
        except OSError as exc:
            raise FetchError(f"cannot read {from_zip}: {exc}") from exc
        if size != want:
            raise FetchError(f"{from_zip.name}: {size} bytes, the manifest says {want}")
        return from_zip.read_bytes()
    url = str(manifest["source_url"])
    try:
        with opener(url) as response:
            data: bytes = response.read(want + 1)
    except OSError as exc:
        raise FetchError(f"download failed: {exc}") from exc
    if len(data) != want:
        raise FetchError(f"download is {len(data)} bytes; the manifest says {want}")
    return data


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, cap: int) -> bytes:
    with archive.open(info) as handle:
        data = handle.read(cap + 1)
    if len(data) > cap:
        raise FetchError(f"archive member {info.filename!r} is larger than {cap} bytes")
    return data


def read_members(
    manifest: dict[str, Any], from_zip: Path | None = None, opener: Opener = _default_opener
) -> dict[str, bytes]:
    """Get the zip, check it, and return ``{name: bytes}`` for the manifest's files.

    Raises ``FetchError`` on a wrong size or sha256, a bad archive, an unsafe, unlisted or
    repeated member name, a missing file, or a member whose sha256 differs.
    """
    raw = _get_zip_bytes(manifest, from_zip, opener)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != manifest["zip"]["sha256"]:
        raise FetchError(
            f"zip sha256 is {actual[:16]}... but the manifest says "
            f"{manifest['zip']['sha256'][:16]}...: not opening it"
        )
    inner_name = str(manifest["inner_zip"])
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            found = [i for i in outer.infolist() if i.filename == inner_name]
            if len(found) != 1:
                raise FetchError(f"expected one {inner_name} in the download, found {len(found)}")
            inner_bytes = _read_member(outer, found[0], MAX_INNER_ZIP_BYTES)
        listed: dict[str, str] = manifest["files"]
        members: dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            for info in inner.infolist():
                check_member_name(info.filename)
                if info.filename not in listed:
                    raise FetchError(f"archive member {info.filename!r} is not in the manifest")
                if info.filename in members:
                    raise FetchError(f"archive member {info.filename!r} appears twice")
                members[info.filename] = _read_member(inner, info, MAX_MEMBER_BYTES)
    except _ARCHIVE_ERRORS as exc:
        raise FetchError(f"not a usable zip archive: {exc!r}") from exc
    missing = sorted(set(listed) - set(members))
    if missing:
        raise FetchError(f"the archive lacks {len(missing)} file(s): {missing}")
    for name, data in members.items():
        digest = hashlib.sha256(data).hexdigest()
        if digest != listed[name]:
            raise FetchError(f"{name}: sha256 differs from the manifest")
    return members


def _replace(src: Path, dst: Path) -> None:
    """``os.replace``, retried: on Windows another process may briefly hold the target."""
    for attempt in range(5):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def write_members(members: dict[str, bytes], dest: Path) -> list[Path]:
    """Write each file into ``dest`` by bare name, through a temporary file and ``os.replace``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written: list[Path] = []
    for name, data in members.items():
        check_member_name(name)
        target = root / name
        if target.resolve().parent != root:
            raise FetchError(f"{name!r} would be written outside {dest}")
        temp = root / f".{name}.part-{os.getpid()}"
        # O_EXCL: refuse anything already at the temporary name, a symlink included, instead of
        # writing through it (review of 10 November, note on the fetch script)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(temp, flags | getattr(os, "O_NOFOLLOW", 0), 0o644)
        except FileExistsError:
            raise FetchError(
                f"{temp.name} already exists in {dest}: remove it and run again"
            ) from None
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _replace(temp, target)
        finally:
            with contextlib.suppress(OSError):
                temp.unlink()
        written.append(target)
    return written


def fetch(
    dest: Path,
    *,
    from_zip: Path | None = None,
    manifest: dict[str, Any] | None = None,
    opener: Opener = _default_opener,
) -> list[Path]:
    """Download or read the zip, check everything, then write the files. Returns their paths."""
    manifest = load_manifest() if manifest is None else manifest
    members = read_members(manifest, from_zip, opener)
    return write_members(members, dest)


def default_dest() -> Path:
    value = os.environ.get(DATA_ENV, "").strip()
    return Path(value) if value else REPO / "data" / "cmapss"


def main(argv: Sequence[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Fetch and check NASA's C-MAPSS data.")
    parser.add_argument("--dest", type=Path, default=None, help="where to write the files")
    parser.add_argument("--from-zip", type=Path, default=None, help="use a local copy of the zip")
    args = parser.parse_args(argv)
    dest = args.dest if args.dest is not None else default_dest()
    try:
        paths = fetch(dest, from_zip=args.from_zip)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(paths)} files to {dest}, each checked against data/MANIFEST.json")
    print("C-MAPSS is simulated by NASA's C-MAPSS model. See the manifest for terms and citations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
