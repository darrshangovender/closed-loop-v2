"""Packaging checks: version, the closed-loop dependency, .gitignore, and machine paths."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

import closed_loop_v2

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"

DIRECT_REFERENCE = re.compile(
    r"closed-loop @ git\+https://github\.com/[A-Za-z0-9_.-]+/closed-loop@v\d+\.\d+\.\d+"
)
IS_CLOSED_LOOP = re.compile(r"closed[-_.]loop(?![-_.A-Za-z0-9])", re.IGNORECASE)


def _project() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- version


def test_the_package_version_matches_pyproject() -> None:
    assert closed_loop_v2.__version__ == _project()["project"]["version"]


# --------------------------------------------------------------------------- closed-loop


def _all_requirements() -> list[str]:
    project = _project()["project"]
    found = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        found.extend(group)
    return found


def test_closed_loop_is_referenced_by_direct_url_and_tag_only() -> None:
    mentions = [r for r in _all_requirements() if IS_CLOSED_LOOP.match(r.strip())]
    assert len(mentions) == 1
    assert DIRECT_REFERENCE.fullmatch(mentions[0]), mentions[0]


def test_pyproject_allows_the_direct_reference() -> None:
    assert _project()["tool"]["hatch"]["metadata"]["allow-direct-references"] is True


def test_the_ci_workflow_installs_closed_loop_from_the_same_url_and_never_by_bare_name() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pinned = next(r for r in _all_requirements() if IS_CLOSED_LOOP.match(r.strip()))
    assert pinned.split(" @ ", 1)[1] in text
    for line in text.splitlines():
        if "pip install" in line:
            assert not re.search(r"closed-loop(?!\s*@)(?!-v2)", line), line


def test_the_ci_workflow_runs_the_documented_steps() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ubuntu-latest" in text
    assert re.search(r'python-version:.*"3\.11".*"3\.12"', text)
    for step in ("ruff check .", "mypy", "pytest tests/ -q", 'pip install -e ".[dev]"'):
        assert step in text, step
    assert re.search(r"permissions:\s*\n\s+contents: read", text)


# --------------------------------------------------------------------------- .gitignore


def test_gitignore_hides_the_data_but_not_the_manifest_by_its_lines() -> None:
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/*" in lines and "!data/MANIFEST.json" in lines
    assert lines.index("data/*") < lines.index("!data/MANIFEST.json")  # the later line wins


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_gitignore_hides_the_data_but_not_the_manifest_according_to_git(tmp_path: Path) -> None:
    """Ask git itself, in a scratch repository made under ``tmp_path``."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / ".gitignore").write_bytes((REPO / ".gitignore").read_bytes())
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=scratch, env=env, capture_output=True, text=True, check=False
        )

    assert run("init", "-q").returncode == 0
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"):
        assert run("check-ignore", "-q", f"data/cmapss/{name}").returncode == 0, name
    assert run("check-ignore", "-q", "data/anything.txt").returncode == 0
    assert run("check-ignore", "-q", "data/MANIFEST.json").returncode == 1  # not ignored


# --------------------------------------------------------------------------- machine paths

# Built from pieces so that this file does not match itself.
WINDOWS_USER_PATH = re.compile("[A-Za-z]:" + r"[\\/]+" + "Users" + r"[\\/]", re.IGNORECASE)
UNIX_HOME_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:" + "Users|home" + r")/[^/\s`'\"]+/")
SHARED_LOCK_FOLDER = re.compile("Dixon" + "-HQ", re.IGNORECASE)
FORBIDDEN = (WINDOWS_USER_PATH, UNIX_HOME_PATH, SHARED_LOCK_FOLDER)

TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini", ".lock"}
TEXT_NAMES = {"Makefile", ".gitignore", ".gitattributes", "LICENSE"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "build",
    "dist",
    ".cache",
    "progress",
}

# Files that still hold a machine path and are not owned by the packaging checks. Each entry is
# a defect to fix at its source, then to delete from here: the second test below fails once the
# file is clean, so this list cannot go stale.
KNOWN_PENDING: set[str] = set()


def scan(root: Path) -> dict[str, list[int]]:
    """Files under ``root`` (as posix paths) with the line numbers that match a forbidden pattern.

    Skips what ``.gitignore`` hides: caches, environments, ``data/*`` except the manifest.
    """
    hits: dict[str, list[int]] = {}
    for folder, dirs, files in os.walk(root):
        rel_folder = Path(folder).relative_to(root).as_posix()
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        if rel_folder == "data":
            dirs[:] = []
        for name in files:
            path = Path(folder) / name
            rel = path.relative_to(root).as_posix()
            if rel_folder == "data" and name != "MANIFEST.json":
                continue
            if path.suffix not in TEXT_SUFFIXES and name not in TEXT_NAMES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = [
                i
                for i, ln in enumerate(text.splitlines(), 1)
                if any(p.search(ln) for p in FORBIDDEN)
            ]
            if lines:
                hits[rel] = lines
    return hits


def test_the_scanner_finds_a_machine_path_and_skips_ignored_folders(tmp_path: Path) -> None:
    windows = "C" + ":" + "\\" + "Users" + "\\someone\\x"
    unix = "/" + "Users" + "/someone/project/"
    (tmp_path / "a.py").write_text(f'P = "{windows}"\n', encoding="utf-8")
    (tmp_path / "b.md").write_text(f"see {unix} here\n", encoding="utf-8")
    (tmp_path / "c.md").write_text("fine\n/ok/path\n", encoding="utf-8")
    (tmp_path / "d.txt").write_text("the Dixon" + "-HQ folder\n", encoding="utf-8")
    (tmp_path / ".hypothesis").mkdir()
    (tmp_path / ".hypothesis" / "e.txt").write_text(windows, encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.txt").write_text(windows, encoding="utf-8")
    assert scan(tmp_path) == {"a.py": [1], "b.md": [1], "d.txt": [1]}


def test_no_machine_path_in_any_text_file_of_the_repository() -> None:
    hits = {k: v for k, v in scan(REPO).items() if k not in KNOWN_PENDING}
    assert hits == {}, "machine path (line numbers) in: " + ", ".join(
        f"{k}:{v}" for k, v in hits.items()
    )


def test_every_known_pending_file_still_has_a_machine_path() -> None:
    hits = scan(REPO)
    clean = sorted(k for k in KNOWN_PENDING if k not in hits)
    assert clean == [], f"now clean, remove from KNOWN_PENDING: {clean}"
