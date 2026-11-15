"""audit.py: where the records live, the code hash, and the heavy-run guard."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from closed_loop.audit.prereg import code_sha256

from closed_loop_v2 import audit
from closed_loop_v2.audit import GuardError, code_sha256_v2, heavy_lock_path, heavy_run


def _make_repo(root: Path, *, run_py: bool = True) -> None:
    pkg = root / "closed_loop_v2"
    (pkg / "data").mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8", newline="\n")
    (pkg / "data" / "load.py").write_text("X = 1\n", encoding="utf-8", newline="\n")
    (root / "benchmarks").mkdir()
    if run_py:
        (root / "benchmarks" / "run.py").write_text("RUN = 1\n", encoding="utf-8", newline="\n")


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    _make_repo(root)
    monkeypatch.setattr(audit, "REPO", root)
    return root


# --------------------------------------------------------------------------- paths


def test_repo_is_the_directory_that_holds_the_package() -> None:
    assert (audit.REPO / "closed_loop_v2" / "audit.py").is_file()
    assert (audit.REPO / "pyproject.toml").is_file()


def test_record_paths_are_the_ones_in_0002() -> None:
    repo = audit.REPO
    assert repo / "docs" / "PREREGISTRATION.md" == audit.PREREG
    assert repo / "benchmarks" / "tuned.json" == audit.TUNED
    assert repo / "benchmarks" / "ledger.jsonl" == audit.LEDGER
    assert repo / "benchmarks" / "results.json" == audit.RESULTS
    assert repo / "benchmarks" / ".cache" == audit.CACHE


# --------------------------------------------------------------------------- code hash


def test_code_hash_is_the_v01_hash_over_the_package_and_run_py(fake_repo: Path) -> None:
    expected = code_sha256([fake_repo / "closed_loop_v2", fake_repo / "benchmarks" / "run.py"])
    assert code_sha256_v2() == expected


def test_code_hash_changes_with_a_package_file(fake_repo: Path) -> None:
    before = code_sha256_v2()
    (fake_repo / "closed_loop_v2" / "data" / "load.py").write_text("X = 2\n", encoding="utf-8")
    assert code_sha256_v2() != before


def test_code_hash_changes_with_a_new_package_file(fake_repo: Path) -> None:
    before = code_sha256_v2()
    (fake_repo / "closed_loop_v2" / "new.py").write_text("Y = 1\n", encoding="utf-8")
    assert code_sha256_v2() != before


def test_code_hash_covers_benchmarks_run_py(fake_repo: Path) -> None:
    before = code_sha256_v2()
    (fake_repo / "benchmarks" / "run.py").write_text("RUN = 2\n", encoding="utf-8")
    assert code_sha256_v2() != before


def test_code_hash_ignores_other_files_in_benchmarks_and_the_docs(fake_repo: Path) -> None:
    before = code_sha256_v2()
    (fake_repo / "benchmarks" / "other.py").write_text("Z = 1\n", encoding="utf-8")
    (fake_repo / "benchmarks" / "tuned.json").write_text("{}", encoding="utf-8")
    (fake_repo / "closed_loop_v2" / "notes.md").write_text("notes", encoding="utf-8")
    cache = fake_repo / "closed_loop_v2" / "__pycache__"
    cache.mkdir()
    (cache / "load.cpython-312.pyc.py").write_text("cache", encoding="utf-8")
    assert code_sha256_v2() == before


def test_code_hash_does_not_depend_on_where_the_checkout_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashes = []
    for place in ("one", "deeper/two"):
        root = tmp_path / place / "repo"
        _make_repo(root)
        monkeypatch.setattr(audit, "REPO", root)
        hashes.append(code_sha256_v2())
    assert hashes[0] == hashes[1]


def test_code_hash_does_not_depend_on_line_endings(fake_repo: Path) -> None:
    before = code_sha256_v2()
    path = fake_repo / "closed_loop_v2" / "data" / "load.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert code_sha256_v2() == before


def test_code_hash_needs_run_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _make_repo(root, run_py=False)
    monkeypatch.setattr(audit, "REPO", root)
    with pytest.raises(FileNotFoundError, match=r"run\.py"):
        code_sha256_v2()


# --------------------------------------------------------------------------- the lock path


def test_no_lock_path_without_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(audit.LOCK_ENV, raising=False)
    assert heavy_lock_path() is None
    monkeypatch.setenv(audit.LOCK_ENV, "  ")
    assert heavy_lock_path() is None


def test_the_lock_path_comes_from_closed_loop_v2_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert audit.LOCK_ENV == "CLOSED_LOOP_V2_LOCK"
    monkeypatch.setenv(audit.LOCK_ENV, str(tmp_path / "heavy.lock"))
    assert heavy_lock_path() == tmp_path / "heavy.lock"


def test_audit_module_holds_no_machine_path() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    strings = [
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Constant)
        if isinstance(n.value, str)
    ]
    assert not [s for s in strings if re.search(r"[A-Za-z]:[\\/]", s)]
    assert not [s for s in strings if s.endswith(".lock")]


# --------------------------------------------------------------------------- heavy_run


def test_heavy_run_takes_the_lock_and_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "heavy.lock"
    monkeypatch.setenv(audit.LOCK_ENV, str(lock))
    with heavy_run("tester", "unit test", reader=lambda: 10.0):
        text = lock.read_text(encoding="utf-8")
        assert "owner=tester" in text and "task=unit test" in text
    assert not lock.exists()


def test_heavy_run_removes_the_lock_when_the_body_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "heavy.lock"
    monkeypatch.setenv(audit.LOCK_ENV, str(lock))
    with pytest.raises(RuntimeError, match="boom"):
        with heavy_run("tester", "unit test", reader=lambda: 10.0):
            raise RuntimeError("boom")
    assert not lock.exists()


def test_heavy_run_refuses_a_held_lock_and_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "heavy.lock"
    lock.write_text("owner=someone else\n", encoding="utf-8")
    monkeypatch.setenv(audit.LOCK_ENV, str(lock))
    with pytest.raises(GuardError, match="held"):
        with heavy_run("tester", "unit test", reader=lambda: 10.0):
            pytest.fail("the body must not run")
    assert lock.read_text(encoding="utf-8") == "owner=someone else\n"


def test_heavy_run_refuses_at_88_percent_before_taking_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "heavy.lock"
    monkeypatch.setenv(audit.LOCK_ENV, str(lock))
    with pytest.raises(GuardError, match="commit charge"):
        heavy_run("tester", "unit test", reader=lambda: 88.0)
    assert not lock.exists()


def test_heavy_run_allows_87_percent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(audit.LOCK_ENV, str(tmp_path / "heavy.lock"))
    with heavy_run("tester", "unit test", reader=lambda: 87.9):
        pass


def test_without_a_lock_path_the_commit_guard_still_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(audit.LOCK_ENV, raising=False)
    with heavy_run("tester", "unit test", reader=lambda: 10.0):
        pass  # no lock file anywhere, and no error
    with pytest.raises(GuardError, match="commit charge"):
        heavy_run("tester", "unit test", reader=lambda: 95.0)


def test_an_unreadable_commit_charge_does_not_stop_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(audit.LOCK_ENV, raising=False)
    with heavy_run("tester", "unit test", reader=lambda: None):
        pass
