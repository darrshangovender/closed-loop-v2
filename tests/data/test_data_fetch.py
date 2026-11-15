"""scripts/fetch_cmapss.py, tested with small zips built in memory and a fake opener.

No test downloads anything, and none touches the real zip.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import sys
import urllib.error
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_cmapss.py"
INNER_PATH = "6. Made-up Data Set/CMAPSSData.zip"
GOOD = {
    "readme.txt": b"made-up readme\r\n",
    "train_FD001.txt": b"1 1 0.5\r\n1 2 0.6\r\n",
    "RUL_FD001.txt": b"112\r\n",
    "Damage Propagation Modeling.pdf": b"%PDF-made-up\n",
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_cmapss_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_script()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_inner(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def make_outer(inner: bytes, inner_path: str = INNER_PATH, extra: int = 0) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(inner_path.rsplit("/", 1)[0] + "/", b"")  # a directory entry, as NASA's
        for _ in range(1 + extra):
            archive.writestr(inner_path, inner)
    return buffer.getvalue()


def make_manifest(
    members: dict[str, bytes], outer: bytes, inner_path: str = INNER_PATH
) -> dict[str, Any]:
    return {
        "source_url": "https://example.invalid/data.zip",
        "zip": {"bytes": len(outer), "sha256": sha(outer)},
        "inner_zip": inner_path,
        "files": {name: sha(data) for name, data in members.items()},
    }


def package(members: dict[str, bytes] | None = None) -> tuple[dict[str, Any], bytes]:
    members = GOOD if members is None else members
    outer = make_outer(make_inner(members))
    return make_manifest(members, outer), outer


class FakeResponse(io.BytesIO):
    """A response that works in a ``with`` block and answers ``read(n)``."""


def opener_for(data: bytes, calls: list[str] | None = None) -> Any:
    def opener(url: str) -> FakeResponse:
        if calls is not None:
            calls.append(url)
        return FakeResponse(data)

    return opener


def no_download(url: str) -> Any:
    raise AssertionError(f"the opener was called for {url}")


def write_zip(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "download.zip"
    path.write_bytes(data)
    return path


def nothing_written(tmp_path: Path, dest: Path) -> bool:
    return (not dest.exists() or not any(dest.iterdir())) and not list(tmp_path.rglob("evil*"))


# --------------------------------------------------------------------------- success


def test_from_zip_writes_every_member_by_bare_name(tmp_path: Path) -> None:
    manifest, outer = package()
    dest = tmp_path / "out" / "cmapss"
    paths = fetch.fetch(
        dest, from_zip=write_zip(tmp_path, outer), manifest=manifest, opener=no_download
    )
    assert sorted(p.name for p in paths) == sorted(GOOD)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == GOOD  # and no temporary files


def test_download_goes_through_the_opener_with_the_manifest_url(tmp_path: Path) -> None:
    manifest, outer = package()
    calls: list[str] = []
    dest = tmp_path / "out"
    fetch.fetch(dest, manifest=manifest, opener=opener_for(outer, calls))
    assert calls == ["https://example.invalid/data.zip"]
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == GOOD


def test_a_second_fetch_replaces_changed_files(tmp_path: Path) -> None:
    manifest, outer = package()
    dest = tmp_path / "out"
    fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
    (dest / "readme.txt").write_bytes(b"edited")
    fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
    assert (dest / "readme.txt").read_bytes() == GOOD["readme.txt"]


# --------------------------------------------------------------------------- the zip itself


def test_a_wrong_zip_sha256_is_refused_before_the_zip_is_opened(tmp_path: Path) -> None:
    manifest, outer = package()
    garbage = b"x" * len(outer)  # right size, not a zip: opening it would raise a zip error
    dest = tmp_path / "out"
    with pytest.raises(fetch.FetchError, match=r"zip sha256 .* not opening"):
        fetch.fetch(dest, manifest=manifest, opener=opener_for(garbage))
    assert nothing_written(tmp_path, dest)


def test_a_valid_zip_with_another_sha256_is_refused(tmp_path: Path) -> None:
    manifest, outer = package()
    manifest["zip"]["sha256"] = "0" * 64
    dest = tmp_path / "out"
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch(dest, from_zip=write_zip(tmp_path, outer), manifest=manifest)
    assert nothing_written(tmp_path, dest)


def test_one_changed_byte_in_the_zip_is_refused(tmp_path: Path) -> None:
    manifest, outer = package()
    changed = bytearray(outer)
    changed[len(changed) // 2] ^= 0x01
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(bytes(changed)))


@pytest.mark.parametrize("delta", [-1, 1, 100])
def test_a_wrong_size_is_refused_for_a_download_and_a_local_copy(
    tmp_path: Path, delta: int
) -> None:
    manifest, outer = package()
    data = outer[:-1] if delta == -1 else outer + b"x" * delta
    # The size message says "bytes"; the sha256 message, which would also catch these, does not.
    with pytest.raises(fetch.FetchError, match=r"bytes.*the manifest says"):
        fetch.fetch(tmp_path / "a", manifest=manifest, opener=opener_for(data))
    with pytest.raises(fetch.FetchError, match=r"bytes.*the manifest says"):
        fetch.fetch(tmp_path / "b", from_zip=write_zip(tmp_path, data), manifest=manifest)


def test_a_local_zip_of_the_wrong_size_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, outer = package()
    path = write_zip(tmp_path, outer + b"x")

    def refuse(self: Path) -> bytes:
        raise AssertionError("the file was read")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    with pytest.raises(fetch.FetchError, match=r"bytes.*the manifest says"):
        fetch.fetch(tmp_path / "out", from_zip=path, manifest=manifest)


def test_a_download_error_is_reported(tmp_path: Path) -> None:
    manifest, _ = package()

    def failing(url: str) -> Any:
        raise urllib.error.URLError("no route")

    with pytest.raises(fetch.FetchError, match="download failed"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=failing)


def test_a_missing_local_zip_is_reported(tmp_path: Path) -> None:
    manifest, _ = package()
    with pytest.raises(fetch.FetchError, match="cannot read"):
        fetch.fetch(tmp_path / "out", from_zip=tmp_path / "absent.zip", manifest=manifest)


def test_a_download_larger_than_the_manifest_size_is_not_read_in_full(tmp_path: Path) -> None:
    manifest, outer = package()
    big = outer + b"x" * 10_000
    reads: list[int] = []

    class Spy(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            reads.append(-1 if size is None else size)
            return super().read(size)

    def opener(url: str) -> Spy:
        return Spy(big)

    with pytest.raises(fetch.FetchError):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener)
    assert reads == [len(outer) + 1]


def test_the_inner_zip_must_be_present_exactly_once(tmp_path: Path) -> None:
    members = GOOD
    inner = make_inner(members)
    wrong_place = make_outer(inner, inner_path="elsewhere/CMAPSSData.zip")
    manifest = make_manifest(members, wrong_place)  # still names INNER_PATH
    with pytest.raises(fetch.FetchError, match="expected one"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(wrong_place))


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_two_inner_zips_with_the_same_name_are_refused(tmp_path: Path) -> None:
    outer = make_outer(make_inner(GOOD), extra=1)
    manifest = make_manifest(GOOD, outer)
    with pytest.raises(fetch.FetchError, match="found 2"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(outer))


def test_an_inner_archive_that_is_not_a_zip_is_refused(tmp_path: Path) -> None:
    outer = make_outer(b"this is not a zip")
    manifest = make_manifest(GOOD, outer)
    with pytest.raises(fetch.FetchError, match="not a usable zip"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(outer))


# --------------------------------------------------------------------------- member names


BAD_NAMES = [
    "../evil.txt",
    "..\\evil.txt",
    "sub/evil.txt",
    "sub\\evil.txt",
    "/evil.txt",
    "\\evil.txt",
    "C:evil.txt",
    "C:/evil.txt",
    "C:\\evil.txt",
    ".evil.txt",
    ".",
    "..",
    "evil/",
    "a/../evil.txt",
    "ev:il.txt",
    "evil.txt.",
    "evil.txt ",
    "evil..txt",  # not a traversal, but the script refuses any ".." in a name
]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_zip_slip_and_other_unsafe_names_are_refused_even_if_the_manifest_lists_them(
    tmp_path: Path, bad: str
) -> None:
    members = {**GOOD, bad: b"payload"}  # the manifest lists the bad name too
    outer = make_outer(make_inner(members))
    manifest = make_manifest(members, outer)
    dest = tmp_path / "out" / "cmapss"
    with pytest.raises(fetch.FetchError, match="unsafe name"):
        fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
    assert nothing_written(tmp_path, dest)
    assert list(tmp_path.rglob("evil*")) == []


@pytest.mark.parametrize(
    "good", ["RUL_FD001.txt", "Damage Propagation Modeling.pdf", "a-b_c.1.txt"]
)
def test_plain_names_pass_the_name_check(good: str) -> None:
    fetch.check_member_name(good)


def test_a_member_the_manifest_does_not_list_is_refused(tmp_path: Path) -> None:
    for name in ("sitecustomize.py", "extra.txt"):
        members = {**GOOD, name: b"print('not run')"}
        outer = make_outer(make_inner(members))
        manifest = make_manifest(GOOD, outer)  # lists only the good files
        dest = tmp_path / name
        with pytest.raises(fetch.FetchError, match="not in the manifest"):
            fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
        assert nothing_written(tmp_path, dest)


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_a_repeated_member_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for _ in range(2):
            archive.writestr("readme.txt", GOOD["readme.txt"])
    outer = make_outer(buffer.getvalue())
    manifest = make_manifest(GOOD, outer)
    with pytest.raises(fetch.FetchError, match="appears twice"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(outer))


def test_a_missing_file_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    members = {k: v for k, v in GOOD.items() if k != "RUL_FD001.txt"}
    outer = make_outer(make_inner(members))
    manifest = make_manifest(GOOD, outer)
    dest = tmp_path / "out"
    with pytest.raises(fetch.FetchError, match=r"lacks 1 file.*RUL_FD001\.txt"):
        fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
    assert nothing_written(tmp_path, dest)


def test_a_member_with_the_wrong_sha256_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    changed = {**GOOD, "train_FD001.txt": b"1 1 0.9\r\n"}
    outer = make_outer(make_inner(changed))
    manifest = make_manifest(GOOD, outer)  # the manifest has the hashes of GOOD
    dest = tmp_path / "out"
    with pytest.raises(fetch.FetchError, match=r"train_FD001\.txt: sha256 differs"):
        fetch.fetch(dest, manifest=manifest, opener=opener_for(outer))
    assert nothing_written(tmp_path, dest)


def test_an_oversized_member_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, outer = package()
    monkeypatch.setattr(fetch, "MAX_MEMBER_BYTES", 5)
    with pytest.raises(fetch.FetchError, match="larger than 5 bytes"):
        fetch.fetch(tmp_path / "out", manifest=manifest, opener=opener_for(outer))


def test_write_members_refuses_unsafe_names_on_its_own(tmp_path: Path) -> None:
    for bad in ("../evil.txt", "sub/evil.txt", "C:evil.txt"):
        with pytest.raises(fetch.FetchError, match="unsafe name"):
            fetch.write_members({bad: b"x"}, tmp_path / "out")
    assert list(tmp_path.rglob("evil*")) == []


def test_write_members_refuses_something_already_at_its_temporary_name(tmp_path: Path) -> None:
    # Review of 10 November: the temporary file is created with O_EXCL, so a file or symlink left
    # at that name is refused, not written through, and is left as it was.
    out = tmp_path / "out"
    out.mkdir()
    squatter = out / f".readme.txt.part-{fetch.os.getpid()}"
    squatter.write_bytes(b"not ours")
    with pytest.raises(fetch.FetchError, match="already exists"):
        fetch.write_members({"readme.txt": b"new"}, out)
    assert squatter.read_bytes() == b"not ours"
    assert not (out / "readme.txt").exists()


def test_writing_retries_a_busy_target_on_windows_and_gives_up_after_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    real = fetch.os.replace
    attempts: list[int] = []

    def flaky(src: Any, dst: Any) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("busy")
        real(src, dst)

    monkeypatch.setattr(fetch.os, "replace", flaky)
    fetch.write_members({"readme.txt": b"hi"}, tmp_path / "a")
    assert len(attempts) == 3 and (tmp_path / "a" / "readme.txt").read_bytes() == b"hi"

    attempts.clear()

    def always(src: Any, dst: Any) -> None:
        attempts.append(1)
        raise PermissionError("busy")

    monkeypatch.setattr(fetch.os, "replace", always)
    with pytest.raises(PermissionError):
        fetch.write_members({"readme.txt": b"hi"}, tmp_path / "b")
    assert len(attempts) == 5
    assert [p.name for p in (tmp_path / "b").iterdir()] == []  # the temporary file is removed


# --------------------------------------------------------------------------- the manifest, main


def test_the_real_manifest_is_accepted_by_the_script() -> None:
    manifest = fetch.load_manifest()
    assert len(manifest["files"]) == 14
    assert manifest["source_url"].startswith("https://")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.pop("files"),
        lambda m: m["files"].update(bad="xyz"),
        lambda m: m["zip"].update(sha256="short"),
        lambda m: m["zip"].update(bytes="12"),
        lambda m: m.pop("inner_zip"),
        lambda m: m.update(source_url="http://example.invalid/x.zip"),
        lambda m: m.update(files={}),
    ],
)
def test_a_malformed_manifest_is_refused(tmp_path: Path, mutate: Any) -> None:
    manifest, _ = package()
    mutate(manifest)
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fetch.FetchError):
        fetch.load_manifest(path)


def test_main_reports_a_failure_with_exit_code_1_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, outer = package()
    dest = tmp_path / "out"
    code = fetch.main(["--from-zip", str(write_zip(tmp_path, outer)), "--dest", str(dest)])
    captured = capsys.readouterr()
    assert code == 1  # the real manifest does not describe this made-up zip
    assert captured.err.startswith("error:") and "manifest says" in captured.err
    assert nothing_written(tmp_path, dest)


def test_main_succeeds_with_a_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, outer = package()
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(fetch, "MANIFEST_PATH", manifest_path)
    dest = tmp_path / "out"
    code = fetch.main(["--from-zip", str(write_zip(tmp_path, outer)), "--dest", str(dest)])
    out = capsys.readouterr().out
    assert code == 0
    assert f"wrote {len(GOOD)} files" in out
    assert "simulated by NASA's C-MAPSS model" in out
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == GOOD


def test_default_dest_follows_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(fetch.DATA_ENV, str(tmp_path / "somewhere"))
    assert fetch.default_dest() == tmp_path / "somewhere"
    monkeypatch.delenv(fetch.DATA_ENV)
    assert fetch.default_dest() == SCRIPT.parents[1] / "data" / "cmapss"


def test_the_script_imports_only_the_standard_library() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    assert modules - {"__future__"} <= set(sys.stdlib_module_names)
    assert "closed_loop_v2" not in modules and "numpy" not in modules


def test_the_script_never_runs_code_from_the_archive() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    functions = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    methods = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    assert not functions & {"exec", "eval", "compile", "__import__"}
    assert not methods & {"import_module", "run", "Popen", "system", "extract", "extractall"}
