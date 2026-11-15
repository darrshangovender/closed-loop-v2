"""The manifest, the data directory and the checksum checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from synthetic_cmapss import Synthetic

from closed_loop_v2.data import paths
from closed_loop_v2.data.errors import DataError

EXPECTED_FILES = {
    "Damage Propagation Modeling.pdf",
    "readme.txt",
    *(f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)),
}


def test_the_manifest_lists_the_14_files_and_the_zip() -> None:
    manifest = json.loads(paths.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["files"]) == EXPECTED_FILES
    assert len(manifest["files"]) == 14
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in manifest["files"].values())
    assert manifest["zip"]["bytes"] == 12429152
    assert manifest["zip"]["sha256"] == (
        "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
    )
    assert re.fullmatch(r"[0-9a-f]{32}", manifest["zip"]["md5"])
    assert manifest["inner_zip"].endswith("CMAPSSData.zip")
    assert manifest["source_url"].startswith("https://")
    assert paths.member_names() == tuple(manifest["files"])


def test_the_manifest_is_valid_for_the_loader() -> None:
    assert paths.expected_sha256("readme.txt") == (
        "4f5270554b775c67e73aff383c5436fd329d6e4cc3d3a116913276fae511269b"
    )
    with pytest.raises(DataError, match=r"notes\.txt"):
        paths.expected_sha256("notes.txt")


def test_data_dir_reads_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_ENV, "somewhere")
    assert paths.data_dir() == Path("somewhere")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_data_dir_default_is_data_cmapss_in_the_repository(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(paths.DATA_ENV, raising=False)
    else:
        monkeypatch.setenv(paths.DATA_ENV, value)
    repo = paths.MANIFEST_PATH.parents[1]
    assert paths.data_dir() == repo / "data" / "cmapss"
    assert (repo / "closed_loop_v2").is_dir()


def test_data_available_needs_every_file(synth: Synthetic) -> None:
    assert paths.data_available()
    (synth.directory / "RUL_FD002.txt").unlink()
    assert not paths.data_available()


def test_data_available_is_false_for_a_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "nowhere"))
    assert not paths.data_available()


def test_verify_accepts_intact_files(synth: Synthetic) -> None:
    paths.verify(synth.directory)


def test_verify_catches_one_changed_byte_and_names_the_file(synth: Synthetic) -> None:
    data = bytearray(synth.read("train_FD001.txt"))
    data[100] ^= 0x01
    synth.write("train_FD001.txt", bytes(data), rehash=False)
    with pytest.raises(DataError, match=r"train_FD001\.txt"):
        paths.verify(synth.directory)


def test_verify_names_a_missing_file(synth: Synthetic) -> None:
    (synth.directory / "test_FD002.txt").unlink()
    with pytest.raises(DataError, match=r"test_FD002\.txt: missing"):
        paths.verify(synth.directory)


def test_verify_names_every_problem(synth: Synthetic) -> None:
    (synth.directory / "test_FD002.txt").unlink()
    synth.write("RUL_FD001.txt", b"1\r\n", rehash=False)
    with pytest.raises(DataError) as info:
        paths.verify(synth.directory)
    assert "test_FD002.txt" in str(info.value)
    assert "RUL_FD001.txt" in str(info.value)


def test_verify_with_names_checks_only_those(synth: Synthetic) -> None:
    synth.write("test_FD001.txt", b"changed\r\n", rehash=False)
    paths.verify(synth.directory, ["train_FD001.txt", "train_FD002.txt"])
    with pytest.raises(DataError, match=r"test_FD001\.txt"):
        paths.verify(synth.directory)


def test_verify_refuses_a_missing_directory(tmp_path: Path, synth: Synthetic) -> None:
    with pytest.raises(DataError, match="does not exist"):
        paths.verify(tmp_path / "nowhere")


def test_read_verified_returns_the_checked_bytes(synth: Synthetic) -> None:
    assert paths.read_verified(synth.directory, "readme.txt") == synth.read("readme.txt")


def test_read_verified_refuses_a_changed_file(synth: Synthetic) -> None:
    synth.write("readme.txt", b"edited\r\n", rehash=False)
    with pytest.raises(DataError, match=r"readme\.txt"):
        paths.read_verified(synth.directory, "readme.txt")


def test_read_verified_refuses_a_missing_file(synth: Synthetic) -> None:
    (synth.directory / "readme.txt").unlink()
    with pytest.raises(DataError, match=r"readme\.txt: missing"):
        paths.read_verified(synth.directory, "readme.txt")


def test_read_verified_refuses_a_name_the_manifest_does_not_list(synth: Synthetic) -> None:
    synth.write("extra.txt", b"x", rehash=False)
    with pytest.raises(DataError, match=r"extra\.txt"):
        paths.read_verified(synth.directory, "extra.txt")


@pytest.mark.data
def test_verify_the_real_raw_folder_hashes_only() -> None:
    """The one test that reads the held-out files, and only to hash them."""
    paths.verify(paths.data_dir())
