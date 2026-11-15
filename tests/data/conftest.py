"""Fixtures for the data tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from synthetic_cmapss import SYNTH_UNITS, Synthetic, make_dataset

from closed_loop_v2.data import load, paths
from closed_loop_v2.data.roles import UNITS_PER_FILE


@pytest.fixture(autouse=True)
def _fresh_cache(request: pytest.FixtureRequest) -> Iterator[None]:
    """Synthetic-file tests start and end with an empty loader cache.

    Tests on the real files share one parse (about 4 s) and clear the cache themselves when they
    need a read to happen.
    """
    if request.node.get_closest_marker("data") is not None:
        yield
        return
    load.clear_cache()
    yield
    load.clear_cache()


@pytest.fixture
def synth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    """Synthetic files for FD001 and FD002, wired in as the data directory and the manifest.

    The unit counts in ``UNITS_PER_FILE`` and the manifest are patched for this test only.
    """
    dataset = make_dataset(tmp_path / "cmapss")
    monkeypatch.setattr(paths, "manifest", dataset.manifest)
    monkeypatch.setenv(paths.DATA_ENV, str(dataset.directory))
    for subset, (n_train, n_test) in SYNTH_UNITS.items():
        monkeypatch.setitem(UNITS_PER_FILE["train"], subset, n_train)
        monkeypatch.setitem(UNITS_PER_FILE["test"], subset, n_test)
    return dataset
