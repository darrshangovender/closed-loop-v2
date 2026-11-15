"""Shared pytest setup: tests marked ``data`` are skipped when the C-MAPSS files are absent."""

from __future__ import annotations

import pytest

from closed_loop_v2.data.paths import DATA_ENV, data_available


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if data_available():
        return
    skip = pytest.mark.skip(reason=f"C-MAPSS files not found ({DATA_ENV} or data/cmapss)")
    for item in items:
        if item.get_closest_marker("data") is not None:
            item.add_marker(skip)
