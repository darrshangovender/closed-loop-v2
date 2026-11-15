"""Errors raised by the data layer."""

from __future__ import annotations


class DataError(Exception):
    """A data file is missing, changed, malformed or inconsistent with the dataset's readme."""


class HeldoutError(Exception):
    """Held-out data was asked for before the freeze, or through a development path."""
