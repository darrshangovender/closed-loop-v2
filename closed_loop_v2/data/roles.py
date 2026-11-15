"""Who is held out, decided by subset and unit number alone (docs/adr/0001-design.md, section 4).

The role comes from a keyed hash of the subset name and the unit number. It does not depend on
any row of the data, so it can be computed, and the counts below checked, before a row is parsed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from closed_loop_v2.types import SUBSETS, Role

__all__ = [
    "CALIB_BELOW",
    "DEV_BELOW",
    "ROLE_COUNTS",
    "UNITS_PER_FILE",
    "check_subsets",
    "role_draw",
    "role_of",
]

DEV_BELOW = 0.30
CALIB_BELOW = 0.50

# Roles of the units in the training files. These are the counts of 0001, section 4, except FD004
# dev: 0001 counted 248 units from the readme, but the file holds 249 (see UNITS_PER_FILE) and
# unit 249 is a dev unit, so dev is 71 and the eval and calib counts are unchanged.
# tests/data/test_data_roles.py recomputes both tables from unit numbers.
ROLE_COUNTS: dict[str, dict[Role, int]] = {
    "FD001": {"dev": 25, "calib": 21, "eval": 54},
    "FD002": {"dev": 64, "calib": 59, "eval": 137},
    "FD003": {"dev": 35, "calib": 16, "eval": 49},
    "FD004": {"dev": 71, "calib": 45, "eval": 133},
}

# Units per file. They are the "Train" and "Test" trajectory counts in the dataset's readme.txt,
# except for FD004: the readme gives 248 train and 249 test, but train_FD004.txt holds units
# 1..249 (the loader's unit-column check found unit 249), so the readme's FD004 pair looks swapped.
# The FD004 test count of 248 follows from that and is not checked against the test files, which
# are held out; HeldoutAccess.test_units checks every count at the first held-out read.
UNITS_PER_FILE: dict[str, dict[str, int]] = {
    "train": {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249},
    "test": {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248},
}


def check_subsets(subsets: Iterable[str]) -> tuple[str, ...]:
    """The subsets as a tuple, or ``ValueError`` if one is unknown, repeated, or none is given."""
    chosen = tuple(subsets)
    if not chosen:
        raise ValueError("no subsets given")
    unknown = [s for s in chosen if s not in SUBSETS]
    if unknown:
        raise ValueError(f"unknown subset(s) {unknown}; choose from {list(SUBSETS)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"subsets repeated in {list(chosen)}")
    return chosen


def role_draw(subset: str, unit: int) -> float:
    """The unit's keyed draw in [0, 1): the first 8 bytes of a SHA-256, big-endian, over 2**64."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")
    digest = hashlib.sha256(f"closed-loop-v2/roles/{subset}/{unit}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def role_of(subset: str, unit: int) -> Role:
    """``"dev"`` below 0.30, ``"calib"`` below 0.50, otherwise ``"eval"``."""
    u = role_draw(subset, unit)
    if u < DEV_BELOW:
        return "dev"
    if u < CALIB_BELOW:
        return "calib"
    return "eval"
