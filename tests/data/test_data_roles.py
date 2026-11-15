"""Roles come from the subset and unit number alone, and the counts match 0001, section 4."""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from closed_loop_v2.data import roles
from closed_loop_v2.data.roles import ROLE_COUNTS, UNITS_PER_FILE, check_subsets, role_of
from closed_loop_v2.types import SUBSETS

# The table in docs/adr/0001-design.md, section 4, which counted units from the readme: 100, 260,
# 100 and 248 training units.
ADR_TABLE = {
    "FD001": {"dev": 25, "calib": 21, "eval": 54},
    "FD002": {"dev": 64, "calib": 59, "eval": 137},
    "FD003": {"dev": 35, "calib": 16, "eval": 49},
    "FD004": {"dev": 70, "calib": 45, "eval": 133},
}
README_TRAIN_UNITS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 248}


def _count(subset: str, n_units: int) -> dict[str, int]:
    counts = Counter(role_of(subset, u) for u in range(1, n_units + 1))
    return {role: counts[role] for role in ("dev", "calib", "eval")}


def _by_hand(subset: str, unit: int) -> str:
    """The formula of 0001, section 4, written out again in the test."""
    digest = hashlib.sha256(f"closed-loop-v2/roles/{subset}/{unit}".encode()).digest()
    u = int.from_bytes(digest[:8], byteorder="big") / 2**64
    return "dev" if u < 0.30 else "calib" if u < 0.50 else "eval"


@pytest.mark.parametrize("subset", SUBSETS)
def test_adr_table_is_reproduced_from_unit_numbers(subset: str) -> None:
    assert _count(subset, README_TRAIN_UNITS[subset]) == ADR_TABLE[subset]


def test_adr_totals() -> None:
    totals = Counter()
    for subset in SUBSETS:
        totals.update(_count(subset, README_TRAIN_UNITS[subset]))
    assert dict(totals) == {"dev": 194, "calib": 141, "eval": 373}


@pytest.mark.parametrize("subset", SUBSETS)
def test_role_counts_match_the_units_per_file(subset: str) -> None:
    n = UNITS_PER_FILE["train"][subset]
    assert _count(subset, n) == ROLE_COUNTS[subset]
    assert sum(ROLE_COUNTS[subset].values()) == n


def test_fd004_has_one_more_unit_than_the_readme_says_and_it_is_a_dev_unit() -> None:
    # train_FD004.txt holds units 1..249; the readme says 248. The eval count is unchanged.
    assert UNITS_PER_FILE["train"]["FD004"] == README_TRAIN_UNITS["FD004"] + 1
    assert role_of("FD004", 249) == "dev"
    assert ROLE_COUNTS["FD004"]["dev"] == ADR_TABLE["FD004"]["dev"] + 1
    assert ROLE_COUNTS["FD004"]["eval"] == ADR_TABLE["FD004"]["eval"]
    assert ROLE_COUNTS["FD004"]["calib"] == ADR_TABLE["FD004"]["calib"]


def test_units_per_file_values() -> None:
    assert UNITS_PER_FILE["train"] == {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}
    assert UNITS_PER_FILE["test"] == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}


def test_role_of_equals_the_formula_written_out_again() -> None:
    for subset in SUBSETS:
        for unit in range(1, 300):
            assert role_of(subset, unit) == _by_hand(subset, unit)


def test_subset_is_part_of_the_key() -> None:
    # The same unit number gets different draws in different subsets.
    draws = {roles.role_draw(s, 7) for s in SUBSETS}
    assert len(draws) == len(SUBSETS)


def test_draw_is_first_eight_bytes_big_endian_over_two_to_the_64() -> None:
    digest = hashlib.sha256(b"closed-loop-v2/roles/FD001/1").digest()
    assert roles.role_draw("FD001", 1) == int.from_bytes(digest[:8], "big") / 2**64
    assert roles.role_draw("FD001", 1) != int.from_bytes(digest[:8], "little") / 2**64


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (0.0, "dev"),
        (0.2999999, "dev"),
        (0.30, "calib"),
        (0.4999999, "calib"),
        (0.50, "eval"),
        (0.999999, "eval"),
    ],
)
def test_thresholds_are_strict_less_than(
    monkeypatch: pytest.MonkeyPatch, draw: float, expected: str
) -> None:
    monkeypatch.setattr(roles, "role_draw", lambda subset, unit: draw)
    assert role_of("FD001", 1) == expected


def test_unknown_subset_is_refused() -> None:
    with pytest.raises(ValueError, match="FD005"):
        role_of("FD005", 1)


def test_check_subsets() -> None:
    assert check_subsets(["FD002", "FD001"]) == ("FD002", "FD001")
    with pytest.raises(ValueError, match="no subsets"):
        check_subsets([])
    with pytest.raises(ValueError, match="FD9"):
        check_subsets(["FD001", "FD9"])
    with pytest.raises(ValueError, match="repeated"):
        check_subsets(["FD001", "FD001"])
