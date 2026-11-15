"""``Trajectory``'s capped truth (ADR 0001, amendment of 10 November 2026)."""

from __future__ import annotations

import numpy as np
import pytest

from closed_loop_v2.types import Trajectory


def unit(n: int, rul_at_end: int = 0) -> Trajectory:
    return Trajectory(
        "FD001", 1, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), rul_at_end
    )


def test_capped_caps_rul_and_keeps_the_uncapped_life() -> None:
    t = unit(6, rul_at_end=2)  # remaining life 7, 6, 5, 4, 3, 2
    c = t.capped(4)
    assert c.rul().tolist() == [4, 4, 4, 4, 3, 2]
    assert c.rul_uncapped().tolist() == [7, 6, 5, 4, 3, 2]
    assert t.rul().tolist() == [7, 6, 5, 4, 3, 2] and t.rul_cap is None
    assert c.rul().dtype == np.int64
    assert c.cycles is t.cycles and c.failure_cycle == t.failure_cycle


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_a_bad_cap_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="rul_cap"):
        unit(3).capped(bad)  # type: ignore[arg-type]
