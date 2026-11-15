"""The bound rule."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from closed_loop_v2.policy import BoundRule


def test_first_index_at_or_below_h() -> None:
    lo = np.array([30.0, 20.0, 10.0, 5.0, 2.0])
    assert BoundRule(10.0).first_trigger(lo) == 2  # equal to h counts
    assert BoundRule(9.99).first_trigger(lo) == 3
    assert BoundRule(100.0).first_trigger(lo) == 0
    assert BoundRule(1.0).first_trigger(lo) is None


def test_first_not_minimum_when_bound_rebounds() -> None:
    lo = np.array([50.0, 8.0, 40.0, 3.0])
    assert BoundRule(10.0).first_trigger(lo) == 1


def test_returns_plain_int() -> None:
    out = BoundRule(1.0).first_trigger(np.array([0.0]))
    assert type(out) is int


def test_minus_infinity_never_fires_on_finite_bounds() -> None:
    assert BoundRule(-math.inf).first_trigger(np.array([-1e300, 0.0, 5.0])) is None


def test_minus_infinite_bound_fires_for_every_h() -> None:
    lo = np.array([4.0, -math.inf])
    assert BoundRule(-math.inf).first_trigger(lo) == 1
    assert BoundRule(0.0).first_trigger(lo) == 1


def test_nan_bound_never_fires_and_empty_input_is_none() -> None:
    assert BoundRule(1e9).first_trigger(np.array([math.nan, math.nan])) is None
    assert BoundRule(1e9).first_trigger(np.array([])) is None


def test_nan_h_rejected_and_rule_is_frozen() -> None:
    with pytest.raises(ValueError):
        BoundRule(math.nan)
    rule = BoundRule(3.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.h = 4.0  # type: ignore[misc]


def test_ignores_everything_but_the_bounds() -> None:
    lo = np.array([9.0, 7.0, 5.0])
    assert BoundRule(7.0).first_trigger(lo) == BoundRule(7.0).first_trigger(lo.copy())
