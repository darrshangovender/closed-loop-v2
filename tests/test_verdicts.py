"""Each verdict rule at and just past each of its boundaries, on an empty interval, and the rebuild
from stored intervals."""

from __future__ import annotations

from typing import Any

import pytest

from closed_loop_v2 import verdicts
from closed_loop_v2.stats import Interval
from closed_loop_v2.verdicts import (
    ext_verdict,
    perc_verdict,
    select_claim_verdict,
    select_verdict,
    verdict_of_claim,
    vw_state,
)

EPS = 1e-9


def iv(low: float, high: float) -> Interval:
    return Interval(low, high, "t", 300, (low + high) / 2, 0.05 / 6)


def test_the_constants_are_the_registered_ones() -> None:
    assert verdicts.PERC_BAND == 0.03
    assert (verdicts.SELECT_MARGIN, verdicts.SELECT_FLOOR) == (0.02, 0.92)
    assert (verdicts.EXT_LO, verdicts.EXT_HI) == (0.92, 0.98)
    assert verdicts.PRIMARY_ALPHA == 0.05 / 6 and verdicts.SECONDARY_ALPHA == 0.05


# ----------------------------------------------------------------------------- C2-PERC


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [
        (0.03 + EPS, 0.2, "win"),  # just past +band
        (0.03, 0.2, "inconclusive"),  # at +band: not wholly above it
        (-0.2, -0.03 - EPS, "win"),  # just past -band
        (-0.2, -0.03, "inconclusive"),
        (-0.03, 0.03, "loss"),  # the band itself
        (-0.03 - EPS, 0.03, "inconclusive"),
        (-0.03, 0.03 + EPS, "inconclusive"),
        (-0.01, 0.01, "loss"),
        (-0.5, 0.5, "inconclusive"),  # too wide to decide: not a loss
    ],
)
def test_perc_edges(low: float, high: float, label: str) -> None:
    assert perc_verdict(iv(low, high)) == label


# ----------------------------------------------------------------------------- C2-SELECT


@pytest.mark.parametrize(
    ("diff", "vw", "label"),
    [
        ((0.02, 0.1), (0.92, 0.99), "win"),  # both lower ends exactly at their thresholds
        ((0.02 - EPS, 0.1), (0.92, 0.99), "inconclusive"),
        ((0.02, 0.1), (0.92 - EPS, 0.99), "inconclusive"),
        ((-0.1, 0.02 - EPS), (0.9, 0.99), "loss"),  # difference wholly below the margin
        ((-0.1, 0.02), (0.9, 0.99), "inconclusive"),  # its upper end at the margin
        ((0.0, 0.1), (0.8, 0.92 - EPS), "loss"),  # VW wholly below the floor
        ((0.0, 0.1), (0.8, 0.92), "inconclusive"),
        ((0.05, 0.1), (0.8, 0.92 - EPS), "loss"),  # a clear difference does not save a low VW
    ],
)
def test_select_edges(diff: tuple[float, float], vw: tuple[float, float], label: str) -> None:
    assert select_verdict(iv(*diff), iv(*vw)) == label


A, B, C = (-2.0, 2.0), (-3.0, 1.5), (-2.5, 1.8)


@pytest.mark.parametrize(
    ("history", "state"),
    [
        ([A], "settled"),  # no step
        ([A, A], "settled"),
        ([A, B, B], "settled"),
        ([A, B, A], "two-cycle"),
        ([A, B, C, B], "two-cycle"),
        ([A, B], "unsettled"),  # one step that moved: nothing to compare two steps back
        ([A, B, C], "unsettled"),
        ([A, B, A, C], "unsettled"),
    ],
)
def test_vw_state_reads_how_the_steps_ended(history: list[tuple[float, float]], state: str) -> None:
    assert vw_state(history) == state


WIN = (iv(0.05, 0.1), iv(0.95, 0.99))
LOSS = (iv(-0.1, 0.0), iv(0.8, 0.9))
UNSURE = (iv(0.0, 0.1), iv(0.9, 0.99))


@pytest.mark.parametrize(
    ("state", "rows", "label"),
    [
        ("settled", [WIN], "win"),
        ("settled", [LOSS], "loss"),
        ("settled", [UNSURE], "inconclusive"),
        ("two-cycle", [WIN, WIN], "win"),
        ("two-cycle", [LOSS, LOSS], "loss"),
        ("two-cycle", [WIN, LOSS], "inconclusive"),  # the pairs disagree
        ("two-cycle", [LOSS, WIN], "inconclusive"),
        ("two-cycle", [WIN, UNSURE], "inconclusive"),
        ("two-cycle", [UNSURE, UNSURE], "inconclusive"),
        ("unsettled", [WIN], "inconclusive"),  # review D, N3: whatever the intervals say
        ("unsettled", [LOSS], "inconclusive"),
        ("unsettled", [], "inconclusive"),
    ],
)
def test_select_claim_verdict_combines_the_vw_pairs(
    state: str, rows: list[tuple[Interval, Interval]], label: str
) -> None:
    assert select_claim_verdict(state, rows) == label


def test_select_claim_verdict_refuses_rows_that_do_not_fit_the_state() -> None:
    with pytest.raises(ValueError, match="needs 2 row"):
        select_claim_verdict("two-cycle", [WIN])
    with pytest.raises(ValueError, match="needs 1 row"):
        select_claim_verdict("settled", [WIN, WIN])
    with pytest.raises(ValueError, match="vw_state must be one of"):
        select_claim_verdict("moved", [WIN])


# ----------------------------------------------------------------------------- C2-EXT


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [
        (0.92, 0.98, "win"),  # exactly the range
        (0.92 - EPS, 0.98, "inconclusive"),
        (0.92, 0.98 + EPS, "inconclusive"),
        (0.80, 0.92 - EPS, "loss"),  # wholly below
        (0.80, 0.92, "inconclusive"),  # touches the range at one point
        (0.98 + EPS, 1.0, "loss"),  # wholly above
        (0.98, 1.0, "inconclusive"),
        (0.85, 0.99, "inconclusive"),
    ],
)
def test_ext_edges(low: float, high: float, label: str) -> None:
    assert ext_verdict(iv(low, high)) == label


# ----------------------------------------------------------------------------- rebuild


def test_verdict_of_claim_rebuilds_every_primary_rule() -> None:
    perc = {"claim": "C2-PERC", "primary": True, "intervals": {"gap": iv(0.05, 0.1).to_dict()}}
    sel = {
        "claim": "C2-SELECT",
        "primary": True,
        "intervals": {"diff": iv(-0.1, 0.0).to_dict(), "vw": iv(0.9, 0.99).to_dict()},
        "vw_state": "settled",
    }
    ext = {
        "claim": "C2-EXT",
        "primary": True,
        "intervals": {"coverage_lo": iv(0.93, 0.97).to_dict()},
    }
    sec = {"claim": "C2-SHIFT", "primary": False, "intervals": {"diff": iv(0, 1).to_dict()}}
    assert [verdict_of_claim(c) for c in (perc, sel, ext, sec)] == ["win", "loss", "win", None]
    with pytest.raises(ValueError, match="unknown"):
        verdict_of_claim({"claim": "C9", "primary": True, "intervals": {}})
    win_ivs = {"diff": iv(0.05, 0.1).to_dict(), "vw": iv(0.95, 0.99).to_dict()}
    win = {**sel, "intervals": win_ivs}
    assert verdict_of_claim(win) == "win"
    assert verdict_of_claim({**win, "vw_state": "unsettled"}) == "inconclusive"
    # a two-cycle reads the other pair's intervals too and keeps the label only if both give it
    both = {**win_ivs, "diff_alt": iv(0.04, 0.2).to_dict(), "vw_alt": iv(0.93, 0.99).to_dict()}
    assert verdict_of_claim({**win, "vw_state": "two-cycle", "intervals": both}) == "win"
    split = {**win_ivs, "diff_alt": iv(-0.1, 0.0).to_dict(), "vw_alt": iv(0.8, 0.9).to_dict()}
    assert verdict_of_claim({**win, "vw_state": "two-cycle", "intervals": split}) == "inconclusive"
    with pytest.raises(KeyError):  # a two-cycle row without the other pair cannot be rebuilt
        verdict_of_claim({**win, "vw_state": "two-cycle"})
    with pytest.raises(KeyError):
        verdict_of_claim({k: v for k, v in win.items() if k != "vw_state"})
    with pytest.raises(ValueError, match="vw_state"):
        verdict_of_claim({**win, "vw_state": "yes"})


def test_the_stratum_row_of_select_is_secondary_and_has_no_label() -> None:
    """C2-SELECT's stratum version carries only a difference; as a secondary row it gives None."""
    row = {
        "claim": "C2-SELECT",
        "row": "VW minus V1, closed-loop stratum c^lo",
        "primary": False,
        "intervals": {"diff": iv(0.05, 0.1).to_dict()},
    }
    assert verdict_of_claim(row) is None


# ----------------------------------------------------------------------------- empty intervals


@pytest.mark.parametrize(
    ("rule", "intervals"),
    [
        (perc_verdict, [iv(0.06, 0.01)]),  # empty above +band: was a win
        (perc_verdict, [iv(-0.01, -0.06)]),  # empty below -band: was a win
        (perc_verdict, [iv(0.02, -0.02)]),  # empty inside the band: was a loss
        (select_verdict, [iv(0.03, 0.01), iv(0.95, 0.99)]),  # empty difference
        (select_verdict, [iv(0.03, 0.1), iv(0.95, 0.93)]),  # empty VW interval
        (select_verdict, [iv(0.01, -0.01), iv(0.95, 0.99)]),  # empty below the margin: was a loss
        (ext_verdict, [iv(0.95, 0.94)]),  # empty inside the range: was a win
        (ext_verdict, [iv(0.90, 0.85)]),  # empty below the range: was a loss
    ],
)
def test_an_empty_interval_is_inconclusive(rule: Any, intervals: list[Interval]) -> None:
    assert rule(*intervals) == "inconclusive"


def test_an_empty_interval_is_inconclusive_when_rebuilt() -> None:
    claim = {"claim": "C2-PERC", "primary": True, "intervals": {"gap": iv(0.06, 0.01).to_dict()}}
    assert verdict_of_claim(claim) == "inconclusive"
