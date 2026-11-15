"""Verdict rules and their constants (docs/adr/0001-design.md, section 9; 0002, section 4).

The constants live here so that they are inside the code hash; the pre-registration repeats them.
Each rule reads only the ends of an interval. The labels are ``win``, ``loss`` and
``inconclusive``; an interval too wide to decide is inconclusive, not a loss.

* **C2-PERC** (``perc_verdict``): win if the interval for ``Gap_B`` lies wholly above ``+band`` or
  wholly below ``-band``; loss if it lies inside ``[-band, +band]``; inconclusive otherwise.
* **C2-SELECT** (``select_verdict``): win if the lower end of the paired difference (VW minus V1)
  is at least ``margin`` and the lower end of VW's coverage is at least ``floor``; loss if the upper
  end of the difference is below ``margin`` or the upper end of VW's coverage is below ``floor``;
  inconclusive otherwise. The row also depends on how VW's fixed-point steps ended (``vw_state``,
  review D, N3): *settled* (the last step did not move the quantiles) gives the rule above;
  *two-cycle* (the steps alternate between two pairs, so the last pair depends only on the parity
  of the step count) applies the rule with each pair and keeps the label only if both agree,
  inconclusive otherwise (``select_claim_verdict``); *unsettled* (still moving, not in a two-cycle)
  is inconclusive whatever the intervals say.
* **C2-EXT** (``ext_verdict``): win if the interval lies inside ``[lo, hi]``; loss if it has no
  point in common with ``[lo, hi]``; inconclusive otherwise.

**An empty interval** (``low > high``) gives ``inconclusive`` in every rule, before any other test.
The betting interval (ADR 0004) can come back empty, with probability at most alpha when the values
share one mean; an empty set says nothing about where the quantity lies, so it cannot be a win or a
loss (review of packet C, finding N1, written for v0.1's running intersection, which had the same
property).

The secondary rows (C2-CENSOR, C2-SHIFT, and the stratum version of C2-SELECT) have no rule in 0001
or 0002: they are reported as descriptive intervals at ``SECONDARY_ALPHA`` and carry no label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from closed_loop_v2.stats import Interval

__all__ = [
    "EXT_HI",
    "EXT_LO",
    "N_PRIMARY",
    "PERC_BAND",
    "PRIMARY_ALPHA",
    "SECONDARY_ALPHA",
    "SELECT_FLOOR",
    "SELECT_MARGIN",
    "VW_SETTLED",
    "VW_STATES",
    "VW_TWO_CYCLE",
    "VW_UNSETTLED",
    "Label",
    "ext_verdict",
    "perc_verdict",
    "select_claim_verdict",
    "select_verdict",
    "verdict_of_claim",
    "vw_state",
]

Label = Literal["win", "loss", "inconclusive"]

PERC_BAND: Final[float] = 0.03
SELECT_MARGIN: Final[float] = 0.02
SELECT_FLOOR: Final[float] = 0.92
EXT_LO: Final[float] = 0.92
EXT_HI: Final[float] = 0.98
# four C2-PERC verdicts, C2-SELECT and C2-EXT
N_PRIMARY: Final[int] = 6
PRIMARY_ALPHA: Final[float] = 0.05 / N_PRIMARY
SECONDARY_ALPHA: Final[float] = 0.05
VW_SETTLED: Final[str] = "settled"
VW_TWO_CYCLE: Final[str] = "two-cycle"
VW_UNSETTLED: Final[str] = "unsettled"
VW_STATES: Final[tuple[str, ...]] = (VW_SETTLED, VW_TWO_CYCLE, VW_UNSETTLED)


def _empty(*intervals: Interval) -> bool:
    """True if any of ``intervals`` is empty (``low > high``)."""
    return any(iv.low > iv.high for iv in intervals)


def perc_verdict(gap: Interval, band: float = PERC_BAND) -> Label:
    """C2-PERC on the interval for ``Gap_B`` (closed minus open lower-bound coverage)."""
    if _empty(gap):
        return "inconclusive"
    if gap.low > band or gap.high < -band:
        return "win"
    if -band <= gap.low and gap.high <= band:
        return "loss"
    return "inconclusive"


def vw_state(history: Sequence[object]) -> str:
    """How VW's fixed-point steps ended, from its step history (V1's quantiles first).

    ``settled`` if the last step did not move the quantiles (or there was no step); ``two-cycle``
    if the last pair equals the one two steps back but not the one just before, which, since each
    step is the same deterministic map, repeats for ever; ``unsettled`` otherwise.
    """
    if len(history) < 2 or history[-1] == history[-2]:
        return VW_SETTLED
    if len(history) >= 3 and history[-1] == history[-3]:
        return VW_TWO_CYCLE
    return VW_UNSETTLED


def select_verdict(
    diff: Interval,
    vw: Interval,
    margin: float = SELECT_MARGIN,
    floor: float = SELECT_FLOOR,
) -> Label:
    """C2-SELECT on the paired difference VW minus V1 and on VW's own closed-loop coverage."""
    if _empty(diff, vw):
        return "inconclusive"
    if diff.low >= margin and vw.low >= floor:
        return "win"
    if diff.high < margin or vw.high < floor:
        return "loss"
    return "inconclusive"


def select_claim_verdict(state: str, rows: Sequence[tuple[Interval, Interval]]) -> Label:
    """C2-SELECT's label given VW's end state and one ``(diff, vw)`` row per VW pair used.

    ``settled`` takes one row, ``two-cycle`` two (the last pair and the one before it), and
    ``unsettled`` is inconclusive. With two rows the label is kept only if both give it.
    """
    if state not in VW_STATES:
        raise ValueError(f"vw_state must be one of {VW_STATES}, got {state!r}")
    if state == VW_UNSETTLED:
        return "inconclusive"
    want = 1 if state == VW_SETTLED else 2
    if len(rows) != want:
        raise ValueError(f"a {state} VW needs {want} row(s), got {len(rows)}")
    labels = {select_verdict(diff, vw) for diff, vw in rows}
    return labels.pop() if len(labels) == 1 else "inconclusive"


def ext_verdict(ci: Interval, lo: float = EXT_LO, hi: float = EXT_HI) -> Label:
    """C2-EXT on the Clopper-Pearson interval for V1's lower-bound coverage on NASA's test units."""
    if _empty(ci):
        return "inconclusive"
    if lo <= ci.low and ci.high <= hi:
        return "win"
    if ci.high < lo or ci.low > hi:
        return "loss"
    return "inconclusive"


def verdict_of_claim(claim: Mapping[str, Any]) -> Label | None:
    """The label the rules give a claim row of ``results.json``, from its stored intervals.

    ``--check`` compares this with the stored label, so an edited label or interval is caught. A
    secondary row (no rule), including the stratum version of C2-SELECT, gives ``None``.
    """
    ivs = claim.get("intervals", {})
    name = claim.get("claim")
    if not claim.get("primary"):
        return None
    if name == "C2-PERC":
        return perc_verdict(Interval.from_dict(ivs["gap"]))
    if name == "C2-SELECT":
        state = claim["vw_state"]  # required: a row without it cannot be rebuilt
        rows = [(Interval.from_dict(ivs["diff"]), Interval.from_dict(ivs["vw"]))]
        if state == VW_TWO_CYCLE:
            rows.append((Interval.from_dict(ivs["diff_alt"]), Interval.from_dict(ivs["vw_alt"])))
        return select_claim_verdict(state, rows)
    if name == "C2-EXT":
        return ext_verdict(Interval.from_dict(ivs["coverage_lo"]))
    raise ValueError(f"unknown primary claim {name!r}")
