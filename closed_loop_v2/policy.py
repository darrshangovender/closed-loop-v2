"""The replacement policy (docs/adr/0001-design.md, sections 6 and 7).

The bound rule replaces a unit at the first checkpoint where the lower bound on its remaining life
is at most ``h`` cycles. It looks at the bounds only, never at the remaining life itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class TriggerRule(Protocol):
    """Anything that can name the first checkpoint at which a unit is replaced."""

    def first_trigger(self, lo: NDArray[np.float64]) -> int | None:
        """Index of the first row at which the rule fires, or ``None`` if it never does."""
        ...


@dataclass(frozen=True)
class BoundRule:
    """Replace at the first row whose lower bound is at most ``h`` cycles.

    ``h = -inf`` never fires on a finite lower bound, which is how the tests switch the policy off.
    A lower bound of ``-inf`` (a calibration set too small for a finite quantile) fires for every
    ``h``, ``-inf`` included.
    """

    h: float

    def __post_init__(self) -> None:
        if math.isnan(self.h):
            raise ValueError("h must not be NaN")

    def first_trigger(self, lo: NDArray[np.float64]) -> int | None:
        """First index with ``lo <= h``; ``None`` when there is none. A NaN bound never fires."""
        hit = np.asarray(lo, dtype=np.float64) <= self.h
        if not hit.any():
            return None
        return int(np.argmax(hit))
