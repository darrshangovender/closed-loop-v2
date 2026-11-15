"""Types shared by every layer (docs/adr/0002-interfaces.md, section 1).

C-MAPSS trajectories are simulated by NASA's C-MAPSS engine model; they are not field data.

**The truth is capped** (ADR 0001, amendment of 10 November 2026). ``Trajectory.capped(cap)``
returns a copy whose ``rul()`` is ``min(remaining life, cap)``: the quantity the estimator, the
conformal scores, the labels and every coverage metric use. It is the one place the cap is applied
to the truth, so that every consumer agrees. ``rul_uncapped()`` keeps the plain remaining life for
the policy's cost (the remaining life at replacement).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
Role = Literal["dev", "calib", "eval"]
Source = Literal["train", "test"]
N_SETTINGS = 3
N_SENSORS = 21


@dataclass(frozen=True, eq=False)
class Trajectory:
    """One engine's rows, in cycle order.

    ``rul_at_end`` is the remaining life after the last row: 0 for a training unit, whose last
    row is its failure cycle, and NASA's published value for a test unit. The true remaining life
    at row ``i`` is ``failure_cycle - cycles[i]``. ``rul_cap`` (``None`` or a positive whole
    number) caps what ``rul()`` returns; set it with ``capped``.
    """

    subset: str
    unit: int
    source: Source
    cycles: NDArray[np.int64]
    settings: NDArray[np.float64]
    sensors: NDArray[np.float64]
    rul_at_end: int
    rul_cap: int | None = None

    def __post_init__(self) -> None:
        n = len(self.cycles)
        if self.subset not in SUBSETS:
            raise ValueError(f"unknown subset {self.subset!r}")
        if n == 0:
            raise ValueError("a trajectory needs at least one row")
        if self.settings.shape != (n, N_SETTINGS) or self.sensors.shape != (n, N_SENSORS):
            raise ValueError(
                f"shape mismatch: {n} cycles, settings {self.settings.shape}, "
                f"sensors {self.sensors.shape}"
            )
        if not np.array_equal(self.cycles, np.arange(1, n + 1)):
            raise ValueError(f"{self.subset} unit {self.unit}: cycles are not 1..{n}")
        if self.rul_at_end < 0:
            raise ValueError("rul_at_end must be non-negative")
        if self.rul_cap is not None and (
            isinstance(self.rul_cap, bool) or not isinstance(self.rul_cap, int) or self.rul_cap < 1
        ):
            raise ValueError(
                f"rul_cap must be a positive whole number or None, got {self.rul_cap!r}"
            )

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.source, self.subset, self.unit)

    @property
    def n(self) -> int:
        return len(self.cycles)

    @property
    def failure_cycle(self) -> int:
        return int(self.cycles[-1]) + self.rul_at_end

    def rul_uncapped(self) -> NDArray[np.int64]:
        """Cycles left until failure at each row, whatever ``rul_cap`` says."""
        return (self.failure_cycle - self.cycles).astype(np.int64)

    def rul(self) -> NDArray[np.int64]:
        """The truth at each row: cycles left until failure, capped at ``rul_cap`` if it is set."""
        raw = self.rul_uncapped()
        return raw if self.rul_cap is None else np.minimum(raw, self.rul_cap)

    def capped(self, cap: int) -> Trajectory:
        """The same rows with the truth capped at ``cap`` (a positive whole number)."""
        return replace(self, rul_cap=cap)
