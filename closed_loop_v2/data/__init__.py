"""C-MAPSS data layer: roles, verified loading, and the held-out gate.

C-MAPSS is simulated by NASA's C-MAPSS model, not field data. Development code reads dev and
calib units through ``development_units``. Eval units and NASA's test files come only from
``HeldoutAccess``, which needs a matching freeze entry in the ledger.
"""

from __future__ import annotations

from closed_loop_v2.data.errors import DataError, HeldoutError
from closed_loop_v2.data.heldout import HeldoutAccess
from closed_loop_v2.data.load import development_units
from closed_loop_v2.data.paths import data_available, data_dir, verify
from closed_loop_v2.data.roles import ROLE_COUNTS, UNITS_PER_FILE, role_of

__all__ = [
    "ROLE_COUNTS",
    "UNITS_PER_FILE",
    "DataError",
    "HeldoutAccess",
    "HeldoutError",
    "data_available",
    "data_dir",
    "development_units",
    "role_of",
    "verify",
]
