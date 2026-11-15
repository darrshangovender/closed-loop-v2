"""Development loader: the training units whose role is ``"dev"`` or ``"calib"``.

This path reads ``train_FD00x.txt`` only. It has no code that names a ``test_`` or ``RUL_`` file,
and it does not split or convert the rows of an eval unit (``parse.parse_training``). The tests
``test_load_never_opens_heldout_files`` (real files) and its synthetic twin patch ``open`` and
record every path. Eval units and the test files go through ``HeldoutAccess``.

Results are cached per process, keyed by the data directory and subset. The arrays are read-only,
so a caller cannot change what the next caller receives.
"""

from __future__ import annotations

from collections.abc import Sequence

from closed_loop_v2.data.errors import HeldoutError
from closed_loop_v2.data.parse import parse_training
from closed_loop_v2.data.paths import data_dir, read_verified
from closed_loop_v2.data.roles import UNITS_PER_FILE, check_subsets
from closed_loop_v2.types import SUBSETS, Role, Trajectory

__all__ = ["clear_cache", "development_units"]

_CACHE: dict[tuple[str, str], dict[Role, list[Trajectory]]] = {}


def clear_cache() -> None:
    """Forget every loaded subset (tests use this to force a read)."""
    _CACHE.clear()


def _subset(directory: str, subset: str) -> dict[Role, list[Trajectory]]:
    key = (directory, subset)
    if key not in _CACHE:
        name = f"train_{subset}.txt"
        data = read_verified(data_dir(), name)
        _CACHE[key] = parse_training(
            data, name, subset, UNITS_PER_FILE["train"][subset], ("dev", "calib")
        )
    return _CACHE[key]


def development_units(role: Role, subsets: Sequence[str] = SUBSETS) -> list[Trajectory]:
    """Training units of ``role`` (``"dev"`` or ``"calib"``) from ``subsets``, in unit order.

    The files are checked against the manifest before they are parsed. Each training file must
    hold unit numbers ``1..N``, with ``N`` from the dataset's readme.txt. ``"eval"`` raises
    ``HeldoutError``: eval units come from ``HeldoutAccess.eval_units`` after the freeze.
    """
    if role == "eval":
        raise HeldoutError(
            "role 'eval' is held out until the pre-registration is frozen: "
            "use HeldoutAccess.eval_units, which checks the ledger and records the read"
        )
    if role not in ("dev", "calib"):
        raise ValueError(f"role must be 'dev' or 'calib', got {role!r}")
    chosen = check_subsets(subsets)
    directory = str(data_dir().resolve())
    units: list[Trajectory] = []
    for subset in chosen:
        units.extend(_subset(directory, subset)[role])
    return units
