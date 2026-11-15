"""The held-out data: eval units and NASA's test units with their remaining lives.

``HeldoutAccess`` hands these out only when the ledger holds a ``freeze`` entry whose
``prereg_sha256`` equals the hash of the pre-registration file as it is now. Each read appends a
``heldout-use`` entry with ``n=1`` before any data file is opened, so a read that fails afterwards
still counts. In v0.2 the ledger's ``heldout_evaluations`` therefore counts reads of held-out data.

Three rules, added after the adversarial review of 10 November (``research`` note
``closed-loop-v2-review-AB.md``, findings 1 and 2):

- Only the **first** ``freeze`` entry counts. A pre-registration edited after the freeze keeps the
  gate shut, and a second freeze does not reopen it.
- When the files being read are the real C-MAPSS files (the package's own ``data/MANIFEST.json`` is
  the manifest in use), the ledger must be the repository's ``benchmarks/ledger.jsonl`` and the
  pre-registration its ``docs/PREREGISTRATION.md``. A scratch ledger holding a freeze line cannot
  open real held-out data, and every real read lands in the one ledger that is published. Tests
  that swap in a synthetic manifest (``tests/data/conftest.py``) may use temporary files.
- If the first freeze entry records ``manifest_sha256`` and the real manifest is in use, the
  manifest file must still have that hash.

What this guards: a code path that goes through ``HeldoutAccess`` gets no held-out rows before a
matching freeze, and every read it makes is on the ledger. It does not stop someone who opens the
raw files directly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from closed_loop.audit.ledger import Ledger, LedgerError
from closed_loop.audit.prereg import file_sha256, prereg_sha256

from closed_loop_v2 import audit
from closed_loop_v2.data import paths
from closed_loop_v2.data.errors import HeldoutError
from closed_loop_v2.data.parse import parse_rul, parse_test, parse_training
from closed_loop_v2.data.paths import data_dir, read_verified
from closed_loop_v2.data.roles import UNITS_PER_FILE, check_subsets, role_of
from closed_loop_v2.types import SUBSETS, Role, Trajectory

__all__ = ["HeldoutAccess"]

_EVAL_ONLY: tuple[Role, ...] = ("eval",)


class HeldoutAccess:
    """Gate and meter for the held-out sets, bound to one ledger and one pre-registration."""

    def __init__(self, ledger: Ledger, prereg_path: Path) -> None:
        self.ledger = ledger
        self.prereg_path = Path(prereg_path)

    def _real_manifest_in_use(self) -> bool:
        """True when the manifest in use is the package's own: the files read would be real.

        If the package's manifest cannot be read, the answer is True, so the stricter rule applies.
        """
        try:
            real = json.loads(paths.MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
        except (OSError, ValueError, KeyError, TypeError):
            return True
        return bool(paths.manifest()["files"] == real)

    def _check_bound_to_repository(self) -> None:
        """With real data, refuse any ledger or pre-registration other than the repository's."""
        if not self._real_manifest_in_use():
            return
        same_ledger = Path(self.ledger.path).resolve() == audit.LEDGER.resolve()
        same_prereg = self.prereg_path.resolve() == audit.PREREG.resolve()
        if not (same_ledger and same_prereg):
            raise HeldoutError(
                "real held-out data opens only against the repository's own ledger "
                f"({audit.LEDGER.relative_to(audit.REPO).as_posix()}) and pre-registration "
                f"({audit.PREREG.relative_to(audit.REPO).as_posix()}); this HeldoutAccess was "
                f"given {Path(self.ledger.path).name} and {self.prereg_path.name}"
            )

    def _first_freeze(self, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((e for e in entries if e["kind"] == "freeze"), None)

    def _frozen_hash(self) -> str | None:
        """The pre-registration's current hash if the ledger's first freeze froze exactly that.

        A ledger that does not verify, or a pre-registration that cannot be read, raises
        ``HeldoutError``: a quiet ``False`` would hide the cause.
        """
        self._check_bound_to_repository()
        try:
            current = prereg_sha256(self.prereg_path)
        except OSError as exc:
            raise HeldoutError(
                f"cannot read the pre-registration {self.prereg_path.name}: {exc}"
            ) from exc
        try:
            entries = self.ledger.entries()
        except LedgerError as exc:
            raise HeldoutError(f"the ledger does not verify: {exc}") from exc
        first = self._first_freeze(entries)
        if first is None or first.get("prereg_sha256") != current:
            return None
        recorded = first.get("manifest_sha256")
        if recorded is not None and self._real_manifest_in_use():
            if file_sha256(paths.MANIFEST_PATH) != recorded:
                raise HeldoutError(
                    f"{paths.MANIFEST_PATH.name} has changed since the freeze: the data checksums "
                    "the pre-registration was frozen with are not the ones in use"
                )
        return current

    def is_frozen(self) -> bool:
        """True if the verified ledger's first ``freeze`` entry is for the file's current hash."""
        return self._frozen_hash() is not None

    def _open(self, what: str, subsets: tuple[str, ...], units: int) -> None:
        """Refuse unless frozen; otherwise write the ledger line, before any data is read."""
        current = self._frozen_hash()
        if current is None:
            raise HeldoutError(
                f"{what} units are held out: the ledger's first freeze entry is not for the "
                f"current pre-registration {self.prereg_path.name} (or there is none). Only the "
                "first freeze counts, so a document edited after the freeze keeps the held-out "
                "data closed."
            )
        self.ledger.append(
            "heldout-use",
            n=1,
            what=what,
            subsets=list(subsets),
            units=units,
            prereg_sha256=current,
        )

    def eval_units(self, subsets: Sequence[str] = SUBSETS) -> list[Trajectory]:
        """The training units whose role is ``"eval"``, in unit order, from ``subsets``."""
        chosen = check_subsets(subsets)
        n_eval = sum(
            role_of(s, unit) == "eval"
            for s in chosen
            for unit in range(1, UNITS_PER_FILE["train"][s] + 1)
        )
        self._open("eval", chosen, n_eval)
        directory = data_dir()
        out: list[Trajectory] = []
        for subset in chosen:
            name = f"train_{subset}.txt"
            data = read_verified(directory, name)
            n_train = UNITS_PER_FILE["train"][subset]
            parsed = parse_training(data, name, subset, n_train, _EVAL_ONLY)
            out.extend(parsed["eval"])
        return out

    def test_units(self, subsets: Sequence[str] = SUBSETS) -> list[Trajectory]:
        """NASA's test units from ``subsets``, with ``rul_at_end`` from ``RUL_FD00x.txt``.

        ``RUL_FD00x.txt`` holds one value per test unit, in unit order.
        """
        chosen = check_subsets(subsets)
        self._open("test", chosen, sum(UNITS_PER_FILE["test"][s] for s in chosen))
        directory = data_dir()
        out: list[Trajectory] = []
        for subset in chosen:
            n_test = UNITS_PER_FILE["test"][subset]
            rul_name = f"RUL_{subset}.txt"
            rul = parse_rul(read_verified(directory, rul_name), rul_name, n_test)
            name = f"test_{subset}.txt"
            out.extend(parse_test(read_verified(directory, name), name, subset, n_test, rul))
        return out
