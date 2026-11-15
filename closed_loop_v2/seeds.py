"""Run seeds, derived from the pre-registration hash (docs/adr/0002-interfaces.md, section 4).

A seed is the first 8 bytes, big-endian, of ``sha256("closed-loop-v2/seed/{prereg_sha}/{label}")``.
The seeds of a full run therefore exist only once the pre-registration is fixed, as in v0.1, and a
changed document has new ones. ``--quick`` uses the key ``"quick"`` in place of the hash.

There are four labels. ``fit-nominal`` and ``fit-shifted`` seed the two estimator fits (the 64-bit
value goes to ``Estimator.fit`` unchanged). ``calib-draw`` seeds every conformal method: V1's one
row per calibration unit, VW's fixed-point steps (VW reproduces V1 only with the same draws), and
V2L and V3L, whose pools start from V1's draw and whose own stream is kept apart inside
``closed_loop_v2.conformal`` by a key of its own. ``orders`` seeds the ``R`` fleet orders.

**Shared streams** (review of packet C, finding N6). Some streams are deliberately shared:

* one ``calib-draw`` seed and one ``orders`` seed serve both conditions. The shifted condition's
  calibration units are the FD001 and FD002 units of the nominal list, so those units get the same
  drawn rows in both conditions, and the two conditions replay their own units in orders drawn from
  the same stream;
* V2L and V3L draw from the same adaptive stream (``calib-draw`` with ``conformal``'s own key) in
  every one of the ``R`` orders and in both loops, so the ``m``-th pool draw uses the same random
  bits in every order. The orders therefore vary the fleet order, not the pool draws.

These are common random numbers: they tighten the paired comparisons (open against closed loop,
one method against another on the same units), and no registered comparison crosses the two
conditions, so no reported number is biased by the sharing. The streams are fixed as they are; this
note documents them and changes none.

``results.json`` records the labels, not the values: anyone holding the frozen document can
recompute them, and nobody can choose them after seeing data.
"""

from __future__ import annotations

import hashlib
from typing import Final

__all__ = ["QUICK_KEY", "SEED_LABELS", "run_seed"]

SEED_LABELS: Final[tuple[str, ...]] = ("fit-nominal", "fit-shifted", "calib-draw", "orders")
QUICK_KEY: Final[str] = "quick"


def run_seed(prereg_sha: str, label: str) -> int:
    """The seed for ``label``: the first 8 bytes, big-endian, of the keyed SHA-256.

    ``ValueError`` for a label outside ``SEED_LABELS`` or an empty key, so that a typing slip cannot
    quietly create a new stream.
    """
    if label not in SEED_LABELS:
        raise ValueError(f"unknown seed label {label!r}; the labels are {list(SEED_LABELS)}")
    if not prereg_sha:
        raise ValueError("the pre-registration key must not be empty")
    digest = hashlib.sha256(f"closed-loop-v2/seed/{prereg_sha}/{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
