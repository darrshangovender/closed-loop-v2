"""The fleet replay: schedule, resolution order, the static fast path, and the open and closed loop.

``replay`` jumps from one resolution to the next instead of walking every step. ``step_replay``
below is a literal, step-by-step reading of the schedule in 0002 section 4, and the tests check that
the two agree on every life, every bound and the order and content of every ``observe_life`` call.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from closed_loop_v2.conformal import (
    AdaptiveConformal,
    LifeLabels,
    Quantiles,
    SplitConformal,
    Status,
    life_labels,
)
from closed_loop_v2.estimator import Prediction
from closed_loop_v2.policy import BoundRule
from closed_loop_v2.replay import End, LifeRun, ReplayRun, orders, replay, replay_static
from closed_loop_v2.types import Trajectory

INF = math.inf


def traj(n: int, unit: int = 1) -> Trajectory:
    return Trajectory(
        "FD001", unit, "train", np.arange(1, n + 1), np.zeros((n, 3)), np.zeros((n, 21)), 0
    )


def life(m: list[float] | NDArray[np.float64], s: float | list[float] = 1.0, unit: int = 1):
    m_arr = np.asarray(m, dtype=np.float64)
    s_arr = np.broadcast_to(np.asarray(s, dtype=np.float64), m_arr.shape).copy()
    return traj(m_arr.size, unit), Prediction(m_arr, s_arr)


class Stub:
    """A stateful method: ``q_lo`` drops by ``step`` after every observed life. It records every
    call to ``observe_life``, with the quantiles in force when it was made."""

    def __init__(self, q_lo: float = 0.0, q_hi: float = 1.0, step: float = 0.5) -> None:
        self.q = Quantiles(q_lo, q_hi)
        self.step = step
        self.calls: list[tuple[LifeLabels, Quantiles]] = []

    def quantiles(self) -> Quantiles:
        return self.q

    def observe_life(self, labels: LifeLabels) -> None:
        self.calls.append((labels, self.q))
        self.q = Quantiles(self.q.q_lo - self.step, self.q.q_hi + self.step)


def step_replay(lives, method, rule: BoundRule | None, *, k: int, order) -> ReplayRun:
    """The schedule of 0002 section 4, one global step at a time."""
    queue = [int(i) for i in order]
    nxt = 0
    slots: list[dict | None] = [None] * k
    done: dict[int, LifeRun] = {}
    resolved: list[int] = []

    def take(j: int, at: int) -> None:
        nonlocal nxt
        if nxt < len(queue):
            idx = queue[nxt]
            nxt += 1
            slots[j] = {"life": idx, "row": 0, "start": at, "lo": [], "hi": []}
        else:
            slots[j] = None

    for j in range(k):
        take(j, 0)
    step = 0
    while any(s is not None for s in slots):
        q = method.quantiles()  # as it stands at the start of the step
        ended: list[tuple[int, End]] = []
        for j in range(k):
            s = slots[j]
            if s is None:
                continue
            t, p = lives[s["life"]]
            i = s["row"]
            lo_i = p.m[i] + p.s[i] * q.q_lo
            hi_i = p.m[i] + p.s[i] * q.q_hi
            if t.rul_cap is not None:  # both ends limited to the cap of the truth
                lo_i, hi_i = min(lo_i, t.rul_cap), min(hi_i, t.rul_cap)
            s["lo"].append(lo_i)
            s["hi"].append(hi_i)
            if rule is not None and lo_i <= rule.h:
                ended.append((j, "replaced"))
            elif i == t.n - 1:
                ended.append((j, "failed"))
            else:
                s["row"] = i + 1
        for j, end in ended:  # slot order
            s = slots[j]
            assert s is not None
            t, p = lives[s["life"]]
            tau = s["row"]
            lo = np.full(t.n, np.nan)
            hi = np.full(t.n, np.nan)
            lo[: tau + 1] = s["lo"]
            hi[: tau + 1] = s["hi"]
            cap = math.inf if t.rul_cap is None else float(t.rul_cap)
            tau_or_none = tau if end == "replaced" else None
            method.observe_life(life_labels(t.rul(), p, lo, hi, replaced_at=tau_or_none, cap=cap))
            done[s["life"]] = LifeRun(
                end, tau, lo[: tau + 1], hi[: tau + 1], slot=j, start_step=s["start"], end_step=step
            )
            resolved.append(s["life"])
            take(j, step + 1)
        step += 1
    return ReplayRun(tuple(done[i] for i in range(len(lives))), tuple(resolved), step)


def random_fleet(rng: np.random.Generator, n_lives: int, max_len: int = 25):
    out = []
    for u in range(n_lives):
        n = int(rng.integers(1, max_len))
        rul = np.arange(n - 1, -1, -1, dtype=np.float64)
        m = rul + rng.normal(0.0, 3.0, size=n)
        s = rng.uniform(1.0, 4.0, size=n)
        out.append(life(m, list(s), unit=u + 1))
    return out


def same_calls(a: Stub, b: Stub) -> bool:
    if len(a.calls) != len(b.calls):
        return False
    for (la, qa), (lb, qb) in zip(a.calls, b.calls, strict=True):
        if qa != qb or la.observed_failure != lb.observed_failure:
            return False
        if not np.array_equal(la.status, lb.status):
            return False
        if (la.scores is None) != (lb.scores is None):
            return False
        if la.scores is not None and not np.array_equal(la.scores, lb.scores):
            return False
    return True


def same_schedule(a: ReplayRun, b: ReplayRun) -> bool:
    return (
        a.same_lives(b)
        and a.resolved == b.resolved
        and a.n_steps == b.n_steps
        and all(
            (x.slot, x.start_step, x.end_step) == (y.slot, y.start_step, y.end_step)
            for x, y in zip(a.lives, b.lives, strict=True)
        )
    )


# ----------------------------------------------------------------------------- the schedule


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("cap", [None, 8])
def test_replay_equals_the_step_by_step_schedule(seed: int, closed: bool, cap: int | None) -> None:
    rng = np.random.default_rng(seed)
    lives = random_fleet(rng, int(rng.integers(1, 14)))
    if cap is not None:  # the truth capped at 8: bounds limited to 8, exact labels past it
        lives = [(t.capped(cap), p) for t, p in lives]
    k = int(rng.integers(1, 6))
    order = rng.permutation(len(lives))
    rule = BoundRule(float(rng.uniform(-2.0, 6.0))) if closed else None
    fast, slow = Stub(step=0.3), Stub(step=0.3)
    got = replay(lives, fast, rule, k=k, order=order)
    want = step_replay(lives, slow, rule, k=k, order=order)
    assert same_schedule(got, want)
    assert same_calls(fast, slow)


@pytest.mark.parametrize("seed", range(6))
def test_replay_equals_the_step_by_step_schedule_with_aci(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    lives = random_fleet(rng, 20, max_len=30)
    pool = rng.normal(0.0, 1.0, size=40)

    def aci() -> AdaptiveConformal:
        return AdaptiveConformal(pool, alpha=0.1, gamma=0.2, window=50, rule="skip", seed=seed)

    rule = BoundRule(2.0)
    order = rng.permutation(len(lives))
    a, b = aci(), aci()
    got = replay(lives, a, rule, k=4, order=order)
    want = step_replay(lives, b, rule, k=4, order=order)
    assert same_schedule(got, want)
    assert a.alpha_t == b.alpha_t and a.pool == b.pool


def test_hand_made_schedule_and_resolution_order() -> None:
    # lives of 3, 1 and 2 rows; two slots; a rule that never fires (open loop)
    lives = [life([5, 4, 3]), life([7]), life([9, 8])]
    stub = Stub(q_lo=0.0, q_hi=1.0, step=1.0)
    run = replay(lives, stub, None, k=2, order=[0, 1, 2])
    # step 0: slot 0 life 0 row 0, slot 1 life 1 row 0 (its last row): life 1 resolves
    # step 1: slot 0 life 0 row 1, slot 1 life 2 row 0
    # step 2: slot 0 life 0 row 2 (last), slot 1 life 2 row 1 (last): slot 0 resolves first
    assert run.resolved == (1, 0, 2)
    assert run.n_steps == 3
    l0, l1, l2 = run.lives
    assert (l0.slot, l0.start_step, l0.end_step) == (0, 0, 2)
    assert (l1.slot, l1.start_step, l1.end_step) == (1, 0, 0)
    assert (l2.slot, l2.start_step, l2.end_step) == (1, 1, 2)
    # the bounds use the quantiles at the start of each step: q_lo 0, then -1 after one life
    np.testing.assert_array_equal(l0.lo, [5.0, 4.0 - 1.0, 3.0 - 1.0])
    np.testing.assert_array_equal(l1.lo, [7.0])
    np.testing.assert_array_equal(l2.lo, [9.0 - 1.0, 8.0 - 1.0])
    # two lives ending in step 2 both saw the pre-step quantiles; observe_life ran in slot order
    assert [q.q_lo for _, q in stub.calls] == [0.0, -1.0, -2.0]
    assert all(lab.observed_failure for lab, _ in stub.calls)


def test_lives_ending_together_resolve_in_slot_order_not_life_order() -> None:
    lives = [life([3, 2]), life([3, 2])]
    stub = Stub()
    run = replay(lives, stub, None, k=2, order=[1, 0])  # slot 0 holds life 1
    assert run.resolved == (1, 0)
    assert run.lives[1].slot == 0 and run.lives[0].slot == 1


def test_a_slot_with_nothing_left_stays_empty() -> None:
    lives = [life([2, 1, 0]), life([4])]
    run = replay(lives, Stub(), None, k=5, order=[0, 1])
    assert run.n_steps == 3
    assert {lr.slot for lr in run.lives} == {0, 1}


def test_a_new_life_starts_on_the_step_after_the_end() -> None:
    lives = [life([1.0]), life([1.0, 1.0])]
    run = replay(lives, Stub(), None, k=1, order=[0, 1])
    assert run.lives[0].end_step == 0
    assert run.lives[1].start_step == 1 and run.lives[1].end_step == 2


# ----------------------------------------------------------------------------- the two loops


def test_the_open_loop_never_replaces() -> None:
    lives = [life([1.0, 0.5, 0.0]), life([-5.0, -6.0])]  # bounds below any h
    run = replay(lives, Stub(), None, k=2, order=[0, 1])
    assert [lr.end for lr in run.lives] == ["failed", "failed"]
    assert [lr.tau for lr in run.lives] == [2, 1]
    static = replay_static(lives, Quantiles(-INF, INF), None)
    assert all(
        lr.end == "failed" and lr.rows == t.n
        for lr, (t, _) in zip(static.lives, lives, strict=True)
    )


def test_the_closed_loop_stops_at_the_first_lower_bound_at_most_h() -> None:
    m = [30.0, 25.0, 21.0, 20.0, 19.0, 25.0, 10.0]
    lives = [life(m)]
    run = replay(lives, SplitConformal(Quantiles(0.0, 1.0)), BoundRule(20.0), k=1, order=[0])
    lr = run.lives[0]
    assert lr.end == "replaced" and lr.tau == 3  # 20.0 <= 20.0, the first such row
    np.testing.assert_array_equal(lr.lo, m[:4])
    assert lr.lo.shape == lr.hi.shape == (4,)


def test_a_trigger_on_the_last_row_counts_as_a_replacement() -> None:
    lives = [life([9.0, 8.0, 1.0])]
    run = replay(lives, Stub(), BoundRule(1.0), k=1, order=[0])
    assert run.lives[0].end == "replaced" and run.lives[0].tau == 2


def test_a_replaced_life_is_labelled_against_its_replacement_row() -> None:
    # true remaining life 4, 3, 2, 1, 0; replaced at row 2 (lo 1 <= h 1)
    lives = [life([6.0, 4.0, 1.0, 9.0, 9.0], s=1.0)]
    stub = Stub(q_lo=0.0, q_hi=1.0)
    run = replay(lives, stub, BoundRule(1.0), k=1, order=[0])
    assert run.lives[0].tau == 2
    labels, _ = stub.calls[0]
    assert not labels.observed_failure and labels.scores is None
    # b = 2, 1, 0 against [6, 7], [4, 5], [1, 2]
    assert list(labels.status) == [Status.UNKNOWN, Status.UNKNOWN, Status.UNKNOWN]


def test_infinite_quantiles() -> None:
    lives = [life([5.0, 4.0, 3.0])]
    closed = replay_static(lives, Quantiles(-INF, INF), BoundRule(0.0))
    assert closed.lives[0].end == "replaced" and closed.lives[0].tau == 0
    assert closed.lives[0].lo[0] == -INF and closed.lives[0].hi[0] == INF
    never = replay_static(lives, Quantiles(INF, -INF), BoundRule(1e9))
    assert never.lives[0].end == "failed"


# ----------------------------------------------------------------------------- the fast path


@pytest.mark.parametrize("closed", [False, True])
def test_replay_static_equals_replay_for_every_k_and_order(closed: bool) -> None:
    rng = np.random.default_rng(7)
    lives = random_fleet(rng, 9)
    q = Quantiles(-1.2, 1.7)
    rule = BoundRule(3.0) if closed else None
    static = replay_static(lives, q, rule)
    for k in range(1, 11):
        for j in range(4):
            order = np.random.default_rng([k, j]).permutation(len(lives))
            run = replay(lives, SplitConformal(q), rule, k=k, order=order)
            assert static.same_lives(run), (k, j)


def test_orders_are_seeded_permutations() -> None:
    a = orders(10, 4, seed=123)
    assert a.shape == (4, 10)
    for row in a:
        assert sorted(row) == list(range(10))
    np.testing.assert_array_equal(a, orders(10, 4, seed=123))
    np.testing.assert_array_equal(a[1], np.random.default_rng([123, 1]).permutation(10))
    assert not np.array_equal(a[0], a[1])
    assert orders(0, 3, seed=1).shape == (3, 0)


def test_input_checks() -> None:
    lives = [life([1.0, 2.0])]
    with pytest.raises(ValueError, match="permutation"):
        replay(lives, Stub(), None, k=1, order=[0, 0])
    with pytest.raises(ValueError, match="at least 1"):
        replay(lives, Stub(), None, k=0, order=[0])
    bad = [(traj(3), Prediction(np.zeros(2), np.ones(2)))]
    with pytest.raises(ValueError, match="predictions"):
        replay_static(bad, Quantiles(0.0, 1.0), None)


def test_an_empty_fleet_is_an_empty_run() -> None:
    run = replay([], Stub(), BoundRule(1.0), k=3, order=[])
    assert run.lives == () and run.n_steps == 0


def test_replayed_bounds_never_exceed_the_cap() -> None:
    """m above the cap gives raw bounds above it; both replays limit them to the cap (S1)."""
    t, p = life([9.0, 9.0, 9.0, 1.0, 9.0, 9.0], s=1.0)
    lives = [(t.capped(4), p)]
    # raw bounds m - 1 .. m + 2: [8, 11] where m = 9; a fresh stub each time (it moves after a life)
    for run in (
        replay(lives, Stub(q_lo=-1.0, q_hi=2.0), BoundRule(-100.0), k=1, order=[0]),
        replay(lives, Stub(q_lo=-1.0, q_hi=2.0), None, k=1, order=[0]),
        replay_static(lives, Quantiles(-1.0, 2.0), None),
    ):
        lr = run.lives[0]
        assert np.all(lr.lo <= 4.0) and np.all(lr.hi <= 4.0)
        assert lr.lo[0] == 4.0 and lr.hi[0] == 4.0  # [8, 11] limited to [4, 4]
        assert lr.lo[3] == 0.0 and lr.hi[3] == 3.0  # m = 1: [0, 3], under the cap


def test_a_capped_replaced_life_has_exact_labels_once_it_outlives_the_cap() -> None:
    """Truth min(RUL, 2): rows 0 and 1 of a life replaced at row 3 are known to have truth 2."""
    t, p = life([9.0, 9.0, 9.0, 1.0, 9.0, 9.0], s=1.0)
    lives = [(t.capped(2), p)]
    stub = Stub(q_lo=-7.5, q_hi=-6.0)  # bounds m - 7.5 .. m - 6: [1.5, 3] on rows 0 to 2
    run = replay(lives, stub, BoundRule(-6.0), k=1, order=[0])
    assert run.lives[0].tau == 3
    labels, _ = stub.calls[0]
    # b = 3, 2, 1, 0: rows 0 and 1 have b >= 2, so the truth is exactly 2, inside [1.5, 3]: HIT;
    # row 2 has b = 1 below lo = 1.5: UNKNOWN; row 3 (lo -6.5) has b = 0 within [-6.5, -5]? no:
    # 0 > hi = -5, so MISS
    assert list(labels.status) == [Status.HIT, Status.HIT, Status.UNKNOWN, Status.MISS]
