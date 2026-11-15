"""Condition model and window features, on synthetic trajectories only."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from closed_loop_v2 import features
from closed_loop_v2.features import ConditionModel, window_features
from closed_loop_v2.types import N_SENSORS, Trajectory

# Six well separated operating conditions, in the style of the multi-condition subsets.
CENTRES = np.array(
    [
        [0.0, 0.0, 100.0],
        [10.0, 0.25, 100.0],
        [20.0, 0.70, 100.0],
        [25.0, 0.62, 60.0],
        [35.0, 0.84, 100.0],
        [42.0, 0.84, 40.0],
    ]
)


def make_traj(
    rng: np.random.Generator,
    unit: int,
    n: int,
    *,
    constant_sensor: int | None = 20,
    conditions: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> Trajectory:
    """A degradation signal plus noise; the failure is at the last row."""
    cond = rng.choice(conditions, size=n)
    settings_arr = CENTRES[cond] + rng.normal(0.0, 0.001, size=(n, 3))
    frac = np.arange(1, n + 1) / n
    offsets = np.linspace(-50.0, 50.0, 6)[cond][:, None] * np.linspace(0.5, 2.0, N_SENSORS)
    gain = np.linspace(1.0, 4.0, N_SENSORS)
    sensors = offsets + gain * (frac[:, None] ** 2) + rng.normal(0.0, 0.05, size=(n, N_SENSORS))
    if constant_sensor is not None:
        sensors[:, constant_sensor] = 7.5
    return Trajectory(
        subset="FD002",
        unit=unit,
        source="train",
        cycles=np.arange(1, n + 1, dtype=np.int64),
        settings=settings_arr,
        sensors=sensors,
        rul_at_end=0,
    )


def make_fleet(seed: int, n_units: int = 12, **kw: object) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    return [make_traj(rng, u + 1, int(rng.integers(30, 60)), **kw) for u in range(n_units)]  # type: ignore[arg-type]


# ------------------------------------------------------------------ window_features


def loop_features(z: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """The plain definition: np.polyfit on each window."""
    n, k = z.shape
    out = np.zeros((n, 2 * k + 1))
    for i in range(n):
        lo = max(0, i - window + 1)
        block = z[lo : i + 1]
        out[i, :k] = block.mean(axis=0)
        if block.shape[0] > 1:
            x = np.arange(lo, i + 1, dtype=np.float64)
            for c in range(k):
                out[i, k + c] = np.polyfit(x, block[:, c], 1)[0]
        out[i, 2 * k] = i + 1
    return out


@pytest.mark.parametrize("window", [1, 2, 3, 7, 30, 59, 60, 61, 500])
def test_window_features_match_loop(window: int) -> None:
    z = np.random.default_rng(window).normal(size=(60, 4))
    np.testing.assert_allclose(window_features(z, window), loop_features(z, window), atol=1e-9)


def test_window_features_trajectory_shorter_than_window_and_single_row() -> None:
    z = np.random.default_rng(1).normal(size=(5, 3))
    np.testing.assert_allclose(window_features(z, 1000), loop_features(z, 1000), atol=1e-12)
    one = window_features(z[:1], 30)
    assert one.shape == (1, 7)
    np.testing.assert_array_equal(one[0, :3], z[0])
    np.testing.assert_array_equal(one[0, 3:6], 0.0)
    assert one[0, 6] == 1.0


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(1, 40),
    k=st.integers(1, 4),
    window=st.integers(1, 50),
    seed=st.integers(0, 10_000),
)
def test_window_features_match_loop_property(n: int, k: int, window: int, seed: int) -> None:
    z = np.random.default_rng(seed).normal(scale=3.0, size=(n, k))
    np.testing.assert_allclose(window_features(z, window), loop_features(z, window), atol=1e-9)


def test_window_one_is_the_row_itself_with_zero_slope() -> None:
    z = np.random.default_rng(2).normal(size=(10, 3))
    out = window_features(z, 1)
    np.testing.assert_allclose(out[:, :3], z, atol=1e-12)
    np.testing.assert_array_equal(out[:, 3:6], 0.0)
    np.testing.assert_array_equal(out[:, 6], np.arange(1, 11))


def test_slope_of_a_ramp_is_its_gradient() -> None:
    z = (3.0 * np.arange(50.0) + 2.0)[:, None]
    out = window_features(z, 10)
    np.testing.assert_allclose(out[1:, 1], 3.0, atol=1e-9)
    assert out[0, 1] == 0.0
    np.testing.assert_allclose(out[20, 0], z[11:21].mean(), atol=1e-9)


def test_window_features_causal_by_truncation_bit_for_bit() -> None:
    z = np.random.default_rng(3).normal(size=(80, 5))
    full = window_features(z, 12)
    for i in (0, 1, 11, 12, 40, 79):
        cut = window_features(z[: i + 1], 12)
        np.testing.assert_array_equal(cut, full[: i + 1])


def test_future_rows_do_not_change_the_past() -> None:
    z = np.random.default_rng(4).normal(size=(50, 3))
    z2 = z.copy()
    z2[30:] = 1e6
    np.testing.assert_array_equal(window_features(z, 8)[:30], window_features(z2, 8)[:30])


def test_old_rows_leave_the_window() -> None:
    z = np.random.default_rng(5).normal(size=(50, 2))
    z2 = z.copy()
    z2[:10] += 1e3  # rows 0..9 are outside every window from row 19 on
    np.testing.assert_allclose(window_features(z, 10)[19:], window_features(z2, 10)[19:], atol=1e-9)


def test_window_features_input_checks() -> None:
    z = np.zeros((4, 2))
    with pytest.raises(ValueError):
        window_features(z, 0)
    with pytest.raises(ValueError):
        window_features(z[:, 0], 3)
    bad = z.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        window_features(bad, 3)


# ------------------------------------------------------------------ ConditionModel


def test_fit_finds_the_six_conditions() -> None:
    fleet = make_fleet(0)
    for seed in range(5):
        model = ConditionModel.fit(fleet, seed=seed)
        assert model.centroids.shape == (6, 3)
        d = np.linalg.norm(model.centroids[:, None, :] - CENTRES[None, :, :], axis=2)
        assert d.min(axis=1).max() < 0.01  # every centroid sits on one centre
        assert sorted(d.argmin(axis=1)) == [0, 1, 2, 3, 4, 5]  # and no two share one


def test_fit_is_deterministic_and_seed_keyed() -> None:
    fleet = make_fleet(1)
    a = ConditionModel.fit(fleet, seed=3)
    b = ConditionModel.fit(fleet, seed=3)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    np.testing.assert_array_equal(a.means, b.means)
    np.testing.assert_array_equal(a.stds, b.stds)
    assert a.kept == b.kept
    other = ConditionModel.fit(fleet, seed=4)
    assert not np.array_equal(a.centroids, other.centroids)  # same clusters, other labelling


def test_constant_sensor_is_dropped_and_others_kept() -> None:
    model = ConditionModel.fit(make_fleet(2), seed=0)
    assert 20 not in model.kept
    assert model.kept == tuple(j for j in range(N_SENSORS) if j != 20)


def test_sensor_is_kept_when_it_varies_in_only_one_condition() -> None:
    rng = np.random.default_rng(3)
    fleet = [make_traj(rng, u, 50) for u in range(6)]
    fleet = [
        Trajectory(
            t.subset,
            t.unit,
            t.source,
            t.cycles,
            t.settings,
            np.where(
                (np.linalg.norm(t.settings - CENTRES[2], axis=1) < 1)[:, None]
                & (np.arange(N_SENSORS) == 5),
                t.sensors,
                np.where(np.arange(N_SENSORS) == 5, 3.0, t.sensors),
            ),
            t.rul_at_end,
        )
        for t in fleet
    ]
    model = ConditionModel.fit(fleet, seed=0)
    assert 5 in model.kept
    # in the five conditions where it is constant its z-score is 0, not nan or huge
    z = model.normalise(fleet[0])
    col = model.kept.index(5)
    c = model.condition(fleet[0].settings)
    assert np.isfinite(z).all()
    flat = np.linalg.norm(fleet[0].settings - CENTRES[2], axis=1) >= 1
    assert flat.any()
    np.testing.assert_array_equal(z[flat, col], 0.0)
    assert (z[~flat, col] != 0.0).any()
    assert c.shape == (fleet[0].n,)


def test_min_std_threshold_drops_a_tiny_sensor() -> None:
    rng = np.random.default_rng(4)
    fleet = []
    for u in range(5):
        t = make_traj(rng, u, 40)
        sens = t.sensors.copy()
        sens[:, 3] = 1.0 + rng.normal(0.0, 1e-9, size=t.n)
        fleet.append(Trajectory(t.subset, t.unit, t.source, t.cycles, t.settings, sens, 0))
    assert 3 not in ConditionModel.fit(fleet, seed=0).kept
    assert 3 in ConditionModel.fit(fleet, seed=0, min_std=1e-12).kept


def test_normalise_gives_unit_scores_within_each_condition() -> None:
    fleet = make_fleet(5, n_units=15)
    model = ConditionModel.fit(fleet, seed=0)
    z = np.concatenate([model.normalise(t) for t in fleet])
    cond = np.concatenate([model.condition(t.settings) for t in fleet])
    assert z.shape[1] == len(model.kept)
    for c in range(6):
        block = z[cond == c]
        assert block.shape[0] > 20
        np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(block.std(axis=0), 1.0, atol=1e-9)


def test_normalise_is_row_local() -> None:
    fleet = make_fleet(6, n_units=4)
    model = ConditionModel.fit(fleet, seed=0)
    t = fleet[0]
    full = model.normalise(t)
    cut = Trajectory(
        t.subset, t.unit, t.source, t.cycles[:10], t.settings[:10], t.sensors[:10], t.rul_at_end
    )
    np.testing.assert_array_equal(model.normalise(cut), full[:10])


def test_condition_assigns_nearest_centroid_ties_to_lowest() -> None:
    model = ConditionModel(
        centroids=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        means=np.zeros((3, N_SENSORS)),
        stds=np.ones((3, N_SENSORS)),
        kept=(0,),
    )
    rows = np.array([[0.4, 0, 0], [1.1, 0, 0], [1.9, 0, 0], [50.0, 0, 0]])
    np.testing.assert_array_equal(model.condition(rows), [0, 1, 1, 1])


def test_fewer_distinct_rows_than_k_does_not_crash() -> None:
    rng = np.random.default_rng(7)
    fleet = [make_traj(rng, u, 30, conditions=(0, 1, 2)) for u in range(4)]
    for t in fleet:  # remove the settings noise: only three distinct rows exist
        nearest = np.linalg.norm(t.settings[:, None, :] - CENTRES[None], axis=2).argmin(axis=1)
        t.settings[:] = CENTRES[nearest]
    model = ConditionModel.fit(fleet, seed=0, k=6)
    assert model.centroids.shape == (6, 3)
    assert np.isfinite(model.means).all() and np.isfinite(model.stds).all()
    assert np.isfinite(model.normalise(fleet[0])).all()
    again = ConditionModel.fit(fleet, seed=0, k=6)
    np.testing.assert_array_equal(model.centroids, again.centroids)


def test_empty_cluster_is_reseeded_on_the_worst_fitted_row() -> None:
    # two blobs; the third centroid starts far from every row, so its cluster is empty
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [13.0, 0, 0]])
    start = np.array([[0.5, 0, 0], [11.5, 0, 0], [1000.0, 0, 0]])
    out = features._lloyd(x, start, max_steps=1)
    # distances to own centroid: 0.25, 0.25, 2.25, 2.25: the tie goes to the lowest row index (2)
    np.testing.assert_array_equal(out[2], x[2])
    full = features._lloyd(x, start, max_steps=100)
    assert np.isfinite(full).all()
    labels = features._sq_dist(x, full).argmin(axis=1)
    assert set(labels.tolist()) == {0, 1, 2}  # the re-seeded cluster is alive at the end


def test_two_empty_clusters_use_two_different_rows() -> None:
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [13.0, 0, 0]])
    start = np.array([[0.5, 0, 0], [11.5, 0, 0], [1000.0, 0, 0], [2000.0, 0, 0]])
    out = features._lloyd(x, start, max_steps=1)
    assert not np.array_equal(out[2], out[3])


def test_lloyd_respects_the_step_cap() -> None:
    assert features.MAX_LLOYD_STEPS == 100
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 3))
    start = features._kmeans_pp(x, 6, np.random.default_rng(0))
    one = features._lloyd(x, start, max_steps=1)
    many = features._lloyd(x, start)
    assert not np.array_equal(one, many)


def test_fit_input_checks() -> None:
    fleet = make_fleet(9, n_units=2)
    with pytest.raises(ValueError):
        ConditionModel.fit([], seed=0)
    with pytest.raises(ValueError):
        ConditionModel.fit(fleet, seed=0, k=0)
    flat = [
        Trajectory(t.subset, t.unit, t.source, t.cycles, t.settings, np.ones_like(t.sensors), 0)
        for t in fleet
    ]
    with pytest.raises(ValueError):
        ConditionModel.fit(flat, seed=0)
