"""The shared estimator, on a few dozen short synthetic units."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from closed_loop_v2.estimator import (
    DEFAULT_PARAMS,
    Estimator,
    EstimatorParams,
    Prediction,
    unit_folds,
)
from closed_loop_v2.types import N_SENSORS, Trajectory

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

SMALL = EstimatorParams(
    window=5, cap=125.0, s_min=1.0, max_iter=30, learning_rate=0.2, max_leaf_nodes=8,
    min_samples_leaf=5,
)  # fmt: skip


def make_traj(rng: np.random.Generator, unit: int, n: int) -> Trajectory:
    """Degradation (a fraction of life, squared) on top of condition offsets, plus noise."""
    cond = rng.integers(0, 6, size=n)
    settings = CENTRES[cond] + rng.normal(0.0, 0.001, size=(n, 3))
    frac = np.arange(1, n + 1) / n
    offsets = np.linspace(-20.0, 20.0, 6)[cond][:, None] * np.linspace(0.5, 2.0, N_SENSORS)
    gain = np.linspace(1.0, 4.0, N_SENSORS)
    sensors = offsets + 10.0 * gain * (frac[:, None] ** 2) + rng.normal(0.0, 0.3, (n, N_SENSORS))
    sensors[:, 20] = 7.5
    return Trajectory("FD002", unit, "train", np.arange(1, n + 1), settings, sensors, 0)


def make_fleet(seed: int, n_units: int = 30) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    return [make_traj(rng, u + 1, int(rng.integers(30, 60))) for u in range(n_units)]


def truncate(t: Trajectory, rows: int) -> Trajectory:
    return Trajectory(
        t.subset, t.unit, t.source, t.cycles[:rows], t.settings[:rows], t.sensors[:rows], 0
    )


@pytest.fixture(scope="module")
def fleet() -> list[Trajectory]:
    return make_fleet(0)


@pytest.fixture(scope="module")
def fitted(fleet: list[Trajectory]) -> Estimator:
    return Estimator.fit(fleet, SMALL, seed=1)


def test_defaults_are_the_registered_starting_point() -> None:
    assert DEFAULT_PARAMS == EstimatorParams(30, 125.0, 5.0, 300, 0.05, 31, 50)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_PARAMS.window = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [{"window": 0}, {"cap": 0.0}, {"s_min": 0.0}, {"max_iter": 0}, {"learning_rate": 0.0}],
)
def test_bad_params_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **change)


def test_prediction_shapes_and_floor(fitted: Estimator, fleet: list[Trajectory]) -> None:
    for t in fleet[:3]:
        p = fitted.predict(t)
        assert p.m.dtype == np.float64 and p.s.dtype == np.float64
        assert p.m.shape == p.s.shape == (t.n,)
        assert (p.s >= SMALL.s_min).all()


def test_scale_floor_binds_when_s_min_is_large(fleet: list[Trajectory]) -> None:
    est = Estimator.fit(fleet, dataclasses.replace(SMALL, s_min=1e6), seed=1)
    np.testing.assert_array_equal(est.predict(fleet[0]).s, 1e6)


def test_the_point_estimate_tracks_the_remaining_life(
    fitted: Estimator,
) -> None:
    held_out = make_fleet(99, n_units=10)
    m = np.concatenate([fitted.predict(t).m for t in held_out])
    rul = np.concatenate([t.rul() for t in held_out]).astype(float)
    assert np.corrcoef(m, rul)[0, 1] > 0.8
    assert np.abs(m - rul).mean() < np.abs(rul - rul.mean()).mean()


def test_m_is_trained_on_the_capped_life(fleet: list[Trajectory]) -> None:
    est = Estimator.fit(fleet, dataclasses.replace(SMALL, cap=8.0), seed=1)
    m_early = np.concatenate([est.predict(t).m[:5] for t in fleet])  # RUL is at least 25 there
    assert m_early.max() < 12.0


def test_a_64_bit_seed_fits_and_repeats(fleet: list[Trajectory]) -> None:
    # Review of 10 November, finding 5: the run's seeds are 64-bit; scikit-learn takes 32-bit ones.
    seed = 2**63 + 12345
    a = Estimator.fit(fleet[:12], SMALL, seed=seed)
    b = Estimator.fit(fleet[:12], SMALL, seed=seed)
    np.testing.assert_array_equal(a.predict(fleet[0]).m, b.predict(fleet[0]).m)
    with pytest.raises(ValueError, match="non-negative"):
        Estimator.fit(fleet[:12], SMALL, seed=-1)


def test_same_seed_gives_bit_identical_output(fleet: list[Trajectory]) -> None:
    a = Estimator.fit(fleet, SMALL, seed=5)
    b = Estimator.fit(fleet, SMALL, seed=5)
    for t in fleet[:4]:
        pa, pb = a.predict(t), b.predict(t)
        np.testing.assert_array_equal(pa.m, pb.m)
        np.testing.assert_array_equal(pa.s, pb.s)


def test_seed_changes_the_cross_fitting(fleet: list[Trajectory]) -> None:
    a = Estimator.fit(fleet, SMALL, seed=5)
    b = Estimator.fit(fleet, SMALL, seed=6)
    assert not np.array_equal(a.predict(fleet[0]).s, b.predict(fleet[0]).s)


def test_predict_is_causal_by_truncation_bit_for_bit(
    fitted: Estimator, fleet: list[Trajectory]
) -> None:
    t = fleet[2]
    full = fitted.predict(t)
    for rows in (1, 2, 5, 6, 20, t.n - 1):
        cut = fitted.predict(truncate(t, rows))
        np.testing.assert_array_equal(cut.m, full.m[:rows])
        np.testing.assert_array_equal(cut.s, full.s[:rows])


def test_predict_ignores_later_rows_and_the_label(
    fitted: Estimator, fleet: list[Trajectory]
) -> None:
    t = fleet[3]
    full = fitted.predict(t)
    i = 15
    sensors = t.sensors.copy()
    settings = t.settings.copy()
    sensors[i + 1 :] = 1e6
    settings[i + 1 :] = -5.0
    altered = Trajectory(t.subset, t.unit, t.source, t.cycles, settings, sensors, 500)
    p = fitted.predict(altered)
    np.testing.assert_array_equal(p.m[: i + 1], full.m[: i + 1])
    np.testing.assert_array_equal(p.s[: i + 1], full.s[: i + 1])


def test_unit_folds_hold_whole_units_and_balance() -> None:
    f = unit_folds(23, 5, seed=3)
    assert f.shape == (23,)
    counts = np.bincount(f, minlength=5)
    assert (
        counts.sum() == 23
        and int(np.max(counts)) - int(np.min(counts)) <= 1
        and int(np.min(counts)) >= 4
    )
    np.testing.assert_array_equal(f, unit_folds(23, 5, seed=3))
    assert not np.array_equal(f, unit_folds(23, 5, seed=4))


def test_scale_is_fitted_on_out_of_fold_residuals() -> None:
    """With an over-flexible m, in-sample residuals are tiny; s must still be large."""
    fleet = make_fleet(7, n_units=15)
    flexible = EstimatorParams(
        window=5, cap=125.0, s_min=1e-3, max_iter=40, learning_rate=0.5,
        max_leaf_nodes=31, min_samples_leaf=1,
    )  # fmt: skip
    est = Estimator.fit(fleet, flexible, seed=2)
    in_sample = np.concatenate([np.abs(t.rul() - est.predict(t).m) for t in fleet]).mean()
    scale = np.concatenate([est.predict(t).s for t in fleet]).mean()
    assert in_sample < 1.0
    assert scale > 5.0 * in_sample


def signature_unit(rng: np.random.Generator, unit: int, life: int) -> Trajectory:
    """Sensors are a constant vector private to the unit: only the unit and the cycle count say
    anything about the remaining life."""
    sig = rng.normal(0.0, 5.0, size=N_SENSORS)
    sensors = sig + rng.normal(0.0, 1e-3, size=(life, N_SENSORS))
    settings = np.tile(CENTRES[0], (life, 1))
    return Trajectory("FD002", unit, "train", np.arange(1, life + 1), settings, sensors, 0)


def test_cross_fitting_holds_out_whole_units() -> None:
    """A unit's rows can be recalled from their neighbours in the same unit. Folds that split a
    unit would let m_oof recall it and make s small; folds of whole units cannot."""
    rng = np.random.default_rng(3)
    fleet = [signature_unit(rng, u, int(rng.integers(30, 90))) for u in range(12)]
    flexible = EstimatorParams(
        window=1, cap=125.0, s_min=1e-3, max_iter=40, learning_rate=0.5,
        max_leaf_nodes=31, min_samples_leaf=1,
    )  # fmt: skip
    est = Estimator.fit(fleet, flexible, seed=1)
    scale = np.concatenate([est.predict(t).s for t in fleet]).mean()
    assert scale > 6.0  # life lengths are 30 to 90: an unseen unit's life is unknown by ~15


def test_fit_input_checks(fleet: list[Trajectory]) -> None:
    with pytest.raises(ValueError):
        Estimator.fit(fleet[:1], SMALL, seed=0)
    with pytest.raises(ValueError):
        Estimator.fit(fleet, SMALL, seed=0, n_folds=1)
    est = Estimator.fit(fleet[:3], SMALL, seed=0, n_folds=9)  # folds are cut down to 3
    assert est.predict(fleet[0]).n == fleet[0].n


def test_prediction_validates() -> None:
    ok = np.ones(3)
    assert Prediction(ok, ok).n == 3
    with pytest.raises(ValueError):
        Prediction(ok, np.ones(4))
    with pytest.raises(ValueError):
        Prediction(ok, np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        Prediction(np.array([1.0, np.nan, 1.0]), ok)


# ----------------------------------------------------------------------------- the capped truth


def test_both_targets_use_the_capped_truth(
    fleet: list[Trajectory], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``s`` is fitted on ``|min(RUL, cap) - m_oof|``: with a cap of 8 and lives of 30 to 60 rows,
    the uncapped errors would reach about 50, the capped ones cannot pass about 2 x 8."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    seen: list[np.ndarray] = []
    real_fit = HistGradientBoostingRegressor.fit

    def recording_fit(self: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(y, dtype=np.float64).copy())
        return real_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", recording_fit)
    Estimator.fit(fleet, dataclasses.replace(SMALL, cap=8.0), seed=1)
    s_target = seen[-1]  # the last fit is the scale model
    assert s_target.max() < 16.0
    assert all(y.max() <= 8.0 for y in seen[:-1])  # the m models: min(RUL, 8)


def test_fit_is_the_same_on_capped_and_uncapped_trajectories(fleet: list[Trajectory]) -> None:
    params = dataclasses.replace(SMALL, cap=20.0)
    a = Estimator.fit(fleet, params, seed=4)
    b = Estimator.fit([t.capped(20) for t in fleet], params, seed=4)
    for t in fleet[:5]:
        pa, pb = a.predict(t), b.predict(t)
        assert np.array_equal(pa.m, pb.m) and np.array_equal(pa.s, pb.s)


def test_fit_refuses_trajectories_capped_at_another_value(fleet: list[Trajectory]) -> None:
    with pytest.raises(ValueError, match="capped at"):
        Estimator.fit([t.capped(30) for t in fleet], dataclasses.replace(SMALL, cap=20.0), seed=4)
    with pytest.raises(ValueError, match="whole number"):
        Estimator.fit(fleet, dataclasses.replace(SMALL, cap=20.5), seed=4)
