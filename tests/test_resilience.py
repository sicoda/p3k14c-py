import importlib.util
import sys

import numpy as np
import pytest

from paleopy.resilience import align, detect_anomalies, fit_resilience, sg_smooth


@pytest.fixture(scope="module")
def orig05():
    spec = importlib.util.spec_from_file_location("orig05_resilience_test", "Scripts/05_Human_Climate_Interaction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig05_resilience_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sg_smooth_parity(orig05):
    rng = np.random.default_rng(0)
    arr = rng.normal(size=100)
    arr[10:15] = np.nan
    np.testing.assert_allclose(sg_smooth(arr), orig05.sg_smooth(arr), equal_nan=True)


def test_sg_smooth_short_array_returns_copy_unchanged():
    arr = np.array([1.0, 2.0, np.nan])
    result = sg_smooth(arr)
    assert result[2] != result[2]  # still NaN
    assert result[0] == 1.0


def test_fit_resilience_parity(orig05):
    rt = np.linspace(0, 500, 20)
    rd = 1 - np.exp(-0.02 * rt) + np.random.default_rng(1).normal(0, 0.01, size=20)
    a = fit_resilience(rt, rd)
    b = orig05.fit_resilience(rt, rd)
    assert a == pytest.approx(b, rel=1e-6)


def test_fit_resilience_too_few_points_returns_nan():
    assert np.isnan(fit_resilience(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_align_parity(orig05):
    years_spd = np.arange(7000, 9000, 10, dtype=float)
    rng = np.random.default_rng(2)
    spd = np.abs(rng.normal(size=len(years_spd)))
    env_ages = np.arange(7000, 9000, 10, dtype=float)
    env_vals = rng.normal(size=len(env_ages))
    env_vals[:20] = np.nan  # simulate a shorter valid env range

    t_new, dz_new, ez_new = align(years_spd, spd, env_ages, env_vals, time_min=7000, time_max=9000, resolution=10)

    orig05.TIME_MIN, orig05.TIME_MAX, orig05.RESOLUTION = 7000, 9000, 10
    t_old, dz_old, ez_old = orig05.align(years_spd, spd, env_ages, env_vals)

    np.testing.assert_allclose(t_new, t_old)
    np.testing.assert_allclose(dz_new, dz_old)
    np.testing.assert_allclose(ez_new, ez_old)


def test_detect_anomalies_finds_episode_and_matches_original(orig05, tmp_path, monkeypatch):
    t = np.arange(7000, 8000, 10, dtype=float)
    rng = np.random.default_rng(3)
    demo = np.abs(rng.normal(0, 0.1, size=len(t))) + 1.0
    env = np.zeros(len(t))
    env[40:70] = -2.0  # a clear anomaly dip

    new_records = detect_anomalies(
        t, demo, env, anomaly_threshold=-1.0, recovery_steps=10, baseline_window=400, resolution=10,
    )

    orig05.ENV_ANOMALY_THRESHOLD, orig05.ENV_RECOVERY_STEPS = -1.0, 10
    orig05.BASELINE_WINDOW, orig05.RESOLUTION = 400, 10
    # The original writes OUTPUT_RES as a side effect if any episodes are
    # found - redirect it to a scratch path so this test doesn't write
    # into the repo.
    monkeypatch.setattr(orig05, "OUTPUT_RES", str(tmp_path / "resilience_test.csv"))
    old_df = orig05.detect_anomalies(t, demo, env)

    assert len(new_records) == len(old_df)
    if new_records:
        assert new_records[0]["onset_bp"] == old_df.iloc[0]["onset_bp"]
        old_resistance = old_df.iloc[0]["resistance"]
        if np.isnan(old_resistance):
            assert np.isnan(new_records[0]["resistance"])
        else:
            assert new_records[0]["resistance"] == pytest.approx(old_resistance)
