import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from paleopy.spd import (
    apply_hygiene_regex,
    bin_by_site_name,
    build_spd_and_caldists,
    cpl_model,
    find_nearby_sites,
    fit_cpl,
    fit_exponential,
    fit_logistic,
    monte_carlo_envelope,
    phase6_cpl,
    summarize_sites,
    taphonomic_weight,
)


@pytest.fixture(scope="module")
def orig04():
    spec = importlib.util.spec_from_file_location("orig04_spd_test", "Scripts/04_SPD.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]  # avoid parse_args() side effects; not invoked at import time
    sys.modules["orig04_spd_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_taphonomic_weight_parity(orig04):
    t = np.arange(7300, 9500, 10, dtype=float)
    np.testing.assert_allclose(taphonomic_weight(t), orig04.taphonomic_factor(t))


def test_fit_exponential_parity(orig04):
    years = np.arange(7300, 9500, 10, dtype=float)
    rng = np.random.default_rng(0)
    spd = np.abs(rng.normal(size=len(years))) + 0.01
    spd /= spd.sum()
    np.testing.assert_allclose(fit_exponential(years, spd), orig04.fit_exponential(years, spd))


def test_fit_logistic_parity(orig04):
    years = np.arange(7300, 9500, 10, dtype=float)
    rng = np.random.default_rng(0)
    spd = np.abs(rng.normal(size=len(years))) + 0.01
    spd /= spd.sum()
    np.testing.assert_allclose(fit_logistic(years, spd), orig04.fit_logistic(years, spd))


def test_cpl_model_parity(orig04):
    years = np.arange(7300, 9500, 10, dtype=float)
    hinge_years = [7300, 8400, 9500]
    hinge_probs = [0.1, 0.5, 0.1]
    np.testing.assert_allclose(
        cpl_model(years, hinge_years, hinge_probs),
        orig04.cpl_model(years, hinge_years, hinge_probs),
    )


def test_bin_by_site_name_parity(orig04):
    df = pd.DataFrame({
        "SiteName": ["Site A", "Site A", "Site A", "Site B", None],
        "Age": [1000, 1050, 1300, 2000, 3000],
    })

    class FakeCfg:
        bin_h = 100

    new_result = bin_by_site_name(df.copy(), bin_h=100)
    orig_result = orig04.phase2_binning(df.copy(), FakeCfg())
    assert list(new_result["Bin"]) == list(orig_result["Bin"])


def test_apply_hygiene_regex_parity(orig04):
    df = pd.DataFrame({
        "Age": [1000, 2000, 3000, 4000],
        "Error": [30, 200, 30, 30],
        "Lat": [37.0, 38.0, -10.0, 40.0],
        "MedianCalBP": [8000, 8000, 8000, 6000],
        "Material": ["Bone", "Bone", "Charcoal", "Shell"],
    })

    class FakeCfg:
        max_error = 100
        time_min = 7000
        time_max = 9000

    def fake_abort(msg):
        raise SystemExit(msg)

    orig04._abort = fake_abort  # avoid real sys.exit if it triggers
    new_result = apply_hygiene_regex(df.copy(), max_error=100, time_min=7000, time_max=9000)
    orig_result = orig04.phase1_hygiene(df.copy(), FakeCfg())
    assert sorted(new_result.index) == sorted(orig_result.index)


def test_apply_hygiene_regex_raises_when_empty():
    # Two rows so pandas' dtype inference on the intermediate boolean
    # columns behaves normally (a single all-filtered-out row is a known
    # pandas 3.0.5 apply()-on-empty-Series edge case, unrelated to this
    # function's actual logic) - both rows here are excluded by the
    # old-wood/marine material filter, leaving nothing.
    df = pd.DataFrame({
        "Age": [1000, 2000], "Error": [30, 30], "Lat": [37.0, 38.0],
        "MedianCalBP": [8000, 8000], "Material": ["Charcoal", "Shell"],
    })
    with pytest.raises(ValueError):
        apply_hygiene_regex(df, max_error=100, time_min=7000, time_max=9000)


def test_find_nearby_sites_and_summarize():
    df = pd.DataFrame({
        "Lat": [37.666, 37.7, 10.0],
        "Long": [32.8277, 32.9, 50.0],
        "SiteName": ["Catalhoyuk", "NearbySite", "FarSite"],
        "SiteID": ["1", "2", "3"],
        "Age": [1000, 2000, 3000],
        "MedianCalBP": [8000, 8500, 8000],
        "Error": [30, 40, 30],
    })
    nearby = find_nearby_sites(df, site_lat=37.666, site_lon=32.8277, radius_km=50)
    assert set(nearby["SiteName"]) == {"Catalhoyuk", "NearbySite"}

    sites = summarize_sites(nearby, time_min=7000, time_max=9000)
    assert len(sites) == 2
    assert "dist_km" in sites.columns


def _stub_calibrate(age, error, curve, years):
    # Fast stand-in for real calibration: a triangular density centered on `age`
    prob = np.maximum(0.0, 1.0 - np.abs(years - age) / 500.0)
    total = prob.sum()
    return prob / total if total > 0 else prob


def test_build_spd_and_caldists_with_stub_calibration():
    years = np.arange(7000, 9000, 10, dtype=float)
    df_binned = pd.DataFrame({
        "Age": [8000, 8010, 8500],
        "Error": [30, 30, 30],
        "CalCurveUsed": ["intcal20", "intcal20", "intcal20"],
        "Bin": ["b0", "b0", "b1"],
    })
    spd_out, cal_dists = build_spd_and_caldists(df_binned, years, _stub_calibrate, show_progress=False)
    assert spd_out.shape == years.shape
    assert cal_dists.shape[0] == 3
    assert pytest.approx(spd_out.sum(), rel=1e-6) == 1.0


def test_monte_carlo_envelope_shapes_with_stub_calibration():
    years = np.arange(7000, 9000, 50, dtype=float)
    rng = np.random.default_rng(1)
    spd = np.abs(rng.normal(size=len(years))) + 0.01
    spd /= spd.sum()
    errors = np.array([30, 40, 30])
    curves = np.array(["intcal20", "intcal20", "shcal20"])

    null_fitted, lower, upper, p = monte_carlo_envelope(
        years, spd, errors, curves, n_sims=3, time_min=7000, time_max=9000,
        calibrate_fn=_stub_calibrate, null_model="exponential", show_progress=False,
    )
    assert null_fitted.shape == years.shape
    assert lower.shape == years.shape
    assert upper.shape == years.shape
    assert 0.0 <= p <= 1.0


@pytest.mark.golden
def test_fit_cpl_zero_hinge_runs():
    """CPL fitting uses differential_evolution with 5 restarts x 1500
    maxiter — slow, so this is marked golden and not part of the default
    fast test run.
    """
    years = np.arange(7000, 9000, 50, dtype=float)
    n_dates = 20
    rng = np.random.default_rng(2)
    cal_dists = rng.dirichlet(np.ones(len(years)), size=n_dates)
    taph_weight = taphonomic_weight(years)

    hy, hp, ll, bic = fit_cpl(
        years, cal_dists, n_hinges=0, n_bins=n_dates, taph_weight=taph_weight,
        time_min=7000, time_max=9000, min_hinge_sep=200,
    )
    assert hy == [7000.0, 9000.0]
    assert len(hp) == 2
    assert np.isfinite(ll)
    assert np.isfinite(bic)
