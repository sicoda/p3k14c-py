import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from paleopy.ccsi import (
    _effective_n,
    _gaussian_kernel_bin,
    align_ccsi_spd,
    bin_proxies,
    build_spd_from_scratch,
    compute_ccsi,
    fit_resilience_with_uncertainty,
    linear_fill_short_gaps,
)


@pytest.fixture(scope="module")
def orig06():
    spec = importlib.util.spec_from_file_location("orig06_ccsi_test", "Scripts/06_Composite_Human_Environment.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig06_ccsi_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_effective_n_parity(orig06):
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert _effective_n(x) == pytest.approx(orig06._effective_n(x))


def test_linear_fill_short_gaps_parity(orig06):
    col = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, 10.0])
    new = linear_fill_short_gaps(col, max_gap=3)
    old = orig06._linear_fill_short_gaps(col, max_gap=3)
    np.testing.assert_allclose(new, old, equal_nan=True)


def test_gaussian_kernel_bin_parity(orig06):
    rng = np.random.default_rng(1)
    ages = np.sort(rng.uniform(7300, 9500, size=50))
    values = rng.normal(size=50)
    grid = np.arange(7300, 9500, 20, dtype=float)
    new = _gaussian_kernel_bin(ages, values, grid, sigma=50)
    old = orig06._gaussian_kernel_bin(ages, values, grid, 50)
    np.testing.assert_allclose(new, old, equal_nan=True)


def test_fit_resilience_with_uncertainty_parity(orig06):
    rt = np.linspace(0, 500, 20)
    rd = 1 - np.exp(-0.02 * rt) + np.random.default_rng(2).normal(0, 0.01, size=20)
    new = fit_resilience_with_uncertainty(rt, rd, cv_threshold=0.5)
    old = orig06.fit_resilience(rt, rd)
    assert new[0] == pytest.approx(old[0], rel=1e-6)
    assert new[2] == old[2]
    assert new[3] == old[3]


def test_fit_resilience_with_uncertainty_too_few_points():
    k, k_se, method, uncertain = fit_resilience_with_uncertainty(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), cv_threshold=0.5
    )
    assert np.isnan(k)
    assert np.isnan(k_se)
    assert method == "undefined"
    assert uncertain is True


def test_build_spd_from_scratch_returns_normalized_series():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame({
        "Age": rng.integers(7500, 8500, size=n).astype(float),
        "Error": rng.integers(20, 60, size=n).astype(float),
        "MedianCalBP": rng.integers(7500, 8500, size=n).astype(float),
        "SiteID": [f"S{i % 3}" for i in range(n)],
    })
    years, spd, lo, hi = build_spd_from_scratch(df, time_min=7300, time_max=9500, bin_h=50, resolution=10)
    assert spd.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(lo == 0)
    assert np.all(hi == 0)


def test_bin_proxies_drops_low_coverage_variables():
    ages_good = np.arange(7300, 9500, 20)
    # Two points clustered at the very start of the range; with a tight
    # kernel (sigma=5yr) relative to the 2200yr grid, this decays to
    # effectively zero coverage well below the 10% threshold.
    ages_sparse = np.array([7300.0, 7310.0])
    df = pd.concat([
        pd.DataFrame({"age_bp": ages_good, "value": np.random.default_rng(0).normal(size=len(ages_good)), "variable": "good_proxy"}),
        pd.DataFrame({"age_bp": ages_sparse, "value": [1.0, 2.0], "variable": "sparse_proxy"}),
    ])
    wide = bin_proxies(df, time_min=7300, time_max=9500, resolution=10, kernel_sigma_yr=5, max_interp_gap_bins=10)
    assert "good_proxy" in wide.columns
    assert "sparse_proxy" not in wide.columns


def test_bin_proxies_raises_if_nothing_survives():
    df = pd.DataFrame({"age_bp": [7300.0], "value": [1.0], "variable": "x"})
    with pytest.raises(ValueError):
        bin_proxies(df, time_min=7300, time_max=9500, resolution=10, kernel_sigma_yr=50, max_interp_gap_bins=10)


def test_compute_ccsi_firewalls_spd_and_returns_diagnostics():
    years = np.arange(7300, 9500, 10, dtype=float)
    rng = np.random.default_rng(0)
    wide_df = pd.DataFrame({
        "CalBP": years,
        "proxy_a": rng.normal(size=len(years)),
        "proxy_b": rng.normal(size=len(years)),
        "GISP2_Temp_C": rng.normal(size=len(years)),
    })
    proxy_df = pd.DataFrame({"age_bp": years, "value": wide_df["GISP2_Temp_C"], "variable": "GISP2_Temp_C"})

    out, diag = compute_ccsi(wide_df, proxy_df, min_proxies_for_pca=2, em_pca_max_iter=10, em_pca_tol=1e-3, warn_neff_threshold=30)

    assert "CCSI" in out.columns
    assert "SPD" not in " ".join(out.columns)  # SPD never enters the PCA matrix
    assert diag["n_proxies"] == 3
    assert 0 <= diag["explained_var_pc1"] <= 1


def test_align_ccsi_spd_shapes():
    years = np.arange(7300, 9500, 10, dtype=float)
    ccsi_df = pd.DataFrame({"CalBP": years, "CCSI": np.random.default_rng(0).normal(size=len(years))})
    spd_un = np.abs(np.random.default_rng(1).normal(size=len(years))) + 0.1

    t, dz, ez = align_ccsi_spd(ccsi_df, years, spd_un, time_min=7300, time_max=9500, resolution=10)
    assert len(t) == len(dz) == len(ez)
