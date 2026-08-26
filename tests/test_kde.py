import importlib.util
import ssl
import sys

import numpy as np
import pandas as pd
import pytest

from paleopy.kde import build_grid, filter_region_and_time, kde_unweighted_sites, kde_weighted_dates


@pytest.fixture(scope="module")
def orig07():
    spec = importlib.util.spec_from_file_location("orig07_kde_test", "Scripts/07_KDE.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig07_kde_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "Lat": rng.uniform(20, 60, size=n),
        "Long": rng.uniform(-120, -60, size=n),
        "MedianCalBP": rng.uniform(1000, 12000, size=n),
    })


def test_paleopy_mapping_import_has_no_global_ssl_or_recursion_side_effects():
    """Regression check: Scripts/07_KDE.py disables SSL verification and
    raises the recursion limit GLOBALLY at import time. The ported
    paleopy.mapping must not do this merely on import - only when
    build_land_mask() is actually called.

    Must run before any test that imports the *original* Scripts/07_KDE.py
    (via the orig07 fixture below), since that original script itself
    mutates this same global state at import time - this test would be
    meaningless once that contamination has already happened.
    """
    previous_ssl = ssl._create_default_https_context
    previous_limit = sys.getrecursionlimit()

    spec = importlib.util.spec_from_file_location("paleopy_mapping_reimport", "src/paleopy/mapping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert ssl._create_default_https_context is previous_ssl
    assert sys.getrecursionlimit() == previous_limit


def test_filter_region_and_time_matches_original_logic(sample_df, orig07):
    lon_range, lat_range = [-170, -50], [15, 80]
    t_min, t_max = 2000, 10000

    df_region, df_sites = filter_region_and_time(sample_df, lon_range, lat_range, t_min, t_max)

    # Replicate the original script's inline filtering (not a function
    # there - it's inlined in generate_map) for a ground-truth comparison.
    df = sample_df.dropna(subset=["Lat", "Long"]).copy()
    df["MedianCalBP"] = pd.to_numeric(df["MedianCalBP"], errors="coerce")
    df = df.dropna(subset=["MedianCalBP"])
    df = df[(df["MedianCalBP"] >= t_min) & (df["MedianCalBP"] <= t_max)]
    mask = (
        (df["Long"] >= lon_range[0]) & (df["Long"] <= lon_range[1])
        & (df["Lat"] >= lat_range[0]) & (df["Lat"] <= lat_range[1])
    )
    expected_region = df[mask]
    expected_sites = expected_region.drop_duplicates(subset=["Lat", "Long"])

    assert len(df_region) == len(expected_region)
    assert len(df_sites) == len(expected_sites)


def test_build_grid_parity(orig07):
    lon_range, lat_range = [-170, -50], [15, 80]
    X_new, Y_new, pos_new = build_grid(lon_range, lat_range, res=20)
    X_old, Y_old, pos_old = orig07.build_grid(lon_range, lat_range, res=20)
    np.testing.assert_allclose(X_new, X_old)
    np.testing.assert_allclose(Y_new, Y_old)
    np.testing.assert_allclose(pos_new, pos_old)


def test_kde_unweighted_sites_parity(orig07, sample_df):
    lon_range, lat_range = [-170, -50], [15, 80]
    X, Y, positions = build_grid(lon_range, lat_range, res=20)
    df_sites = sample_df.drop_duplicates(subset=["Lat", "Long"])
    new_Z = kde_unweighted_sites(df_sites, positions, X)
    old_Z = orig07.kde_unweighted_sites(df_sites, positions, X)
    np.testing.assert_allclose(new_Z, old_Z)


def test_kde_weighted_dates_parity(orig07, sample_df):
    lon_range, lat_range = [-170, -50], [15, 80]
    X, Y, positions = build_grid(lon_range, lat_range, res=20)
    new_Z = kde_weighted_dates(sample_df, positions, X)
    old_Z = orig07.kde_weighted_dates(sample_df, positions, X)
    np.testing.assert_allclose(new_Z, old_Z)
