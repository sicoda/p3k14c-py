import numpy as np
import pandas as pd
import pytest

from paleopy.calibration import load_intcal20
from paleopy.human_climate import build_spd, load_spd_from_04, spd_significance_envelope


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 15
    return pd.DataFrame({
        "Age": rng.integers(7500, 8500, size=n).astype(float),
        "Error": rng.integers(20, 60, size=n).astype(float),
        "MedianCalBP": rng.integers(7500, 8500, size=n).astype(float),
        "SiteID": [f"S{i % 3}" for i in range(n)],
    })


def test_build_spd_returns_normalized_series(synthetic_df):
    years, spd_un, spd_no = build_spd(synthetic_df, time_min=7300, time_max=9500, bin_h=50, resolution=10)
    assert len(years) == len(spd_un) == len(spd_no)
    assert spd_un.sum() == pytest.approx(1.0, rel=1e-6)
    assert spd_no.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(spd_un >= 0)


def test_build_spd_uses_gaussian_fallback_when_age_error_missing():
    df = pd.DataFrame({
        "Age": [np.nan], "Error": [np.nan], "MedianCalBP": [8000.0], "SiteID": ["S1"],
    })
    years, spd_un, spd_no = build_spd(df, time_min=7300, time_max=9500, bin_h=50, resolution=10)
    assert spd_un.sum() == pytest.approx(1.0, rel=1e-6)
    peak_year = years[np.argmax(spd_un)]
    assert abs(peak_year - 8000) < 200  # peak roughly where the Gaussian fallback was centered


def test_spd_significance_envelope_shapes(synthetic_df):
    years, spd_un, _ = build_spd(synthetic_df, time_min=7300, time_max=9500, bin_h=50, resolution=10)
    curve = load_intcal20()
    lo, hi, pvals = spd_significance_envelope(
        synthetic_df, years, spd_un, bin_h=50, n_sim=5, model="exponential", intcal_curve=curve,
    )
    assert lo.shape == years.shape
    assert hi.shape == years.shape
    assert pvals.shape == years.shape
    assert np.all(lo <= hi + 1e-9)


def test_load_spd_from_04_roundtrip(tmp_path):
    path = tmp_path / "site_spd_for_06.csv"
    years = np.arange(7300, 9500, 10, dtype=float)
    pd.DataFrame({
        "CalBP": years,
        "SPD_TaphCorrected": np.ones_like(years) / len(years),
        "MC_lo_2.5pct": np.zeros_like(years),
        "MC_hi_97.5pct": np.ones_like(years),
    }).to_csv(path, index=False)

    y, spd_un, lo, hi = load_spd_from_04(str(path))
    np.testing.assert_allclose(y, years)
    assert spd_un.sum() == pytest.approx(1.0, rel=1e-6)
