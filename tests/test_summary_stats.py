import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from paleopy.summary_stats import compute_summary, default_output_name


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [1000, 2000, 3000, 4000, 5000],
        "Error": [30, 40, 50, 20, 25],
        "Lat": [37.0, 38.0, None, 40.0, 41.0],
        "Long": [32.0, 33.0, 34.0, None, 36.0],
        "MedianCalBP": [1100, 2200, 3300, 4400, 5500],
        "Country": ["Turkey", "Turkey", "Greece", "Greece", "Turkey"],
        "SiteName": ["Catalhoyuk", "Catalhoyuk", "Knossos", "Knossos", "Catalhoyuk"],
        "Continent": ["Asia", "Asia", "Europe", "Europe", "Asia"],
        "Material": ["Bone", "Charcoal", "Bone", "Shell", "Charcoal"],
    })


def test_default_output_name_global():
    assert default_output_name(None, None) == "summary_global.png"


def test_default_output_name_country():
    assert default_output_name("Turkey", None) == "summary_Turkey.png"


def test_default_output_name_site_name_replaces_spaces():
    assert default_output_name(None, "Site With Spaces") == "summary_Site_With_Spaces.png"


def test_compute_summary_global(sample_df):
    stats, fig = compute_summary(sample_df)
    assert stats["total_records"] == 5
    assert "continental_breakdown" in stats
    assert "top_countries" in stats
    assert stats["missing_values"].sum() == 2  # one missing Lat, one missing Long
    assert fig is not None


def test_compute_summary_country_filter_excludes_continental_breakdown(sample_df):
    stats, fig = compute_summary(sample_df, country="Turkey")
    assert stats["total_records"] == 3
    assert "continental_breakdown" not in stats
    # top_countries is only excluded by a site_name filter, not a country filter
    assert "top_countries" in stats


def test_compute_summary_site_name_filter(sample_df):
    stats, fig = compute_summary(sample_df, site_name="Knossos")
    assert stats["total_records"] == 2
    assert "top_countries" not in stats


def test_compute_summary_case_insensitive_filter(sample_df):
    stats, fig = compute_summary(sample_df, country="turkey")
    assert stats["total_records"] == 3


def test_compute_summary_raises_on_empty_result(sample_df):
    with pytest.raises(ValueError):
        compute_summary(sample_df, country="Nonexistent Country")
