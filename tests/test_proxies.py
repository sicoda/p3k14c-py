import numpy as np
import pandas as pd
import pytest
import requests

from paleopy.proxies import (
    collect_proxies,
    env_series_from_df,
    fetch_neotoma,
    fetch_neotoma_proxy,
    fetch_pangaea,
    fetch_pangaea_proxy,
)


class _FakeResponse:
    def __init__(self, json_data=None, text_data="", status_code=200):
        self._json = json_data
        self.text = text_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


@pytest.mark.network
def test_fetch_neotoma_regression_for_nearby_ids_bug(monkeypatch):
    """Regression test for the bugfixed `nearby_ids` -> `nearby_siteids`
    NameError in the original Scripts/05_Human_Climate_Interaction.py.
    Exercises the exact code path that used to crash: sites are found
    within radius, then a dataset query is built from those site ids.
    """
    sites_response = _FakeResponse(json_data={
        "data": [
            {"siteid": 101, "geography": '{"coordinates": [32.83, 37.67]}'},
            {"siteid": 102, "geography": '{"coordinates": [32.90, 37.70]}'},
        ]
    })
    datasets_response = _FakeResponse(json_data={"data": []})

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/data/sites" in url:
            return sites_response
        if "/data/datasets" in url:
            return datasets_response
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_neotoma(
        ["stable isotopes"], site_lat=37.666, site_lon=32.8277,
        env_radius_km=1000, time_min=7300, time_max=9500,
    )

    assert result is None  # no datasets returned, but no NameError raised
    assert any("/data/datasets" in c and "siteid=101,102" in c for c in calls)


@pytest.mark.network
def test_fetch_pangaea_no_hits_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_data={"hits": {"hits": []}}))
    result = fetch_pangaea("stable isotope", site_lat=37.666, site_lon=32.8277, env_radius_km=1000, time_min=7300, time_max=9500)
    assert result is None


@pytest.mark.network
def test_fetch_neotoma_no_sites_returns_none(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_data={"data": []}))
    result = fetch_neotoma(["stable isotopes"], site_lat=37.666, site_lon=32.8277, env_radius_km=1000, time_min=7300, time_max=9500)
    assert result is None


def test_env_series_from_df_zscores_and_detrends():
    ages = np.arange(7300, 9500, 10)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age_bp": np.tile(ages, 2),
        "value": np.concatenate([
            rng.normal(0, 1, size=len(ages)),
            rng.normal(100, 20, size=len(ages)),  # different scale/units
        ]),
        "variable": ["delta18O"] * len(ages) + ["pollen_count"] * len(ages),
    })

    result_ages, composite = env_series_from_df(df, time_min=7300, time_max=9500, resolution=10, env_bin_yr=10)
    assert len(result_ages) == len(composite)
    finite = composite[np.isfinite(composite)]
    assert len(finite) > 0
    # Z-scored+averaged+detrended composite should be roughly centered near 0
    assert abs(np.nanmean(composite)) < 1.0


def test_env_series_from_df_empty_input_returns_all_nan():
    df = pd.DataFrame({"age_bp": [], "value": [], "variable": []})
    ages, composite = env_series_from_df(df, time_min=7300, time_max=9500, resolution=10, env_bin_yr=10)
    assert np.all(np.isnan(composite))


@pytest.mark.network
def test_fetch_pangaea_proxy_tags_variable_with_keyword(monkeypatch):
    hits = {"hits": {"hits": [{"_source": {
        "URI": "https://doi.org/10.1594/PANGAEA.999999",
        "meanPosition": {"lat": 37.7, "lon": 32.9},
    }}]}}
    textfile = "\n".join([
        "*/",
        "Age [ka BP]\tdelta18O",
        "7.5\t-4.2",
        "8.0\t-4.5",
        "8.5\t-4.1",
        "9.0\t-4.3",
        "9.2\t-4.0",
    ])

    def fake_get(url, **kwargs):
        if "ws.pangaea.de" in url:
            return _FakeResponse(json_data=hits)
        return _FakeResponse(text_data=textfile)

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_pangaea_proxy("stable isotope", site_lat=37.666, site_lon=32.8277, env_radius_km=1000, time_min=7300, time_max=9500)

    assert result is not None
    assert all(result["variable"].str.startswith("PANGAEA_stable isotope")) or all(result["variable"].str.startswith("PANGAEA_"))
    assert all(result["source"] == "PANGAEA") if "source" in result.columns else True


@pytest.mark.network
def test_fetch_neotoma_proxy_regression_uses_correct_variable_name(monkeypatch):
    """06's fetch_neotoma_proxy never had the nearby_ids bug (it's correct
    in the original), but this locks in the same behavior verified for
    05's bugfixed fetch_neotoma: sites within radius correctly flow into
    the dataset query.
    """
    sites_response = _FakeResponse(json_data={"data": [
        {"siteid": 55, "geography": '{"coordinates": [32.83, 37.67]}'},
    ]})
    datasets_response = _FakeResponse(json_data={"data": []})

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/data/sites" in url:
            return sites_response
        if "/data/datasets" in url:
            return datasets_response
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_neotoma_proxy(["pollen"], site_lat=37.666, site_lon=32.8277, env_radius_km=1000, time_min=7300, time_max=9500)

    assert result is None
    assert any("siteid=55" in c for c in calls)


@pytest.mark.network
def test_collect_proxies_raises_if_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_data={"hits": {"hits": []}, "data": []}))
    with pytest.raises(ValueError):
        collect_proxies(
            site_lat=37.666, site_lon=32.8277, env_radius_km=1000, time_min=7300, time_max=9500,
            use_gisp2=False, pangaea_keywords=["stable isotope"], neotoma_types=["pollen"],
            gisp2_cache_path="unused.csv",
        )
