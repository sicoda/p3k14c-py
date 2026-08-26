from pathlib import Path

import pytest

from paleopy.config import Config


def test_defaults_match_catalhoyuk_case_study():
    cfg = Config()
    assert cfg.site_name == "Catalhoyuk"
    assert cfg.site_lat == pytest.approx(37.6660)
    assert cfg.site_lon == pytest.approx(32.8277)
    assert cfg.archaeo_radius_km == 5
    assert cfg.time_min == 7300
    assert cfg.time_max == 9500
    assert cfg.max_error == 100
    assert cfg.bin_h == 100


def test_slug_replaces_spaces():
    cfg = Config(site_name="New Site Name")
    assert cfg.slug == "New_Site_Name"


def test_outdir_coerced_to_path_and_created(tmp_path):
    target = tmp_path / "some" / "dir"
    cfg = Config(outdir=target)
    assert isinstance(cfg.outdir, Path)
    assert target.is_dir()


def test_invalid_time_window_raises():
    with pytest.raises(ValueError):
        Config(time_min=9000, time_max=8000)
