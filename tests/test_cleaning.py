import numpy as np
import pandas as pd
import pytest

from paleopy.cleaning import (
    _col_fix,
    _convert_coord,
    _deg_min_sec_to_dec,
    _is_integer,
    _solheim_to_dec,
    apply_misc_scrubbing,
    clean,
    code_from_lab_num,
    is_corrupted_unicode,
    standardize_lab_id,
)


def test_code_from_lab_num_strips_digits_and_symbols():
    assert code_from_lab_num("Beta-123456") == "beta"
    assert code_from_lab_num("GXO-676") == "gxo"


def test_standardize_lab_id_inserts_dash_between_letters_and_digits():
    assert standardize_lab_id("beta123456") == "BETA-123456"
    assert standardize_lab_id("beta-123456") == "BETA-123456"


def test_standardize_lab_id_no_digits_returns_unchanged():
    assert standardize_lab_id("nodigits") == "NODIGITS"


def test_is_corrupted_unicode_detects_garbled_text():
    assert is_corrupted_unicode("normal text 123") is False
    assert is_corrupted_unicode("â€™garbled") is True


def test_deg_min_sec_to_dec_basic():
    # 37 degrees 30 minutes north -> 37.5
    assert _deg_min_sec_to_dec("37*30'", "lat") == pytest.approx(37.5)


def test_deg_min_sec_to_dec_longitude_is_negated():
    assert _deg_min_sec_to_dec("32*30'", "long") == pytest.approx(-32.5)


def test_solheim_to_dec():
    result = _solheim_to_dec("37 30N")
    assert result == pytest.approx(37.5)


def test_convert_coord_passes_through_plain_float():
    assert _convert_coord(37.5, "lat") == 37.5


def test_is_integer_true_for_whole_numbers():
    assert _is_integer(5) is True
    assert _is_integer("5.0") is True
    assert _is_integer(5.5) is False
    assert _is_integer("not a number") is False


def test_col_fix_strips_quotes_and_exotic_whitespace():
    assert _col_fix('say "hi"') == "say hi"
    assert _col_fix(float("nan")) is np.nan or (
        isinstance(_col_fix(float("nan")), float) and _col_fix(float("nan")) != _col_fix(float("nan"))
    )


def test_clean_end_to_end_on_synthetic_dataframe():
    df = pd.DataFrame({
        "LabID": ["Beta-1", "Beta-2", "???-3", "Beta-4"],
        "Age": ["1000", "2000", "3000", "not_a_number"],
        "Error": ["30", "40", "50", "30"],
        "Lat": ["37.5", "38.0", "39.0", "40.0"],
        "Long": ["32.5", "33.0", "34.0", "35.0"],
        "SiteName": [" Site A ", "Site B", "Site C", "Site D"],
        "SiteID": ["1", "2", "3", "4"],
        "Country": ["United States", "Turkey", "Turkey", "Turkey"],
    })

    cleaned, graveyard, unknown_codes = clean(df, labs_path=None, family_tree_path=None)

    # Row with unparseable Age should be removed (non-integer age)
    assert len(cleaned) < len(df)
    assert "USA" in cleaned["Country"].values
    assert not graveyard.empty
