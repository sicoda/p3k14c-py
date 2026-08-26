import pandas as pd
import pytest

from paleopy.calibration import (
    CURVE_NORTH,
    CURVE_SOUTH,
    calibrate_dataframe_iosacal,
    calibrate_row_iosacal,
    choose_curve_iosacal,
)


def test_choose_curve_northern_hemisphere():
    assert choose_curve_iosacal(37.666) == CURVE_NORTH
    assert choose_curve_iosacal(0.0) == CURVE_NORTH


def test_choose_curve_southern_hemisphere():
    assert choose_curve_iosacal(-10.0) == CURVE_SOUTH


def test_calibrate_row_iosacal_returns_expected_keys():
    result = calibrate_row_iosacal("TEST-1", age=3000, error=30, lat=37.666)
    assert result is not None
    assert set(result.keys()) == {
        "CalCurve", "MedianCalBP", "CI68_Lower", "CI68_Upper", "CI95_Lower", "CI95_Upper",
    }
    assert result["CalCurve"] == CURVE_NORTH
    assert result["CI68_Lower"] <= result["MedianCalBP"] <= result["CI68_Upper"]
    assert result["CI95_Lower"] <= result["CI68_Lower"]
    assert result["CI68_Upper"] <= result["CI95_Upper"]


def test_calibrate_row_iosacal_southern_hemisphere_uses_shcal20():
    result = calibrate_row_iosacal("TEST-2", age=3000, error=30, lat=-20.0)
    assert result["CalCurve"] == CURVE_SOUTH


def test_calibrate_dataframe_iosacal_skips_missing_fields():
    df = pd.DataFrame({
        "Age": [3000, None, 4000],
        "Error": [30, 40, 30],
        "Lat": [37.666, 10.0, 37.666],
    }, index=["A-1", "A-2", "A-3"])

    out_df, fail_df = calibrate_dataframe_iosacal(df, show_progress=False)

    assert "A-2" in fail_df.index
    assert set(out_df.index) <= {"A-1", "A-3"}
    for col in ["CalCurve", "MedianCalBP", "CI68_Lower", "CI68_Upper", "CI95_Lower", "CI95_Upper"]:
        assert col in out_df.columns


def test_calibrate_dataframe_iosacal_requires_age_error_lat_columns():
    df = pd.DataFrame({"Age": [3000]}, index=["A-1"])
    with pytest.raises(ValueError):
        calibrate_dataframe_iosacal(df, show_progress=False)
