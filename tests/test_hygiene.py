import pandas as pd
import pytest

from paleopy.calibration import apply_chronometric_hygiene


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [1000, 2000, 3000, 4000, 5000, 6000],
        "Error": [30, 40, 200, 30, 30, 30],  # row 2 has too-high error
        "Lat": [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
        "MedianCalBP": [8000, 8500, 8000, 8000, 6000, 8000],  # row 4 outside window
        "Material": ["Bone", "Charcoal", "Shell", "Bone", "Bone", "Bone"],
        "LocAccuracy": [1, 1, 1, 0, 1, 1],  # row 3 has bad LocAccuracy
    })


def test_drops_high_error_rows(sample_df):
    result = apply_chronometric_hygiene(sample_df, max_error=100, time_min=7000, time_max=9000)
    assert 2 not in result.index or sample_df.loc[2, "Error"] <= 100


def test_drops_old_wood_and_marine_materials(sample_df):
    result = apply_chronometric_hygiene(sample_df, max_error=1000, time_min=0, time_max=10000)
    assert not result["Material"].str.lower().isin(["charcoal", "shell"]).any()


def test_drops_outside_time_window(sample_df):
    result = apply_chronometric_hygiene(sample_df, max_error=1000, time_min=7000, time_max=9000)
    assert (result["MedianCalBP"].between(7000, 9000)).all()


def test_drops_bad_loc_accuracy(sample_df):
    result = apply_chronometric_hygiene(sample_df, max_error=1000, time_min=0, time_max=10000)
    assert (result["LocAccuracy"] >= 1).all()


def test_missing_loc_accuracy_column_is_fine():
    df = pd.DataFrame({
        "Age": [1000], "Error": [30], "Lat": [37.0],
        "MedianCalBP": [8000], "Material": ["Bone"],
    })
    result = apply_chronometric_hygiene(df, max_error=100, time_min=7000, time_max=9000)
    assert len(result) == 1


def test_parity_with_original_scripts_05_and_06():
    """Direct numeric/behavioral parity check against the actual
    Scripts/05 and Scripts/06 hygiene functions on identical input,
    confirming they're truly interchangeable before being unified here.
    """
    import importlib.util
    import sys

    df = pd.DataFrame({
        "Age": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "Error": [30, 40, 200, 30, 30, 30, 9999],
        "Lat": [37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0],
        "MedianCalBP": [8000, 8500, 8000, 8000, 6000, 8000, 8000],
        "Material": ["Bone", "Charcoal", "Shell", "Timber", "Bone", "Coral", "Bone"],
        "LocAccuracy": [1, 1, 1, 0, 1, 1, 1],
    })

    spec5 = importlib.util.spec_from_file_location(
        "orig05_hygiene_test", "Scripts/05_Human_Climate_Interaction.py"
    )
    orig5 = importlib.util.module_from_spec(spec5)
    sys.modules["orig05_hygiene_test"] = orig5
    spec5.loader.exec_module(orig5)

    spec6 = importlib.util.spec_from_file_location(
        "orig06_hygiene_test", "Scripts/06_Composite_Human_Environment.py"
    )
    orig6 = importlib.util.module_from_spec(spec6)
    sys.modules["orig06_hygiene_test"] = orig6
    spec6.loader.exec_module(orig6)

    result_new = apply_chronometric_hygiene(
        df.copy(), max_error=orig5.MAX_ERROR, time_min=orig5.TIME_MIN, time_max=orig5.TIME_MAX
    )
    result_05 = orig5.apply_hygiene(df.copy())
    result_06 = orig6._apply_hygiene(df.copy())

    assert sorted(result_new.index) == sorted(result_05.index)
    assert sorted(result_05.index) == sorted(result_06.index)
