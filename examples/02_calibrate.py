"""Example: calibrate a cleaned p3k14c dataset with IOSACal using the
paleopy API directly.

Replicates Scripts/02_Calibrating.py without going through the
paleopy-calibrate console script. Run from the repo root:

    python examples/02_calibrate.py
"""

from pathlib import Path

import pandas as pd

from paleopy.calibration import CURVE_NORTH, CURVE_SOUTH, calibrate_dataframe_iosacal

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    # Uses the repo's pre-cleaned dataset directly; swap for
    # examples_output/cleaned_p3k14c.csv if you ran 01_clean.py first.
    input_path = REPO_ROOT / "Datasets" / "cleaned_p3k14c.csv"

    df = pd.read_csv(input_path, low_memory=False, index_col=0)
    print(f"Loaded {len(df):,} records")

    # Calibrating the full dataset takes hours; this example uses a small
    # sample so it runs in seconds. Drop .head(200) to calibrate everything.
    sample = df.head(200)

    out_df, fail_df = calibrate_dataframe_iosacal(sample)

    out_df.to_csv(OUTDIR / "p3k14c_pristine_sample.csv")
    if not fail_df.empty:
        fail_df.to_csv(OUTDIR / "calibration_failures_sample.csv")

    n_nh = int((out_df["CalCurve"] == CURVE_NORTH).sum())
    n_sh = int((out_df["CalCurve"] == CURVE_SOUTH).sum())
    print(f"Calibrated (NH): {n_nh}  Calibrated (SH): {n_sh}  Failed: {len(fail_df)}")
    print(f"Wrote p3k14c_pristine_sample.csv -> {OUTDIR}")


if __name__ == "__main__":
    main()
