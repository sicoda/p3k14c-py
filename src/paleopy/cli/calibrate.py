"""paleopy-calibrate : calibrates every CRA in a cleaned p3k14c dataset
using IOSACal.

CLI wrapper around paleopy.calibration.calibrate_dataframe_iosacal();
mirrors Scripts/02_Calibrating.py's behavior with named arguments instead
of positional sys.argv parsing.
"""

import argparse
import sys

import pandas as pd

from paleopy.calibration import CURVE_NORTH, CURVE_SOUTH, calibrate_dataframe_iosacal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paleopy-calibrate",
        description="Calibrate every radiocarbon age in a cleaned p3k14c dataset using IOSACal.",
    )
    parser.add_argument("--input", required=True, help="Path to the cleaned p3k14c CSV (output of paleopy-clean)")
    parser.add_argument("--output", required=True, help="Path to write the calibrated ('pristine') CSV")
    parser.add_argument("--failures", required=True, help="Path to write the calibration-failures CSV")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print(f"[calibrate] Reading   : {args.input}")
    df = pd.read_csv(args.input, low_memory=False, index_col=0)
    print(f"[calibrate] Loaded    : {len(df):,} records")

    n_in = len(df)
    out_df, fail_df = calibrate_dataframe_iosacal(df, show_progress=not args.no_progress)

    if not fail_df.empty:
        fail_df.to_csv(args.failures, encoding="utf-8")
        print(f"[calibrate] Failures  : {len(fail_df):,} rows -> {args.failures}")

    out_df.to_csv(args.output, encoding="utf-8")

    n_nh = int((out_df["CalCurve"] == CURVE_NORTH).sum()) if not out_df.empty else 0
    n_sh = int((out_df["CalCurve"] == CURVE_SOUTH).sum()) if not out_df.empty else 0

    print(f"\n{'=' * 60}")
    print(f"  Input records     : {n_in:,}")
    print(f"  Calibrated (NH)   : {n_nh:,}   [{CURVE_NORTH}]")
    print(f"  Calibrated (SH)   : {n_sh:,}   [{CURVE_SOUTH}]")
    print(f"  Failed / skipped  : {n_in - len(out_df):,}")
    print(f"  Output            : {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main())
