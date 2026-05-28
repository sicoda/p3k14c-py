"""
02_Calibrate_Dates.py : Calibrates every conventional radiocarbon age (CRA) in cleaned_p3k14c.csv using IOSACal and the appropriate calibration curve:
 
 - Lat >= 0  (Northern Hemisphere)  →  IntCal20
 - Lat <  0  (Southern Hemisphere)  →  SHCal20

Input  : cleaned_p3k14c.csv (output of 01_Data_Cleaning_and_Prep)
Output : p3k14c_pristine_dates.csv, calibration_failures.csv
 
DEPENDENCIES : pip install iosacal pandas tqdm
PYTHON       : 3.12+
IOSACal      : 0.7+ / 0.8+
"""
 
import sys
import warnings
from pathlib import Path
import os
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable
 
try:
    from iosacal import R
except ImportError:
    sys.exit(
        "ERROR: IOSACal is not installed.\n"
        "Install it with:  pip install iosacal"

 
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
 
INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_SCRIPT_DIR, "cleaned_p3k14c.csv")
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv")
FAIL_CSV   = os.path.join(_SCRIPT_DIR, "calibration_failures.csv")
 
CURVE_NORTH = "intcal20"   # Northern Hemisphere  (Lat >= 0)
CURVE_SOUTH = "shcal20"    # Southern Hemisphere  (Lat <  0)
 
NEW_COLS = [
    "CalCurve",     # which calibration curve was used
    "MedianCalBP",  # median calibrated age in Cal BP
    "CI68_Lower",   # 68.2% (1σ) lower bound  (Cal BP)
    "CI68_Upper",   # 68.2% (1σ) upper bound  (Cal BP)
    "CI95_Lower",   # 95.4% (2σ) lower bound  (Cal BP)
    "CI95_Upper",   # 95.4% (2σ) upper bound  (Cal BP)
]
 
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
 
def choose_curve(lat: float) -> str:
    """Return IntCal20 for Northern Hemisphere, SHCal20 for Southern"""
    return CURVE_SOUTH if lat < 0 else CURVE_NORTH
 
 
def calibrate_row(lab_id: str, age: int, error: int, lat: float) -> dict | None:
    """
    Calibrate one CRA with IOSACal.
 
    Uses R(age, error, id).calibrate(curve) then CalAge.quantiles() to
    extract the median and 68 / 95% confidence intervals.
 
    Returns a dict of new-column values, None if calibration fails.
    """
    curve = choose_curve(lat)
    try:
        det    = R(age, error, lab_id)
        cal    = det.calibrate(curve)
        quants = cal.quantiles()
 
        # quants[50]  -> scalar  median Cal BP
        # quants[68]  -> [lower, upper]  68.2% CI
        # quants[95]  -> [lower, upper]  95.4% CI
        median = float(quants[50])
        ci68   = quants[68]
        ci95   = quants[95]
 
        return {
            "CalCurve":    curve,
            "MedianCalBP": round(median, 1),
            "CI68_Lower":  round(float(ci68[0]), 1),
            "CI68_Upper":  round(float(ci68[1]), 1),
            "CI95_Lower":  round(float(ci95[0]), 1),
            "CI95_Upper":  round(float(ci95[1]), 1),
        }
 
    except Exception as exc:
        warnings.warn(
            f"Calibration failed for {lab_id} "
            f"({age} +/- {error} BP, Lat={lat:.2f}): {exc}"
        )
        return None
 
 
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
 
def main() -> None:
 
    # -- Load -------------------------------------------------------------------
    if not Path(INPUT_CSV).is_file():
        sys.exit(f"ERROR: Input file not found: {INPUT_CSV}")
 
    print(f"[calibrate] Reading   : {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False, index_col=0)
    print(f"[calibrate] Loaded    : {len(df):,} records")
 
    # -- Validate required columns ----------------------------------------------
    required = {"Age", "Error", "Lat"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Input CSV is missing required columns: {missing}")
 
    df["Age"]   = pd.to_numeric(df["Age"],   errors="coerce")
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")
    df["Lat"]   = pd.to_numeric(df["Lat"],   errors="coerce")
 
    # -- Separate rows that can't be calibrated ---------------------------------
    invalid_mask = df[["Age", "Error", "Lat"]].isna().any(axis=1)
    invalid_df   = df[invalid_mask].copy()
    valid_df     = df[~invalid_mask].copy()
 
    if len(invalid_df):
        print(f"[calibrate] Skipping  : {len(invalid_df):,} rows "
              f"(missing Age / Error / Lat)")
 
    print(f"[calibrate] Calibrating {len(valid_df):,} dates ...")
 
    # -- Calibrate row by row ---------------------------------------------------
    cal_records = []   # (lab_id, result_dict)  for successes
    fail_ids    = []   # lab_ids for failures
 
    iterator = tqdm(
        valid_df.iterrows(),
        total=len(valid_df),
        desc="Calibrating",
        unit="date",
    ) if HAS_TQDM else valid_df.iterrows()
 
    for lab_id, row in iterator:
        result = calibrate_row(
            lab_id=str(lab_id),
            age=int(row["Age"]),
            error=int(row["Error"]),
            lat=float(row["Lat"]),
        )
        if result is not None:
            cal_records.append((lab_id, result))
        else:
            fail_ids.append(lab_id)
 
    # -- Build output DataFrame ------------------------------------------------
    if cal_records:
        good_index  = [r[0] for r in cal_records]
        good_values = [r[1] for r in cal_records]
        cal_df      = pd.DataFrame(good_values, index=good_index)
 
        out_df = valid_df.loc[good_index].copy()
        for col in NEW_COLS:
            out_df[col] = cal_df[col]
    else:
        out_df = pd.DataFrame(columns=list(df.columns) + NEW_COLS)
 
    # -- Save failures ---------------------------------------------------------
    fail_parts = []
    if fail_ids:
        fail_parts.append(valid_df.loc[fail_ids])
    if len(invalid_df):
        fail_parts.append(invalid_df)
 
    if fail_parts:
        pd.concat(fail_parts).to_csv(FAIL_CSV, encoding="utf-8")
        total_fails = sum(len(p) for p in fail_parts)
        print(f"[calibrate] Failures  : {total_fails:,} rows → {FAIL_CSV}")
 
    # -- Save output -----------------------------------------------------------
    out_df.to_csv(OUTPUT_CSV, encoding="utf-8")
 
    # -- Summary ---------------------------------------------------------------
    n_nh = int((out_df["CalCurve"] == CURVE_NORTH).sum()) if not out_df.empty else 0
    n_sh = int((out_df["CalCurve"] == CURVE_SOUTH).sum()) if not out_df.empty else 0
 
    print(f"\n{'='*60}")
    print(f"  Input records     : {len(df):,}")
    print(f"  Calibrated (NH)   : {n_nh:,}   [{CURVE_NORTH}]")
    print(f"  Calibrated (SH)   : {n_sh:,}   [{CURVE_SOUTH}]")
    print(f"  Failed / skipped  : {len(df) - len(out_df):,}")
    print(f"  Output            : {OUTPUT_CSV}")
    print(f"{'='*60}")
    print("\nNew columns in output:")
    print("  CalCurve     — calibration curve used (intcal20 or shcal20)")
    print("  MedianCalBP  — median calibrated age  (Cal BP)")
    print("  CI68_Lower   — 68.2% (1σ) lower bound (Cal BP)")
    print("  CI68_Upper   — 68.2% (1σ) upper bound (Cal BP)")
    print("  CI95_Lower   — 95.4% (2σ) lower bound (Cal BP)")
    print("  CI95_Upper   — 95.4% (2σ) upper bound (Cal BP)")
 
 
if __name__ == "__main__":
    main()