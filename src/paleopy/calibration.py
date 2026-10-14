"""Radiocarbon calibration methods and chronometric hygiene filtering.

Per an explicit project decision, the three calibration methods used
across the original scripts are NOT consolidated into one shared
implementation — they use different libraries/algorithms and give
numerically different results:

- IOSACal (Bayesian), from Scripts/02_Calibrating.py
- the third-party `radiocarbon` package (grid-integration), from
  Scripts/04_SPD.py
- a hand-rolled Gaussian-quadrature calibration against a self-downloaded
  IntCal20 curve, from Scripts/05/06

Each lives in its own clearly-separated section below. Chronometric
hygiene filtering (old-wood/marine material exclusion, error thresholds,
time-window filtering) is folded into this module too since it's
conceptually part of "is this date usable / how do we calibrate it."
"""

import os
import warnings

import numpy as np
import pandas as pd

from paleopy.utils import tqdm

# ---------------------------------------------------------------------------
# IOSACal calibration (Scripts/02_Calibrating.py)
# ---------------------------------------------------------------------------

CURVE_NORTH = "intcal20"  # Northern Hemisphere (Lat >= 0)
CURVE_SOUTH = "shcal20"  # Southern Hemisphere (Lat < 0)

IOSACAL_NEW_COLS = [
    "CalCurve",     # which calibration curve was used
    "MedianCalBP",  # median calibrated age in Cal BP
    "CI68_Lower",   # 68.2% (1sigma) lower bound (Cal BP)
    "CI68_Upper",   # 68.2% (1sigma) upper bound (Cal BP)
    "CI95_Lower",   # 95.4% (2sigma) lower bound (Cal BP)
    "CI95_Upper",   # 95.4% (2sigma) upper bound (Cal BP)
]


def choose_curve_iosacal(lat: float) -> str:
    """Return IntCal20 for Northern Hemisphere, SHCal20 for Southern."""
    return CURVE_SOUTH if lat < 0 else CURVE_NORTH


def calibrate_row_iosacal(lab_id: str, age: int, error: int, lat: float) -> dict | None:
    """Calibrate one conventional radiocarbon age (CRA) with IOSACal.

    Uses R(age, error, id).calibrate(curve) then CalAge.quantiles() to
    extract the median and 68 / 95% confidence intervals.

    Returns a dict of new-column values, None if calibration fails. Import
    of `iosacal` is deferred to call time since it's an optional extra
    (`paleopy[calibration]`).
    """
    try:
        from iosacal import R
    except ImportError as exc:
        raise ImportError(
            "IOSACal is required for calibrate_row_iosacal(). "
            "Install it with: pip install paleopy[calibration]"
        ) from exc

    curve = choose_curve_iosacal(lat)
    try:
        det = R(age, error, lab_id)
        cal = det.calibrate(curve)
        quants = cal.quantiles()

        # quants[50] -> scalar  median Cal BP
        # quants[68] -> [lower, upper]  68.2% CI
        # quants[95] -> [lower, upper]  95.4% CI
        median = float(quants[50])
        ci68 = quants[68]
        ci95 = quants[95]

        return {
            "CalCurve": curve,
            "MedianCalBP": round(median, 1),
            "CI68_Lower": round(float(ci68[0]), 1),
            "CI68_Upper": round(float(ci68[1]), 1),
            "CI95_Lower": round(float(ci95[0]), 1),
            "CI95_Upper": round(float(ci95[1]), 1),
        }
    except Exception as exc:
        warnings.warn(
            f"Calibration failed for {lab_id} "
            f"({age} +/- {error} BP, Lat={lat:.2f}): {exc}"
        )
        return None


def calibrate_dataframe_iosacal(
    df: pd.DataFrame, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate every row of a cleaned p3k14c dataframe (indexed by LabID)
    using IOSACal.

    Returns (out_df, fail_df): out_df has the original columns plus
    IOSACAL_NEW_COLS for successfully calibrated rows; fail_df holds rows
    that were skipped (missing Age/Error/Lat) or failed calibration.

    Pure port of Scripts/02_Calibrating.py's main() body, minus the
    file I/O and CLI concerns (argument parsing, printing, saving), which
    live in paleopy.cli.calibrate.
    """
    required = {"Age", "Error", "Lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Error"] = pd.to_numeric(df["Error"], errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")

    invalid_mask = df[["Age", "Error", "Lat"]].isna().any(axis=1)
    invalid_df = df[invalid_mask].copy()
    valid_df = df[~invalid_mask].copy()

    cal_records = []  # (lab_id, result_dict) for successes
    fail_ids = []  # lab_ids for failures

    iterator = valid_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(valid_df), desc="Calibrating", unit="date")

    for lab_id, row in iterator:
        result = calibrate_row_iosacal(
            lab_id=str(lab_id),
            age=int(row["Age"]),
            error=int(row["Error"]),
            lat=float(row["Lat"]),
        )
        if result is not None:
            cal_records.append((lab_id, result))
        else:
            fail_ids.append(lab_id)

    if cal_records:
        good_index = [r[0] for r in cal_records]
        good_values = [r[1] for r in cal_records]
        cal_df = pd.DataFrame(good_values, index=good_index)

        out_df = valid_df.loc[good_index].copy()
        for col in IOSACAL_NEW_COLS:
            out_df[col] = cal_df[col]
    else:
        out_df = pd.DataFrame(columns=list(df.columns) + IOSACAL_NEW_COLS)

    fail_parts = []
    if fail_ids:
        fail_parts.append(valid_df.loc[fail_ids])
    if len(invalid_df):
        fail_parts.append(invalid_df)
    fail_df = pd.concat(fail_parts) if fail_parts else pd.DataFrame(columns=df.columns)

    return out_df, fail_df


# ---------------------------------------------------------------------------
# Chronometric hygiene (Scripts/05_Human_Climate_Interaction.py /
# 06_Composite_Human_Environment.py)
# ---------------------------------------------------------------------------
#
# NOTE: this is NOT the same algorithm as Scripts/04_SPD.py's
# phase1_hygiene() — 04 detects "old wood" via a word-boundary regex
# (`\bcharcoal\b|charcoal-|\btimber\b|\bwood\b`) applied only to that
# check, has no LocAccuracy filter, and aborts the run if nothing
# survives. 05 and 06 are verified byte-identical to each other (same
# OLD_WOOD/MARINE term sets, same substring-containment check combining
# both categories in one pass, same LocAccuracy>=1 filter, no abort), so
# only those two are unified here. 04's distinct version stays local to
# paleopy.spd (ported in a later stage) rather than being forced into
# this shared function.

OLD_WOOD_TERMS = {
    "charcoal", "wood", "timber",
    "unidentified wood", "unidentified charcoal", "charred wood",
}
MARINE_TERMS = {
    "shell", "marine shell", "marine", "coral",
    "rangia", "macoma", "oyster", "mussel", "clam",
}


def apply_chronometric_hygiene(
    df: pd.DataFrame,
    max_error: float,
    time_min: float,
    time_max: float,
    old_wood_terms: set = None,
    marine_terms: set = None,
) -> pd.DataFrame:
    """Chronometric hygiene filter shared by the human-climate (05) and
    CCSI (06) pipelines: drops rows with missing Age/Error/Lat/MedianCalBP,
    high lab error, old-wood/marine materials, dates outside the Cal BP
    window, and (if present) LocAccuracy < 1.
    """
    old_wood_terms = OLD_WOOD_TERMS if old_wood_terms is None else old_wood_terms
    marine_terms = MARINE_TERMS if marine_terms is None else marine_terms

    df = df.dropna(subset=["Age", "Error", "Lat", "MedianCalBP"])
    df = df[df["Error"] <= max_error].copy()
    mat = df["Material"].fillna("").str.lower().str.strip()
    excluded_terms = old_wood_terms | marine_terms
    df = df[~mat.apply(lambda m: any(t in m for t in excluded_terms))].copy()
    df = df[df["MedianCalBP"].between(time_min, time_max)].copy()
    if "LocAccuracy" in df.columns:
        df["LocAccuracy"] = pd.to_numeric(df["LocAccuracy"], errors="coerce")
        df = df[df["LocAccuracy"] >= 1].copy()
    return df


# ---------------------------------------------------------------------------
# `radiocarbon` package calibration (Scripts/04_SPD.py)
# ---------------------------------------------------------------------------
#
# Distinct from both the IOSACal (Bayesian) method above and the
# hand-rolled Gaussian-quadrature method used by 05/06 — this uses true
# grid integration against the `radiocarbon` package's bundled
# IntCal20/SHCal20 curves. Kept as its own section per the project
# decision not to consolidate calibration methods.


def choose_curve_radiocarbon_pkg(lat: float) -> str:
    return "shcal20" if lat < 0 else "intcal20"


def calibrate_date_radiocarbon_pkg(
    c14_age: int, c14_error: int, curve_name: str, years: np.ndarray
) -> np.ndarray:
    """Calibrate a single radiocarbon determination onto a fixed Cal BP
    grid via Bayesian grid integration against IntCal20/SHCal20
    (Reimer et al. 2020), using the `radiocarbon` package.

    Preserves multi-modality and plateau artefacts. The resulting
    probability array is used both for SPD construction and the CPL
    likelihood.
    """
    try:
        from radiocarbon import Date as RC14Date
    except ImportError as exc:
        raise ImportError(
            "The 'radiocarbon' package is required for calibrate_date_radiocarbon_pkg(). "
            "Install it with: pip install paleopy[calibration]"
        ) from exc

    d = RC14Date(c14_age, c14_error, curve=curve_name)
    d.calibrate()
    cd = d.cal_date  # shape (n, 3): [cal_bp, unnorm, norm]
    order = np.argsort(cd[:, 0])
    prob = np.interp(years, cd[order, 0], cd[order, 2], left=0.0, right=0.0)
    total = prob.sum()
    return prob / total if total > 0 else prob


def calibration_curve_grid(curve_name: str, t_min: int, t_max: int) -> tuple:
    """Extract the (t_grid, mu_grid, sigma_grid, c14_grid) arrays for one
    calibration curve, windowed to [t_min, t_max], from the `radiocarbon`
    package's bundled CALIBRATION_CURVES table. Used by the Monte Carlo
    NHST envelope in paleopy.spd to back-project null-model densities
    onto 14C-age space.
    """
    try:
        from radiocarbon import CALIBRATION_CURVES
    except ImportError as exc:
        raise ImportError(
            "The 'radiocarbon' package is required for calibration_curve_grid(). "
            "Install it with: pip install paleopy[calibration]"
        ) from exc

    curve = CALIBRATION_CURVES[curve_name]
    cal_bp_arr = curve[:, 0][::-1]
    c14_arr = curve[:, 1][::-1]
    sig_arr = curve[:, 2][::-1]
    mask = (cal_bp_arr >= t_min) & (cal_bp_arr <= t_max)
    t_grid, mu_grid, sg_grid = cal_bp_arr[mask], c14_arr[mask], sig_arr[mask]
    c14_grid = np.arange(
        int(mu_grid.min() - 5 * sg_grid.max()),
        int(mu_grid.max() + 5 * sg_grid.max()) + 1, dtype=float,
    )
    return t_grid, mu_grid, sg_grid, c14_grid


# ---------------------------------------------------------------------------
# Gaussian-quadrature calibration against a self-downloaded IntCal20 curve
# (Scripts/05_Human_Climate_Interaction.py / 06_Composite_Human_Environment.py)
# ---------------------------------------------------------------------------
#
# Distinct from both IOSACal (Bayesian) and the `radiocarbon` package
# (grid-integration) above — this combines measurement + curve
# uncertainty in quadrature and computes a Gaussian likelihood, not true
# calibration-curve integration. Per project decision, kept separate
# rather than swapped for either of the other two methods.
#
# load_intcal20() is lazy/memoized here (unlike the original scripts,
# which called it as an IMPORT-TIME side effect — merely importing
# Scripts/05 or 06 triggered a network download if no cache existed).
# Importing paleopy.calibration never touches the network; the curve is
# only fetched the first time calibrate_gaussian_intcal() (or
# load_intcal20() directly) is actually called.

_INTCAL20_CACHE_STATE = {"loaded": None}


def load_intcal20(cache_path: str = "intcal20.npz") -> tuple:
    """Load the IntCal20 calibration curve, downloading and caching it on
    first use. Returns (cal_bp, c14_age, c14_error) arrays sorted
    ascending by cal_bp. Memoized in-process after the first call.
    """
    if _INTCAL20_CACHE_STATE["loaded"] is not None:
        return _INTCAL20_CACHE_STATE["loaded"]

    from paleopy.net import fetch_with_fallback

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        cal_bp, c14_age, c14_error = d["cal_bp"], d["c14_age"], d["c14_error"]
    else:
        url = "https://intcal.org/curves/intcal20.14c"
        resp = fetch_with_fallback(url, timeout=60)
        text = resp.text

        cal_bp_l, c14_age_l, c14_error_l = [], [], []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                cal_bp_l.append(float(parts[0]))
                c14_age_l.append(float(parts[1]))
                c14_error_l.append(float(parts[2]))
            except ValueError:
                continue

        cal_bp = np.array(cal_bp_l)
        c14_age = np.array(c14_age_l)
        c14_error = np.array(c14_error_l)
        np.savez(cache_path, cal_bp=cal_bp, c14_age=c14_age, c14_error=c14_error)

    sort_idx = np.argsort(cal_bp)
    result = (cal_bp[sort_idx], c14_age[sort_idx], c14_error[sort_idx])
    _INTCAL20_CACHE_STATE["loaded"] = result
    return result


def calibrate_gaussian_intcal(
    c14_age: float, c14_error: float, cal_range: np.ndarray, curve: tuple = None
) -> np.ndarray:
    """Calibrate a single radiocarbon measurement against IntCal20 via
    Gaussian likelihood (measurement + curve uncertainty combined in
    quadrature). Properly handles multimodal distributions caused by
    calibration plateaus (e.g. Hallstatt plateau) - something a naive
    Gaussian on cal_range alone cannot do.

    curve defaults to the lazily-loaded module IntCal20 curve if not
    provided (via load_intcal20()).
    """
    intcal_cal, intcal_c14, intcal_err = curve if curve is not None else load_intcal20()

    curve_c14 = np.interp(cal_range, intcal_cal, intcal_c14)
    curve_err = np.interp(cal_range, intcal_cal, intcal_err)

    combined_sigma = np.sqrt(c14_error ** 2 + curve_err ** 2)
    prob = np.exp(-0.5 * ((c14_age - curve_c14) / combined_sigma) ** 2) / combined_sigma

    total = prob.sum()
    if total > 0:
        prob /= total
    return prob
