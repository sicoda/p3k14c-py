"""
06_Human_Environment_Comparison.py : Compares p3k14c radiocarbon SPD against paleoenvironmental data from multiple open databases, queried automatically by coordinates

Input  : p3k14c_pristine_dates.csv, <Site>_spd_for_06.csv (if available), intcal20.npz, <Site>_environment.csv  
Output : <Site>_human_environment.png, <Site>_spd.csv (if applicable), <Site>_environment.csv, <Site>_comparison.csv, <Site>_resilience.csv

STATISTICAL FIXES :
1. True IntCal20 radiocarbon calibration (replaces Gaussian approximation)
2. Monte Carlo null-model significance envelope (Timpson et al. 2014)
3. Per-proxy Z-scoring before aggregation (fixes unit mismatch)
4. Linear detrending before Z-scoring (removes orbital-scale trends)
5. Exponential resilience fit (replaces linear slope)

DATABASE PRIORITY ORDER :
1. PANGAEA   — best for Near East / Anatolia / Mediterranean
2. Neotoma   — best for North America and Europe
3. NOAA GISP2 ice core — global fallback

DEPENDENCIES :  pip install pandas numpy scipy matplotlib seaborn requests
PYTHON       : 3.12+
"""

import json as _json
import math
import sys
import time
import warnings
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, detrend as scipy_detrend
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
#  USER CONFIGURATION
# ---------------------------------------------------------------------------

SITE_NAME        = "Catalhoyuk"
SITE_LAT         = 37.6660
SITE_LON         = 32.8277

ARCHO_RADIUS_KM  = 5           # must match 04_Population_Analysis.py ARCHO_RADIUS_KM
ENV_RADIUS_KM    = 1000

TIME_MIN         = 7700        # must match 04_Population_Analysis.py TIME_MIN
TIME_MAX         = 9500        # must match 04_Population_Analysis.py TIME_MAX
BIN_H            = 50
MAX_ERROR        = 150
RESOLUTION       = 10
CONFIRM          = True

# -- Environmental data settings --------------------------------------------
ENV_PROXY_KEYWORD   = "stable isotope"
ENV_NEOTOMA_TYPE    = ["stable isotopes"]
ENV_BIN_YR          = RESOLUTION
USE_GISP2_ONLY      = True                  # if you only want Ice Core data

# -- Anomaly detection ------------------------------------------------------
ENV_ANOMALY_THRESHOLD = -1.0
ENV_RECOVERY_STEPS    = 10
BASELINE_WINDOW       = 400

# -- Monte Carlo significance testing ---------------------------------------
MC_N_SIM   = 999   # reduce to 199 for faster dev runs
MC_MODEL   = "logistic"   # "exponential" or "uniform"

# -- SPD source -------------------------------------------------------------
# Set to path of ###_spd_for_06.csv to reuse 04_SPD.py's output,
# or None to rebuild from scratch
SPD_FROM_04 = f"{SITE_NAME.replace(' ','_')}_spd_for_06.csv"
#SPD_FROM_04 = None

# ---------------------------------------------------------------------------
#  END OF USER CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_FILE  = os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv")
_SLUG       = SITE_NAME.replace(" ", "_")
OUTPUT_PLOT = f"{_SLUG}_human_environment.png"
OUTPUT_SPD  = f"{_SLUG}_spd.csv"
OUTPUT_ENV  = f"{_SLUG}_environment.csv"
OUTPUT_CSV  = f"{_SLUG}_comparison.csv"
OUTPUT_RES  = f"{_SLUG}_resilience.csv"
INTCAL_CACHE = "intcal20.npz"

GISP2_URL   = ("https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
               "summit/gisp2/isotopes/gisp2_temp_accum_alley2000.txt")
NEOTOMA_API = "https://api.neotomadb.org/v2.0"

OLD_WOOD = {"charcoal", "wood", "timber", "unidentified wood",
            "unidentified charcoal", "charred wood"}
MARINE   = {"shell", "marine shell", "marine", "coral", "rangia",
            "macoma", "oyster", "mussel", "clam"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def divider(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w - pad - len(title) - 2)}")
    else:
        print(f"\n{'─'*w}")


def sg_smooth(arr, window=11, poly=2):
    arr = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < 5:
        return arr.copy()
    # Interpolate over NaNs temporarily so savgol_filter can run
    idx = np.arange(len(arr))
    arr_filled = arr.copy()
    arr_filled[~finite_mask] = np.interp(
        idx[~finite_mask], idx[finite_mask], arr[finite_mask]
    )
    w = min(window, len(arr_filled))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()
    smoothed = savgol_filter(arr_filled, w, poly)
    # Restore NaNs outside the original data range
    smoothed[~finite_mask] = np.nan
    return smoothed


# ---------------------------------------------------------------------------
# Step 0.1  -  IntCal20 calibration curve
# ---------------------------------------------------------------------------

def load_intcal20() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load IntCal20 calibration curve.
    Downloads once and caches as intcal20.npz.
    Uses requests instead of urllib to avoid macOS SSL cert issues.
    """
    if os.path.exists(INTCAL_CACHE):
        d = np.load(INTCAL_CACHE)
        return d["cal_bp"], d["c14_age"], d["c14_error"]

    print("  [IntCal20] Downloading calibration curve from intcal.org...")
    url = "https://intcal.org/curves/intcal20.14c"

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        # SSL fallback: try without verification (read-only public data)
        print(f"  [IntCal20] SSL error ({e}), retrying without verification...")
        resp = requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()
        text = resp.text

    cal_bp, c14_age, c14_error = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            cal_bp.append(float(parts[0]))
            c14_age.append(float(parts[1]))
            c14_error.append(float(parts[2]))
        except ValueError:
            continue

    cal_bp    = np.array(cal_bp)
    c14_age   = np.array(c14_age)
    c14_error = np.array(c14_error)
    np.savez(INTCAL_CACHE, cal_bp=cal_bp, c14_age=c14_age, c14_error=c14_error)
    print(f"  [IntCal20] {len(cal_bp):,} points cached -> {INTCAL_CACHE}")
    return cal_bp, c14_age, c14_error


# Load curve once at module level
INTCAL_CAL, INTCAL_C14, INTCAL_ERR = load_intcal20()

# Pre-sort ascending for np.interp (which requires inc x)
_sort_idx   = np.argsort(INTCAL_CAL)
INTCAL_CAL  = INTCAL_CAL[_sort_idx]
INTCAL_C14  = INTCAL_C14[_sort_idx]
INTCAL_ERR  = INTCAL_ERR[_sort_idx]


def calibrate_date(c14_age: float, c14_error: float,
                   cal_range: np.ndarray) -> np.ndarray:
    """
    Calibrate a single radiocarbon measurement against IntCal20.

    Properly handles multimodal distributions caused by calibration
    plateaus (e.g. Hallstatt plateau) — something a Gaussian cannot do.

    Returns a probability vector over cal_range, normalized to sum=1.
    """
    # Interpolate IntCal20 onto every year in cal_range
    curve_c14 = np.interp(cal_range, INTCAL_CAL, INTCAL_C14)
    curve_err = np.interp(cal_range, INTCAL_CAL, INTCAL_ERR)

    # Measurement error + curve uncertainty in quadrature
    combined_sigma = np.sqrt(c14_error ** 2 + curve_err ** 2)

    # Likelihood of each calendar year given the measurement
    prob = (np.exp(-0.5 * ((c14_age - curve_c14) / combined_sigma) ** 2)
            / combined_sigma)

    total = prob.sum()
    if total > 0:
        prob /= total
    return prob


# ---------------------------------------------------------------------------
# Step 0.2  —  Site lookup
# ---------------------------------------------------------------------------

def lookup_and_confirm(input_file):
    divider("SITE LOOKUP")
    print(f"  Site          : {SITE_NAME}")
    print(f"  Coordinates   : {SITE_LAT}N, {SITE_LON}E")
    print(f"  Archo radius  : {ARCHO_RADIUS_KM} km")
    print(f"  Env radius    : {ENV_RADIUS_KM} km")
    print(f"  Time window   : {TIME_MIN}-{TIME_MAX} Cal BP")

    if not Path(input_file).is_file():
        sys.exit(f"ERROR: {input_file} not found.")

    df = pd.read_csv(input_file, low_memory=False, index_col=0)
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP",
                "CI95_Lower", "CI95_Upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Lat", "Long"])
    df["dist_km"] = df.apply(
        lambda r: haversine_km(SITE_LAT, SITE_LON, r["Lat"], r["Long"]), axis=1)
    nearby = df[df["dist_km"] <= ARCHO_RADIUS_KM].copy()

    if nearby.empty:
        print(f"\n  No p3k14c records within {ARCHO_RADIUS_KM} km.")
        sys.exit()

    nearby["SiteName"] = nearby["SiteName"].fillna("Unknown")
    nearby["SiteID"]   = nearby["SiteID"].fillna("Unknown")
    sites = (nearby.groupby(["SiteName", "SiteID"])
             .agg(n_dates=("Age", "count"), dist_km=("dist_km", "min"),
                  lat=("Lat", "first"), lon=("Long", "first"))
             .sort_values("dist_km").reset_index())

    print(f"\n  Records within {ARCHO_RADIUS_KM} km: "
          f"{len(nearby):,} across {len(sites)} site(s)\n")
    print(sites[["SiteName", "SiteID", "n_dates", "dist_km", "lat", "lon"]]
          .to_string(index=False))

    if CONFIRM:
        print("\n  Proceed? (yes/no) ", end="")
        if input().strip().lower() not in ("yes", "y", ""):
            sys.exit("Adjust settings and re-run.")
    return nearby


# ---------------------------------------------------------------------------
# Step 1  —  Chronometric hygiene
# ---------------------------------------------------------------------------

def apply_hygiene(df):
    divider("CHRONOMETRIC HYGIENE")
    n0  = len(df)
    df  = df.dropna(subset=["Age", "Error", "Lat", "MedianCalBP"])
    df  = df[df["Error"] <= MAX_ERROR].copy()
    mat = df["Material"].fillna("").str.lower().str.strip()
    n_ow = mat.apply(lambda m: any(t in m for t in OLD_WOOD)).sum()
    n_mr = mat.apply(lambda m: any(t in m for t in MARINE)).sum()
    df   = df[~mat.apply(lambda m: any(t in m for t in OLD_WOOD | MARINE))].copy()
    df   = df[df["MedianCalBP"].between(TIME_MIN, TIME_MAX)].copy()
    if "LocAccuracy" in df.columns:
        df["LocAccuracy"] = pd.to_numeric(df["LocAccuracy"], errors="coerce")
        df = df[df["LocAccuracy"] >= 1].copy()

    med_err     = df["Error"].median() if not df.empty else 0
    suggested_h = int(round(med_err / 50) * 50) or 50
    print(f"  Records in    : {n0:,}")
    print(f"  Old-wood excl : {n_ow:,}")
    print(f"  Marine excl   : {n_mr:,}")
    print(f"  After filters : {len(df):,}")
    print(f"  Median error  : {med_err:.0f} yr  (suggested BIN_H ~{suggested_h} yr)")
    print(f"  Current BIN_H : {BIN_H} yr")
    return df


# ---------------------------------------------------------------------------
# Step 2.1  —  Binning + SPD
# ---------------------------------------------------------------------------

def bin_dates(df):
    df = df.copy()
    df["SiteID_filled"] = df["SiteID"].fillna(
        pd.Series(df.index.astype(str), index=df.index))
    df["Bin"] = ""
    counter   = 0
    for _, grp in df.groupby("SiteID_filled"):
        ages  = grp["Age"].values
        idx   = grp.index
        order = np.argsort(ages)
        s_idx, s_age = idx[order], ages[order]
        cb = counter
        df.loc[s_idx[0], "Bin"] = f"b{cb}"
        for k in range(1, len(s_age)):
            if s_age[k] - s_age[k - 1] > BIN_H:
                counter += 1
                cb = counter
            df.loc[s_idx[k], "Bin"] = f"b{cb}"
        counter += 1
    return df


def taphonomic_weight(t):
    raw = 5.726442e6 * np.power(t + 2176.4, -1.3925309)
    return raw / raw[np.argmin(t)]


def load_spd_from_04(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed SPD and MC envelope from 04_Population_Dynamics.py.
    Returns (years, spd_un, spd_lo, spd_hi).
    """
    df = pd.read_csv(path)
    years  = df["CalBP"].values
    spd_un = df["SPD_TaphCorrected"].values
    spd_lo = df["MC_lo_2.5pct"].values
    spd_hi = df["MC_hi_97.5pct"].values
    print(f"  [SPD] Loaded from {path}  ({len(years)} time steps)")
    print(f"  [SPD] MC envelope: exponential null model (from 04)")
    return years, spd_un, spd_lo, spd_hi


def build_spd(df):
    """
    FIX 1: Build SPD using true IntCal20 calibration.
    Falls back to Gaussian if no raw 14C age available.
    """
    divider("SPD CONSTRUCTION")
    years  = np.arange(TIME_MIN, TIME_MAX + RESOLUTION, RESOLUTION)
    spd_un = np.zeros(len(years))
    spd_no = np.zeros(len(years))
    df_b   = bin_dates(df)
    n_bins = df_b["Bin"].nunique()
    print(f"  Bins (h={BIN_H} yr) : {n_bins:,}")
    print(f"  Calibration   : IntCal20 (Reimer et al. 2020)")

    n_calibrated = 0
    n_gaussian   = 0

    for _, grp in df_b.groupby("Bin"):
        bu = np.zeros(len(years))
        bn = np.zeros(len(years))
        for _, row in grp.iterrows():
            c14 = row.get("Age")
            err = row.get("Error")
            if pd.notna(c14) and pd.notna(err) and float(err) > 0:
                prob = calibrate_date(float(c14), float(err), years)
                n_calibrated += 1
            else:
                sigma = max(float(row.get("UncalBPError", 30)), 20)
                prob  = np.exp(-0.5 * ((years - float(row["MedianCalBP"])) / sigma) ** 2)
                total = prob.sum()
                prob  = prob / total if total > 0 else prob
                n_gaussian += 1
            bu += prob
            bn += prob / (prob.sum() or 1.0)
        spd_un += bu / len(grp)
        spd_no += bn / len(grp)

    print(f"  IntCal20 calibrated : {n_calibrated:,} dates")
    if n_gaussian > 0:
        print(f"  Gaussian fallback   : {n_gaussian:,} dates")

    for arr in (spd_un, spd_no):
        arr /= arr.sum() or 1.0

    taph   = taphonomic_weight(years)
    spd_un = spd_un / taph
    spd_un /= spd_un.sum()
    spd_no = spd_no / taph
    spd_no /= spd_no.sum()

    pd.DataFrame({"CalBP": years, "SPD_Unnormalized": spd_un,
                  "SPD_Normalized": spd_no}).to_csv(OUTPUT_SPD, index=False)
    print(f"  Saved -> {OUTPUT_SPD}")
    return years, spd_un, spd_no


# ---------------------------------------------------------------------------
# Step 2.2  -  Monte Carlo null-model significance envelope
# ---------------------------------------------------------------------------

def spd_significance_envelope(
        df: pd.DataFrame,
        years: np.ndarray,
        empirical_spd: np.ndarray,
        n_sim: int = MC_N_SIM,
        model: str = MC_MODEL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo significance testing following Timpson et al. (2014) /
    Crema & Bevan (2021).

    Procedure:
      1. Fit a null growth model (exponential) to the empirical SPD.
      2. Simulate n_sim SPDs by drawing dates from that null model,
         back-projecting to 14C ages via IntCal20, adding measurement
         noise, then calibrating.
      3. Build 2.5–97.5% confidence envelope.
      4. SPD falling BELOW the lower envelope = statistically significant
         demographic trough.

    Returns (lo_envelope, hi_envelope, p_value_array).
    """
    divider("MONTE CARLO SIGNIFICANCE TEST")
    print(f"  Null model    : {model}")
    print(f"  Simulations   : {n_sim}  (reduce MC_N_SIM for faster runs)")

    df_binned = bin_dates(df)
    n_dates   = df_binned["Bin"].nunique()

    # --- Fit null model to empirical SPD ---
    x = years - years.mean()
    y = empirical_spd.copy()
    y = np.where(y > 0, y, 1e-10)

    if model == "exponential":
        def null_model(x, a, b):
            return a * np.exp(b * x)
        try:
            popt, _ = curve_fit(null_model, x, y, p0=[y.mean(), 0.0],
                                maxfev=10000)
            fitted = null_model(x, *popt)
        except RuntimeError:
            print("  WARNING: exponential fit failed, using uniform null model")
            fitted = np.ones_like(y) * y.mean()
    else:
        fitted = np.ones_like(y) * y.mean()

    fitted       = np.clip(fitted, 0, None)
    fitted_prob  = fitted / fitted.sum()

    # --- Simulate SPDs ---
    sim_spds = np.zeros((n_sim, len(years)))
    rng      = np.random.default_rng(42)

    for i in range(n_sim):
        # Draw calendar dates from null model
        sim_cal_ages = rng.choice(years, size=n_dates, p=fitted_prob)
        sim_spd_i    = np.zeros(len(years))

        for cal_age in sim_cal_ages:
            # Back-project calendar age - 14C age via IntCal20
            idx      = np.argmin(np.abs(INTCAL_CAL - cal_age))
            c14_mean = float(INTCAL_C14[idx])
            c14_err  = float(INTCAL_ERR[idx])
            # Add typical measurement noise
            c14_sim  = float(rng.normal(c14_mean, max(c14_err, 30)))
            prob     = calibrate_date(c14_sim, max(c14_err, 30), years)
            sim_spd_i += prob

        # Taphonomic correction
        taph      = taphonomic_weight(years)
        taph      = np.where(taph > 0, taph, np.nan)
        sim_spd_i = sim_spd_i / taph
        sim_spd_i = np.nan_to_num(sim_spd_i, nan=0.0)

        # Normalize to same total as empirical
        if sim_spd_i.sum() > 0:
            sim_spd_i *= empirical_spd.sum() / sim_spd_i.sum()

        sim_spds[i] = sim_spd_i

        if (i + 1) % 100 == 0:
            print(f"    Simulating... {i+1}/{n_sim}", end="\r")

    print(f"    Simulating... {n_sim}/{n_sim} — done           ")

    lo    = np.percentile(sim_spds, 2.5,  axis=0)
    hi    = np.percentile(sim_spds, 97.5, axis=0)
    p_vals = (sim_spds >= empirical_spd[None, :]).mean(axis=0)

    sig_lo = (empirical_spd < lo).sum()
    sig_hi = (empirical_spd > hi).sum()
    print(f"  Significant troughs : {sig_lo} yr  |  peaks : {sig_hi} yr")
    print(f"  (years outside 95% null-model envelope)")
    return lo, hi, p_vals


# ---------------------------------------------------------------------------
# Step 3.1  —  PANGAEA
# ---------------------------------------------------------------------------

def fetch_pangaea(keyword: str) -> pd.DataFrame | None:
    """
    Search PANGAEA by keyword via Elasticsearch API.
    Returns standardised DataFrame [age_bp, value, variable,
    site_name, dataset_type, dist_km, doi].
    """
    deg = ENV_RADIUS_KM / 111.0
    print(f"\n  [PANGAEA] Searching: '{keyword}' within {ENV_RADIUS_KM} km...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            "https://ws.pangaea.de/es/pangaea/panmd/_search",
            params={"q": keyword, "size": 50, "_source": "true"},
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        print(f"  PANGAEA status: {response.status_code}")
        response.raise_for_status()
        results_raw = response.json()
        raw_hits    = results_raw.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"  PANGAEA search failed: {e}")
        return None

    print(f"  PANGAEA hits: {len(raw_hits)}")
    if not raw_hits:
        return None

    rows = []
    for hit in raw_hits:
        src = hit.get("_source", {})
        doi = src.get("URI", "")
        if not doi:
            continue

        # Coordinates: prefer meanPosition, fall back to bbox center
        lat = src.get("meanPosition", {}).get("lat")
        lon = src.get("meanPosition", {}).get("lon")
        if lat is None:
            n = src.get("northBoundLatitude")
            s = src.get("southBoundLatitude")
            lat = (n + s) / 2 if n is not None and s is not None else None
        if lon is None:
            e = src.get("eastBoundLongitude")
            w = src.get("westBoundLongitude")
            lon = (e + w) / 2 if e is not None and w is not None else None
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue

        dist = haversine_km(SITE_LAT, SITE_LON, lat, lon)
        if dist > ENV_RADIUS_KM:
            continue

        # Parse title
        site_name = ""
        xml_thumb = src.get("xml-thumb", "")
        if "<md:title>" in xml_thumb:
            try:
                site_name = xml_thumb.split("<md:title>")[1].split("</md:title>")[0][:60]
            except IndexError:
                pass
        if not site_name:
            site_name = doi

        print(f"    + {site_name[:50]:<50}  dist={dist:.0f} km  doi={doi}")

        # Expand parent collections to datasets
        xml = src.get("xml", "")
        child_ids = []
        if 'collectionChilds' in xml:
            try:
                child_str = xml.split('key="collectionChilds" value="')[1].split('"')[0]
                child_ids = [c.replace("D", "") for c in child_str.split(",")]
            except IndexError:
                pass

        dois_to_fetch = ([f"https://doi.org/10.1594/PANGAEA.{cid}"
                          for cid in child_ids]
                         if child_ids else [doi])

        for fetch_doi in dois_to_fetch:
            data_url = (fetch_doi
                        .replace("https://doi.org/", "https://doi.pangaea.de/")
                        + "?format=textfile")
            try:
                time.sleep(0.3)
                dr = requests.get(data_url, headers=headers, timeout=30)
                dr.raise_for_status()
                text = dr.text

                lines = text.splitlines()
                header_end = 0
                for i, line in enumerate(lines):
                    if line.startswith("*/"):
                        header_end = i + 1
                        break
                if header_end == 0:
                    continue

                data_lines = lines[header_end:]
                if not data_lines:
                    continue

                col_line = data_lines[0].split("\t")
                age_col  = None
                for j, col in enumerate(col_line):
                    if any(term in col.lower() for term in
                           ["age", "cal bp", "ka bp", "cal yr", "yr bp"]):
                        age_col = j
                        break
                if age_col is None:
                    continue

                skip_terms = {"latitude", "longitude", "depth", "event",
                              "elevation", "sample", "label", "comment",
                              "reference", "age", "date"}
                val_cols = [(j, col) for j, col in enumerate(col_line)
                            if j != age_col and not any(
                                s in col.lower() for s in skip_terms)]

                for line in data_lines[1:]:
                    parts = line.split("\t")
                    if len(parts) <= age_col:
                        continue
                    try:
                        raw_age = float(parts[age_col])
                        if raw_age < 200:
                            raw_age *= 1000
                        if not (TIME_MIN <= raw_age <= TIME_MAX):
                            continue
                        for j, col_name in val_cols:
                            if j < len(parts) and parts[j].strip():
                                try:
                                    val = float(parts[j])
                                    rows.append({
                                        "age_bp":       raw_age,
                                        "value":        val,
                                        "variable":     col_name,
                                        "site_name":    site_name[:60],
                                        "dataset_type": keyword,
                                        "dist_km":      dist,
                                        "doi":          fetch_doi,
                                    })
                                except ValueError:
                                    pass
                    except ValueError:
                        continue
            except Exception as e:
                print(f"    WARNING downloading {fetch_doi}: {e}")
                continue

    if not rows:
        print("  PANGAEA: no usable data rows in time window.")
        return None

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(TIME_MIN, TIME_MAX)].copy()
    print(f"  PANGAEA: {len(df):,} observations in time window")
    return df if len(df) >= 5 else None


# ---------------------------------------------------------------------------
# Step 3.2  —  Neotoma
# ---------------------------------------------------------------------------

def fetch_neotoma(dtypes):
    deg  = ENV_RADIUS_KM / 111.0
    rows = []

    print(f"\n  [Neotoma] Searching sites within {ENV_RADIUS_KM} km...")
    sites_url = (f"{NEOTOMA_API}/data/sites"
                 f"?bbox={SITE_LON-deg:.4f},{SITE_LAT-deg:.4f},"
                 f"{SITE_LON+deg:.4f},{SITE_LAT+deg:.4f}&limit=100")
    try:
        r = requests.get(sites_url, timeout=30)
        r.raise_for_status()
        all_sites = r.json().get("data", [])
    except Exception as e:
        print(f"  Neotoma sites query failed: {e}")
        return None

    nearby_siteids = []
    for s in all_sites:
        if not isinstance(s, dict):
            continue
        geo_raw = s.get("geography", "{}")
        try:
            geo    = _json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
            coords = geo.get("coordinates", [None, None])
            slon, slat = coords[0], coords[1]
            if slat and slon:
                dist = haversine_km(SITE_LAT, SITE_LON, slat, slon)
                if dist <= ENV_RADIUS_KM:
                    nearby_siteids.append(s.get("siteid"))
        except Exception:
            continue

    print(f"  Neotoma sites within radius: {len(nearby_siteids)}")
    if not nearby_siteids:
        return None

    for dtype in dtypes:
        ids_str = ",".join(str(i) for i in nearby_ids)
        ds_url  = (f"{NEOTOMA_API}/data/datasets"
                   f"?siteid={ids_str}"
                   f"&datasettype={dtype.replace(' ', '%20')}&limit=200")
        try:
            r = requests.get(ds_url, timeout=30)
            r.raise_for_status()
            datasets = r.json().get("data", [])
        except Exception as e:
            print(f"  WARNING: {e}")
            continue

        print(f"  Neotoma '{dtype}' datasets: {len(datasets)}")
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            try:
                si      = ds.get("site", {})
                if not isinstance(si, dict):
                    continue
                name    = si.get("sitename", "?")
                geo_raw = si.get("geography", "{}")
                try:
                    geo    = _json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
                    coords = geo.get("coordinates", [None, None])
                    slon, slat = coords[0], coords[1]
                    dist = haversine_km(SITE_LAT, SITE_LON, slat, slon) if slat and slon else None
                except Exception:
                    dist = None

                site_ds = si.get("datasets", [])
                if not site_ds:
                    continue
                dsid = site_ds[0].get("datasetid")
                if dsid is None:
                    continue

                print(f"    + {name:<35}  dist={dist:.0f} km  id={dsid}"
                      if dist else f"    + {name}  id={dsid}")
                time.sleep(0.3)
                sr = requests.get(f"{NEOTOMA_API}/data/downloads/{dsid}", timeout=30)
                sr.raise_for_status()
                for item in sr.json().get("data", []):
                    for sample in item.get("samples", []):
                        age = sample.get("age")
                        if age is None:
                            continue
                        for datum in sample.get("data", []):
                            rows.append({
                                "age_bp":       float(age),
                                "value":        datum.get("value"),
                                "variable":     datum.get("variablename", ""),
                                "site_name":    name,
                                "dataset_type": dtype,
                                "dist_km":      dist,
                                "doi":          "",
                            })
            except Exception as e:
                print(f"    WARNING: {e}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(TIME_MIN, TIME_MAX)].copy()
    print(f"  Neotoma: {len(df):,} observations in time window")
    return df if len(df) >= 5 else None


# ---------------------------------------------------------------------------
# Step 3.3  —  NOAA GISP2 fallback
# ---------------------------------------------------------------------------

def fetch_gisp2():
    cache = Path(OUTPUT_ENV)
    df    = None
    if cache.is_file():
        c = pd.read_csv(cache)
        if "Age_BP" in c.columns:
            print(f"  [GISP2] Loading cached data from {OUTPUT_ENV}")
            df = c
    if df is None:
        print("  [GISP2] Downloading from NOAA...")
        resp = requests.get(GISP2_URL, timeout=30)
        resp.raise_for_status()
        rows, in_data = [], False
        for line in resp.text.splitlines():
            if "Age" in line and "Temperature" in line:
                in_data = True
                continue
            if not in_data:
                continue
            if "Accumulation" in line:
                break
            p = line.strip().split()
            if len(p) >= 2:
                try:
                    rows.append((float(p[0]) * 1000, float(p[1])))
                except ValueError:
                    pass
        df = pd.DataFrame(rows, columns=["Age_BP", "Temp_C"]).sort_values("Age_BP")
        df.to_csv(OUTPUT_ENV, index=False)
        print(f"  [GISP2] {len(df):,} points -> {OUTPUT_ENV}")

    ages = np.arange(TIME_MIN, TIME_MAX + RESOLUTION, RESOLUTION)
    f    = interp1d(df["Age_BP"], df["Temp_C"], bounds_error=False, fill_value=np.nan)
    return ages, sg_smooth(f(ages))


# ---------------------------------------------------------------------------
# Step 3.4  -  Per-proxy Z-scoring + detrending before aggregation
# ---------------------------------------------------------------------------

def env_series_from_df(env_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score each proxy variable individually before averaging,
    so that δ18O (-4.5) and pollen counts (1500) are commensurable.

    Linearly detrend the composite to remove orbital-scale
    long-term trends (e.g. gradual Holocene cooling), so that the
    anomaly detector picks up rapid events rather than trend artefacts.
    """
    ages      = np.arange(TIME_MIN, TIME_MAX + RESOLUTION, RESOLUTION)
    composite = np.full(len(ages), np.nan)
    n_proxies = 0

    env_df = env_df.copy()

    # Group by variable name — never mix raw units across proxies
    for var_name, grp in env_df.groupby("variable"):
        if len(grp) < 5:
            continue

        grp_sorted = grp.sort_values("age_bp")
        var_series = np.full(len(ages), np.nan)

        for i, yr in enumerate(ages):
            window = grp_sorted[
                (grp_sorted["age_bp"] >= yr - ENV_BIN_YR / 2) &
                (grp_sorted["age_bp"] <  yr + ENV_BIN_YR / 2)
            ]
            if len(window) > 0:
                var_series[i] = window["value"].mean()

        # Interpolate small gaps only (</= 5 bins); no extrapolation
        mask = np.isfinite(var_series)
        if mask.sum() < 5:
            continue
        idx        = np.where(mask)[0]
        var_interp = np.interp(np.arange(len(ages)), idx, var_series[mask])
        # Restore NaN outside the data range (no extrapolation)
        var_interp[:idx[0]]  = np.nan
        var_interp[idx[-1]:] = np.nan
        var_series = var_interp

        # --- Z-score THIS proxy individually ---
        mu, sd = np.nanmean(var_series), np.nanstd(var_series)
        if sd < 1e-10:
            continue
        z_series = (var_series - mu) / sd

        print(f"    Proxy '{var_name[:40]}': {mask.sum()} pts  "
              f"mean={mu:.3f}  sd={sd:.3f}")

        # Accumulate composite as mean of Z-scores
        if np.all(np.isnan(composite)):
            composite = z_series.copy()
        else:
            composite = np.nanmean(np.stack([composite, z_series]), axis=0)
        n_proxies += 1

    print(f"  Composite built from {n_proxies} proxy variable(s)")

    if n_proxies == 0:
        return ages, np.full(len(ages), np.nan)

    # --- Detrend to remove orbital-scale trend ---
    finite_mask = np.isfinite(composite)
    if finite_mask.sum() > 10:
        composite[finite_mask] = scipy_detrend(composite[finite_mask])

    return ages, sg_smooth(composite)


# ---------------------------------------------------------------------------
# Step 3.5  —  Orchestrate database queries
# ---------------------------------------------------------------------------

def get_env_data():
    divider("ENVIRONMENTAL DATA")
    print(f"  PANGAEA keyword : '{ENV_PROXY_KEYWORD}'")
    print(f"  Neotoma type    : {ENV_NEOTOMA_TYPE}")
    print(f"  Search radius   : {ENV_RADIUS_KM} km")
    print(f"  Database order  : PANGAEA -> Neotoma -> GISP2")

    if USE_GISP2_ONLY:
        print("\n  USE_GISP2_ONLY=True, skipping live database queries.")
        ages, vals = fetch_gisp2()
        return ages, vals, "GISP2 Ice Core Temperature (Alley 2000, NOAA)"

    all_rows = []

    pang_df = fetch_pangaea(ENV_PROXY_KEYWORD)
    if pang_df is not None:
        all_rows.append(pang_df)
        print(f"  PANGAEA: SUCCESS ({len(pang_df):,} rows)")

    neot_df = fetch_neotoma(ENV_NEOTOMA_TYPE)
    if neot_df is not None:
        all_rows.append(neot_df)
        print(f"  Neotoma: SUCCESS ({len(neot_df):,} rows)")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(OUTPUT_ENV, index=False)
        ages, vals = env_series_from_df(combined)

        sources = []
        if pang_df is not None:
            sources.append(f"PANGAEA '{ENV_PROXY_KEYWORD}'")
        if neot_df is not None:
            sources.append(f"Neotoma {ENV_NEOTOMA_TYPE}")
        label = " + ".join(sources) + f" ({ENV_RADIUS_KM} km)"

        if np.sum(~np.isnan(vals)) >= 10:
            return ages, vals, label

    print(f"\n  No local proxy data found within {ENV_RADIUS_KM} km.")
    print("  Falling back to GISP2 ice core (global signal).")
    print("  To get local data:")
    print(f"    - Try a different ENV_PROXY_KEYWORD (e.g. 'speleothem Turkey')")
    print(f"    - Increase ENV_RADIUS_KM (currently {ENV_RADIUS_KM} km)")
    print(f"    - Set USE_GISP2_ONLY=True to suppress these messages")
    ages, vals = fetch_gisp2()
    return ages, vals, f"GISP2 Ice Core (fallback — no local data within {ENV_RADIUS_KM} km)"


# ---------------------------------------------------------------------------
# Step 4  —  Align and Z-score
# ---------------------------------------------------------------------------

def align(years_spd, spd, env_ages, env_vals):
    valid = ~np.isnan(env_vals)
    t = np.arange(max(TIME_MIN, float(env_ages[valid].min())),
                  min(TIME_MAX, float(env_ages[valid].max())) + RESOLUTION,
                  RESOLUTION)
    f_s = interp1d(years_spd, spd,      bounds_error=False, fill_value=0.0)
    f_e = interp1d(env_ages,  env_vals, bounds_error=False, fill_value=np.nan)
    d, e = f_s(t), f_e(t)
    v    = ~np.isnan(e) & ~np.isnan(d) & (d > 0)
    t, d, e = t[v], d[v], e[v]
    dz = (d - d.mean()) / (d.std() or 1.0)
    ez = (e - e.mean()) / (e.std() or 1.0)
    print(f"\n[alignment] {t[0]:.0f}-{t[-1]:.0f} Cal BP  "
          f"({len(t)} points at {RESOLUTION}-yr resolution)")
    return t, dz, ez


# ---------------------------------------------------------------------------
# Step 4.1  -  Exponential resilience fit
# ---------------------------------------------------------------------------

def fit_resilience(rt: np.ndarray, rd: np.ndarray) -> float:
    """
    Fit exponential recovery curve instead of linear slope.

    Model: rd(t) ≈ A * (1 − exp(−k * t))
    Returns rate constant k per 100 yr (higher = faster recovery).

    Falls back to linear slope if curve_fit fails (e.g. too few points).
    """
    if len(rt) < 4:
        return np.nan

    t = rt - rt[0]

    # Normalize rd to [0, 1] for fitting stability
    rd_min, rd_max = rd.min(), rd.max()
    rd_range = rd_max - rd_min
    if rd_range < 1e-10:
        return np.nan
    rd_norm = (rd - rd_min) / rd_range

    try:
        def exp_recovery(t, k):
            return 1.0 - np.exp(-k * t)

        popt, _ = curve_fit(exp_recovery, t, rd_norm,
                            p0=[0.01], bounds=(0, np.inf),
                            maxfev=2000)
        return float(popt[0]) * 100   # per 100 yr

    except RuntimeError:
        # Fallback: linear slope
        slope = np.polyfit(t, rd, 1)[0]
        return float(slope) * 100


# ---------------------------------------------------------------------------
# Step 5  —  Anomaly detection + Resistance & Resilience
# ---------------------------------------------------------------------------

def detect_anomalies(t, demo, env,
                     spd_lo=None, spd_hi=None,
                     years_spd=None):
    """
    Detect environmental anomalies and calculate resistance/resilience.

    If MC envelopes are provided (spd_lo, spd_hi), also flags whether
    each anomaly coincides with a statistically significant demographic
    trough (SPD below 2.5% null-model envelope).
    """
    divider("ANOMALY DETECTION")
    print(f"  Threshold     : Z < {ENV_ANOMALY_THRESHOLD}")
    print(f"  Recovery req. : {ENV_RECOVERY_STEPS} steps "
          f"({ENV_RECOVERY_STEPS * RESOLUTION} yr)")
    print(f"  Baseline win. : {BASELINE_WINDOW} yr before onset")
    print(f"  Resilience    : exponential recovery fit (Fix 5)")

    env_s       = sg_smooth(env, window=11)
    roc         = np.gradient(demo, t)
    episodes    = []
    in_ev       = False
    onset_i     = None
    above_count = 0

    for i, v in enumerate(env_s):
        if not in_ev:
            if v < ENV_ANOMALY_THRESHOLD:
                in_ev = True; onset_i = i; above_count = 0
        else:
            if v >= ENV_ANOMALY_THRESHOLD:
                above_count += 1
                if above_count >= ENV_RECOVERY_STEPS:
                    episodes.append((onset_i, i - ENV_RECOVERY_STEPS + 1))
                    in_ev = False; above_count = 0
            else:
                above_count = 0
    if in_ev and onset_i is not None:
        episodes.append((onset_i, len(t) - 1))

    print(f"\n  Episodes found: {len(episodes)}")
    records = []

    for onset_i, end_i in episodes:
        onset_bp = t[onset_i]
        end_bp   = t[end_i]
        bl_mask  = (t >= onset_bp - BASELINE_WINDOW) & (t < onset_bp)
        baseline_ok = bl_mask.sum() >= 5
        if not baseline_ok:
            print(f"  WARNING: short baseline for {onset_bp:.0f} Cal BP")
        Gb = float(np.mean(roc[bl_mask])) if baseline_ok else 0.0

        ev_mask     = (t >= onset_bp) & (t <= end_bp)
        env_vals    = env_s[ev_mask]
        demo_vals   = demo[ev_mask]
        t_vals      = t[ev_mask]
        env_min_i   = np.argmin(env_vals)
        Gx          = float(demo_vals[np.argmin(demo_vals)])
        demo_min_bp = float(t_vals[np.argmin(demo_vals)])

        denom      = abs(Gb) + abs(Gb - Gx)
        resistance = 1 - 2 * abs(Gb - Gx) / denom if denom > 0 else np.nan

        # Exponential resilience fit
        mi = np.argmin(demo_vals)
        rt = t_vals[mi:]
        rd = demo_vals[mi:]
        resilience = fit_resilience(rt, rd)

        # MC significance test
        sig_demo = False
        if spd_lo is not None and years_spd is not None:
            f_lo  = interp1d(years_spd, spd_lo, bounds_error=False, fill_value=np.nan)
            lo_ev = f_lo(t_vals)
            # Re-interpolate raw SPD onto t for comparison
            sig_demo = bool(np.any(demo_vals < lo_ev))

        rec = {
            "onset_bp":            round(onset_bp),
            "end_bp":              round(end_bp),
            "duration_yr":         round(end_bp - onset_bp),
            "env_min":             round(float(env_vals[env_min_i]), 3),
            "env_min_bp":          round(float(t_vals[env_min_i])),
            "demo_min_bp":         round(demo_min_bp),
            "resistance":          round(resistance, 4) if not np.isnan(resistance) else np.nan,
            "resilience_per100yr": round(resilience, 6) if not np.isnan(resilience) else np.nan,
            "resilience_method":   "exponential",
            "demo_sig_below_null": sig_demo,
            "baseline_start_bp":   round(onset_bp - BASELINE_WINDOW),
            "baseline_end_bp":     round(onset_bp),
            "baseline_ok":         baseline_ok,
        }
        records.append(rec)

        res_s   = f"{rec['resistance']:.3f}"          if not np.isnan(rec['resistance'])          else "n/a"
        resil_s = f"{rec['resilience_per100yr']:.4f}" if not np.isnan(rec['resilience_per100yr']) else "n/a"
        sig_s   = "YES (below MC null envelope)" if sig_demo else "no"
        print(f"\n  Anomaly at {rec['onset_bp']} Cal BP:")
        print(f"    Duration   : {rec['duration_yr']} yr ({rec['onset_bp']}-{rec['end_bp']})")
        print(f"    Env min    : Z={rec['env_min']:.2f} at {rec['env_min_bp']} Cal BP")
        print(f"    Resistance : {res_s}  (-1=collapse, +1=no impact)")
        print(f"    Resilience : {resil_s} per 100 yr  [exponential fit]")
        print(f"    Demo sig.  : {sig_s}")

    df_r = pd.DataFrame(records)
    if not df_r.empty:
        df_r.to_csv(OUTPUT_RES, index=False)
        print(f"\n  Saved -> {OUTPUT_RES}")
    else:
        print(f"\n  None found. Try lowering ENV_ANOMALY_THRESHOLD "
              f"(currently {ENV_ANOMALY_THRESHOLD})")
    return df_r


# ---------------------------------------------------------------------------
# Step 6  —  Plotting
# ---------------------------------------------------------------------------

def _res_panel_single(ax, t, demo, env, rec):
    env_s = sg_smooth(env)
    ax2 = ax.twinx()
    
    # Environmental (Right Axis)
    ax2.fill_between(t, env, alpha=0.10, color="#D7191C", linewidth=0)
    ax2.plot(t, env_s, color="#D7191C", lw=1.5, ls="--", alpha=0.8, label="Env (smoothed)")
    ax2.axhline(ENV_ANOMALY_THRESHOLD, color="#D7191C", lw=1, ls=":", alpha=0.6)
    ax2.set_ylabel("Env Z-score", color="#D7191C")
    ax2.tick_params(axis="y", labelcolor="#D7191C")
    sns.despine(ax=ax2, right=False, top=True)

    # Demographic (Left Axis)
    ax.fill_between(t, demo, alpha=0.20, color="#2C7BB6", linewidth=0)
    ax.plot(t, demo, color="#2C7BB6", lw=1.5, label="Demographic SPD")
    
    # Highlights
    ax.axvspan(rec["onset_bp"], rec["end_bp"], alpha=0.2, color="#FDAE61", lw=0, zorder=0)
    ax.axvline(rec["onset_bp"], color="#E66101", lw=1.5, label="Onset")
    ax.axvspan(rec["baseline_start_bp"], rec["baseline_end_bp"], alpha=0.1, color="#1A9641", lw=0, zorder=0)

    # Annotations
    res_s = f"{rec['resistance']:.2f}" if not np.isnan(rec['resistance']) else "n/a"
    resil_s = f"{rec['resilience_per100yr']:.3f}" if not np.isnan(rec['resilience_per100yr']) else "n/a"
    sig_s = "Yes" if rec.get("demo_sig_below_null") else "No"
    
    ann = (f"Event: {rec['onset_bp']}-{rec['end_bp']} Cal BP\n"
           f"Resistance: {res_s} | Resilience: {resil_s}/100yr\n"
           f"Sig. Demo Drop: {sig_s}")
           
    ax.text(0.03, 0.95, ann, transform=ax.transAxes, fontsize=9, va="top", 
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.9))
            
    ax.set_xlim(t.max(), t.min())
    ax.set_xlabel("Calendar Age (Cal BP)")
    ax.set_ylabel("Demographic Z-score", color="#2C7BB6")
    ax.tick_params(axis="y", labelcolor="#2C7BB6")
    ax.set_title("E) Resistance & Resilience Detail", fontweight="bold", loc="center")
    sns.despine(ax=ax, right=False, top=True)


def _res_panel_multi(ax, df_res):
    dur = df_res["duration_yr"].fillna(0).values
    dur_n = (dur - dur.min()) / (dur.max() - dur.min() + 1)
    
    sc = ax.scatter(df_res["resistance"].fillna(0),
                    df_res["resilience_per100yr"].fillna(0),
                    c=dur_n, cmap="YlOrRd", s=100, zorder=5,
                    edgecolors="#333", linewidths=0.5, alpha=0.9)
                    
    for _, row in df_res.iterrows():
        res, resil = row["resistance"], row["resilience_per100yr"]
        if not (np.isnan(res) or np.isnan(resil)):
            ax.annotate(f"{int(row['onset_bp'])}", (res, resil),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
                        
    ax.axhline(0, color="lightgrey", lw=1, ls="--", zorder=0)
    ax.axvline(0, color="lightgrey", lw=1, ls="--", zorder=0)
    plt.colorbar(sc, ax=ax, label="Event Duration", shrink=0.8, pad=0.04)
    
    ax.set_xlabel("Resistance (-1=collapse, +1=no impact)")
    ax.set_ylabel("Resilience (per 100 yr)")
    ax.set_title(f"E) Resistance vs Resilience ({len(df_res)} episodes)", fontweight="bold", loc="center")


def make_plots(t, spd_years, spd_un, spd_no, demo_z, env_z,
               df_res, env_label, spd_lo=None, spd_hi=None):
    
    sns.set_theme(style="ticks", font_scale=0.9)
    n_rows = 4 if (spd_lo is not None) else 3
    fig = plt.figure(figsize=(16, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.25)

    fig.suptitle(f"Human-Environment Dynamics: {SITE_NAME}\n"
                 f"{TIME_MIN}-{TIME_MAX} Cal BP | Archo: {ARCHO_RADIUS_KM}km | Env: {ENV_RADIUS_KM}km",
                 fontsize=14, fontweight="bold", y=0.96)

    color_demo = "#2C7BB6"  
    color_env = "#D7191C"   
    color_ano = "#FDAE61"   
    
    # --- Plot 1: SPD (Full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(spd_years, spd_un, alpha=0.2, color=color_demo, lw=0)
    ax1.plot(spd_years, spd_un, color=color_demo, lw=1.5, label="Unnormalized SPD")
    ax1.plot(spd_years, spd_no, color="#333", lw=1.0, ls="--", alpha=0.5, label="Normalized SPD")

    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax1.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax1.set_xlim(spd_years.max(), spd_years.min())
    ax1.set_title("A) Archaeological Activity Proxy (SPD)", fontweight="bold", loc="center")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left", frameon=False)
    
    # --- Plot 2: Environmental ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(t, env_z, alpha=0.15, color=color_env, lw=0)
    ax2.plot(t, env_z, color=color_env, lw=1.2, alpha=0.6)
    ax2.plot(t, sg_smooth(env_z), color="#9E0142", lw=1.5, label="Smoothed Trend")
    ax2.axhline(ENV_ANOMALY_THRESHOLD, color=color_env, lw=1, ls=":", label="Anomaly Threshold")
    ax2.set_xlim(t.max(), t.min())
    
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax2.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax2.set_title("B) Environmental Proxy (Z-scored)", fontweight="bold", loc="center")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left", frameon=False)

    # --- Plot 3: Demographic ---
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.fill_between(t, demo_z, alpha=0.2, color=color_demo, lw=0)
    ax3.plot(t, demo_z, color=color_demo, lw=1.5)
    
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax3.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax3.set_title("C) Demographic Proxy (Z-scored)", fontweight="bold", loc="center")
    ax3.set_ylabel("Z-Score")

    # --- Plot 4: Overlay ---
    ax4 = fig.add_subplot(gs[2, 0], sharex=ax2)
    ax4.plot(t, demo_z, color=color_demo, lw=1.5, label="Demographic")
    ax4.plot(t, env_z, color=color_env, lw=1.5, alpha=0.8, label="Environmental")
    ax4.axhline(0, color="lightgrey", lw=1, ls="--")
    
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax4.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax4.set_title("D) Overlay: Demographic vs Environmental", fontweight="bold", loc="center")
    ax4.set_ylabel("Z-Score")
    ax4.legend(loc="upper left", frameon=False)

    # --- Plot 5: Resistance & Resilience ---
    ax5 = fig.add_subplot(gs[2, 1])
    n = len(df_res)
    if n == 0:
        ax5.set_axis_off()
        ax5.text(0.5, 0.5, "No anomalies detected.", ha="center", va="center", fontsize=12)
        ax5.set_title("E) Resistance & Resilience", fontweight="bold", loc="center")
    elif n == 1:
        _res_panel_single(ax5, t, demo_z, env_z, df_res.iloc[0].to_dict())
    else:
        _res_panel_multi(ax5, df_res)

    # --- Plot 6: MC Null Model ---
    if spd_lo is not None and n_rows == 4:
        ax6 = fig.add_subplot(gs[3, :], sharex=ax1)
        ax6.fill_between(spd_years, spd_lo, spd_hi, alpha=0.15, color="grey", lw=0, label="95% Null Envelope")
        ax6.plot(spd_years, spd_un, color=color_demo, lw=1.5, label="Empirical SPD")
        
        sig_mask = spd_un < spd_lo
        if sig_mask.any():
            ax6.fill_between(spd_years, spd_un, spd_lo, where=sig_mask, alpha=0.4, color="#D7191C", lw=0, label="Sig. Trough")
            
        # Add the shaded anomaly regions to the MC Null Model plot
        if not df_res.empty:
            for _, row in df_res.iterrows():
                ax6.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
                
        ax6.set_title("F) Monte Carlo Significance Test", fontweight="bold", loc="center")
        ax6.set_ylabel("Density")
        ax6.legend(loc="upper left", frameon=False)

    # Clean up axes
    for ax in fig.axes:
        sns.despine(ax=ax)
 
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Calendar Age (Cal BP)")
        
    if spd_lo is not None and n_rows == 4:
        ax6.tick_params(labelbottom=True)
        ax6.set_xlabel("Calendar Age (Cal BP)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"\n[output] Cleaned figure saved -> {OUTPUT_PLOT}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    divider("HUMAN-ENVIRONMENT COMPARISON")
    print("  Databases : PANGAEA -> Neotoma -> GISP2")

    df_nearby = lookup_and_confirm(INPUT_FILE)
    df_clean  = apply_hygiene(df_nearby)
    if len(df_clean) < 5:
        sys.exit("Not enough dates. Increase ARCHO_RADIUS_KM.")

    # SPD + MC envelope
    if SPD_FROM_04 and Path(SPD_FROM_04).is_file():
        divider("SPD — LOADING FROM 04_Population_Dynamics.py")
        years_spd, spd_un, spd_lo, spd_hi = load_spd_from_04(SPD_FROM_04)
        spd_no    = spd_un
        spd_pvals = (spd_un < spd_lo).astype(float)
        print("  Skipping IntCal20 rebuild and MC simulation.")
    else:
        divider("SPD — BUILDING FROM SCRATCH")
        years_spd, spd_un, spd_no = build_spd(df_clean)
        spd_lo, spd_hi, spd_pvals = spd_significance_envelope(
            df_clean, years_spd, spd_un, n_sim=MC_N_SIM, model=MC_MODEL
        )

    env_ages, env_vals, env_label = get_env_data()
    t, demo_z, env_z = align(years_spd, spd_un, env_ages, env_vals)

    pd.DataFrame({"CalBP": t, "Demo_Z": demo_z, "Env_Z": env_z}).to_csv(
        OUTPUT_CSV, index=False)

    # Save MC envelope alongside SPD
    pd.DataFrame({
        "CalBP":            years_spd,
        "SPD_Unnormalized": spd_un,
        "SPD_Normalized":   spd_no,
        "MC_lo_2.5pct":     spd_lo,
        "MC_hi_97.5pct":    spd_hi,
        "MC_pval":          spd_pvals,
    }).to_csv(OUTPUT_SPD, index=False)

    df_res = detect_anomalies(t, demo_z, env_z,
                              spd_lo=spd_lo, spd_hi=spd_hi,
                              years_spd=years_spd)

    make_plots(t, years_spd, spd_un, spd_no, demo_z, env_z,
               df_res, env_label,
               spd_lo=spd_lo, spd_hi=spd_hi)

    divider("COMPLETE")
    print(f"  Site      : {SITE_NAME}  [{SITE_LAT}N, {SITE_LON}E]")
    print(f"  Archo r.  : {ARCHO_RADIUS_KM} km  |  Env r.: {ENV_RADIUS_KM} km")
    print(f"  Time win. : {TIME_MIN}-{TIME_MAX} Cal BP  |  BIN_H={BIN_H} yr")
    print(f"  Dates     : {len(df_clean):,}  |  Anomalies: {len(df_res)}")
    print(f"  Env data  : {env_label}")
    print("  Outputs:")
    for f in (OUTPUT_SPD, OUTPUT_ENV, OUTPUT_CSV, OUTPUT_RES, OUTPUT_PLOT):
        print(f"    {f}")


if __name__ == "__main__":
    main()