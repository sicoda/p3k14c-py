"""
07_ClimateStress_PCA.py : Composite Climate Stress Index via PCA; an alternative to / extension of 06_Human_Environment.py

Input  : p3k14c_pristine_dates.csv, <Site>_spd_for_06.csv (if available), intcal20.npz, gisp2_cache.csv
Output : <Site>_ccsi.csv, <Site>_ccsi_resilience.csv, <Site>_ccsi.png

KEY ASPECTS :

- The SPD (demographic data) is NEVER entered into the PCA matrix.
  The PCA reduces only the environmental proxies (ice core, pollen, isotopes)
  into a Composite Climate Stress Index (CCSI).  The SPD is then correlated
  against that independent CCSI — keeping the climate and human datasets
  strictly firewalled.  Adding SPD to the PCA would create circular
  self-correlation (data leakage).

- EM-PCA gap-fill replaces mean imputation so that missing intervals
  during genuine anomalies are not artificially pulled to zero.

- Gaussian-kernel binning replaces rigid box averaging so that coarse
  proxies (pollen, ~56 yr spacing) are not falsely sharpened to decadal
  resolution.

- Effective-N (Bretherton 1999) is reported to flag when autocorrelation
  makes the PCA's independence assumption untenable.

- Resilience curve-fitting returns SE(k); results are flagged "uncertain"
  when SE/k > threshold.

DEPENDENCIES : pip install pandas numpy scipy matplotlib seaborn scikit-learn requests (cartopy / shapely NOT required)
PYTHON       : 3.12+
"""

import json as _json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, detrend as scipy_detrend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

SITE_NAME       = "Catalhoyuk"
SITE_LAT        = 37.6660
SITE_LON        = 32.8277

ARCHO_RADIUS_KM = 5             # ← must match 04_Population_Analysis.py ARCHO_RADIUS_KM
ENV_RADIUS_KM   = 1000

TIME_MIN    = 7700              # ← must match 04_Population_Analysis.py TIME_MIN
TIME_MAX    = 9500              # ← must match 04_Population_Analysis.py TIME_MAX
RESOLUTION  = 10
MAX_ERROR   = 150
BIN_H       = 50

KERNEL_SIGMA_YR     = 50
MAX_INTERP_GAP_BINS = 10

USE_GISP2        = True
PANGAEA_KEYWORDS = ["stable isotope", "speleothem", "pollen"]
NEOTOMA_TYPES    = ["pollen", "stable isotopes"]

MIN_PROXIES_FOR_PCA  = 2
EM_PCA_MAX_ITER      = 50
EM_PCA_TOL           = 1e-4

WARN_NEFF_THRESHOLD     = 30
RESILIENCE_CV_THRESHOLD = 0.5

CCSI_ANOMALY_THRESHOLD = -1.0
ENV_RECOVERY_STEPS     = 10
BASELINE_WINDOW        = 400

SPD_FROM_04 = f"{SITE_NAME.replace(' ', '_')}_spd_for_06.csv"

# ---------------------------------------------------------------------------
# END OF USER CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_FILE   = os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv")
INTCAL_CACHE = "intcal20.npz"
_SLUG        = SITE_NAME.replace(" ", "_")
OUTPUT_CCSI  = f"{_SLUG}_ccsi.csv"
OUTPUT_RES   = f"{_SLUG}_ccsi_resilience.csv"
OUTPUT_PLOT  = f"{_SLUG}_ccsi.png"

GISP2_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
    "summit/gisp2/isotopes/gisp2_temp_accum_alley2000.txt"
)
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
    arr    = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 5:
        return arr.copy()
    idx    = np.arange(len(arr))
    filled = arr.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    w = min(window, len(filled))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()
    out          = savgol_filter(filled, w, poly)
    out[~finite] = np.nan
    return out


def taphonomic_weight(t):
    raw = 5.726442e6 * np.power(t + 2176.4, -1.3925309)
    return raw / raw[np.argmin(t)]


# ---------------------------------------------------------------------------
# Step 0.1  -  Time gap filling
# ---------------------------------------------------------------------------

def _linear_fill_short_gaps(col: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap; leave longer ones."""
    out    = col.copy()
    finite = np.where(np.isfinite(out))[0]
    if len(finite) < 2:
        return out
    for i in range(len(finite) - 1):
        left, right = finite[i], finite[i + 1]
        gap = right - left - 1
        if 0 < gap <= max_gap:
            xs = np.arange(left + 1, right)
            out[xs] = np.interp(xs, [left, right], [out[left], out[right]])
    return out


def em_pca_impute(X: np.ndarray,
                  n_components: int,
                  max_iter: int = EM_PCA_MAX_ITER,
                  tol: float    = EM_PCA_TOL):
    """
    EM-PCA gap-fill (Schneider 2001 / DINEOF-style).

    Returns
    -------
    X_filled  : fully imputed (T, P) array
    converged : bool — False means max_iter reached without satisfying tol.
                Caller stores this flag in diag and warns the user;
                results are best-available but not guaranteed stable.
    """
    X_filled  = X.copy()
    miss_mask = np.isnan(X)

    if not miss_mask.any():
        return X_filled, True

    # Bootstrap: fill with column means (this is only the starting point,
    # not the final imputation — subsequent EM iterations replace these values)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    for j in range(X.shape[1]):
        X_filled[miss_mask[:, j], j] = col_means[j]

    n_comp    = min(n_components, X.shape[1], X.shape[0] - 1)
    prev_fill = X_filled[miss_mask].copy()
    delta     = np.inf
    converged = False

    for iteration in range(max_iter):
        mu     = X_filled.mean(axis=0)
        X_c    = X_filled - mu
        pca    = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(X_c)
        X_rec  = pca.inverse_transform(scores) + mu
        X_filled[miss_mask] = X_rec[miss_mask]
        delta = np.linalg.norm(X_filled[miss_mask] - prev_fill)
        if delta < tol:
            print(f"     EM-PCA converged after {iteration + 1} iterations "
                  f"(delta={delta:.2e})")
            converged = True
            break
        prev_fill = X_filled[miss_mask].copy()

    if not converged:
        print(f"     WARNING: EM-PCA did NOT converge after {max_iter} "
              f"iterations (final delta={delta:.2e}, tol={tol:.2e}).")
        print(f"       Gap-filled values are provisional. Consider raising "
              f"EM_PCA_MAX_ITER or increasing KERNEL_SIGMA_YR.")

    return X_filled, converged


# ---------------------------------------------------------------------------
# Step 0.2  -  IntCal20 calibration curve
# ---------------------------------------------------------------------------

def load_intcal20():
    if os.path.exists(INTCAL_CACHE):
        d = np.load(INTCAL_CACHE)
        return d["cal_bp"], d["c14_age"], d["c14_error"]
    print(" [IntCal20] Downloading ...")
    url = "https://intcal.org/curves/intcal20.14c"
    try:
        resp = requests.get(url, timeout=60); resp.raise_for_status()
    except Exception:
        resp = requests.get(url, timeout=60, verify=False); resp.raise_for_status()
    cal_bp, c14_age, c14_err = [], [], []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        p = line.split(",")
        if len(p) < 3:
            continue
        try:
            cal_bp.append(float(p[0])); c14_age.append(float(p[1])); c14_err.append(float(p[2]))
        except ValueError:
            continue
    arr = np.array(cal_bp), np.array(c14_age), np.array(c14_err)
    np.savez(INTCAL_CACHE, cal_bp=arr[0], c14_age=arr[1], c14_error=arr[2])
    return arr


INTCAL_CAL, INTCAL_C14, INTCAL_ERR = load_intcal20()
_si = np.argsort(INTCAL_CAL)
INTCAL_CAL, INTCAL_C14, INTCAL_ERR = INTCAL_CAL[_si], INTCAL_C14[_si], INTCAL_ERR[_si]


def calibrate_date(c14_age, c14_error, cal_range):
    cc  = np.interp(cal_range, INTCAL_CAL, INTCAL_C14)
    ce  = np.interp(cal_range, INTCAL_CAL, INTCAL_ERR)
    sig = np.sqrt(c14_error ** 2 + ce ** 2)
    p   = np.exp(-0.5 * ((c14_age - cc) / sig) ** 2) / sig
    s   = p.sum()
    return p / s if s > 0 else p


# ---------------------------------------------------------------------------
# Step 1 — Load / build SPD
# ---------------------------------------------------------------------------

def load_spd_from_04(path):
    df     = pd.read_csv(path)
    years  = df["CalBP"].values
    spd_un = df["SPD_TaphCorrected"].values
    spd_lo = df["MC_lo_2.5pct"].values
    spd_hi = df["MC_hi_97.5pct"].values
    print(f"  [SPD] Loaded from {path}  ({len(years)} time steps)")
    return years, spd_un, spd_lo, spd_hi


def build_spd_from_scratch(df_clean):
    divider("BUILDING SPD FROM SCRATCH")
    years = np.arange(TIME_MIN, TIME_MAX + RESOLUTION, RESOLUTION)
    df    = df_clean.copy()
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
                counter += 1; cb = counter
            df.loc[s_idx[k], "Bin"] = f"b{cb}"
        counter += 1

    spd = np.zeros(len(years))
    for _, grp in df.groupby("Bin"):
        bu = np.zeros(len(years))
        for _, row in grp.iterrows():
            c14 = row.get("Age"); err = row.get("Error")
            if pd.notna(c14) and pd.notna(err) and float(err) > 0:
                prob = calibrate_date(float(c14), float(err), years)
            else:
                sigma = max(float(row.get("UncalBPError", 30)), 20)
                prob  = np.exp(-0.5 * ((years - float(row["MedianCalBP"])) / sigma) ** 2)
            prob = prob / (prob.sum() or 1.0)
            bu  += prob
        spd += bu / len(grp)

    spd  /= spd.sum() or 1.0
    taph  = taphonomic_weight(years)
    spd   = spd / taph; spd /= spd.sum()
    lo    = np.zeros_like(spd); hi = np.zeros_like(spd)
    return years, spd, lo, hi


def get_spd(df_nearby):
    if SPD_FROM_04 and Path(SPD_FROM_04).is_file():
        divider("SPD — LOADING FROM 04_Population_Dynamics.py")
        return load_spd_from_04(SPD_FROM_04)
    divider("SPD — BUILDING FROM SCRATCH")
    df_clean = _apply_hygiene(df_nearby)
    if len(df_clean) < 5:
        sys.exit("Too few dates. Increase ARCHO_RADIUS_KM.")
    return build_spd_from_scratch(df_clean)


def _apply_hygiene(df):
    df  = df.dropna(subset=["Age", "Error", "Lat", "MedianCalBP"])
    df  = df[df["Error"] <= MAX_ERROR].copy()
    mat = df["Material"].fillna("").str.lower().str.strip()
    df  = df[~mat.apply(lambda m: any(t in m for t in OLD_WOOD | MARINE))].copy()
    df  = df[df["MedianCalBP"].between(TIME_MIN, TIME_MAX)].copy()
    if "LocAccuracy" in df.columns:
        df["LocAccuracy"] = pd.to_numeric(df["LocAccuracy"], errors="coerce")
        df = df[df["LocAccuracy"] >= 1].copy()
    return df


# ---------------------------------------------------------------------------
# Step 2 — Fetch palaeoclimate proxies
# ---------------------------------------------------------------------------

def fetch_gisp2() -> pd.DataFrame:
    print(" [GISP2] Fetching temperature ...")
    cache = Path("gisp2_cache.csv")
    if cache.is_file():
        df = pd.read_csv(cache)
        print(f" [GISP2] Loaded from cache ({len(df):,} pts)")
    else:
        resp = requests.get(GISP2_URL, timeout=30); resp.raise_for_status()
        rows, in_data = [], False
        for line in resp.text.splitlines():
            if "Age" in line and "Temperature" in line:
                in_data = True; continue
            if not in_data: continue
            if "Accumulation" in line: break
            p = line.strip().split()
            if len(p) >= 2:
                try: rows.append((float(p[0]) * 1000, float(p[1])))
                except ValueError: pass
        df = pd.DataFrame(rows, columns=["age_bp", "value"]).sort_values("age_bp")
        df["variable"] = "GISP2_Temp_C"
        df["source"]   = "GISP2"
        df.to_csv(cache, index=False)
        print(f" [GISP2] {len(df):,} points cached")
    df["variable"] = "GISP2_Temp_C"
    df["source"]   = "GISP2"
    return df[["age_bp", "value", "variable", "source"]]


def fetch_pangaea_proxy(keyword: str):
    print(f"\n [PANGAEA] '{keyword}' within {ENV_RADIUS_KM} km ...")
    try:
        resp = requests.get(
            "https://ws.pangaea.de/es/pangaea/panmd/_search",
            params={"q": keyword, "size": 30, "_source": "true"},
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
    except Exception as e:
        print(f" PANGAEA failed: {e}"); return None

    rows    = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    for hit in hits:
        src = hit.get("_source", {})
        doi = src.get("URI", "")
        if not doi: continue
        lat = src.get("meanPosition", {}).get("lat")
        lon = src.get("meanPosition", {}).get("lon")
        if lat is None:
            n = src.get("northBoundLatitude"); s = src.get("southBoundLatitude")
            lat = (n + s) / 2 if n is not None and s is not None else None
        if lon is None:
            e = src.get("eastBoundLongitude"); w = src.get("westBoundLongitude")
            lon = (e + w) / 2 if e is not None and w is not None else None
        if lat is None or lon is None: continue
        try: lat, lon = float(lat), float(lon)
        except (TypeError, ValueError): continue
        if haversine_km(SITE_LAT, SITE_LON, lat, lon) > ENV_RADIUS_KM: continue

        data_url = (doi.replace("https://doi.org/", "https://doi.pangaea.de/")
                    + "?format=textfile")
        try:
            time.sleep(0.3)
            dr = requests.get(data_url, headers=headers, timeout=30)
            dr.raise_for_status(); text = dr.text
            lines      = text.splitlines()
            he         = next((i + 1 for i, l in enumerate(lines) if l.startswith("*/")), 0)
            if he == 0: continue
            data_lines = lines[he:]
            if not data_lines: continue
            col_line   = data_lines[0].split("\t")
            age_col    = next((j for j, c in enumerate(col_line)
                               if any(t in c.lower() for t in
                                      ["age", "cal bp", "ka bp", "cal yr", "yr bp"])), None)
            if age_col is None: continue
            skip = {"latitude","longitude","depth","event","elevation",
                    "sample","label","comment","reference","age","date"}
            val_cols = [(j, c) for j, c in enumerate(col_line)
                        if j != age_col and not any(s in c.lower() for s in skip)]
            for line in data_lines[1:]:
                parts = line.split("\t")
                if len(parts) <= age_col: continue
                try:
                    raw_age = float(parts[age_col])
                    if raw_age < 200: raw_age *= 1000
                    if not (TIME_MIN <= raw_age <= TIME_MAX): continue
                    for j, col_name in val_cols:
                        if j < len(parts) and parts[j].strip():
                            try:
                                rows.append({
                                    "age_bp":   raw_age,
                                    "value":    float(parts[j]),
                                    "variable": f"PANGAEA_{keyword[:15]}_{col_name[:20]}",
                                    "source":   "PANGAEA",
                                })
                            except ValueError: pass
                except ValueError: continue
        except Exception: continue

    if not rows: return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(TIME_MIN, TIME_MAX)]
    print(f" PANGAEA '{keyword}': {len(df):,} observations")
    return df if len(df) >= 5 else None


def fetch_neotoma_proxy(dtypes: list):
    deg  = ENV_RADIUS_KM / 111.0
    rows = []
    print(f"\n [Neotoma] Searching within {ENV_RADIUS_KM} km ...")
    try:
        r = requests.get(
            f"{NEOTOMA_API}/data/sites"
            f"?bbox={SITE_LON-deg:.4f},{SITE_LAT-deg:.4f},"
            f"{SITE_LON+deg:.4f},{SITE_LAT+deg:.4f}&limit=100",
            timeout=30)
        r.raise_for_status()
        all_sites = r.json().get("data", [])
    except Exception as e:
        print(f" Neotoma failed: {e}"); return None

    nearby_ids = []
    for s in all_sites:
        if not isinstance(s, dict): continue
        geo_raw = s.get("geography", "{}")
        try:
            geo    = _json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
            coords = geo.get("coordinates", [None, None])
            if coords[0] and coords[1]:
                if haversine_km(SITE_LAT, SITE_LON, coords[1], coords[0]) <= ENV_RADIUS_KM:
                    nearby_ids.append(s.get("siteid"))
        except Exception: continue

    print(f" Neotoma sites: {len(nearby_ids)}")
    if not nearby_ids: return None

    for dtype in dtypes:
        ids_str = ",".join(str(i) for i in nearby_ids)
        try:
            r = requests.get(
                f"{NEOTOMA_API}/data/datasets?siteid={ids_str}"
                f"&datasettype={dtype.replace(' ','%20')}&limit=200",
                timeout=30)
            r.raise_for_status()
            datasets = r.json().get("data", [])
        except Exception: continue
        print(f" Neotoma '{dtype}': {len(datasets)} datasets")
        for ds in datasets:
            if not isinstance(ds, dict): continue
            try:
                si      = ds.get("site", {})
                if not isinstance(si, dict): continue
                site_ds = si.get("datasets", [])
                if not site_ds: continue
                dsid    = site_ds[0].get("datasetid")
                if dsid is None: continue
                time.sleep(0.3)
                sr = requests.get(f"{NEOTOMA_API}/data/downloads/{dsid}", timeout=30)
                sr.raise_for_status()
                for item in sr.json().get("data", []):
                    for sample in item.get("samples", []):
                        age = sample.get("age")
                        if age is None: continue
                        for datum in sample.get("data", []):
                            rows.append({
                                "age_bp":   float(age),
                                "value":    datum.get("value"),
                                "variable": f"Neotoma_{dtype[:10]}_{datum.get('variablename','')[:20]}",
                                "source":   "Neotoma",
                            })
            except Exception: continue

    if not rows: return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(TIME_MIN, TIME_MAX)]
    print(f" Neotoma total: {len(df):,} observations")
    return df if len(df) >= 5 else None


def collect_proxies() -> pd.DataFrame:
    divider("PROXY COLLECTION")
    print(f"  PANGAEA keywords : {PANGAEA_KEYWORDS}")
    print(f"  Neotoma types    : {NEOTOMA_TYPES}")
    print(f"  Search radius    : {ENV_RADIUS_KM} km")
    print(f"  Database order   : PANGAEA -> Neotoma -> GISP2")
    frames = []
    if USE_GISP2:
        frames.append(fetch_gisp2())
    for kw in PANGAEA_KEYWORDS:
        df = fetch_pangaea_proxy(kw)
        if df is not None:
            frames.append(df)
    df_neo = fetch_neotoma_proxy(NEOTOMA_TYPES)
    if df_neo is not None:
        frames.append(df_neo)
    if not frames:
        sys.exit("ERROR: No proxy data retrieved.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["age_bp"].between(TIME_MIN, TIME_MAX)]
    n_vars   = combined["variable"].nunique()
    print(f"\n  Total proxy observations : {len(combined):,}")
    print(f"  Distinct proxy variables : {n_vars}")
    for v in combined["variable"].unique():
        n = (combined["variable"] == v).sum()
        print(f"    {v:<55} {n:>5} obs")
    return combined


# ---------------------------------------------------------------------------
# Step 3 — Gaussian-kernel binning
# ---------------------------------------------------------------------------

def _gaussian_kernel_bin(ages, values, grid, sigma):
    MIN_WEIGHT = 1e-6
    out    = np.full(len(grid), np.nan)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out
    ages_f  = ages[finite]
    vals_f  = values[finite]
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((ages_f - g) / sigma) ** 2)
        W = w.sum()
        if W > MIN_WEIGHT:
            out[i] = (w * vals_f).sum() / W
    return out


def bin_proxies(proxy_df: pd.DataFrame) -> pd.DataFrame:
    divider("GAUSSIAN-KERNEL BINNING (sigma = %d yr)" % KERNEL_SIGMA_YR)
    years  = np.arange(TIME_MIN, TIME_MAX + RESOLUTION, RESOLUTION)
    result = pd.DataFrame({"CalBP": years})

    for var_name, grp in proxy_df.groupby("variable"):
        grp        = grp.sort_values("age_bp")
        raw_ages   = grp["age_bp"].values
        raw_values = grp["value"].values

        col = _gaussian_kernel_bin(raw_ages, raw_values, years, KERNEL_SIGMA_YR)
        col = _linear_fill_short_gaps(col, MAX_INTERP_GAP_BINS)

        coverage = np.isfinite(col).sum() / len(years)
        if coverage < 0.10:
            print(f" SKIP {var_name:<55} (coverage={coverage:.0%} < 10%)")
            continue

        raw_col   = _gaussian_kernel_bin(raw_ages, raw_values, years, KERNEL_SIGMA_YR)
        valid_idx = np.where(np.isfinite(raw_col))[0]
        if len(valid_idx) == 0:
            continue
        col[years < years[valid_idx[0]]]  = np.nan
        col[years > years[valid_idx[-1]]] = np.nan

        print(f" KEEP {var_name:<55} coverage={coverage:.0%}  n_raw={len(raw_ages)}")
        result[var_name] = col

    proxy_cols = [c for c in result.columns if c != "CalBP"]
    print(f"\n Proxies retained: {len(proxy_cols)}")
    if not proxy_cols:
        sys.exit("ERROR: No proxy variables survived binning.")
    return result


# ---------------------------------------------------------------------------
# Step 4 — PCA on climate proxies ONLY → CCSI
#
# IMPORTANT: The SPD is NOT included here.  Adding population data to the
# PCA matrix would make the CCSI partially self-correlated with the SPD,
# invalidating any subsequent climate-demography comparison (data leakage).
# The SPD enters only in Step 5 (correlation) as an independent variable.
# ---------------------------------------------------------------------------

def _effective_n(x: np.ndarray) -> float:
    """N_eff = N*(1-r1)/(1+r1)  (Bretherton et al. 1999)."""
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 4:
        return float(N)
    x_c = x - x.mean()
    r1  = float(np.corrcoef(x_c[:-1], x_c[1:])[0, 1])
    r1  = max(-0.999, min(0.999, r1))
    return N * (1 - r1) / (1 + r1)


def compute_ccsi(wide_df: pd.DataFrame,
                 proxy_df: pd.DataFrame) -> tuple:
    """
    Reduce climate proxies to a single Composite Climate Stress Index via PCA.

    The SPD is deliberately excluded from this function.  The demographic
    signal is correlated against the resulting CCSI in a separate step
    (compute_proxy_spd_correlations) so that the two datasets remain
    statistically independent.
    """
    divider("PCA — COMPOSITE CLIMATE STRESS INDEX (climate proxies only)")

    # Climate-proxy columns only — no SPD
    proxy_cols = [c for c in wide_df.columns if c != "CalBP"]
    n_proxies  = len(proxy_cols)
    print(f" Proxy columns entering PCA : {n_proxies}")

    X_raw = wide_df[proxy_cols].values.astype(float)
    scaler = StandardScaler()
    X_z    = scaler.fit_transform(np.where(np.isfinite(X_raw), X_raw, np.nan))
    X_z[np.isnan(X_raw)] = np.nan

    if n_proxies < MIN_PROXIES_FOR_PCA:
        print(f" Only {n_proxies} proxy — skipping PCA, using mean Z-score.")
        ccsi_raw  = np.nanmean(X_z, axis=1)
        explained = [1.0]
        loadings  = {proxy_cols[0]: 1.0}
        em_converged = True
    else:
        n_miss_before = np.isnan(X_z).sum()
        print(f"\n Missing cells before EM-PCA : {n_miss_before:,} "
              f"({n_miss_before / X_z.size:.1%})")
        n_comp_em = min(n_proxies, 3)
        print(f" Running EM-PCA with {n_comp_em} components ...")
        X_imp, em_converged = em_pca_impute(X_z, n_components=n_comp_em)
        print(f" Missing cells after EM-PCA  : {np.isnan(X_imp).sum():,}")
        if not em_converged:
            print(" *** EM-PCA gap-fill did not converge — "
                  "flag 'em_pca_converged=False' stored in diagnostics ***")

        pca    = PCA(n_components=min(n_proxies, 5), random_state=42)
        scores = pca.fit_transform(X_imp)

        pc1      = scores[:, 0]
        explained = pca.explained_variance_ratio_.tolist()
        loadings  = dict(zip(proxy_cols, pca.components_[0]))

        print(f"\n Explained variance by PC1 : {explained[0]:.1%}")
        if len(explained) > 1:
            print(f" Explained variance by PC2 : {explained[1]:.1%}")
        print(f"\n PC1 loadings:")
        for proxy, loading in sorted(loadings.items(), key=lambda x: -abs(x[1])):
            bar  = "X" * int(abs(loading) * 20)
            sign = "+" if loading >= 0 else "-"
            print(f"   {sign}{bar:<20} {loading:+.3f}  {proxy}")

        # Sign convention
        gisp2_candidates = [p for p in proxy_cols if "GISP2" in p or "Temp" in p]
        if gisp2_candidates:
            ref_proxy   = gisp2_candidates[0]
            ref_loading = loadings[ref_proxy]
            if ref_loading < 0:
                pc1      = -pc1
                loadings = {k: -v for k, v in loadings.items()}
                print(f"\n PC1 sign FLIPPED — {ref_proxy} loading was {ref_loading:+.3f}")
            else:
                print(f"\n PC1 sign OK — {ref_proxy} loading is {ref_loading:+.3f}")
        else:
            gisp2_raw = proxy_df[proxy_df["variable"] == "GISP2_Temp_C"].copy()
            if not gisp2_raw.empty:
                gi = gisp2_raw.sort_values("age_bp")
                gisp2_interp = np.interp(wide_df["CalBP"].values,
                                          gi["age_bp"].values, gi["value"].values)
                valid = np.isfinite(pc1) & np.isfinite(gisp2_interp)
                if valid.sum() > 5:
                    r = np.corrcoef(pc1[valid], gisp2_interp[valid])[0, 1]
                    if r < 0:
                        pc1      = -pc1
                        loadings = {k: -v for k, v in loadings.items()}
                        print(f"\n PC1 sign FLIPPED (r={r:.3f} with GISP2)")
                    else:
                        print(f"\n PC1 sign OK (r={r:.3f} with GISP2)")
            else:
                print("\n WARNING: No GISP2 data for sign check.")

        ccsi_raw = pc1

    # Detrend orbital-scale trend
    finite_mask = np.isfinite(ccsi_raw)
    if finite_mask.sum() > 10:
        ccsi_raw[finite_mask] = scipy_detrend(ccsi_raw[finite_mask])

    mu, sd = np.nanmean(ccsi_raw), np.nanstd(ccsi_raw)
    ccsi_z = (ccsi_raw - mu) / (sd or 1.0)
    ccsi_z = sg_smooth(ccsi_z)

    # Autocorrelation / effective N
    n_eff = _effective_n(ccsi_z)
    print(f"\n Autocorrelation diagnostic:")
    print(f"   N (time steps) : {len(ccsi_z)}")
    print(f"   N_eff          : {n_eff:.1f}")
    if n_eff < WARN_NEFF_THRESHOLD:
        print(f"   WARNING: N_eff < {WARN_NEFF_THRESHOLD}. Series is heavily autocorrelated.")
        print(f"   Standard PCA independence assumption is VIOLATED.")
        print(f"   Consider MSSA for a fully time-series-aware decomposition.")
    else:
        print(f"   N_eff is adequate for exploratory PCA.")

    # Sign diagnostic vs raw GISP2
    gisp2_raw = proxy_df[proxy_df["variable"] == "GISP2_Temp_C"].copy()
    if not gisp2_raw.empty:
        gi = gisp2_raw.sort_values("age_bp")
        gisp2_interp = np.interp(wide_df["CalBP"].values,
                                  gi["age_bp"].values, gi["value"].values)
        gisp2_z = ((gisp2_interp - np.nanmean(gisp2_interp))
                   / (np.nanstd(gisp2_interp) or 1.0))
        valid = np.isfinite(ccsi_z) & np.isfinite(gisp2_z)
        if valid.sum() > 5:
            r_final = np.corrcoef(ccsi_z[valid], gisp2_z[valid])[0, 1]
            print(f"\n Sign diagnostic — CCSI vs GISP2 r = {r_final:.3f}")
            if r_final > 0:
                print(f"   Positive: warm = high CCSI = low stress (correct)")
            else:
                print(f"   Negative: CCSI may be inverted. Check loadings.")

    diag = {
        "n_proxies":         n_proxies,
        "proxy_names":       proxy_cols,
        "explained_var_pc1": explained[0],
        "loadings":          loadings,
        "n_eff":             n_eff,
        "em_pca_converged":  em_converged,
        "spd_correlations":  {},   # filled in main() after firewall check
    }

    out = wide_df[["CalBP"]].copy()
    for i, col in enumerate(proxy_cols):
        out[f"Z_{col}"] = X_z[:, i]
    out["CCSI"] = ccsi_z

    out.to_csv(OUTPUT_CCSI, index=False)
    print(f"\n Saved CCSI -> {OUTPUT_CCSI}")
    return out, diag


# ---------------------------------------------------------------------------
# Step 5 — Proxy–SPD correlations
# ---------------------------------------------------------------------------

def compute_proxy_spd_correlations(wide_df: pd.DataFrame,
                                    years_spd: np.ndarray,
                                    spd_un: np.ndarray) -> dict:
    """
    For each climate proxy in wide_df compute N_eff-corrected Pearson r
    against the SPD, plus a peak cross-correlation search over +/-500 yr.

    The SPD is kept strictly separate from the PCA (no data leakage).
    np.corrcoef handles standardisation internally; no manual Z-scoring needed.
    """
    from scipy.stats import t as t_dist

    MAX_LAG_YR   = 500
    max_lag_bins = int(MAX_LAG_YR / RESOLUTION)

    # Columns to correlate: exclude CalBP only (no SPD_demographic present here)
    proxy_cols  = [c for c in wide_df.columns if c != "CalBP"]
    grid        = wide_df["CalBP"].values

    spd_on_grid = np.interp(grid, years_spd, spd_un, left=np.nan, right=np.nan)

    results = {}

    for col in proxy_cols:
        px    = wide_df[col].values.astype(float)
        valid = np.isfinite(px) & np.isfinite(spd_on_grid)

        if valid.sum() < 10:
            results[col] = dict(r=np.nan, p_corrected=np.nan,
                                peak_r=np.nan, peak_lag_yr=np.nan, n_eff=np.nan)
            continue

        px_v   = px[valid]
        spd_v  = spd_on_grid[valid]

        # np.corrcoef normalizes internally — no manual Z-score needed
        r     = float(np.corrcoef(px_v, spd_v)[0, 1])
        n_eff = max(4.0, _effective_n(px_v))

        t_stat = r * math.sqrt(n_eff - 2) / math.sqrt(max(1 - r ** 2, 1e-12))
        p_val  = float(2 * t_dist.sf(abs(t_stat), df=n_eff - 2))

        # Peak cross-correlation
        best_r, best_lag = r, 0
        for lag in range(-max_lag_bins, max_lag_bins + 1):
            if lag == 0:
                continue
            if lag > 0:
                a, b = px[lag:], spd_on_grid[:-lag]
            else:
                a, b = px[:lag], spd_on_grid[-lag:]
            v2 = np.isfinite(a) & np.isfinite(b)
            if v2.sum() < 10:
                continue
            rc = float(np.corrcoef(a[v2], b[v2])[0, 1])
            if abs(rc) > abs(best_r):
                best_r, best_lag = rc, lag

        results[col] = dict(
            r           = round(r, 4),
            p_corrected = round(p_val, 6),
            peak_r      = round(best_r, 4),
            peak_lag_yr = int(best_lag * RESOLUTION),
            n_eff       = round(n_eff, 1),
        )

    divider("PROXY-SPD CORRELATIONS (N_eff-corrected, SPD firewall)")
    print(f"  {'Proxy':<40} {'r':>6}  {'p':>8}  {'peak_r':>7}  "
          f"{'lag_yr':>7}  {'N_eff':>6}")
    print(f"  {'─'*40} {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
    for col, d in results.items():
        def _fmt(v, w=6, dec=3):
            return f"{v:{w}.{dec}f}" if not np.isnan(v) else f"{'n/a':>{w}}"
        sig = "*" if (not np.isnan(d['p_corrected']) and d['p_corrected'] < 0.05) else " "
        print(f"  {col[:40]:<40} {_fmt(d['r'])}  {_fmt(d['p_corrected'],8,4)}  "
              f"{_fmt(d['peak_r'],7,3)}  {d['peak_lag_yr']:>7}  "
              f"{_fmt(d['n_eff'],6,1)} {sig}")
    print("  (* p < 0.05 after N_eff correction)")
    return results


# ---------------------------------------------------------------------------
# Step 6 — Align CCSI with SPD
# ---------------------------------------------------------------------------

def align_ccsi_spd(ccsi_df, years_spd, spd_un):
    ccsi_years = ccsi_df["CalBP"].values
    ccsi_vals  = ccsi_df["CCSI"].values
    valid      = np.isfinite(ccsi_vals)
    t = np.arange(
        max(TIME_MIN,  float(ccsi_years[valid].min())),
        min(TIME_MAX,  float(ccsi_years[valid].max())) + RESOLUTION,
        RESOLUTION,
    )
    f_spd  = interp1d(years_spd, spd_un, bounds_error=False, fill_value=0.0)
    f_ccsi = interp1d(ccsi_years, ccsi_vals, bounds_error=False, fill_value=np.nan)
    d, e   = f_spd(t), f_ccsi(t)
    v      = np.isfinite(e) & np.isfinite(d) & (d > 0)
    t, d, e = t[v], d[v], e[v]
    dz = (d - d.mean()) / (d.std() or 1.0)
    ez = (e - e.mean()) / (e.std() or 1.0)
    print(f"[alignment] {t[0]:.0f}-{t[-1]:.0f} Cal BP  ({len(t)} points at {RESOLUTION}-yr resolution)")
    return t, dz, ez


# ---------------------------------------------------------------------------
# Step 7 — Resilience with uncertainty
# ---------------------------------------------------------------------------

def fit_resilience(rt, rd):
    if len(rt) < 4:
        return np.nan, np.nan, "undefined", True
    t   = rt - rt[0]
    rng = rd.max() - rd.min()
    if rng < 1e-10:
        return np.nan, np.nan, "undefined", True
    rd_n = (rd - rd.min()) / rng
    try:
        popt, pcov = curve_fit(
            lambda t, k: 1 - np.exp(-k * t),
            t, rd_n, p0=[0.01], bounds=(0, np.inf), maxfev=5000,
        )
        k     = float(popt[0])
        k_var = float(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.inf
        k_se  = math.sqrt(k_var) if k_var < 1e12 else np.inf
        cv    = k_se / k if k > 0 else np.inf
        return k * 100, k_se * 100, "exponential", bool(cv > RESILIENCE_CV_THRESHOLD)
    except RuntimeError:
        slope = float(np.polyfit(t, rd_n, 1)[0])
        return slope * 100, np.nan, "linear", True


def detect_episodes(t, demo_z, ccsi_z, spd_lo, spd_hi, years_spd):
    divider("EPISODE DETECTION")
    print(f"  Threshold     : Z < {CCSI_ANOMALY_THRESHOLD}")
    print(f"  Recovery req. : {ENV_RECOVERY_STEPS} steps ({ENV_RECOVERY_STEPS * RESOLUTION} yr)")
    print(f"  Baseline win. : {BASELINE_WINDOW} yr before onset")
    print(f"  Resilience    : exponential recovery fit")
    ccsi_s = sg_smooth(ccsi_z)
    roc    = np.gradient(demo_z, t)

    episodes, in_ev, onset_i, above_count = [], False, None, 0
    for i, v in enumerate(ccsi_s):
        if not in_ev:
            if v < CCSI_ANOMALY_THRESHOLD:
                in_ev = True; onset_i = i; above_count = 0
        else:
            above_count = (above_count + 1) if v >= CCSI_ANOMALY_THRESHOLD else 0
            if above_count >= ENV_RECOVERY_STEPS:
                episodes.append((onset_i, i - ENV_RECOVERY_STEPS + 1))
                in_ev = False; above_count = 0
    if in_ev and onset_i is not None:
        episodes.append((onset_i, len(t) - 1))

    print(f"  Episodes found: {len(episodes)}")
    records = []

    for onset_i, end_i in episodes:
        onset_bp = t[onset_i]; end_bp = t[end_i]
        bl_mask  = (t >= onset_bp - BASELINE_WINDOW) & (t < onset_bp)
        ok       = bl_mask.sum() >= 5
        Gb       = float(np.mean(roc[bl_mask])) if ok else 0.0

        ev_mask    = (t >= onset_bp) & (t <= end_bp)
        cv, dv, tv = ccsi_s[ev_mask], demo_z[ev_mask], t[ev_mask]

        Gx    = float(dv[np.argmin(dv)])
        denom = abs(Gb) + abs(Gb - Gx)
        resistance = 1 - 2 * abs(Gb - Gx) / denom if denom > 0 else np.nan

        mi = np.argmin(dv)
        k100, k_se100, method, uncertain = fit_resilience(tv[mi:], dv[mi:])

        sig_demo = False
        if spd_lo is not None and years_spd is not None and spd_lo.any():
            f_lo     = interp1d(years_spd, spd_lo, bounds_error=False, fill_value=np.nan)
            lo_ev    = f_lo(tv)
            sig_demo = bool(np.any(dv < lo_ev))

        rec = {
            "onset_bp":               round(onset_bp),
            "end_bp":                 round(end_bp),
            "duration_yr":            round(end_bp - onset_bp),
            "ccsi_min":               round(float(cv[np.argmin(cv)]), 3),
            "ccsi_min_bp":            round(float(tv[np.argmin(cv)])),
            "demo_min_bp":            round(float(tv[np.argmin(dv)])),
            "resistance":             round(resistance, 4) if not np.isnan(resistance) else np.nan,
            "resilience_per100yr":    round(k100, 6)    if not np.isnan(k100)    else np.nan,
            "resilience_SE_per100yr": round(k_se100, 6) if not np.isnan(k_se100) else np.nan,
            "resilience_method":      method,
            "resilience_uncertain":   uncertain,
            "demo_sig_below_null":    sig_demo,
            "baseline_ok":            ok,
        }
        records.append(rec)

        res_s   = f"{rec['resistance']:.3f}" if not np.isnan(rec['resistance']) else "n/a"
        k_s     = f"{k100:.4f}" if not np.isnan(k100) else "n/a"
        se_s    = f"+-{k_se100:.4f}" if not np.isnan(k_se100) else ""
        unc_tag = " [UNCERTAIN]" if uncertain else ""
        print(f"  Anomaly at {rec['onset_bp']} Cal BP:")
        print(f"    Duration   : {rec['duration_yr']} yr ({rec['onset_bp']}-{rec['end_bp']})")
        print(f"    CCSI min   : Z={rec['ccsi_min']:.2f} at {rec['ccsi_min_bp']} Cal BP")
        print(f"    Resistance : {res_s}  (-1=collapse, +1=no impact)")
        print(f"    Resilience : {k_s}{se_s} per 100 yr  [{method}]{unc_tag}")
        print(f"    Demo sig.  : {'YES (below MC null envelope)' if sig_demo else 'no'}")

    df_r = pd.DataFrame(records)
    if not df_r.empty:
        df_r.to_csv(OUTPUT_RES, index=False)
        print(f"  Saved -> {OUTPUT_RES}")
    else:
        print(f"\n  No episodes found. Try lowering CCSI_ANOMALY_THRESHOLD "
              f"(currently {CCSI_ANOMALY_THRESHOLD})")
    return df_r


# ---------------------------------------------------------------------------
# Step 8 — Plotting
# ---------------------------------------------------------------------------

def _short_label(k):
    for prefix in ("GISP2_", "PANGAEA_", "Neotoma_", "Z_"):
        k = k.replace(prefix, "")
    return k[:22].strip("_")


def _res_panel_single(ax, t, demo, env, rec):
    ax2 = ax.twinx()
    
    # Environmental (Right Axis)
    ax2.fill_between(t, env, alpha=0.10, color="#D7191C", linewidth=0)
    ax2.plot(t, env, color="#D7191C", lw=1.5, ls="--", alpha=0.8, label="CCSI (smoothed)")
    ax2.axhline(CCSI_ANOMALY_THRESHOLD, color="#D7191C", lw=1, ls=":", alpha=0.6)
    ax2.set_ylabel("CCSI Z-score", color="#D7191C")
    ax2.tick_params(axis="y", labelcolor="#D7191C")
    sns.despine(ax=ax2, right=False, top=True)

    # Demographic (Left Axis)
    ax.fill_between(t, demo, alpha=0.20, color="#2C7BB6", linewidth=0)
    ax.plot(t, demo, color="#2C7BB6", lw=1.5, label="Demographic SPD")
    
    # Highlights
    ax.axvspan(rec["onset_bp"], rec["end_bp"], alpha=0.2, color="#FDAE61", lw=0, zorder=0)
    ax.axvline(rec["onset_bp"], color="#E66101", lw=1.5, label="Onset")
    
    baseline_start = rec["onset_bp"] - BASELINE_WINDOW
    ax.axvspan(baseline_start, rec["onset_bp"], alpha=0.1, color="#1A9641", lw=0, zorder=0)

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
    ax.set_title("E) Resistance & Resilience Detail", fontweight="bold", loc="center")
    ax.set_ylabel("Demographic Z-score", color="#2C7BB6")
    ax.tick_params(axis="y", labelcolor="#2C7BB6")
    sns.despine(ax=ax, right=False, top=True)


def _res_panel_multi(ax, df_res):
    dur = df_res["duration_yr"].fillna(0).values
    dur_n = (dur - dur.min()) / (dur.max() - dur.min() + 1) if (dur.max() - dur.min()) > 0 else dur * 0 + 0.5
    
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


def make_figure(t, years_spd, spd_un, spd_lo, spd_hi,
                demo_z, ccsi_z, ccsi_df, diag, df_res):
    
    sns.set_theme(style="ticks", font_scale=0.9)
    has_mc = spd_lo is not None and spd_lo.any()
    n_rows = 5 if has_mc else 4
    fig = plt.figure(figsize=(16, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.25)

    neff_str = f"N_eff={diag['n_eff']:.0f}"
    conv_str = " | EM-PCA: NOT CONVERGED" if not diag["em_pca_converged"] else ""
    
    fig.suptitle(f"Paleodemographic Climate Stress Analysis: {SITE_NAME}\n"
                 f"{TIME_MIN}-{TIME_MAX} Cal BP | {diag['n_proxies']} climate proxies | PC1={diag['explained_var_pc1']:.0%} | {neff_str} | Archo: {ARCHO_RADIUS_KM}km | Env: {ENV_RADIUS_KM}km{conv_str}",
                 fontsize=14, fontweight="bold", y=0.96)

    color_demo = "#2C7BB6"  
    color_env = "#D7191C"   
    color_ano = "#FDAE61"   
    ccsi_sm = sg_smooth(ccsi_z)

    # --- Plot 1: SPD ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(years_spd, spd_un, alpha=0.2, color=color_demo, lw=0)
    ax1.plot(years_spd, spd_un, color=color_demo, lw=1.5, label="Empirical SPD")

    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax1.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax1.set_xlim(years_spd.max(), years_spd.min())
    ax1.set_title("A) Archaeological Activity Proxy (SPD)", fontweight="bold", loc="center")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left", frameon=False)
    
    # --- Plot 2: CCSI Environmental ---
    # Decoupled from ax1 sharex, bound explicitly by t instead
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(t, ccsi_sm, alpha=0.15, color=color_env, lw=0)
    ax2.plot(t, ccsi_z, color=color_env, lw=1.2, alpha=0.6)
    ax2.plot(t, ccsi_sm, color="#9E0142", lw=1.5, label="Smoothed Trend")
    ax2.axhline(CCSI_ANOMALY_THRESHOLD, color=color_env, lw=1, ls=":", label="Anomaly Threshold")
    ax2.set_xlim(t.max(), t.min())
    
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax2.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            
    ax2.set_title("B) Environmental Proxy (CCSI Z-scored)", fontweight="bold", loc="center")
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
    ax4.plot(t, ccsi_sm, color=color_env, lw=1.5, alpha=0.8, label="Environmental (CCSI)")
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
        _res_panel_single(ax5, t, demo_z, ccsi_sm, df_res.iloc[0].to_dict())
    else:
        _res_panel_multi(ax5, df_res)

    # --- Plot 6: Best-Proxy & SPD ---
    spd_corrs = diag["spd_correlations"]
    corr_cols, corr_names, corr_r, corr_p, corr_lag = [], [], [], [], []
    for col, d in spd_corrs.items():
        corr_cols.append(col)
        corr_names.append(_short_label(col))
        corr_r.append(d["r"] if not np.isnan(d["r"]) else 0.0)
        corr_p.append(d["p_corrected"])
        corr_lag.append(d["peak_lag_yr"])

    best_idx     = int(np.argmax(np.abs(corr_r))) if corr_r else 0
    best_col_raw = corr_cols[best_idx] if corr_cols else None
    best_z_col   = f"Z_{best_col_raw}" if best_col_raw else None
    best_name    = corr_names[best_idx] if corr_names else "n/a"
    best_r_val   = corr_r[best_idx]     if corr_r     else np.nan
    best_p_val   = corr_p[best_idx]     if corr_p     else np.nan

    proxy_years = ccsi_df["CalBP"].values
    if best_z_col and best_z_col in ccsi_df.columns:
        proxy_series = sg_smooth(ccsi_df[best_z_col].values)
    else:
        proxy_series = np.full(len(proxy_years), np.nan)

    spd_z_full = np.interp(proxy_years, years_spd, spd_un, left=np.nan, right=np.nan)
    spd_z_full = ((spd_z_full - np.nanmean(spd_z_full)) / (np.nanstd(spd_z_full) or 1.0))

    ax6 = fig.add_subplot(gs[3, :], sharex=ax1)
    ax6r = ax6.twinx()
    
    ax6.fill_between(proxy_years, spd_z_full, alpha=0.2, color=color_demo, lw=0)
    ax6.plot(proxy_years, spd_z_full, color=color_demo, lw=1.5, label="Demographic Z-score")
    ax6r.plot(proxy_years, proxy_series, color=color_env, lw=2.0, ls="--", label=f"{best_name} Z-score")
    
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax6.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
            onset_bp = float(row["onset_bp"]); end_bp = float(row["end_bp"])
                    
            # Info box
            res_s = f"{row['resistance']:.2f}" if not np.isnan(row["resistance"]) else "n/a"
            rel_s = f"{row['resilience_per100yr']:.3f}/100yr" if not np.isnan(row["resilience_per100yr"]) else "n/a"
            sig_s = "Yes" if row["demo_sig_below_null"] else "No"
            box_text = (f"Event: {int(onset_bp)}\u2013{int(end_bp)} Cal BP\n"
                        f"Resistance: {res_s} | Resilience: {rel_s}\n"
                        f"Sig. Demo Drop: {sig_s}")
            ax6.text(0.02, 0.95, box_text, transform=ax6.transAxes, fontsize=9, va="top",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.9), zorder=10)
    
    ax6.set_ylabel("Demographic Z-score", color=color_demo)
    ax6r.set_ylabel(f"{best_name} Z-score", color=color_env)
    ax6.tick_params(axis="y", labelcolor=color_demo)
    ax6r.tick_params(axis="y", labelcolor=color_env)
    
    sns.despine(ax=ax6, right=False, top=True)
    sns.despine(ax=ax6r, right=False, top=True)
    ax6.set_title(f"F) Best-proxy & SPD | {best_name} r={best_r_val:+.3f} p={best_p_val:.3f}", fontweight="bold", loc="center")
    
    lines_l, labels_l = ax6.get_legend_handles_labels()
    lines_r, labels_r = ax6r.get_legend_handles_labels()
    ax6.legend(lines_l + lines_r, labels_l + labels_r, loc="lower right", frameon=False)

    # --- Plot 7: MC Null Model ---
    if has_mc:
        ax7 = fig.add_subplot(gs[4, :], sharex=ax1)
        ax7.fill_between(years_spd, spd_lo, spd_hi, alpha=0.15, color="grey", lw=0, label="95% Null Envelope")
        ax7.plot(years_spd, spd_un, color=color_demo, lw=1.5, label="Empirical SPD")
        
        sig_mask = spd_un < spd_lo
        if sig_mask.any():
            ax7.fill_between(years_spd, spd_un, spd_lo, where=sig_mask, alpha=0.4, color=color_env, lw=0, label="Sig. Trough")
            
        if not df_res.empty:
            for _, row in df_res.iterrows():
                ax7.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
                
        ax7.set_title("G) Monte Carlo Significance Test", fontweight="bold", loc="center")
        ax7.set_ylabel("Density")
        ax7.legend(loc="upper left", frameon=False)

    # Clean up axes 
    for ax in fig.axes:
        if ax != ax6r: # Skip twinx to prevent despine glitches
            sns.despine(ax=ax)
    for ax in [ax1, ax2, ax3, ax4, ax6]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Calendar Age (Cal BP)")
    if has_mc:
        ax7.tick_params(labelbottom=True)
        ax7.set_xlabel("Calendar Age (Cal BP)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"[output] Cleaned figure saved -> {OUTPUT_PLOT}")

# ---------------------------------------------------------------------------
# Site review + approval
# ---------------------------------------------------------------------------

def review_nearby_sites(df_nearby):
    """Print a summary table of all archaeological sites within ARCHO_RADIUS_KM
    and prompt the user to confirm before proceeding — mirrors 06_DataMerge.py."""
    site_col  = next((c for c in ("SiteName", "Site", "site_name") if c in df_nearby.columns), None)
    id_col    = next((c for c in ("SiteID",   "Site_ID", "site_id")  if c in df_nearby.columns), None)
    lat_col   = next((c for c in ("Lat", "lat", "latitude")          if c in df_nearby.columns), None)
    lon_col   = next((c for c in ("Long", "lon", "longitude")        if c in df_nearby.columns), None)

    # Build per-site summary
    rows = []
    if site_col:
        for name, grp in df_nearby.groupby(site_col, sort=False):
            sid     = grp[id_col].iloc[0]  if id_col  else "Unknown"
            n       = len(grp)
            dist    = grp["dist_km"].min()
            lat_v   = grp[lat_col].iloc[0] if lat_col else float("nan")
            lon_v   = grp[lon_col].iloc[0] if lon_col else float("nan")
            rows.append({"SiteName": name, "SiteID": sid,
                         "n_dates": n, "dist_km": dist,
                         "lat": lat_v, "lon": lon_v})
    else:
        # No site name column — fall back to one row per record
        for _, r in df_nearby.iterrows():
            rows.append({"SiteName": r.get("SiteName", "?"),
                         "SiteID":   r.get(id_col, "Unknown") if id_col else "Unknown",
                         "n_dates":  1,
                         "dist_km":  r["dist_km"],
                         "lat":      r.get(lat_col, float("nan")) if lat_col else float("nan"),
                         "lon":      r.get(lon_col, float("nan")) if lon_col else float("nan")})

    df_sites = pd.DataFrame(rows).sort_values("dist_km")
    n_sites  = len(df_sites)
    n_dates  = df_sites["n_dates"].sum()

    print(f"  Records within {ARCHO_RADIUS_KM} km: {n_dates} across {n_sites} site(s)")

    # Pretty-print the table (right-align numbers, left-align names)
    name_w = max(df_sites["SiteName"].astype(str).str.len().max(), 8)
    id_w   = max(df_sites["SiteID"].astype(str).str.len().max(),   6)
    header = (f"  {'SiteName':>{name_w}}  {'SiteID':>{id_w}}  "
              f"{'n_dates':>7}  {'dist_km':>9}  {'lat':>8}  {'lon':>9}")
    print(header)
    for _, row in df_sites.iterrows():
        print(f"  {str(row['SiteName']):>{name_w}}  {str(row['SiteID']):>{id_w}}  "
              f"{int(row['n_dates']):>7}  {row['dist_km']:>9.6f}  "
              f"{row['lat']:>8.4f}  {row['lon']:>9.4f}")

    # Prompt
    while True:
        ans = input("  Proceed? (yes/no) ").strip().lower()
        if ans in ("yes", "y"):
            return
        if ans in ("no", "n"):
            print("  Aborted by user.")
            sys.exit(0)
        print("  Please type 'yes' or 'no'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    divider("COMPOSITE CLIMATE STRESS INDEX — PCA PIPELINE")
    print(f"  Databases    : PANGAEA -> Neotoma -> GISP2")
    divider("SITE LOOKUP")
    print(f"  Site         : {SITE_NAME}")
    print(f"  Coordinates  : {SITE_LAT}N, {SITE_LON}E")
    print(f"  Archo radius : {ARCHO_RADIUS_KM} km")
    print(f"  Env radius   : {ENV_RADIUS_KM} km")
    print(f"  Time window  : {TIME_MIN}-{TIME_MAX} Cal BP")
    print(f"  Resolution   : {RESOLUTION} yr  |  Kernel sigma: {KERNEL_SIGMA_YR} yr")
    print(f"  SPD firewall : SPD excluded from PCA matrix")

    # 1. Load SPD
    df_input = pd.read_csv(INPUT_FILE, low_memory=False, index_col=0)
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP"):
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
    df_input = df_input.dropna(subset=["Lat", "Long"])
    df_input["dist_km"] = df_input.apply(
        lambda r: haversine_km(SITE_LAT, SITE_LON, r["Lat"], r["Long"]), axis=1)
    df_nearby = df_input[df_input["dist_km"] <= ARCHO_RADIUS_KM].copy()
    print(f"  Dates within {ARCHO_RADIUS_KM} km : {len(df_nearby):,}")
    review_nearby_sites(df_nearby)

    years_spd, spd_un, spd_lo, spd_hi = get_spd(df_nearby)

    # 2. Fetch proxies
    proxy_df = collect_proxies()

    # 3. Gaussian-kernel bin
    wide_df = bin_proxies(proxy_df)

    # 4. PCA on climate proxies ONLY (SPD excluded — no leakage)
    ccsi_df, diag = compute_ccsi(wide_df, proxy_df)

    # 5. Correlate each climate proxy against SPD independently
    #    (SPD was never in the PCA matrix so this is a clean external test)
    spd_correlations = compute_proxy_spd_correlations(wide_df, years_spd, spd_un)
    diag["spd_correlations"] = spd_correlations

    # 6. Align CCSI with SPD for episode detection
    t, demo_z, ccsi_z = align_ccsi_spd(ccsi_df, years_spd, spd_un)

    # 7. Detect climate stress episodes + resilience
    df_res = detect_episodes(t, demo_z, ccsi_z, spd_lo, spd_hi, years_spd)

    # 8. Figure
    make_figure(t, years_spd, spd_un, spd_lo, spd_hi,
                demo_z, ccsi_z, ccsi_df, diag, df_res)

    divider("COMPLETE")
    print(f"  Site         : {SITE_NAME}  [{SITE_LAT}N, {SITE_LON}E]")
    print(f"  Archo r.     : {ARCHO_RADIUS_KM} km  |  Env r.: {ENV_RADIUS_KM} km")
    print(f"  Time win.    : {TIME_MIN}-{TIME_MAX} Cal BP  |  Resolution: {RESOLUTION} yr")
    print(f"  Proxies (PCA): {diag['n_proxies']} (climate only, SPD excluded)")
    print(f"  PC1 variance : {diag['explained_var_pc1']:.1%}")
    print(f"  N_eff        : {diag['n_eff']:.1f}")
    print(f"  EM-PCA conv. : {diag['em_pca_converged']}")
    sig_corrs = {k: v for k, v in diag['spd_correlations'].items()
                 if not np.isnan(v['p_corrected']) and v['p_corrected'] < 0.05}
    print(f"  Sig. proxy-SPD correlations (p<0.05): {len(sig_corrs)}")
    for k, v in sig_corrs.items():
        print(f"    {k[:45]:<45}  r={v['r']:+.3f}  p={v['p_corrected']:.4f}  "
              f"peak_lag={v['peak_lag_yr']:+d}yr")
    print(f"  Episodes     : {len(df_res)}")
    if not df_res.empty:
        n_unc = df_res["resilience_uncertain"].sum()
        print(f"    of which {n_unc} have uncertain resilience estimates")
    print("  Outputs:")
    for f in (OUTPUT_CCSI, OUTPUT_RES, OUTPUT_PLOT):
        print(f"    {f}")


if __name__ == "__main__":
    main()