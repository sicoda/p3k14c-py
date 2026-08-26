"""
04_SPD.py : Designed for general research use with any site from the p3k14c database.

Input  : p3k14c_pristine_dates.csv   (output of 02_Calibrating.py)
Output : <Site>_population_dynamics.csv, <Site>_population_dynamics.png, <Site>_spd_for_06.csv
 
DEPENDENCIES :  pip install argparse math re sys time warnings os dataclasses pathlilb
PYTHON       :  3.12+
"""
 
import argparse
import math
import re
import sys
import time
import warnings
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from dataclasses import dataclass, field
from pathlib import Path
 
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    from scipy.optimize import differential_evolution, curve_fit
    from scipy.stats import linregress
    from tqdm import tqdm
except ImportError as e:
    sys.exit(
        f"ERROR: a required package is missing: {e}\n"
        "  Install all dependencies with:\n"
        "    pip install numpy pandas matplotlib scipy tqdm\n"
    )
 
try:
    from radiocarbon import CALIBRATION_CURVES, Date as RC14Date
except ImportError:
    sys.exit(
        "ERROR: the 'radiocarbon' package is required.\n"
        "  Install with:  pip install radiocarbon\n"
    )
 
warnings.filterwarnings("ignore")
 
 
# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
 

INPUT_FILE      = os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv")  # path to the cleaned p3k14c dataset
 
SITE_NAME       = "Catalhoyuk"
SITE_LAT        = 37.6660    # decimal degrees, positive = North
SITE_LON        = 32.8277    # decimal degrees, positive = East
ARCHO_RADIUS_KM = 5          # search radius around the site coordinates (km)
 
TIME_MIN        = 7300       # Cal BP — younger end of analysis window
TIME_MAX        = 9500       # Cal BP — older end of analysis window
 
MAX_ERROR       = 100        # maximum accepted lab 14C error (yr).
                             # modern AMS dates carry +/-15–40 yr; legacy dates
                             # with +/-150 yr add >600 cal yr of uncertainty and
                             # degrade SPD resolution. Raise only for sparse data.
 
BIN_H           = 100        # temporal binning threshold (yr). Dates from the
                             # same site within BIN_H 14C yr are pooled into one
                             # bin to correct for excavation-intensity bias.
                             # Default follows rcarbon::binPrep() convention.
 
RESOLUTION      = 10         # SPD grid step size (cal yr). Finer = slower.
N_SIMULATIONS   = 5000       # Monte Carlo iterations for NHST envelope.
                             # 5000 is the publication-standard minimum; use
                             # --fast for exploratory work (200 iterations).
MIN_HINGE_SEP   = 200        # minimum years between consecutive CPL hinges
 
# ---------------------------------------------------------------------------
# END OF USER CONFIGURATION
# ---------------------------------------------------------------------------

OLD_WOOD_TERMS = {
    "charcoal", "wood", "timber",
    "unidentified wood", "unidentified charcoal", "charred wood",
}
MARINE_TERMS = {
    "shell", "marine shell", "marine", "coral",
    "rangia", "macoma", "oyster", "mussel", "clam",
}
 
# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
 
@dataclass
class Config:
    input_file     : str   = INPUT_FILE
    outdir         : Path  = field(default_factory=Path.cwd)
    site_name      : str   = SITE_NAME
    site_lat       : float = SITE_LAT
    site_lon       : float = SITE_LON
    radius_km      : float = ARCHO_RADIUS_KM
    time_min       : int   = TIME_MIN
    time_max       : int   = TIME_MAX
    max_error      : int   = MAX_ERROR
    bin_h          : int   = BIN_H
    n_sims         : int   = N_SIMULATIONS
    resolution     : int   = RESOLUTION
    min_hinge_sep  : int   = MIN_HINGE_SEP
    fast           : bool  = False
    run_cpl        : bool  = True
    confirm        : bool  = False
    show_plot      : bool  = True
 
    # Derived output paths (set after outdir is known)
    output_csv     : Path  = field(init=False)
    output_plot    : Path  = field(init=False)
 
    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        slug = self.site_name.replace(" ", "_")
        self.output_csv  = self.outdir / f"{slug}_population_dynamics.csv"
        self.output_plot = self.outdir / f"{slug}_population_dynamics.png"
 
    @property
    def effective_sims(self) -> int:
        return 200 if self.fast else self.n_sims
 
    @property
    def max_hinges(self) -> int:
        return 2 if self.fast else 4
 
 
# -- CLI -----------------------------------------------------------------------
         
def parse_args() -> Config:
    p = argparse.ArgumentParser(
        prog="paleodem_pipeline.py",
        description="Radiocarbon paleodemographic analysis (dates-as-data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Use all CONFIG defaults:\n"
            "  python paleodem_pipeline.py\n\n"
            "  # Target a different site:\n"
            "  python paleodem_pipeline.py --lat 36.86 --lon 30.87 --name Sagalassos\n\n"
            "  # Quick exploratory run:\n"
            "  python paleodem_pipeline.py --fast\n\n"
            "  # Full run, save to specific folder:\n"
            "  python paleodem_pipeline.py --sims 5000 --outdir /results/my_site\n"
        ),
    )
 
    p.add_argument("--input",      metavar="FILE",  default=INPUT_FILE,
                   help=f"Path to p3k14c CSV  [default: {INPUT_FILE}]")
    p.add_argument("--outdir",     metavar="DIR",   default=str(Path.cwd()),
                   help="Output directory for CSV and PNG  [default: current dir]")
 
    g = p.add_argument_group("site")
    g.add_argument("--name",       metavar="TEXT",  default=SITE_NAME,
                   help=f"Site label used in plot titles  [default: {SITE_NAME}]")
    g.add_argument("--lat",        metavar="FLOAT", type=float, default=SITE_LAT,
                   help=f"Site latitude  (+ = North)  [default: {SITE_LAT}]")
    g.add_argument("--lon",        metavar="FLOAT", type=float, default=SITE_LON,
                   help=f"Site longitude (+ = East)   [default: {SITE_LON}]")
    g.add_argument("--radius",     metavar="KM",    type=float, default=ARCHO_RADIUS_KM,
                   help=f"Search radius around coordinates (km)  [default: {ARCHO_RADIUS_KM}]")
 
    g2 = p.add_argument_group("time window and data quality")
    g2.add_argument("--time-min",  metavar="CAL_BP", type=int, default=TIME_MIN,
                    help=f"Younger Cal BP boundary  [default: {TIME_MIN}]")
    g2.add_argument("--time-max",  metavar="CAL_BP", type=int, default=TIME_MAX,
                    help=f"Older Cal BP boundary    [default: {TIME_MAX}]")
    g2.add_argument("--max-error", metavar="YR",    type=int, default=MAX_ERROR,
                    help=f"Max accepted lab ¹⁴C error (yr)  [default: {MAX_ERROR}]")
    g2.add_argument("--bin-h",     metavar="YR",    type=int, default=BIN_H,
                    help=f"Temporal bin width (yr)  [default: {BIN_H}]")
 
    g3 = p.add_argument_group("analysis options")
    g3.add_argument("--sims",      metavar="N",     type=int, default=N_SIMULATIONS,
                    help=f"Monte Carlo iterations  [default: {N_SIMULATIONS}]")
    g3.add_argument("--fast",      action="store_true",
                    help="Quick mode: 200 MC iterations, max 2 CPL hinges (for exploration)")
    g3.add_argument("--no-cpl",    action="store_true",
                    help="Skip CPL modelling (saves time on large datasets)")
    g3.add_argument("--confirm",   action="store_true",
                    help="Pause for y/n confirmation before the long computation steps")
    g3.add_argument("--no-show",   action="store_true",
                    help="Save plot without opening an interactive window")
 
    args = p.parse_args()
 
    cfg = Config(
        input_file    = args.input,
        outdir        = args.outdir,
        site_name     = args.name,
        site_lat      = args.lat,
        site_lon      = args.lon,
        radius_km     = args.radius,
        time_min      = args.time_min,
        time_max      = args.time_max,
        max_error     = args.max_error,
        bin_h         = args.bin_h,
        n_sims        = args.sims,
        fast          = args.fast,
        run_cpl       = not args.no_cpl,
        confirm       = args.confirm,
        show_plot     = not args.no_show,
    )
 
    # Sanity-check time window
    if cfg.time_min >= cfg.time_max:
        p.error("--time-min must be less than --time-max")
 
    return cfg
 
 
# -- Console helpers -----------------------------------------------------------
 
def _divider(title: str = "", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("\n" + "─" * width)
 
 
def _warn(msg: str) -> None:
    print(f"\n  WARNING: {msg}")
 
 
def _abort(msg: str) -> None:
    sys.exit(f"\n  ERROR: {msg}\n")
 
 
def _confirm_or_abort(prompt: str) -> None:
    print(f"\n  {prompt} [yes/no] ", end="", flush=True)
    if input().strip().lower() not in ("yes", "y", ""):
        _abort("Aborted by user.")
 
 
# -- Geography ----------------------------------------------------------------
 
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
 
 
def _get_curve(lat: float) -> str:
    return "shcal20" if lat < 0 else "intcal20"
 
 
# -- True IntCal20/SHCal20 calibration -----------------------------------------
 
def calibrate_date(c14_age: int, c14_error: int, curve_name: str,
                   years: np.ndarray) -> np.ndarray:
    """
    Calibrate a single radiocarbon determination onto a fixed Cal BP grid
    via Bayesian grid integration against IntCal20/SHCal20 (Reimer et al. 2020).
 
    Preserves multi-modality and plateau artefacts. The resulting probability
    array is used both for SPD construction and the CPL likelihood.
    """
    d = RC14Date(c14_age, c14_error, curve=curve_name)
    d.calibrate()
    cd    = d.cal_date                     # shape (n, 3): [cal_bp, unnorm, norm]
    order = np.argsort(cd[:, 0])
    prob  = np.interp(years, cd[order, 0], cd[order, 2], left=0.0, right=0.0)
    total = prob.sum()
    return prob / total if total > 0 else prob
 
 
# ---------------------------------------------------------------------------
# Step 0 — Site lookup
# ---------------------------------------------------------------------------
 
def lookup_and_confirm(cfg: Config) -> pd.DataFrame:
    _divider("SITE LOOKUP")
    print(f"  Site          : {cfg.site_name}")
    print(f"  Coordinates   : {cfg.site_lat}°N, {cfg.site_lon}°E")
    print(f"  Search radius : {cfg.radius_km} km")
    print(f"  Time window   : {cfg.time_min}–{cfg.time_max} Cal BP")
    print(f"  Bin width     : {cfg.bin_h} yr   |   Max error: {cfg.max_error} yr")
    print(f"  MC iterations : {cfg.effective_sims}"
          + ("  (fast mode)" if cfg.fast else ""))
    print(f"  CPL modelling : {'yes (up to ' + str(cfg.max_hinges) + ' hinges)' if cfg.run_cpl else 'skipped'}")
    print(f"  Input         : {cfg.input_file}")
    print(f"  Output dir    : {cfg.outdir}")
 
    if not Path(cfg.input_file).is_file():
        _abort(
            f"{cfg.input_file!r} not found.\n"
            "  Pass the correct path with --input /path/to/p3k14c_pristine_dates.csv"
        )
 
    print(f"\n  Loading dataset ...", end=" ", flush=True)
    df = pd.read_csv(cfg.input_file, low_memory=False)
    print(f"{len(df):,} total records")
 
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP",
                "CI95_Lower", "CI95_Upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    df = df.dropna(subset=["Lat", "Long"])
    df["dist_km"] = df.apply(
        lambda r: haversine_km(cfg.site_lat, cfg.site_lon, r["Lat"], r["Long"]), axis=1
    )
    nearby = df[df["dist_km"] <= cfg.radius_km].copy()
 
    if nearby.empty:
        _abort(
            f"No records found within {cfg.radius_km} km of "
            f"({cfg.site_lat}, {cfg.site_lon}).\n"
            "  Try increasing --radius, or check --lat / --lon."
        )
 
    nearby["SiteName"] = nearby["SiteName"].fillna("Unknown")
    nearby["SiteID"]   = nearby["SiteID"].fillna("—")
 
    sites = (
        nearby
        .groupby(["SiteName", "SiteID"], dropna=False)
        .agg(
            n_total     = ("Age",         "count"),
            n_in_window = ("MedianCalBP",
                           lambda x: int(((x >= cfg.time_min)
                                          & (x <= cfg.time_max)).sum())),
            age_min     = ("MedianCalBP", "min"),
            age_max     = ("MedianCalBP", "max"),
            dist_km     = ("dist_km",     "min"),
            lat         = ("Lat",         "first"),
            lon         = ("Long",        "first"),
        )
        .sort_values("dist_km")
        .reset_index()
    )
    sites["dist_km"]   = sites["dist_km"].round(2)
    sites["age_range"] = (sites["age_min"].round(0).astype(int).astype(str)
                          + "–" + sites["age_max"].round(0).astype(int).astype(str))
 
    print(f"\n  Records within {cfg.radius_km} km : "
          f"{len(nearby):,} across {len(sites)} site(s)\n")
 
    display_df = sites[["SiteName", "SiteID", "n_total", "n_in_window",
                         "age_range", "dist_km", "lat", "lon"]].copy()
    display_df.columns = ["SiteName", "SiteID", "N total",
                          f"N {cfg.time_min}–{cfg.time_max}",
                          "Cal BP range", "dist (km)", "Lat", "Lon"]
    print(display_df.to_string(index=False))
 
    in_window = nearby[(nearby["MedianCalBP"] >= cfg.time_min)
                       & (nearby["MedianCalBP"] <= cfg.time_max)]
    if not in_window.empty and "Material" in in_window.columns:
        mat_counts = in_window["Material"].fillna("(blank)").value_counts().head(15)
        print(f"\n  Top materials in {cfg.time_min}–{cfg.time_max} Cal BP window:")
        for mat, n in mat_counts.items():
            print(f"    {n:>4}  {mat}")
 
    n_usable = int(in_window["Error"].le(cfg.max_error).sum()) if not in_window.empty else 0
    print(f"\n  Dates in window with error ≤ {cfg.max_error} yr: {n_usable}")
    if n_usable < 10:
        _warn(
            f"Only {n_usable} usable dates in the analysis window. "
            "Results may be unreliable.\n"
            "  Consider: increasing --radius, widening --time-min/--time-max, "
            "or raising --max-error."
        )
 
    if cfg.confirm:
        _confirm_or_abort("Proceed with these settings?")
 
    _divider()
    return nearby
 
 
# ---------------------------------------------------------------------------
# Step 1 — Chronometric Hygiene
# ---------------------------------------------------------------------------
 
def phase1_hygiene(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Remove radiocarbon determinations susceptible to systematic age biases.
 
    Filters
    -------
    1. Old-wood / long-lived charcoal (old-wood effect)
    2. Marine / estuarine samples    (marine reservoir effect)
    3. High laboratory error > MAX_ERROR yr
    4. Outside the Cal BP analysis window
    """
    _divider("Phase 1: Chronometric Hygiene")
    df = df.copy()
    for col in ("Age", "Error", "Lat", "MedianCalBP"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Age", "Error", "Lat", "MedianCalBP"])
 
    n_high_error = int((df["Error"] > cfg.max_error).sum())
    df = df[df["Error"] <= cfg.max_error].copy()
 
    df["Material_norm"] = df["Material"].fillna("").str.lower().str.strip()
    charcoal_re = re.compile(
        r'\bcharcoal\b|charcoal-|\btimber\b|\bwood\b', re.IGNORECASE)
    df["is_old_wood"] = df["Material_norm"].apply(
        lambda m: bool(charcoal_re.search(m)))
    df["is_marine"] = df["Material_norm"].apply(
        lambda m: any(t in m for t in MARINE_TERMS))
 
    n_old_wood = int(df["is_old_wood"].sum())
    n_marine   = int(df["is_marine"].sum())
 
    df_clean = df[~df["is_old_wood"] & ~df["is_marine"]].copy()
    df_clean["CalCurveUsed"] = df_clean["Lat"].apply(_get_curve)
    df_clean = df_clean[
        (df_clean["MedianCalBP"] >= cfg.time_min)
        & (df_clean["MedianCalBP"] <= cfg.time_max)
    ].copy()
 
    print(f"  Removed: {n_old_wood} old-wood/charcoal, "
          f"{n_marine} marine, {n_high_error} high-error (>{cfg.max_error} yr)")
    print(f"  Retained: {len(df_clean):,} dates for analysis")
 
    if len(df_clean) == 0:
        _abort(
            "No dates survived chronometric hygiene.\n"
            "  Try: raising --max-error, widening the time window, "
            "or increasing --radius."
        )
    return df_clean
 
 
# ---------------------------------------------------------------------------
# Step 2 — Spatial-temporal Binning
# ---------------------------------------------------------------------------
 
def phase2_binning(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Group dates into site-phase bins to correct for excavation-intensity bias
    (Shennan et al. 2013; Timpson et al. 2014). Dates from the same site
    within BIN_H ¹⁴C yr are pooled; their SPD contribution is averaged so
    that densely sampled phases do not dominate the aggregate signal.
    """
    _divider(f"Phase 2: Binning  (h={cfg.bin_h} yr)")
    df = df.copy()
    df["_site_key"] = (
        df["SiteName"].fillna("UNKNOWN")
        .str.encode("ascii", errors="ignore").str.decode("ascii")
        .str.lower().str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df["Bin"] = ""
    bc = 0
    for _, group in df.groupby("_site_key"):
        ages  = group["Age"].values
        idx   = group.index
        if len(ages) == 1:
            df.loc[idx[0], "Bin"] = f"b{bc}"; bc += 1; continue
        order = np.argsort(ages)
        sidx  = idx[order]; sages = ages[order]
        cb    = bc
        df.loc[sidx[0], "Bin"] = f"b{cb}"
        for k in range(1, len(sages)):
            if sages[k] - sages[k - 1] > cfg.bin_h:
                bc += 1; cb = bc
            df.loc[sidx[k], "Bin"] = f"b{cb}"
        bc += 1
    df.drop(columns=["_site_key"], inplace=True)
    n_bins = df["Bin"].nunique()
    print(f"  {len(df):,} dates → {n_bins} site-phase bins  "
          f"(effective n for BIC = {n_bins})")
    return df
 
 
# ---------------------------------------------------------------------------
# Step 3 — Taphonomic weight
# ---------------------------------------------------------------------------
 
def taphonomic_factor(t: np.ndarray) -> np.ndarray:
    """
    Bluhm & Surovell (2018) power-law taphonomic survival curve,
    normalized so the youngest time step = 1.0.
    Used as a generative likelihood weight in the CPL model.
    """
    raw = 5.726442e6 * np.power(t + 2176.4, -1.3925309)
    return raw / raw[np.argmin(t)]
 
 
# ---------------------------------------------------------------------------
# Step 4 — SPD
# ---------------------------------------------------------------------------
 
def build_spd_and_caldists(df: pd.DataFrame, cfg: Config) -> tuple:
    """
    Build the binned SPD and the (N_dates x T) calibrated probability matrix.
 
    Each date is calibrated via true grid integration (not Gaussian CI
    approximation). Within each bin, distributions are averaged before
    summing, so sampling intensity does not bias the SPD.
    """
    _divider("Phase 4: SPD Construction")
    years = np.arange(cfg.time_min, cfg.time_max + cfg.resolution,
                      cfg.resolution)
    spd       = np.zeros(len(years))
    cal_dists = []
 
    print(f"  Calibrating {len(df):,} dates against IntCal20/SHCal20 ...",
          end=" ", flush=True)
    t0 = time.time()
 
    for _bin_id, group in df.groupby("Bin"):
        bin_density = np.zeros(len(years))
        for _, row in group.iterrows():
            prob = calibrate_date(
                int(row["Age"]), int(row["Error"]),
                str(row.get("CalCurveUsed", "intcal20")), years,
            )
            cal_dists.append(prob)
            bin_density += prob
        bin_density /= len(group)
        spd += bin_density
 
    if spd.sum() > 0:
        spd /= spd.sum()
 
    elapsed = time.time() - t0
    print(f"done  ({elapsed:.1f}s)")
    print(f"  {len(years)} time steps | "
          f"{df['Bin'].nunique()} bins | "
          f"{len(cal_dists)} calibrated distributions")
    return years, spd, np.array(cal_dists)
 
 
# ---------------------------------------------------------------------------
# Step 5 — NHST
# ---------------------------------------------------------------------------
 
def fit_exponential(years: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """Exponential null via log-linear regression (Shennan et al. 2013)."""
    pos = spd > 0
    if pos.sum() < 3:
        return np.ones_like(spd) / len(spd)
    try:
        slope, intercept, *_ = linregress(years[pos], np.log(spd[pos]))
        fitted = np.exp(intercept + slope * years)
        return np.abs(fitted) / np.abs(fitted).sum()
    except Exception:
        return np.ones_like(spd) / len(spd)
 
 
def fit_logistic(years: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """Logistic null — sigmoid growth in forward time (Zahid et al. 2016)."""
    t_fwd = years.max() - years
 
    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
 
    try:
        t_span  = t_fwd.max() - t_fwd.min()
        popt, _ = curve_fit(
            logistic, t_fwd, spd,
            p0=[spd.max(), 0.001, t_fwd[np.argmax(spd)]],
            bounds=([0, -0.05, -t_span], [1, 0.05, t_fwd.max() + t_span]),
            maxfev=10000,
        )
        fitted   = logistic(t_fwd, *popt)
        t0_calbp = years.max() - popt[2]
        print(f"  Logistic fit: L={popt[0]:.5f}, k={popt[1]:.5f}, "
              f"inflection ≈ {t0_calbp:.0f} Cal BP")
        return np.abs(fitted) / np.abs(fitted).sum()
    except Exception as e:
        _warn(f"Logistic fit failed ({e}) — falling back to exponential.")
        return fit_exponential(years, spd)
 
 
def _build_uncalsample_weights(null_prob_cal, t_grid, mu_grid,
                                sg_grid, c14_grid):
    """
    Compute w(r) ∝ Σ_t Pr(t|null) x p(r|μ_t, o_t²)  over c14_grid.
    Drawing simulated ¹⁴C ages from w(r) and forward-calibrating ensures
    the MC envelope inherits IntCal artefacts, eliminating Type I errors.
    """
    diff   = (c14_grid[:, None] - mu_grid[None, :]) ** 2
    lp     = np.exp(-0.5 * diff / sg_grid[None, :] ** 2) / sg_grid[None, :]
    weight = lp @ null_prob_cal
    total  = weight.sum()
    return weight / total if total > 0 else weight
 
 
def _calibrate_for_curve(curve_name: str, t_min: int, t_max: int) -> tuple:
    curve      = CALIBRATION_CURVES[curve_name]
    cal_bp_arr = curve[:, 0][::-1]
    c14_arr    = curve[:, 1][::-1]
    sig_arr    = curve[:, 2][::-1]
    mask       = (cal_bp_arr >= t_min) & (cal_bp_arr <= t_max)
    t_grid, mu_grid, sg_grid = (
        cal_bp_arr[mask], c14_arr[mask], sig_arr[mask])
    c14_grid = np.arange(
        int(mu_grid.min() - 5 * sg_grid.max()),
        int(mu_grid.max() + 5 * sg_grid.max()) + 1, dtype=float,
    )
    return t_grid, mu_grid, sg_grid, c14_grid
 
 
def monte_carlo_envelope(
    years: np.ndarray,
    spd: np.ndarray,
    df: pd.DataFrame,
    cfg: Config,
    null_model: str = "exponential",
) -> tuple:
    _divider(f"Phase 5: NHST — {null_model} null  ({cfg.effective_sims:,} iterations)")
 
    null_fitted = (fit_exponential(years, spd) if null_model == "exponential"
                   else fit_logistic(years, spd))
 
    errors = df["Error"].values
    curves = (df["CalCurveUsed"].values if "CalCurveUsed" in df.columns
              else np.array(["intcal20"] * len(df)))
 
    unique_curves = list(set(curves))
    curve_counts  = {c: int((curves == c).sum()) for c in unique_curves}
    print(f"  Calibration curves: {curve_counts}")
 
    curve_grids = {c: _calibrate_for_curve(c, cfg.time_min, cfg.time_max)
                   for c in unique_curves}
 
    def _null_on_tgrid(cname):
        t_grid = curve_grids[cname][0]
        probs  = np.interp(t_grid, years, null_fitted)
        total  = probs.sum()
        return probs / total if total > 0 else np.ones(len(t_grid)) / len(t_grid)
 
    rng      = np.random.default_rng(42)
    sim_spds = np.zeros((cfg.effective_sims, len(years)))
 
    for i in tqdm(range(cfg.effective_sims),
                  desc=f"  Simulating ({null_model})", unit="iter"):
        sim_density = np.zeros(len(years))
        for cname, n_c in curve_counts.items():
            t_grid, mu_grid, sg_grid, c14_grid = curve_grids[cname]
            weight  = _build_uncalsample_weights(
                _null_on_tgrid(cname), t_grid, mu_grid, sg_grid, c14_grid)
            sim_c14 = rng.choice(c14_grid.astype(int), size=n_c, p=weight)
            sim_err = rng.choice(errors, size=n_c, replace=True)
            for c14age, err in zip(sim_c14, sim_err):
                sim_density += calibrate_date(
                    int(c14age), max(int(err), 15), cname, years)
        if sim_density.sum() > 0:
            sim_spds[i] = sim_density / sim_density.sum()
 
    lower_95 = np.percentile(sim_spds, 2.5,  axis=0)
    upper_95 = np.percentile(sim_spds, 97.5, axis=0)
    exceeds  = np.sum((spd > upper_95) | (spd < lower_95))
    global_p = 1.0 - (exceeds / len(years))
    print(f"  Global p-value = {global_p:.4f}")
    return null_fitted, lower_95, upper_95, global_p
 
 
# ---------------------------------------------------------------------------
# Step 6 — CPL modeling
# ---------------------------------------------------------------------------
 
def cpl_model(years, hinge_years, hinge_probs) -> np.ndarray:
    density = np.interp(years, sorted(hinge_years), hinge_probs)
    density = np.maximum(density, 0)
    total   = density.sum()
    return density / total if total > 0 else density
 
 
def fit_cpl(years, cal_dists, n_hinges, n_bins, taph_weight, cfg) -> tuple:
    """
    Maximize ADMUR-style log-likelihood:
        LL = Σ_d log( Σ_t  p_d(t) · m(t) · τ(t) )
    BIC sample size = n_bins (effective count after binning, per ADMUR).
    """
    n_params = n_hinges + (n_hinges + 2)
    t_min, t_max = cfg.time_min, cfg.time_max
 
    def neg_log_likelihood(params):
        if n_hinges == 0:
            hy = [float(t_min), float(t_max)]
            hp = np.abs(params) + 1e-10
        else:
            raw  = sorted(params[:n_hinges])
            iy   = []; prev = t_min
            for v in raw:
                v = max(v, prev + cfg.min_hinge_sep)
                iy.append(v); prev = v
            if iy and iy[-1] > t_max - cfg.min_hinge_sep:
                return 1e15
            hy = [float(t_min)] + iy + [float(t_max)]
            hp = np.abs(params[n_hinges:]) + 1e-10
 
        m      = cpl_model(years, hy, hp) * taph_weight
        total  = m.sum()
        if total <= 0:
            return 1e15
        m     /= total
        ll     = np.sum(np.log(np.maximum(cal_dists @ m, 1e-300)))
        return -ll
 
    if n_hinges == 0:
        bounds = [(0.0, 0.05)] * 2
    else:
        bounds = ([(t_min + cfg.min_hinge_sep,
                    t_max - cfg.min_hinge_sep)] * n_hinges
                  + [(0.0, 0.05)] * (n_hinges + 2))
 
    best_ll = -np.inf; best_r = None
    for seed in [42, 123, 7, 99, 314]:
        try:
            r = differential_evolution(
                neg_log_likelihood, bounds, seed=seed,
                maxiter=1500, tol=1e-9, popsize=20,
                mutation=(0.5, 1.5), recombination=0.9, workers=1,
            )
            if -r.fun > best_ll:
                best_ll = -r.fun; best_r = r
        except Exception:
            pass
 
    if best_r is None:
        return [float(t_min), float(t_max)], [1.0, 1.0], -np.inf, np.inf
 
    if n_hinges == 0:
        hy = [float(t_min), float(t_max)]
        hp = list(np.abs(best_r.x) + 1e-10)
    else:
        raw  = sorted(best_r.x[:n_hinges]); iy = []; prev = t_min
        for v in raw:
            v = max(v, prev + cfg.min_hinge_sep); iy.append(v); prev = v
        hy = [float(t_min)] + iy + [float(t_max)]
        hp = list(np.abs(best_r.x[n_hinges:]) + 1e-10)
 
    bic = n_params * np.log(n_bins) - 2 * best_ll
    return hy, hp, best_ll, bic
 
 
def phase6_cpl(years, spd, cal_dists, df_clean, df_binned,
               taph_weight, cfg) -> tuple:
    _divider("Phase 6: CPL Modelling")
    n_bins = df_binned["Bin"].nunique()
    print(f"  n_bins (BIC effective n) = {n_bins}  |  "
          f"n_dates (raw) = {len(df_clean)}  |  "
          f"max hinges = {cfg.max_hinges}")
 
    results = {}
    for n in range(0, cfg.max_hinges + 1):
        print(f"  Fitting {n}-hinge ...", end=" ", flush=True)
        hy, hp, ll, bic = fit_cpl(years, cal_dists, n, n_bins,
                                   taph_weight, cfg)
        results[n] = (hy, hp, ll, bic)
        print(f"LL={ll:.2f}  BIC={bic:.2f}  "
              f"hinges @ {[round(y) for y in hy]}")
 
    best_n      = min(results, key=lambda n: results[n][3])
    bh_years    = results[best_n][0]
    bh_probs    = results[best_n][1]
    best_fitted = cpl_model(years, bh_years, bh_probs)
 
    print(f"\n  ★ Best: {best_n} internal hinge(s)  "
          f"BIC={results[best_n][3]:.2f}  "
          f"ΔBIC vs 0-hinge={results[best_n][3]-results[0][3]:+.2f}")
 
    sorted_pairs = sorted(zip(bh_years, bh_probs))
    print("\n  Segment growth rates:")
    for i in range(len(sorted_pairs) - 1):
        t0, p0 = sorted_pairs[i]; t1, p1 = sorted_pairs[i + 1]
        n_seg = int(((df_clean["MedianCalBP"] >= t0)
                     & (df_clean["MedianCalBP"] < t1)).sum())
        rate  = (p1 - p0) / (t1 - t0) * 1000
        arrow = "▲" if rate > 0 else "▼"
        print(f"    {round(t0)}–{round(t1)} Cal BP "
              f"({round(t0-1950)}–{round(t1-1950)} BCE)  "
              f"n={n_seg}  {arrow} {rate:+.5f}/1000yr")
 
    return best_fitted, bh_years, bh_probs, best_n
 
 
# ---------------------------------------------------------------------------
# Step 7  -  Plotting
# ---------------------------------------------------------------------------
 
_C = dict(
    spd_fill  = "#4C9BE8",
    spd_line  = "#1A5FA8",
    taph_fill = "#F4A261",
    taph_line = "#C8610A",
    null      = "#333333",
    env       = "#E0E0E0", 
    sig_hi    = "#D62728",
    sig_lo    = "#2CA02C",
    cpl       = "#E05C00",
    hinge     = "#8B2FC9",
    bg        = "#FFFFFF", 
)
 
 
def _style_ax(ax: plt.Axes, title: str, y_ceil: float, cfg: Config) -> None:
    """Applies a minimalist, user-friendly style to the axes."""
    ax.set_facecolor(_C["bg"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", labelsize=10, color="#CCCCCC", length=0)
    ax.set_xlim(cfg.time_max, cfg.time_min)
    ax.set_ylim(0, y_ceil)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
 
    # Simplified X-axis using only Cal BP to reduce vertical clutter
    ax.set_xlabel("Calibrated Years BP", fontsize=10, labelpad=8, fontweight="medium")
    ax.set_ylabel("Probability Density", fontsize=10, labelpad=8, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left", color="#333333")
    
    # Minimal grid
    ax.grid(axis="y", color="#F0F0F0", lw=0.8, zorder=0)
    ax.grid(axis="x", color="#F9F9F9", lw=0.5, zorder=0)
 
 
def make_plots(
    years, spd_raw, spd_taph,
    null_exp, lower_exp, upper_exp,
    null_log, lower_log, upper_log,
    cpl_fitted, hinge_years, hinge_probs, best_n,
    df_clean, cfg: Config,
) -> None:
    y_ceil = max(spd_taph.max(), spd_raw.max(),
                 upper_exp.max(), upper_log.max()) * 1.15
 
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.35, wspace=0.20,
                            left=0.05, right=0.95,
                            top=0.90, bottom=0.08)
    ax_taph = fig.add_subplot(gs[0, 0])
    ax_exp  = fig.add_subplot(gs[0, 1])
    ax_log  = fig.add_subplot(gs[1, 0])
    ax_cpl  = fig.add_subplot(gs[1, 1])
 
    # Panel A
    ax = ax_taph
    ax.fill_between(years, spd_raw,  alpha=0.25, color=_C["spd_fill"],  zorder=2)
    ax.plot(years, spd_raw,  color=_C["spd_line"],  lw=1.5, zorder=3)
    ax.fill_between(years, spd_taph, alpha=0.25, color=_C["taph_fill"], zorder=2)
    ax.plot(years, spd_taph, color=_C["taph_line"], lw=1.8, ls="--",   zorder=3)
    _style_ax(ax, "A) Raw & Taphonomically Corrected SPD", y_ceil, cfg)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=_C["spd_fill"],  alpha=0.4, label="Raw SPD"),
            mpatches.Patch(facecolor=_C["taph_fill"], alpha=0.4, label="Taphonomically Corrected"),
        ],
        loc="upper left", fontsize=9, frameon=False
    )
 
    # Panel B
    ax = ax_exp
    ax.fill_between(years, lower_exp, upper_exp,
                    color=_C["env"], alpha=0.60, label="95% MC Envelope", zorder=2)
    ax.fill_between(years, upper_exp, spd_taph,
                    where=spd_taph > upper_exp, color=_C["sig_hi"],
                    alpha=0.60, label="Above Null", zorder=3)
    ax.fill_between(years, spd_taph, lower_exp,
                    where=spd_taph < lower_exp, color=_C["sig_lo"],
                    alpha=0.60, label="Below Null", zorder=3)
    ax.plot(years, null_exp, color=_C["null"], lw=1.5, ls="--",
            zorder=4, label="Exponential Null")
    ax.plot(years, spd_taph, color=_C["spd_line"], lw=1.8,
            zorder=5, label="Empirical SPD")
    _style_ax(ax, "B) NHST — Exponential Null", y_ceil, cfg)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
 
    # Panel C
    ax = ax_log
    ax.fill_between(years, lower_log, upper_log,
                    color=_C["env"], alpha=0.60, label="95% MC Envelope", zorder=2)
    ax.fill_between(years, upper_log, spd_taph,
                    where=spd_taph > upper_log, color=_C["sig_hi"],
                    alpha=0.60, label="Above Null", zorder=3)
    ax.fill_between(years, spd_taph, lower_log,
                    where=spd_taph < lower_log, color=_C["sig_lo"],
                    alpha=0.60, label="Below Null", zorder=3)
    ax.plot(years, null_log, color=_C["null"], lw=1.5, ls="--",
            zorder=4, label="Logistic Null")
    ax.plot(years, spd_taph, color=_C["spd_line"], lw=1.8,
            zorder=5, label="Empirical SPD")
    _style_ax(ax, "C) NHST — Logistic Null", y_ceil, cfg)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
 
    # Panel D
    ax = ax_cpl
    ax.fill_between(years, spd_taph, alpha=0.15, color=_C["spd_fill"], zorder=2)
    ax.plot(years, spd_taph,   color=_C["spd_line"], lw=1.5, zorder=3,
            label="Empirical SPD")
    ax.plot(years, cpl_fitted, color=_C["cpl"],      lw=2.5, zorder=4,
            label=f"CPL Best Fit ({best_n} hinges)")
    _style_ax(ax, f"D) CPL Model Best Fit", y_ceil, cfg)
 
    # Hinge Annotations
    sorted_pairs    = sorted(zip(hinge_years, hinge_probs))
    internal_hinges = sorted_pairs[1:-1]
    
    hinge_labels = []
    for rank, (hy, _) in enumerate(internal_hinges):
        y_dot = float(np.interp(hy, years, cpl_fitted))
        ax.axvline(hy, color=_C["hinge"], lw=1.0, ls="--", alpha=0.5, zorder=3)
        ax.scatter([hy], [y_dot], color=_C["hinge"], s=50,
                   zorder=6, edgecolors="white", lw=1.5)
        hinge_labels.append(f"{round(hy)} Cal BP")
        
    # Add a clean text box in the top right corner
    if hinge_labels:
        box_text = "CPL Hinge Dates:\n" + "\n".join(f"• {label}" for label in hinge_labels)
        ax.text(0.96, 0.96, box_text, transform=ax.transAxes,
                fontsize=9, color=_C["hinge"], ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#E0E0E0", alpha=0.95),
                zorder=10)
        
    ax.legend(loc="upper left", fontsize=9, frameon=False)
 
    # Streamlined Title
    fig.text(0.5, 0.96, f"Paleodemographic Population Dynamics: {cfg.site_name}",
             ha="center", va="bottom", fontsize=16,
             fontweight="bold", color="#222222")
    fig.text(0.5, 0.935,
             f"{cfg.time_min}-{cfg.time_max} Cal BP | Monte Carlo N={cfg.effective_sims}",
             ha="center", va="bottom", fontsize=11, color="#666666")
 
    plt.savefig(cfg.output_plot, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Plot saved → {cfg.output_plot}")
    if cfg.show_plot:
        plt.show()
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Step 8  -  Summary
# ---------------------------------------------------------------------------
 
def _print_summary(cfg: Config, df_clean: pd.DataFrame, df_binned: pd.DataFrame,
                   p_exp: float, p_log: float,
                   best_n: int, bh_years: list, elapsed: float) -> None:
    _divider("RUN SUMMARY")
    print(f"  Site             : {cfg.site_name}")
    print(f"  Time window      : {cfg.time_min}–{cfg.time_max} Cal BP")
    print(f"  Dates analysed   : {len(df_clean):,}")
    print(f"  Site-phase bins  : {df_binned['Bin'].nunique()}")
    print(f"  MC iterations    : {cfg.effective_sims:,}")
    print(f"  Global p (exp.)  : {p_exp:.4f}")
    print(f"  Global p (log.)  : {p_log:.4f}")
    if cfg.run_cpl:
        hinge_str = ", ".join(
            f"{round(h)} Cal BP (≈ {abs(round(h-1950))} BCE)"
            for h in bh_years[1:-1]
        ) or "none (0-hinge model)"
        print(f"  CPL best model   : {best_n} internal hinge(s)")
        print(f"  Hinge location(s): {hinge_str}")
    print(f"  Total runtime    : {elapsed:.0f}s")
    print(f"  Outputs")
    print(f"    CSV  → {cfg.output_csv}")
    print(f"    Plot → {cfg.output_plot}")
    _divider()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    cfg = parse_args()
    t_start = time.time()
 
    df_spatial = lookup_and_confirm(cfg)
    df_clean   = phase1_hygiene(df_spatial, cfg)
    df_binned  = phase2_binning(df_clean, cfg)
 
    years, spd_raw, cal_dists = build_spd_and_caldists(df_binned, cfg)
 
    _divider("Phase 3: Taphonomic Weight")
    taph_weight = taphonomic_factor(years)
    spd_taph    = spd_raw / taph_weight
    spd_taph   /= spd_taph.sum()
    print("  Bluhm & Surovell (2018) survival curve applied.")
    print("  Generative weight embedded in CPL likelihood.")
    print("  Divisive-corrected SPD kept for visualisation only.")
 
    if cfg.confirm:
        _confirm_or_abort(
            f"About to run {cfg.effective_sims:,} MC iterations "
            f"(~{cfg.effective_sims * len(df_clean) * 0.013 / 60:.1f} min est.). Proceed?"
        )
 
    null_exp, lower_exp, upper_exp, p_exp = monte_carlo_envelope(
        years, spd_taph, df_binned, cfg, null_model="exponential")
    null_log, lower_log, upper_log, p_log = monte_carlo_envelope(
        years, spd_taph, df_binned, cfg, null_model="logistic")
 
    if cfg.run_cpl:
        cpl_fitted, hinge_years, hinge_probs, best_n = phase6_cpl(
            years, spd_taph, cal_dists, df_clean, df_binned, taph_weight, cfg)
    else:
        print("\n  CPL modelling skipped (--no-cpl).")
        cpl_fitted  = np.zeros_like(years)
        hinge_years = [float(cfg.time_min), float(cfg.time_max)]
        hinge_probs = [0.0, 0.0]
        best_n      = 0
 
    pd.DataFrame({
        "CalBP":               years,
        "SPD_Raw":             spd_raw,
        "SPD_TaphCorrected":   spd_taph,
        "NullModel_Exp":       null_exp,
        "Envelope_Exp_Lower":  lower_exp,
        "Envelope_Exp_Upper":  upper_exp,
        "NullModel_Log":       null_log,
        "Envelope_Log_Lower":  lower_log,
        "Envelope_Log_Upper":  upper_log,
        "CPL_Fitted":          cpl_fitted,
    }).to_csv(cfg.output_csv, index=False)
    print(f"  CSV  saved → {cfg.output_csv}")

    # Export a companion file formatted for 06_DataMerge.py
    spd_slug = cfg.site_name.replace(" ", "_")
    spd_for_06_path = cfg.outdir / f"{spd_slug}_spd_for_06.csv"
    pd.DataFrame({
        "CalBP":             years,
        "SPD_TaphCorrected": spd_taph,
        "MC_lo_2.5pct":      lower_exp,
        "MC_hi_97.5pct":     upper_exp,
    }).to_csv(spd_for_06_path, index=False)
    print(f"  SPD for 06  → {spd_for_06_path}")
 
    _divider("Saving outputs")
    print(f"  CSV  saved → {cfg.output_csv}")
    make_plots(
        years, spd_raw, spd_taph,
        null_exp, lower_exp, upper_exp,
        null_log, lower_log, upper_log,
        cpl_fitted, hinge_years, hinge_probs, best_n,
        df_clean, cfg,
    )
 
    _print_summary(cfg, df_clean, df_binned,
                   p_exp, p_log, best_n, hinge_years,
                   elapsed=time.time() - t_start)
 
 
if __name__ == "__main__":
    main()