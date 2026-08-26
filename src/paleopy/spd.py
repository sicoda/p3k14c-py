"""SPD (Summed Probability Distribution) paleodemography pipeline.

Ported from Scripts/04_SPD.py. Binning and taphonomic-correction are
folded in here (per project decision) since they exist only to serve SPD
construction. Chronometric hygiene here (apply_hygiene_regex) is
algorithmically DIFFERENT from paleopy.calibration.apply_chronometric_hygiene
(05/06's shared version) — this one uses a word-boundary regex for
old-wood detection and has no LocAccuracy filter, verified during
porting (Stage 6) — so it is NOT unified with that function.

Calibration is injected as a callable (calibrate_fn) rather than hardcoded,
so the SPD-building plumbing here can be reused by scripts that use a
different calibration method, without changing calibration math itself.
"""

import re

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import linregress

from paleopy.calibration import calibration_curve_grid
from paleopy.geo import haversine_km
from paleopy.utils import tqdm
from paleopy.viz import style_spd_ax

_PLOT_COLORS = dict(
    spd_fill="#4C9BE8", spd_line="#1A5FA8",
    taph_fill="#F4A261", taph_line="#C8610A",
    null="#333333", env="#E0E0E0",
    sig_hi="#D62728", sig_lo="#2CA02C",
    cpl="#E05C00", hinge="#8B2FC9", bg="#FFFFFF",
)

MARINE_TERMS = {
    "shell", "marine shell", "marine", "coral",
    "rangia", "macoma", "oyster", "mussel", "clam",
}

_OLD_WOOD_RE = re.compile(r"\bcharcoal\b|charcoal-|\btimber\b|\bwood\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Site lookup
# ---------------------------------------------------------------------------


def find_nearby_sites(
    df: pd.DataFrame, site_lat: float, site_lon: float, radius_km: float,
    site_id_fillna: str = "-",
) -> pd.DataFrame:
    """Filter records to those within radius_km of (site_lat, site_lon).

    site_id_fillna differs between callers: Scripts/04_SPD.py used "-",
    Scripts/05_Human_Climate_Interaction.py used "Unknown" — preserved as
    a parameter rather than silently picking one.
    """
    df = df.copy()
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP", "CI95_Lower", "CI95_Upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Lat", "Long"])
    df["dist_km"] = df.apply(
        lambda r: haversine_km(site_lat, site_lon, r["Lat"], r["Long"]), axis=1
    )
    nearby = df[df["dist_km"] <= radius_km].copy()
    nearby["SiteName"] = nearby["SiteName"].fillna("Unknown")
    nearby["SiteID"] = nearby["SiteID"].fillna(site_id_fillna)
    return nearby


def summarize_sites(nearby: pd.DataFrame, time_min: float, time_max: float) -> pd.DataFrame:
    """Per-site summary table (counts, date range, distance) used for the
    site-lookup confirmation step.
    """
    sites = (
        nearby
        .groupby(["SiteName", "SiteID"], dropna=False)
        .agg(
            n_total=("Age", "count"),
            n_in_window=("MedianCalBP", lambda x: int(((x >= time_min) & (x <= time_max)).sum())),
            age_min=("MedianCalBP", "min"),
            age_max=("MedianCalBP", "max"),
            dist_km=("dist_km", "min"),
            lat=("Lat", "first"),
            lon=("Long", "first"),
        )
        .sort_values("dist_km")
        .reset_index()
    )
    sites["dist_km"] = sites["dist_km"].round(2)
    sites["age_range"] = (
        sites["age_min"].round(0).astype(int).astype(str)
        + "-" + sites["age_max"].round(0).astype(int).astype(str)
    )
    return sites


# ---------------------------------------------------------------------------
# Chronometric hygiene (04-specific — see module docstring)
# ---------------------------------------------------------------------------


def apply_hygiene_regex(
    df: pd.DataFrame,
    max_error: float,
    time_min: float,
    time_max: float,
    marine_terms: set = None,
) -> pd.DataFrame:
    """04's chronometric hygiene: word-boundary regex for old-wood
    detection, substring match for marine materials, high-error and
    time-window filtering. No LocAccuracy filter (unlike 05/06).

    Raises ValueError if nothing survives (the original aborted the run).
    """
    marine_terms = MARINE_TERMS if marine_terms is None else marine_terms

    df = df.copy()
    for col in ("Age", "Error", "Lat", "MedianCalBP"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Age", "Error", "Lat", "MedianCalBP"])

    df = df[df["Error"] <= max_error].copy()

    df["Material_norm"] = df["Material"].fillna("").str.lower().str.strip()
    df["is_old_wood"] = df["Material_norm"].apply(lambda m: bool(_OLD_WOOD_RE.search(m)))
    df["is_marine"] = df["Material_norm"].apply(lambda m: any(t in m for t in marine_terms))

    df_clean = df[~df["is_old_wood"] & ~df["is_marine"]].copy()
    df_clean["CalCurveUsed"] = df_clean["Lat"].apply(
        lambda lat: "shcal20" if lat < 0 else "intcal20"
    )
    df_clean = df_clean[
        (df_clean["MedianCalBP"] >= time_min) & (df_clean["MedianCalBP"] <= time_max)
    ].copy()

    if len(df_clean) == 0:
        raise ValueError(
            "No dates survived chronometric hygiene. Try: raising max_error, "
            "widening the time window, or increasing the search radius."
        )
    return df_clean


# ---------------------------------------------------------------------------
# Spatial-temporal binning (04-specific — groups by SiteName; 05/06 group
# by SiteID instead, verified during porting to be a genuine difference)
# ---------------------------------------------------------------------------


def bin_by_site_id(df: pd.DataFrame, bin_h: float) -> pd.DataFrame:
    """Group dates into site-phase bins by SiteID (falling back to the row
    index when SiteID is missing). Shared, verified-identical binning
    algorithm from Scripts/05_Human_Climate_Interaction.py and
    06_Composite_Human_Environment.py — distinct from bin_by_site_name
    (04's SiteName-based binning) below.
    """
    df = df.copy()
    df["SiteID_filled"] = df["SiteID"].fillna(pd.Series(df.index.astype(str), index=df.index))
    df["Bin"] = ""
    counter = 0
    for _, grp in df.groupby("SiteID_filled"):
        ages = grp["Age"].values
        idx = grp.index
        order = np.argsort(ages)
        s_idx, s_age = idx[order], ages[order]
        cb = counter
        df.loc[s_idx[0], "Bin"] = f"b{cb}"
        for k in range(1, len(s_age)):
            if s_age[k] - s_age[k - 1] > bin_h:
                counter += 1
                cb = counter
            df.loc[s_idx[k], "Bin"] = f"b{cb}"
        counter += 1
    return df


def bin_by_site_name(df: pd.DataFrame, bin_h: float) -> pd.DataFrame:
    """Group dates into site-phase bins by normalized SiteName, correcting
    for excavation-intensity bias (Shennan et al. 2013; Timpson et al. 2014).
    """
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
        ages = group["Age"].values
        idx = group.index
        if len(ages) == 1:
            df.loc[idx[0], "Bin"] = f"b{bc}"
            bc += 1
            continue
        order = np.argsort(ages)
        sidx = idx[order]
        sages = ages[order]
        cb = bc
        df.loc[sidx[0], "Bin"] = f"b{cb}"
        for k in range(1, len(sages)):
            if sages[k] - sages[k - 1] > bin_h:
                bc += 1
                cb = bc
            df.loc[sidx[k], "Bin"] = f"b{cb}"
        bc += 1
    df.drop(columns=["_site_key"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Taphonomic weight (shared identical formula across 04/05/06)
# ---------------------------------------------------------------------------


def taphonomic_weight(t: np.ndarray) -> np.ndarray:
    """Bluhm & Surovell (2018) power-law taphonomic survival curve,
    normalized so the youngest time step = 1.0.
    """
    raw = 5.726442e6 * np.power(t + 2176.4, -1.3925309)
    return raw / raw[np.argmin(t)]


# ---------------------------------------------------------------------------
# SPD construction
# ---------------------------------------------------------------------------


def build_spd_and_caldists(
    df_binned: pd.DataFrame, years: np.ndarray, calibrate_fn, show_progress: bool = True
) -> tuple:
    """Build the binned SPD and the (n_dates x T) calibrated probability
    matrix. Each date is calibrated via calibrate_fn(age, error, curve,
    years). Within each bin, distributions are averaged before summing, so
    sampling intensity does not bias the SPD.
    """
    spd = np.zeros(len(years))
    cal_dists = []

    groups = df_binned.groupby("Bin")
    iterator = tqdm(groups, desc="Calibrating bins", unit="bin") if show_progress else groups

    for _bin_id, group in iterator:
        bin_density = np.zeros(len(years))
        for _, row in group.iterrows():
            prob = calibrate_fn(
                int(row["Age"]), int(row["Error"]),
                str(row.get("CalCurveUsed", "intcal20")), years,
            )
            cal_dists.append(prob)
            bin_density += prob
        bin_density /= len(group)
        spd += bin_density

    if spd.sum() > 0:
        spd /= spd.sum()

    return spd, np.array(cal_dists)


# ---------------------------------------------------------------------------
# NHST null models
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
    """Logistic null - sigmoid growth in forward time (Zahid et al. 2016)."""
    t_fwd = years.max() - years

    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    try:
        t_span = t_fwd.max() - t_fwd.min()
        popt, _ = curve_fit(
            logistic, t_fwd, spd,
            p0=[spd.max(), 0.001, t_fwd[np.argmax(spd)]],
            bounds=([0, -0.05, -t_span], [1, 0.05, t_fwd.max() + t_span]),
            maxfev=10000,
        )
        fitted = logistic(t_fwd, *popt)
        return np.abs(fitted) / np.abs(fitted).sum()
    except Exception:
        return fit_exponential(years, spd)


# ---------------------------------------------------------------------------
# Monte Carlo NHST envelope
# ---------------------------------------------------------------------------


def _build_uncalsample_weights(null_prob_cal, t_grid, mu_grid, sg_grid, c14_grid):
    """Compute w(r) ~ sum_t Pr(t|null) * p(r|mu_t, sigma_t^2) over c14_grid.

    Drawing simulated 14C ages from w(r) and forward-calibrating ensures
    the MC envelope inherits IntCal artefacts, eliminating Type I errors.
    """
    diff = (c14_grid[:, None] - mu_grid[None, :]) ** 2
    lp = np.exp(-0.5 * diff / sg_grid[None, :] ** 2) / sg_grid[None, :]
    weight = lp @ null_prob_cal
    total = weight.sum()
    return weight / total if total > 0 else weight


def monte_carlo_envelope(
    years: np.ndarray,
    spd: np.ndarray,
    errors: np.ndarray,
    curves: np.ndarray,
    n_sims: int,
    time_min: float,
    time_max: float,
    calibrate_fn,
    null_model: str = "exponential",
    rng_seed: int = 42,
    show_progress: bool = True,
) -> tuple:
    """Monte Carlo NHST envelope against an exponential or logistic null
    model. Returns (null_fitted, lower_95, upper_95, global_p).
    """
    null_fitted = fit_exponential(years, spd) if null_model == "exponential" else fit_logistic(years, spd)

    unique_curves = list(set(curves))
    curve_counts = {c: int((curves == c).sum()) for c in unique_curves}
    curve_grids = {c: calibration_curve_grid(c, time_min, time_max) for c in unique_curves}

    def _null_on_tgrid(cname):
        t_grid = curve_grids[cname][0]
        probs = np.interp(t_grid, years, null_fitted)
        total = probs.sum()
        return probs / total if total > 0 else np.ones(len(t_grid)) / len(t_grid)

    rng = np.random.default_rng(rng_seed)
    sim_spds = np.zeros((n_sims, len(years)))

    iterator = range(n_sims)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Simulating ({null_model})", unit="iter")

    for i in iterator:
        sim_density = np.zeros(len(years))
        for cname, n_c in curve_counts.items():
            t_grid, mu_grid, sg_grid, c14_grid = curve_grids[cname]
            weight = _build_uncalsample_weights(_null_on_tgrid(cname), t_grid, mu_grid, sg_grid, c14_grid)
            sim_c14 = rng.choice(c14_grid.astype(int), size=n_c, p=weight)
            sim_err = rng.choice(errors, size=n_c, replace=True)
            for c14age, err in zip(sim_c14, sim_err):
                sim_density += calibrate_fn(int(c14age), max(int(err), 15), cname, years)
        if sim_density.sum() > 0:
            sim_spds[i] = sim_density / sim_density.sum()

    lower_95 = np.percentile(sim_spds, 2.5, axis=0)
    upper_95 = np.percentile(sim_spds, 97.5, axis=0)
    exceeds = np.sum((spd > upper_95) | (spd < lower_95))
    global_p = 1.0 - (exceeds / len(years))
    return null_fitted, lower_95, upper_95, global_p


# ---------------------------------------------------------------------------
# CPL (Continuous Piecewise Linear) demographic regime-shift model
# ---------------------------------------------------------------------------


def cpl_model(years, hinge_years, hinge_probs) -> np.ndarray:
    density = np.interp(years, sorted(hinge_years), hinge_probs)
    density = np.maximum(density, 0)
    total = density.sum()
    return density / total if total > 0 else density


def fit_cpl(
    years, cal_dists, n_hinges, n_bins, taph_weight, time_min, time_max, min_hinge_sep
) -> tuple:
    """Maximize ADMUR-style log-likelihood:
        LL = sum_d log( sum_t  p_d(t) * m(t) * tau(t) )
    BIC sample size = n_bins (effective count after binning, per ADMUR).
    """
    n_params = n_hinges + (n_hinges + 2)
    t_min, t_max = time_min, time_max

    def neg_log_likelihood(params):
        if n_hinges == 0:
            hy = [float(t_min), float(t_max)]
            hp = np.abs(params) + 1e-10
        else:
            raw = sorted(params[:n_hinges])
            iy = []
            prev = t_min
            for v in raw:
                v = max(v, prev + min_hinge_sep)
                iy.append(v)
                prev = v
            if iy and iy[-1] > t_max - min_hinge_sep:
                return 1e15
            hy = [float(t_min)] + iy + [float(t_max)]
            hp = np.abs(params[n_hinges:]) + 1e-10

        m = cpl_model(years, hy, hp) * taph_weight
        total = m.sum()
        if total <= 0:
            return 1e15
        m /= total
        ll = np.sum(np.log(np.maximum(cal_dists @ m, 1e-300)))
        return -ll

    if n_hinges == 0:
        bounds = [(0.0, 0.05)] * 2
    else:
        bounds = (
            [(t_min + min_hinge_sep, t_max - min_hinge_sep)] * n_hinges
            + [(0.0, 0.05)] * (n_hinges + 2)
        )

    best_ll = -np.inf
    best_r = None
    for seed in [42, 123, 7, 99, 314]:
        try:
            r = differential_evolution(
                neg_log_likelihood, bounds, seed=seed,
                maxiter=1500, tol=1e-9, popsize=20,
                mutation=(0.5, 1.5), recombination=0.9, workers=1,
            )
            if -r.fun > best_ll:
                best_ll = -r.fun
                best_r = r
        except Exception:
            pass

    if best_r is None:
        return [float(t_min), float(t_max)], [1.0, 1.0], -np.inf, np.inf

    if n_hinges == 0:
        hy = [float(t_min), float(t_max)]
        hp = list(np.abs(best_r.x) + 1e-10)
    else:
        raw = sorted(best_r.x[:n_hinges])
        iy = []
        prev = t_min
        for v in raw:
            v = max(v, prev + min_hinge_sep)
            iy.append(v)
            prev = v
        hy = [float(t_min)] + iy + [float(t_max)]
        hp = list(np.abs(best_r.x[n_hinges:]) + 1e-10)

    bic = n_params * np.log(n_bins) - 2 * best_ll
    return hy, hp, best_ll, bic


def phase6_cpl(
    years, spd, cal_dists, df_binned, taph_weight, time_min, time_max, min_hinge_sep, max_hinges
) -> tuple:
    """Fit CPL models with 0..max_hinges internal hinges, select the best
    by BIC. Returns (best_fitted, best_hinge_years, best_hinge_probs, best_n, results).
    """
    n_bins = df_binned["Bin"].nunique()

    results = {}
    for n in range(0, max_hinges + 1):
        hy, hp, ll, bic = fit_cpl(years, cal_dists, n, n_bins, taph_weight, time_min, time_max, min_hinge_sep)
        results[n] = (hy, hp, ll, bic)

    best_n = min(results, key=lambda n: results[n][3])
    bh_years = results[best_n][0]
    bh_probs = results[best_n][1]
    best_fitted = cpl_model(years, bh_years, bh_probs)

    return best_fitted, bh_years, bh_probs, best_n, results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plots(
    years, spd_raw, spd_taph,
    null_exp, lower_exp, upper_exp,
    null_log, lower_log, upper_log,
    cpl_fitted, hinge_years, hinge_probs, best_n,
    site_name: str, time_min: float, time_max: float, n_sims: int,
    output_plot,
) -> "plt.Figure":
    """Builds the 4-panel SPD dashboard figure and saves it to output_plot.
    Returns the Figure (caller decides whether to show/close it).
    """
    c = _PLOT_COLORS
    y_ceil = max(spd_taph.max(), spd_raw.max(), upper_exp.max(), upper_log.max()) * 1.15

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.35, wspace=0.20,
        left=0.05, right=0.95, top=0.90, bottom=0.08,
    )
    ax_taph = fig.add_subplot(gs[0, 0])
    ax_exp = fig.add_subplot(gs[0, 1])
    ax_log = fig.add_subplot(gs[1, 0])
    ax_cpl = fig.add_subplot(gs[1, 1])

    # Panel A
    ax = ax_taph
    ax.fill_between(years, spd_raw, alpha=0.25, color=c["spd_fill"], zorder=2)
    ax.plot(years, spd_raw, color=c["spd_line"], lw=1.5, zorder=3)
    ax.fill_between(years, spd_taph, alpha=0.25, color=c["taph_fill"], zorder=2)
    ax.plot(years, spd_taph, color=c["taph_line"], lw=1.8, ls="--", zorder=3)
    style_spd_ax(ax, "A) Raw & Taphonomically Corrected SPD", y_ceil, time_min, time_max)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=c["spd_fill"], alpha=0.4, label="Raw SPD"),
            mpatches.Patch(facecolor=c["taph_fill"], alpha=0.4, label="Taphonomically Corrected"),
        ],
        loc="upper left", fontsize=9, frameon=False,
    )

    # Panel B
    ax = ax_exp
    ax.fill_between(years, lower_exp, upper_exp, color=c["env"], alpha=0.60, label="95% MC Envelope", zorder=2)
    ax.fill_between(years, upper_exp, spd_taph, where=spd_taph > upper_exp, color=c["sig_hi"], alpha=0.60, label="Above Null", zorder=3)
    ax.fill_between(years, spd_taph, lower_exp, where=spd_taph < lower_exp, color=c["sig_lo"], alpha=0.60, label="Below Null", zorder=3)
    ax.plot(years, null_exp, color=c["null"], lw=1.5, ls="--", zorder=4, label="Exponential Null")
    ax.plot(years, spd_taph, color=c["spd_line"], lw=1.8, zorder=5, label="Empirical SPD")
    style_spd_ax(ax, "B) NHST - Exponential Null", y_ceil, time_min, time_max)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    # Panel C
    ax = ax_log
    ax.fill_between(years, lower_log, upper_log, color=c["env"], alpha=0.60, label="95% MC Envelope", zorder=2)
    ax.fill_between(years, upper_log, spd_taph, where=spd_taph > upper_log, color=c["sig_hi"], alpha=0.60, label="Above Null", zorder=3)
    ax.fill_between(years, spd_taph, lower_log, where=spd_taph < lower_log, color=c["sig_lo"], alpha=0.60, label="Below Null", zorder=3)
    ax.plot(years, null_log, color=c["null"], lw=1.5, ls="--", zorder=4, label="Logistic Null")
    ax.plot(years, spd_taph, color=c["spd_line"], lw=1.8, zorder=5, label="Empirical SPD")
    style_spd_ax(ax, "C) NHST - Logistic Null", y_ceil, time_min, time_max)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    # Panel D
    ax = ax_cpl
    ax.fill_between(years, spd_taph, alpha=0.15, color=c["spd_fill"], zorder=2)
    ax.plot(years, spd_taph, color=c["spd_line"], lw=1.5, zorder=3, label="Empirical SPD")
    ax.plot(years, cpl_fitted, color=c["cpl"], lw=2.5, zorder=4, label=f"CPL Best Fit ({best_n} hinges)")
    style_spd_ax(ax, "D) CPL Model Best Fit", y_ceil, time_min, time_max)

    sorted_pairs = sorted(zip(hinge_years, hinge_probs))
    internal_hinges = sorted_pairs[1:-1]

    hinge_labels = []
    for _rank, (hy, _) in enumerate(internal_hinges):
        y_dot = float(np.interp(hy, years, cpl_fitted))
        ax.axvline(hy, color=c["hinge"], lw=1.0, ls="--", alpha=0.5, zorder=3)
        ax.scatter([hy], [y_dot], color=c["hinge"], s=50, zorder=6, edgecolors="white", lw=1.5)
        hinge_labels.append(f"{round(hy)} Cal BP")

    if hinge_labels:
        box_text = "CPL Hinge Dates:\n" + "\n".join(f"- {label}" for label in hinge_labels)
        ax.text(
            0.96, 0.96, box_text, transform=ax.transAxes,
            fontsize=9, color=c["hinge"], ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#E0E0E0", alpha=0.95),
            zorder=10,
        )

    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.text(0.5, 0.96, f"Paleodemographic Population Dynamics: {site_name}",
              ha="center", va="bottom", fontsize=16, fontweight="bold", color="#222222")
    fig.text(0.5, 0.935, f"{time_min}-{time_max} Cal BP | Monte Carlo N={n_sims}",
              ha="center", va="bottom", fontsize=11, color="#666666")

    fig.savefig(output_plot, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
