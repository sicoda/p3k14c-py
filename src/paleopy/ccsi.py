"""Composite Climate Stress Index (CCSI): multi-proxy PCA reduction of
paleoclimate proxies, explicitly firewalled from the SPD (demographic
data is never entered into the PCA matrix — see compute_ccsi).

Ported from Scripts/06_Composite_Human_Environment.py.
"""

import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import detrend as scipy_detrend
from scipy.stats import t as t_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paleopy.calibration import calibrate_gaussian_intcal
from paleopy.resilience import sg_smooth
from paleopy.spd import bin_by_site_id, taphonomic_weight
from paleopy.viz import res_panel_multi, res_panel_single

# ---------------------------------------------------------------------------
# SPD construction (06's simpler from-scratch builder; see
# paleopy.human_climate.build_spd for 05's richer version with dual
# un/normalized tracking and n_calibrated/n_gaussian counts)
# ---------------------------------------------------------------------------


def build_spd_from_scratch(df_clean: pd.DataFrame, time_min: float, time_max: float, bin_h: float, resolution: float) -> tuple:
    years = np.arange(time_min, time_max + resolution, resolution)
    df = bin_by_site_id(df_clean, bin_h)

    spd = np.zeros(len(years))
    for _, grp in df.groupby("Bin"):
        bu = np.zeros(len(years))
        for _, row in grp.iterrows():
            c14 = row.get("Age")
            err = row.get("Error")
            if pd.notna(c14) and pd.notna(err) and float(err) > 0:
                prob = calibrate_gaussian_intcal(float(c14), float(err), years)
            else:
                sigma = max(float(row.get("UncalBPError", 30)), 20)
                prob = np.exp(-0.5 * ((years - float(row["MedianCalBP"])) / sigma) ** 2)
            prob = prob / (prob.sum() or 1.0)
            bu += prob
        spd += bu / len(grp)

    spd /= spd.sum() or 1.0
    taph = taphonomic_weight(years)
    spd = spd / taph
    spd /= spd.sum()
    lo = np.zeros_like(spd)
    hi = np.zeros_like(spd)
    return years, spd, lo, hi


# ---------------------------------------------------------------------------
# Gaussian-kernel binning + gap-filling
# ---------------------------------------------------------------------------


def linear_fill_short_gaps(col: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap; leave longer ones."""
    out = col.copy()
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


def _gaussian_kernel_bin(ages, values, grid, sigma):
    min_weight = 1e-6
    out = np.full(len(grid), np.nan)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out
    ages_f = ages[finite]
    vals_f = values[finite]
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((ages_f - g) / sigma) ** 2)
        W = w.sum()
        if W > min_weight:
            out[i] = (w * vals_f).sum() / W
    return out


def bin_proxies(proxy_df: pd.DataFrame, time_min: float, time_max: float, resolution: float, kernel_sigma_yr: float, max_interp_gap_bins: int) -> pd.DataFrame:
    """Gaussian-kernel-bin every proxy variable onto a common grid,
    dropping variables with <10% coverage.
    """
    years = np.arange(time_min, time_max + resolution, resolution)
    result = pd.DataFrame({"CalBP": years})

    for var_name, grp in proxy_df.groupby("variable"):
        grp = grp.sort_values("age_bp")
        raw_ages = grp["age_bp"].values
        raw_values = grp["value"].values

        col = _gaussian_kernel_bin(raw_ages, raw_values, years, kernel_sigma_yr)
        col = linear_fill_short_gaps(col, max_interp_gap_bins)

        coverage = np.isfinite(col).sum() / len(years)
        if coverage < 0.10:
            print(f" SKIP {var_name:<55} (coverage={coverage:.0%} < 10%)")
            continue

        raw_col = _gaussian_kernel_bin(raw_ages, raw_values, years, kernel_sigma_yr)
        valid_idx = np.where(np.isfinite(raw_col))[0]
        if len(valid_idx) == 0:
            continue
        col[years < years[valid_idx[0]]] = np.nan
        col[years > years[valid_idx[-1]]] = np.nan

        print(f" KEEP {var_name:<55} coverage={coverage:.0%}  n_raw={len(raw_ages)}")
        result[var_name] = col

    proxy_cols = [c for c in result.columns if c != "CalBP"]
    print(f"\n Proxies retained: {len(proxy_cols)}")
    if not proxy_cols:
        raise ValueError("No proxy variables survived binning.")
    return result


# ---------------------------------------------------------------------------
# EM-PCA gap-fill
# ---------------------------------------------------------------------------


def em_pca_impute(X: np.ndarray, n_components: int, max_iter: int = 50, tol: float = 1e-4) -> tuple:
    """EM-PCA gap-fill (Schneider 2001 / DINEOF-style).

    Returns (X_filled, converged). converged=False means max_iter was
    reached without satisfying tol; results are best-available but not
    guaranteed stable.
    """
    X_filled = X.copy()
    miss_mask = np.isnan(X)

    if not miss_mask.any():
        return X_filled, True

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    for j in range(X.shape[1]):
        X_filled[miss_mask[:, j], j] = col_means[j]

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    prev_fill = X_filled[miss_mask].copy()
    delta = np.inf
    converged = False

    for iteration in range(max_iter):
        mu = X_filled.mean(axis=0)
        X_c = X_filled - mu
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(X_c)
        X_rec = pca.inverse_transform(scores) + mu
        X_filled[miss_mask] = X_rec[miss_mask]
        delta = np.linalg.norm(X_filled[miss_mask] - prev_fill)
        if delta < tol:
            print(f"     EM-PCA converged after {iteration + 1} iterations (delta={delta:.2e})")
            converged = True
            break
        prev_fill = X_filled[miss_mask].copy()

    if not converged:
        print(f"     WARNING: EM-PCA did NOT converge after {max_iter} iterations (final delta={delta:.2e}, tol={tol:.2e}).")

    return X_filled, converged


# ---------------------------------------------------------------------------
# PCA -> Composite Climate Stress Index
# ---------------------------------------------------------------------------


def _effective_n(x: np.ndarray) -> float:
    """N_eff = N*(1-r1)/(1+r1) (Bretherton et al. 1999)."""
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 4:
        return float(N)
    x_c = x - x.mean()
    r1 = float(np.corrcoef(x_c[:-1], x_c[1:])[0, 1])
    r1 = max(-0.999, min(0.999, r1))
    return N * (1 - r1) / (1 + r1)


def compute_ccsi(
    wide_df: pd.DataFrame, proxy_df: pd.DataFrame,
    min_proxies_for_pca: int, em_pca_max_iter: int, em_pca_tol: float,
    warn_neff_threshold: float,
) -> tuple:
    """Reduce climate proxies to a single Composite Climate Stress Index
    via PCA. The SPD is deliberately excluded here — it's correlated
    against the resulting CCSI in a separate step
    (compute_proxy_spd_correlations) so the two datasets stay
    statistically independent (no data leakage).

    Returns (out_df, diag) where out_df has CalBP, Z_<proxy> columns, and
    CCSI; diag is a dict of PCA diagnostics.
    """
    proxy_cols = [c for c in wide_df.columns if c != "CalBP"]
    n_proxies = len(proxy_cols)
    print(f" Proxy columns entering PCA : {n_proxies}")

    X_raw = wide_df[proxy_cols].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(np.where(np.isfinite(X_raw), X_raw, np.nan))
    X_z[np.isnan(X_raw)] = np.nan

    if n_proxies < min_proxies_for_pca:
        print(f" Only {n_proxies} proxy - skipping PCA, using mean Z-score.")
        ccsi_raw = np.nanmean(X_z, axis=1)
        explained = [1.0]
        loadings = {proxy_cols[0]: 1.0}
        em_converged = True
    else:
        n_miss_before = np.isnan(X_z).sum()
        print(f"\n Missing cells before EM-PCA : {n_miss_before:,} ({n_miss_before / X_z.size:.1%})")
        n_comp_em = min(n_proxies, 3)
        print(f" Running EM-PCA with {n_comp_em} components ...")
        X_imp, em_converged = em_pca_impute(X_z, n_components=n_comp_em, max_iter=em_pca_max_iter, tol=em_pca_tol)
        print(f" Missing cells after EM-PCA  : {np.isnan(X_imp).sum():,}")

        pca = PCA(n_components=min(n_proxies, 5), random_state=42)
        scores = pca.fit_transform(X_imp)

        pc1 = scores[:, 0]
        explained = pca.explained_variance_ratio_.tolist()
        loadings = dict(zip(proxy_cols, pca.components_[0]))

        print(f"\n Explained variance by PC1 : {explained[0]:.1%}")

        gisp2_candidates = [p for p in proxy_cols if "GISP2" in p or "Temp" in p]
        if gisp2_candidates:
            ref_proxy = gisp2_candidates[0]
            ref_loading = loadings[ref_proxy]
            if ref_loading < 0:
                pc1 = -pc1
                loadings = {k: -v for k, v in loadings.items()}
        else:
            gisp2_raw = proxy_df[proxy_df["variable"] == "GISP2_Temp_C"].copy()
            if not gisp2_raw.empty:
                gi = gisp2_raw.sort_values("age_bp")
                gisp2_interp = np.interp(wide_df["CalBP"].values, gi["age_bp"].values, gi["value"].values)
                valid = np.isfinite(pc1) & np.isfinite(gisp2_interp)
                if valid.sum() > 5:
                    r = np.corrcoef(pc1[valid], gisp2_interp[valid])[0, 1]
                    if r < 0:
                        pc1 = -pc1
                        loadings = {k: -v for k, v in loadings.items()}

        ccsi_raw = pc1

    finite_mask = np.isfinite(ccsi_raw)
    if finite_mask.sum() > 10:
        ccsi_raw[finite_mask] = scipy_detrend(ccsi_raw[finite_mask])

    mu, sd = np.nanmean(ccsi_raw), np.nanstd(ccsi_raw)
    ccsi_z = (ccsi_raw - mu) / (sd or 1.0)
    ccsi_z = sg_smooth(ccsi_z)

    n_eff = _effective_n(ccsi_z)
    print(f"\n Autocorrelation diagnostic: N={len(ccsi_z)}  N_eff={n_eff:.1f}")
    if n_eff < warn_neff_threshold:
        print(f"   WARNING: N_eff < {warn_neff_threshold}. Series is heavily autocorrelated.")

    diag = {
        "n_proxies": n_proxies,
        "proxy_names": proxy_cols,
        "explained_var_pc1": explained[0],
        "loadings": loadings,
        "n_eff": n_eff,
        "em_pca_converged": em_converged,
        "spd_correlations": {},
    }

    out = wide_df[["CalBP"]].copy()
    for i, col in enumerate(proxy_cols):
        out[f"Z_{col}"] = X_z[:, i]
    out["CCSI"] = ccsi_z

    return out, diag


# ---------------------------------------------------------------------------
# Proxy-SPD correlations (SPD firewall: correlated externally, never
# entered into the PCA matrix above)
# ---------------------------------------------------------------------------


def compute_proxy_spd_correlations(wide_df: pd.DataFrame, years_spd: np.ndarray, spd_un: np.ndarray, resolution: float) -> dict:
    """For each climate proxy in wide_df, compute N_eff-corrected Pearson
    r against the SPD, plus a peak cross-correlation search over +/-500 yr.
    """
    max_lag_yr = 500
    max_lag_bins = int(max_lag_yr / resolution)

    proxy_cols = [c for c in wide_df.columns if c != "CalBP"]
    grid = wide_df["CalBP"].values
    spd_on_grid = np.interp(grid, years_spd, spd_un, left=np.nan, right=np.nan)

    results = {}
    for col in proxy_cols:
        px = wide_df[col].values.astype(float)
        valid = np.isfinite(px) & np.isfinite(spd_on_grid)

        if valid.sum() < 10:
            results[col] = dict(r=np.nan, p_corrected=np.nan, peak_r=np.nan, peak_lag_yr=np.nan, n_eff=np.nan)
            continue

        px_v = px[valid]
        spd_v = spd_on_grid[valid]

        r = float(np.corrcoef(px_v, spd_v)[0, 1])
        n_eff = max(4.0, _effective_n(px_v))

        t_stat = r * math.sqrt(n_eff - 2) / math.sqrt(max(1 - r ** 2, 1e-12))
        p_val = float(2 * t_dist.sf(abs(t_stat), df=n_eff - 2))

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
            r=round(r, 4), p_corrected=round(p_val, 6),
            peak_r=round(best_r, 4), peak_lag_yr=int(best_lag * resolution),
            n_eff=round(n_eff, 1),
        )

    return results


def align_ccsi_spd(ccsi_df: pd.DataFrame, years_spd: np.ndarray, spd_un: np.ndarray, time_min: float, time_max: float, resolution: float) -> tuple:
    from scipy.interpolate import interp1d

    ccsi_years = ccsi_df["CalBP"].values
    ccsi_vals = ccsi_df["CCSI"].values
    valid = np.isfinite(ccsi_vals)
    t = np.arange(
        max(time_min, float(ccsi_years[valid].min())),
        min(time_max, float(ccsi_years[valid].max())) + resolution,
        resolution,
    )
    f_spd = interp1d(years_spd, spd_un, bounds_error=False, fill_value=0.0)
    f_ccsi = interp1d(ccsi_years, ccsi_vals, bounds_error=False, fill_value=np.nan)
    d, e = f_spd(t), f_ccsi(t)
    v = np.isfinite(e) & np.isfinite(d) & (d > 0)
    t, d, e = t[v], d[v], e[v]
    dz = (d - d.mean()) / (d.std() or 1.0)
    ez = (e - e.mean()) / (e.std() or 1.0)
    return t, dz, ez


# ---------------------------------------------------------------------------
# Resilience with uncertainty (extends paleopy.resilience.fit_resilience
# with SE(k)/CV-based "uncertain" flagging)
# ---------------------------------------------------------------------------


def fit_resilience_with_uncertainty(rt: np.ndarray, rd: np.ndarray, cv_threshold: float) -> tuple:
    """Returns (k_per100yr, k_se_per100yr, method, uncertain)."""
    if len(rt) < 4:
        return np.nan, np.nan, "undefined", True
    t = rt - rt[0]
    rng = rd.max() - rd.min()
    if rng < 1e-10:
        return np.nan, np.nan, "undefined", True
    rd_n = (rd - rd.min()) / rng
    try:
        popt, pcov = curve_fit(lambda t, k: 1 - np.exp(-k * t), t, rd_n, p0=[0.01], bounds=(0, np.inf), maxfev=5000)
        k = float(popt[0])
        k_var = float(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.inf
        k_se = math.sqrt(k_var) if k_var < 1e12 else np.inf
        cv = k_se / k if k > 0 else np.inf
        return k * 100, k_se * 100, "exponential", bool(cv > cv_threshold)
    except RuntimeError:
        slope = float(np.polyfit(t, rd_n, 1)[0])
        return slope * 100, np.nan, "linear", True


def detect_episodes(
    t, demo_z, ccsi_z, spd_lo, years_spd,
    anomaly_threshold: float, recovery_steps: int, baseline_window: float, resilience_cv_threshold: float,
) -> list:
    """CCSI-specific anomaly/episode detection, extending
    paleopy.resilience.detect_anomalies with SE(k)/CV uncertainty
    flagging on the resilience fit.
    """
    ccsi_s = sg_smooth(ccsi_z)
    roc = np.gradient(demo_z, t)

    episodes, in_ev, onset_i, above_count = [], False, None, 0
    for i, v in enumerate(ccsi_s):
        if not in_ev:
            if v < anomaly_threshold:
                in_ev = True
                onset_i = i
                above_count = 0
        else:
            above_count = (above_count + 1) if v >= anomaly_threshold else 0
            if above_count >= recovery_steps:
                episodes.append((onset_i, i - recovery_steps + 1))
                in_ev = False
                above_count = 0
    if in_ev and onset_i is not None:
        episodes.append((onset_i, len(t) - 1))

    records = []
    for onset_i, end_i in episodes:
        onset_bp = t[onset_i]
        end_bp = t[end_i]
        bl_mask = (t >= onset_bp - baseline_window) & (t < onset_bp)
        ok = bl_mask.sum() >= 5
        Gb = float(np.mean(roc[bl_mask])) if ok else 0.0

        ev_mask = (t >= onset_bp) & (t <= end_bp)
        cv, dv, tv = ccsi_s[ev_mask], demo_z[ev_mask], t[ev_mask]

        Gx = float(dv[np.argmin(dv)])
        denom = abs(Gb) + abs(Gb - Gx)
        resistance = 1 - 2 * abs(Gb - Gx) / denom if denom > 0 else np.nan

        mi = np.argmin(dv)
        k100, k_se100, method, uncertain = fit_resilience_with_uncertainty(tv[mi:], dv[mi:], resilience_cv_threshold)

        sig_demo = False
        if spd_lo is not None and years_spd is not None and spd_lo.any():
            from scipy.interpolate import interp1d
            f_lo = interp1d(years_spd, spd_lo, bounds_error=False, fill_value=np.nan)
            lo_ev = f_lo(tv)
            sig_demo = bool(np.any(dv < lo_ev))

        records.append({
            "onset_bp": round(onset_bp),
            "end_bp": round(end_bp),
            "duration_yr": round(end_bp - onset_bp),
            "ccsi_min": round(float(cv[np.argmin(cv)]), 3),
            "ccsi_min_bp": round(float(tv[np.argmin(cv)])),
            "demo_min_bp": round(float(tv[np.argmin(dv)])),
            "resistance": round(resistance, 4) if not np.isnan(resistance) else np.nan,
            "resilience_per100yr": round(k100, 6) if not np.isnan(k100) else np.nan,
            "resilience_SE_per100yr": round(k_se100, 6) if not np.isnan(k_se100) else np.nan,
            "resilience_method": method,
            "resilience_uncertain": uncertain,
            "demo_sig_below_null": sig_demo,
            "baseline_ok": ok,
        })

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _short_label(k: str) -> str:
    for prefix in ("GISP2_", "PANGAEA_", "Neotoma_", "Z_"):
        k = k.replace(prefix, "")
    return k[:22].strip("_")


def make_figure(
    t, years_spd, spd_un, spd_lo, spd_hi, demo_z, ccsi_z, ccsi_df, diag, df_res,
    site_name: str, time_min: float, time_max: float, archaeo_radius_km: float, env_radius_km: float,
    anomaly_threshold: float, baseline_window: float, output_plot,
) -> "plt.Figure":
    sns.set_theme(style="ticks", font_scale=0.9)
    has_mc = spd_lo is not None and spd_lo.any()
    n_rows = 5 if has_mc else 4
    fig = plt.figure(figsize=(16, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.25)

    neff_str = f"N_eff={diag['n_eff']:.0f}"
    conv_str = " | EM-PCA: NOT CONVERGED" if not diag["em_pca_converged"] else ""

    fig.suptitle(
        f"Paleodemographic Climate Stress Analysis: {site_name}\n"
        f"{time_min}-{time_max} Cal BP | {diag['n_proxies']} climate proxies | "
        f"PC1={diag['explained_var_pc1']:.0%} | {neff_str} | "
        f"Archo: {archaeo_radius_km}km | Env: {env_radius_km}km{conv_str}",
        fontsize=14, fontweight="bold", y=0.96,
    )

    color_demo = "#2C7BB6"
    color_env = "#D7191C"
    color_ano = "#FDAE61"
    ccsi_sm = sg_smooth(ccsi_z)

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

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(t, ccsi_sm, alpha=0.15, color=color_env, lw=0)
    ax2.plot(t, ccsi_z, color=color_env, lw=1.2, alpha=0.6)
    ax2.plot(t, ccsi_sm, color="#9E0142", lw=1.5, label="Smoothed Trend")
    ax2.axhline(anomaly_threshold, color=color_env, lw=1, ls=":", label="Anomaly Threshold")
    ax2.set_xlim(t.max(), t.min())
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax2.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
    ax2.set_title("B) Environmental Proxy (CCSI Z-scored)", fontweight="bold", loc="center")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper left", frameon=False)

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.fill_between(t, demo_z, alpha=0.2, color=color_demo, lw=0)
    ax3.plot(t, demo_z, color=color_demo, lw=1.5)
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax3.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
    ax3.set_title("C) Demographic Proxy (Z-scored)", fontweight="bold", loc="center")
    ax3.set_ylabel("Z-Score")

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

    ax5 = fig.add_subplot(gs[2, 1])
    n = len(df_res)
    if n == 0:
        ax5.set_axis_off()
        ax5.text(0.5, 0.5, "No anomalies detected.", ha="center", va="center", fontsize=12)
        ax5.set_title("E) Resistance & Resilience", fontweight="bold", loc="center")
    elif n == 1:
        res_panel_single(
            ax5, t, demo_z, ccsi_sm, df_res.iloc[0].to_dict(), anomaly_threshold,
            env_label="CCSI", smooth_internally=False, baseline_window=baseline_window,
        )
    else:
        res_panel_multi(ax5, df_res, guard_zero_duration_range=True)

    spd_corrs = diag["spd_correlations"]
    corr_cols, corr_names, corr_r, corr_p = [], [], [], []
    for col, d in spd_corrs.items():
        corr_cols.append(col)
        corr_names.append(_short_label(col))
        corr_r.append(d["r"] if not np.isnan(d["r"]) else 0.0)
        corr_p.append(d["p_corrected"])

    best_idx = int(np.argmax(np.abs(corr_r))) if corr_r else 0
    best_col_raw = corr_cols[best_idx] if corr_cols else None
    best_z_col = f"Z_{best_col_raw}" if best_col_raw else None
    best_name = corr_names[best_idx] if corr_names else "n/a"
    best_r_val = corr_r[best_idx] if corr_r else np.nan
    best_p_val = corr_p[best_idx] if corr_p else np.nan

    proxy_years = ccsi_df["CalBP"].values
    if best_z_col and best_z_col in ccsi_df.columns:
        proxy_series = sg_smooth(ccsi_df[best_z_col].values)
    else:
        proxy_series = np.full(len(proxy_years), np.nan)

    spd_z_full = np.interp(proxy_years, years_spd, spd_un, left=np.nan, right=np.nan)
    spd_z_full = (spd_z_full - np.nanmean(spd_z_full)) / (np.nanstd(spd_z_full) or 1.0)

    ax6 = fig.add_subplot(gs[3, :], sharex=ax1)
    ax6r = ax6.twinx()
    ax6.fill_between(proxy_years, spd_z_full, alpha=0.2, color=color_demo, lw=0)
    ax6.plot(proxy_years, spd_z_full, color=color_demo, lw=1.5, label="Demographic Z-score")
    ax6r.plot(proxy_years, proxy_series, color=color_env, lw=2.0, ls="--", label=f"{best_name} Z-score")
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax6.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
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

    for ax in fig.axes:
        if ax != ax6r:
            sns.despine(ax=ax)
    for ax in [ax1, ax2, ax3, ax4, ax6]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Calendar Age (Cal BP)")
    if has_mc:
        ax7.tick_params(labelbottom=True)
        ax7.set_xlabel("Calendar Age (Cal BP)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    return fig
