"""Human-climate interaction analysis: 05-specific SPD construction (via
the Gaussian/IntCal20 calibration method) and its own independent Monte
Carlo null-model significance test.

Ported from Scripts/05_Human_Climate_Interaction.py. Kept separate from
paleopy.spd's monte_carlo_envelope (04's MC implementation) per project
decision — same statistical idea (Timpson et al. 2014 / Crema & Bevan
2021), genuinely different code, not a copy-paste duplicate.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from paleopy.calibration import calibrate_gaussian_intcal
from paleopy.resilience import sg_smooth
from paleopy.spd import bin_by_site_id, taphonomic_weight
from paleopy.viz import res_panel_multi, res_panel_single


def load_spd_from_04(path: str) -> tuple:
    """Load pre-computed SPD and MC envelope from paleopy-spd's output
    (<Site>_spd_for_06.csv). Returns (years, spd_un, spd_lo, spd_hi).
    """
    df = pd.read_csv(path)
    years = df["CalBP"].values
    spd_un = df["SPD_TaphCorrected"].values
    spd_lo = df["MC_lo_2.5pct"].values
    spd_hi = df["MC_hi_97.5pct"].values
    return years, spd_un, spd_lo, spd_hi


def build_spd(df: pd.DataFrame, time_min: float, time_max: float, bin_h: float, resolution: float) -> tuple:
    """Build SPD using true IntCal20 (Gaussian-quadrature) calibration.
    Falls back to a Gaussian-on-MedianCalBP approximation if no raw 14C
    age/error is available for a row. Returns (years, spd_un, spd_no).
    """
    years = np.arange(time_min, time_max + resolution, resolution)
    spd_un = np.zeros(len(years))
    spd_no = np.zeros(len(years))
    df_b = bin_by_site_id(df, bin_h)
    n_bins = df_b["Bin"].nunique()
    print(f"  Bins (h={bin_h} yr) : {n_bins:,}")
    print("  Calibration   : IntCal20 (Reimer et al. 2020)")

    n_calibrated = 0
    n_gaussian = 0

    for _, grp in df_b.groupby("Bin"):
        bu = np.zeros(len(years))
        bn = np.zeros(len(years))
        for _, row in grp.iterrows():
            c14 = row.get("Age")
            err = row.get("Error")
            if pd.notna(c14) and pd.notna(err) and float(err) > 0:
                prob = calibrate_gaussian_intcal(float(c14), float(err), years)
                n_calibrated += 1
            else:
                sigma = max(float(row.get("UncalBPError", 30)), 20)
                prob = np.exp(-0.5 * ((years - float(row["MedianCalBP"])) / sigma) ** 2)
                total = prob.sum()
                prob = prob / total if total > 0 else prob
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

    taph = taphonomic_weight(years)
    spd_un = spd_un / taph
    spd_un /= spd_un.sum()
    spd_no = spd_no / taph
    spd_no /= spd_no.sum()

    return years, spd_un, spd_no


def spd_significance_envelope(
    df: pd.DataFrame,
    years: np.ndarray,
    empirical_spd: np.ndarray,
    bin_h: float,
    n_sim: int,
    model: str,
    intcal_curve: tuple,
) -> tuple:
    """Monte Carlo significance testing following Timpson et al. (2014) /
    Crema & Bevan (2021): fit a null growth model to the empirical SPD,
    simulate n_sim SPDs by drawing dates from that null model,
    back-projecting to 14C ages via IntCal20 with added noise, then
    re-calibrating. Returns (lo_envelope, hi_envelope, p_value_array).
    """
    intcal_cal, intcal_c14, intcal_err = intcal_curve

    df_binned = bin_by_site_id(df, bin_h)
    n_dates = df_binned["Bin"].nunique()

    x = years - years.mean()
    y = empirical_spd.copy()
    y = np.where(y > 0, y, 1e-10)

    if model == "exponential":
        def null_model(x, a, b):
            return a * np.exp(b * x)
        try:
            popt, _ = curve_fit(null_model, x, y, p0=[y.mean(), 0.0], maxfev=10000)
            fitted = null_model(x, *popt)
        except RuntimeError:
            fitted = np.ones_like(y) * y.mean()
    else:
        fitted = np.ones_like(y) * y.mean()

    fitted = np.clip(fitted, 0, None)
    fitted_prob = fitted / fitted.sum()

    sim_spds = np.zeros((n_sim, len(years)))
    rng = np.random.default_rng(42)
    taph = taphonomic_weight(years)
    taph = np.where(taph > 0, taph, np.nan)

    for i in range(n_sim):
        sim_cal_ages = rng.choice(years, size=n_dates, p=fitted_prob)
        sim_spd_i = np.zeros(len(years))

        for cal_age in sim_cal_ages:
            idx = np.argmin(np.abs(intcal_cal - cal_age))
            c14_mean = float(intcal_c14[idx])
            c14_err = float(intcal_err[idx])
            c14_sim = float(rng.normal(c14_mean, max(c14_err, 30)))
            prob = calibrate_gaussian_intcal(c14_sim, max(c14_err, 30), years, curve=intcal_curve)
            sim_spd_i += prob

        sim_spd_i = sim_spd_i / taph
        sim_spd_i = np.nan_to_num(sim_spd_i, nan=0.0)

        if sim_spd_i.sum() > 0:
            sim_spd_i *= empirical_spd.sum() / sim_spd_i.sum()

        sim_spds[i] = sim_spd_i

    lo = np.percentile(sim_spds, 2.5, axis=0)
    hi = np.percentile(sim_spds, 97.5, axis=0)
    p_vals = (sim_spds >= empirical_spd[None, :]).mean(axis=0)

    return lo, hi, p_vals


def make_plots(
    t, spd_years, spd_un, spd_no, demo_z, env_z,
    df_res: pd.DataFrame, site_name: str, time_min: float, time_max: float,
    archaeo_radius_km: float, env_radius_km: float, anomaly_threshold: float,
    output_plot,
    spd_lo=None, spd_hi=None,
) -> "plt.Figure":
    """Builds the human-environment comparison dashboard (SPD, environmental
    proxy, demographic proxy, overlay, resistance/resilience, and
    optionally the MC null-model panel) and saves it to output_plot.
    """
    sns.set_theme(style="ticks", font_scale=0.9)
    n_rows = 4 if (spd_lo is not None) else 3
    fig = plt.figure(figsize=(16, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.25)

    fig.suptitle(
        f"Human-Environment Dynamics: {site_name}\n"
        f"{time_min}-{time_max} Cal BP | Archo: {archaeo_radius_km}km | Env: {env_radius_km}km",
        fontsize=14, fontweight="bold", y=0.96,
    )

    color_demo = "#2C7BB6"
    color_env = "#D7191C"
    color_ano = "#FDAE61"

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

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(t, env_z, alpha=0.15, color=color_env, lw=0)
    ax2.plot(t, env_z, color=color_env, lw=1.2, alpha=0.6)
    ax2.plot(t, sg_smooth(env_z), color="#9E0142", lw=1.5, label="Smoothed Trend")
    ax2.axhline(anomaly_threshold, color=color_env, lw=1, ls=":", label="Anomaly Threshold")
    ax2.set_xlim(t.max(), t.min())
    if not df_res.empty:
        for _, row in df_res.iterrows():
            ax2.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
    ax2.set_title("B) Environmental Proxy (Z-scored)", fontweight="bold", loc="center")
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
    ax4.plot(t, env_z, color=color_env, lw=1.5, alpha=0.8, label="Environmental")
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
        res_panel_single(ax5, t, demo_z, env_z, df_res.iloc[0].to_dict(), anomaly_threshold)
    else:
        res_panel_multi(ax5, df_res)

    if spd_lo is not None and n_rows == 4:
        ax6 = fig.add_subplot(gs[3, :], sharex=ax1)
        ax6.fill_between(spd_years, spd_lo, spd_hi, alpha=0.15, color="grey", lw=0, label="95% Null Envelope")
        ax6.plot(spd_years, spd_un, color=color_demo, lw=1.5, label="Empirical SPD")
        sig_mask = spd_un < spd_lo
        if sig_mask.any():
            ax6.fill_between(spd_years, spd_un, spd_lo, where=sig_mask, alpha=0.4, color="#D7191C", lw=0, label="Sig. Trough")
        if not df_res.empty:
            for _, row in df_res.iterrows():
                ax6.axvspan(row["onset_bp"], row["end_bp"], alpha=0.15, color=color_ano, lw=0)
        ax6.set_title("F) Monte Carlo Significance Test", fontweight="bold", loc="center")
        ax6.set_ylabel("Density")
        ax6.legend(loc="upper left", frameon=False)

    for ax in fig.axes:
        sns.despine(ax=ax)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Calendar Age (Cal BP)")

    if spd_lo is not None and n_rows == 4:
        ax6.tick_params(labelbottom=True)
        ax6.set_xlabel("Calendar Age (Cal BP)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    return fig
