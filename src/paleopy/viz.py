"""Shared plotting helpers used across the SPD/human-climate/CCSI figures.

_style_ax was duplicated (as _style_ax in Scripts/04_SPD.py) and is
reused here for the SPD dashboard; the resistance/resilience panel
helpers (_res_panel_single/_res_panel_multi) are added when scripts 05/06
are ported.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


def style_spd_ax(ax, title: str, y_ceil: float, time_min: float, time_max: float) -> None:
    """Applies the minimalist SPD-dashboard axis style used across
    Scripts/04_SPD.py's 4-panel figure.
    """
    ax.set_facecolor("#FFFFFF")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", labelsize=10, color="#CCCCCC", length=0)
    ax.set_xlim(time_max, time_min)
    ax.set_ylim(0, y_ceil)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))

    ax.set_xlabel("Calibrated Years BP", fontsize=10, labelpad=8, fontweight="medium")
    ax.set_ylabel("Probability Density", fontsize=10, labelpad=8, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left", color="#333333")

    ax.grid(axis="y", color="#F0F0F0", lw=0.8, zorder=0)
    ax.grid(axis="x", color="#F9F9F9", lw=0.5, zorder=0)


def res_panel_single(
    ax, t, demo, env, rec, anomaly_threshold: float, env_label: str = "Env",
    smooth_internally: bool = True, baseline_window: float = None,
) -> None:
    """Single-episode resistance/resilience detail panel: demographic
    series on the left axis, environmental series (smoothed) on the
    right, with the episode window/onset/baseline highlighted.

    Shared between Scripts/05's and 06's near-identical _res_panel_single.
    They differ in two real ways, not just cosmetics:
    - 05 smooths `env` internally (raw series filled, smoothed series
      plotted as the line); 06 receives an already-smoothed CCSI series
      and uses it as-is for both the fill and the line
      (smooth_internally=False reproduces that).
    - 05's episode records carry precomputed baseline_start_bp/
      baseline_end_bp keys; 06's records don't have those keys at all —
      06 computes the baseline span on the fly as
      onset_bp - baseline_window. Pass baseline_window to get 06's
      behavior; leave it None to read the precomputed keys (05's
      behavior).
    """
    from paleopy.resilience import sg_smooth

    env_s = sg_smooth(env) if smooth_internally else env
    ax2 = ax.twinx()

    ax2.fill_between(t, env, alpha=0.10, color="#D7191C", linewidth=0)
    ax2.plot(t, env_s, color="#D7191C", lw=1.5, ls="--", alpha=0.8, label=f"{env_label} (smoothed)")
    ax2.axhline(anomaly_threshold, color="#D7191C", lw=1, ls=":", alpha=0.6)
    ax2.set_ylabel(f"{env_label} Z-score", color="#D7191C")
    ax2.tick_params(axis="y", labelcolor="#D7191C")
    sns.despine(ax=ax2, right=False, top=True)

    ax.fill_between(t, demo, alpha=0.20, color="#2C7BB6", linewidth=0)
    ax.plot(t, demo, color="#2C7BB6", lw=1.5, label="Demographic SPD")

    ax.axvspan(rec["onset_bp"], rec["end_bp"], alpha=0.2, color="#FDAE61", lw=0, zorder=0)
    ax.axvline(rec["onset_bp"], color="#E66101", lw=1.5, label="Onset")
    if baseline_window is None:
        baseline_start, baseline_end = rec["baseline_start_bp"], rec["baseline_end_bp"]
    else:
        baseline_start, baseline_end = rec["onset_bp"] - baseline_window, rec["onset_bp"]
    ax.axvspan(baseline_start, baseline_end, alpha=0.1, color="#1A9641", lw=0, zorder=0)

    res_s = f"{rec['resistance']:.2f}" if not np.isnan(rec["resistance"]) else "n/a"
    resil_s = f"{rec['resilience_per100yr']:.3f}" if not np.isnan(rec["resilience_per100yr"]) else "n/a"
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


def res_panel_multi(ax, df_res, guard_zero_duration_range: bool = False) -> None:
    """Multi-episode resistance-vs-resilience scatter panel, shared
    between Scripts/05 and 06.

    06 guards against all episodes having the same duration (division
    would otherwise collapse all points to color 0 instead of blowing
    up, since +1 in the denominator avoids an actual ZeroDivisionError);
    05 does not have this guard. guard_zero_duration_range=True
    reproduces 06's behavior (falls back to a uniform 0.5 color).
    """
    dur = df_res["duration_yr"].fillna(0).values
    dur_range = dur.max() - dur.min()
    if guard_zero_duration_range and dur_range <= 0:
        dur_n = dur * 0 + 0.5
    else:
        dur_n = (dur - dur.min()) / (dur_range + 1)

    sc = ax.scatter(
        df_res["resistance"].fillna(0), df_res["resilience_per100yr"].fillna(0),
        c=dur_n, cmap="YlOrRd", s=100, zorder=5, edgecolors="#333", linewidths=0.5, alpha=0.9,
    )

    for _, row in df_res.iterrows():
        res, resil = row["resistance"], row["resilience_per100yr"]
        if not (np.isnan(res) or np.isnan(resil)):
            ax.annotate(f"{int(row['onset_bp'])}", (res, resil), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(0, color="lightgrey", lw=1, ls="--", zorder=0)
    ax.axvline(0, color="lightgrey", lw=1, ls="--", zorder=0)
    plt.colorbar(sc, ax=ax, label="Event Duration", shrink=0.8, pad=0.04)

    ax.set_xlabel("Resistance (-1=collapse, +1=no impact)")
    ax.set_ylabel("Resilience (per 100 yr)")
    ax.set_title(f"E) Resistance vs Resilience ({len(df_res)} episodes)", fontweight="bold", loc="center")
