"""Demographic/environmental alignment, anomaly detection, and
resistance/resilience analysis.

Ported from Scripts/05_Human_Climate_Interaction.py. sg_smooth is folded
in here (per project decision) since it's only ever used for smoothing
series ahead of resilience/anomaly analysis, not inside SPD-building.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def sg_smooth(arr, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing with NaN handling: interpolates over NaNs
    temporarily so savgol_filter can run, then restores NaNs outside the
    original finite-data range.
    """
    arr = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < 5:
        return arr.copy()

    idx = np.arange(len(arr))
    arr_filled = arr.copy()
    arr_filled[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], arr[finite_mask])

    w = min(window, len(arr_filled))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()

    smoothed = savgol_filter(arr_filled, w, poly)
    smoothed[~finite_mask] = np.nan
    return smoothed


def align(years_spd, spd, env_ages, env_vals, time_min, time_max, resolution):
    """Interpolate the demographic (SPD) and environmental series onto a
    common grid spanning the overlap of their valid ranges, then Z-score
    both.
    """
    valid = ~np.isnan(env_vals)
    t = np.arange(
        max(time_min, float(env_ages[valid].min())),
        min(time_max, float(env_ages[valid].max())) + resolution,
        resolution,
    )
    f_s = interp1d(years_spd, spd, bounds_error=False, fill_value=0.0)
    f_e = interp1d(env_ages, env_vals, bounds_error=False, fill_value=np.nan)
    d, e = f_s(t), f_e(t)
    v = ~np.isnan(e) & ~np.isnan(d) & (d > 0)
    t, d, e = t[v], d[v], e[v]
    dz = (d - d.mean()) / (d.std() or 1.0)
    ez = (e - e.mean()) / (e.std() or 1.0)
    return t, dz, ez


def fit_resilience(rt: np.ndarray, rd: np.ndarray) -> float:
    """Fit an exponential recovery curve: rd(t) ~ A * (1 - exp(-k*t)).
    Returns the rate constant k per 100 yr (higher = faster recovery).
    Falls back to a linear slope if curve_fit fails (e.g. too few points).
    """
    if len(rt) < 4:
        return np.nan

    t = rt - rt[0]

    rd_min, rd_max = rd.min(), rd.max()
    rd_range = rd_max - rd_min
    if rd_range < 1e-10:
        return np.nan
    rd_norm = (rd - rd_min) / rd_range

    try:
        def exp_recovery(t, k):
            return 1.0 - np.exp(-k * t)

        popt, _ = curve_fit(exp_recovery, t, rd_norm, p0=[0.01], bounds=(0, np.inf), maxfev=2000)
        return float(popt[0]) * 100
    except RuntimeError:
        slope = np.polyfit(t, rd, 1)[0]
        return float(slope) * 100


def detect_anomalies(
    t, demo, env,
    anomaly_threshold: float,
    recovery_steps: int,
    baseline_window: float,
    resolution: float,
    spd_lo=None,
    years_spd=None,
):
    """Detect environmental anomaly episodes (Z < anomaly_threshold for
    at least recovery_steps consecutive above-threshold steps to close),
    then compute resistance and resilience for each. If spd_lo/years_spd
    are provided, also flags whether the demographic minimum during the
    episode fell below the Monte Carlo null-model envelope.

    Returns a list of episode-record dicts (caller decides whether/how
    to persist them, e.g. as a DataFrame).
    """
    env_s = sg_smooth(env, window=11)
    roc = np.gradient(demo, t)
    episodes = []
    in_ev = False
    onset_i = None
    above_count = 0

    for i, v in enumerate(env_s):
        if not in_ev:
            if v < anomaly_threshold:
                in_ev = True
                onset_i = i
                above_count = 0
        else:
            if v >= anomaly_threshold:
                above_count += 1
                if above_count >= recovery_steps:
                    episodes.append((onset_i, i - recovery_steps + 1))
                    in_ev = False
                    above_count = 0
            else:
                above_count = 0
    if in_ev and onset_i is not None:
        episodes.append((onset_i, len(t) - 1))

    records = []
    for onset_i, end_i in episodes:
        onset_bp = t[onset_i]
        end_bp = t[end_i]
        bl_mask = (t >= onset_bp - baseline_window) & (t < onset_bp)
        baseline_ok = bl_mask.sum() >= 5
        Gb = float(np.mean(roc[bl_mask])) if baseline_ok else 0.0

        ev_mask = (t >= onset_bp) & (t <= end_bp)
        env_vals = env_s[ev_mask]
        demo_vals = demo[ev_mask]
        t_vals = t[ev_mask]
        env_min_i = np.argmin(env_vals)
        Gx = float(demo_vals[np.argmin(demo_vals)])
        demo_min_bp = float(t_vals[np.argmin(demo_vals)])

        denom = abs(Gb) + abs(Gb - Gx)
        resistance = 1 - 2 * abs(Gb - Gx) / denom if denom > 0 else np.nan

        mi = np.argmin(demo_vals)
        rt = t_vals[mi:]
        rd = demo_vals[mi:]
        resilience = fit_resilience(rt, rd)

        sig_demo = False
        if spd_lo is not None and years_spd is not None:
            f_lo = interp1d(years_spd, spd_lo, bounds_error=False, fill_value=np.nan)
            lo_ev = f_lo(t_vals)
            sig_demo = bool(np.any(demo_vals < lo_ev))

        records.append({
            "onset_bp": round(onset_bp),
            "end_bp": round(end_bp),
            "duration_yr": round(end_bp - onset_bp),
            "env_min": round(float(env_vals[env_min_i]), 3),
            "env_min_bp": round(float(t_vals[env_min_i])),
            "demo_min_bp": round(demo_min_bp),
            "resistance": round(resistance, 4) if not np.isnan(resistance) else np.nan,
            "resilience_per100yr": round(resilience, 6) if not np.isnan(resilience) else np.nan,
            "resilience_method": "exponential",
            "demo_sig_below_null": sig_demo,
            "baseline_start_bp": round(onset_bp - baseline_window),
            "baseline_end_bp": round(onset_bp),
            "baseline_ok": baseline_ok,
        })

    return records
