"""Example: Composite Climate Stress Index (CCSI) via PCA using the
paleopy API directly.

Replicates Scripts/06_Composite_Human_Environment.py's Catalhoyuk case
study, using an empty PANGAEA keyword list and Neotoma type list so no
proxy data is actually retrieved from either (GISP2 is the only proxy
that ends up in the PCA). Note collect_proxies() still makes one live
Neotoma site-lookup API call regardless (unlike paleopy.proxies.
get_env_data's use_gisp2_only short-circuit used in 05_climate.py) -
this is a real network dependency of Scripts/06's own collect_proxies()
that was preserved as-is during porting. Run from the repo root:

    python examples/06_ccsi.py
"""

from pathlib import Path

import pandas as pd

from paleopy.calibration import apply_chronometric_hygiene
from paleopy.ccsi import (
    align_ccsi_spd,
    bin_proxies,
    build_spd_from_scratch,
    compute_ccsi,
    compute_proxy_spd_correlations,
    detect_episodes,
    make_figure,
)
from paleopy.config import Config
from paleopy.geo import haversine_km
from paleopy.human_climate import load_spd_from_04
from paleopy.proxies import collect_proxies

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    cfg = Config(time_min=7700, time_max=9500, bin_h=50, max_error=150, outdir=OUTDIR)
    input_path = REPO_ROOT / "Datasets" / "p3k14c_pristine_dates.csv"

    df_input = pd.read_csv(input_path, low_memory=False, index_col=0)
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP"):
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
    df_input = df_input.dropna(subset=["Lat", "Long"])
    df_input["dist_km"] = df_input.apply(lambda r: haversine_km(cfg.site_lat, cfg.site_lon, r["Lat"], r["Long"]), axis=1)
    df_nearby = df_input[df_input["dist_km"] <= cfg.archaeo_radius_km].copy()
    print(f"Dates within {cfg.archaeo_radius_km} km: {len(df_nearby):,}")

    spd_for_06 = REPO_ROOT / "Catalhoyuk Data" / "Catalhoyuk_spd_for_06.csv"
    if spd_for_06.is_file():
        print(f"Reusing precomputed SPD from {spd_for_06}")
        years_spd, spd_un, spd_lo, spd_hi = load_spd_from_04(str(spd_for_06))
    else:
        print("Building SPD from scratch (no precomputed SPD found)")
        df_clean = apply_chronometric_hygiene(df_nearby, cfg.max_error, cfg.time_min, cfg.time_max)
        years_spd, spd_un, spd_lo, spd_hi = build_spd_from_scratch(df_clean, cfg.time_min, cfg.time_max, cfg.bin_h, cfg.resolution)

    proxy_df = collect_proxies(
        cfg.site_lat, cfg.site_lon, cfg.env_radius_km, cfg.time_min, cfg.time_max,
        use_gisp2=True, pangaea_keywords=[], neotoma_types=[],
        gisp2_cache_path=str(OUTDIR / "gisp2_cache.csv"),
    )
    wide_df = bin_proxies(proxy_df, cfg.time_min, cfg.time_max, cfg.resolution, cfg.kernel_sigma_yr, cfg.max_interp_gap_bins)

    ccsi_df, diag = compute_ccsi(wide_df, proxy_df, cfg.min_proxies_for_pca, cfg.em_pca_max_iter, cfg.em_pca_tol, cfg.warn_neff_threshold)
    diag["spd_correlations"] = compute_proxy_spd_correlations(wide_df, years_spd, spd_un, cfg.resolution)

    t, demo_z, ccsi_z = align_ccsi_spd(ccsi_df, years_spd, spd_un, cfg.time_min, cfg.time_max, cfg.resolution)

    records = detect_episodes(
        t, demo_z, ccsi_z, spd_lo, years_spd,
        cfg.anomaly_threshold, cfg.recovery_steps, cfg.baseline_window, cfg.resilience_cv_threshold,
    )
    df_res = pd.DataFrame(records)
    print(f"PCA proxies: {diag['n_proxies']}  |  N_eff: {diag['n_eff']:.1f}  |  Episodes: {len(df_res)}")

    ccsi_df.to_csv(OUTDIR / f"{cfg.slug}_ccsi.csv", index=False)
    if not df_res.empty:
        df_res.to_csv(OUTDIR / f"{cfg.slug}_ccsi_resilience.csv", index=False)

    output_plot = OUTDIR / f"{cfg.slug}_ccsi.png"
    make_figure(
        t, years_spd, spd_un, spd_lo, spd_hi, demo_z, ccsi_z, ccsi_df, diag, df_res,
        cfg.site_name, cfg.time_min, cfg.time_max, cfg.archaeo_radius_km, cfg.env_radius_km,
        cfg.anomaly_threshold, cfg.baseline_window, output_plot,
    )

    print(f"Wrote CCSI CSVs and {output_plot.name} -> {OUTDIR}")


if __name__ == "__main__":
    main()
