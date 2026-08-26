"""paleopy-ccsi : Composite Climate Stress Index (CCSI) via PCA.

CLI wrapper around paleopy.ccsi/proxies; mirrors
Scripts/06_Composite_Human_Environment.py's main() orchestration.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
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
from paleopy.cli._console import add_site_arguments, divider
from paleopy.config import Config
from paleopy.geo import haversine_km
from paleopy.human_climate import load_spd_from_04
from paleopy.proxies import collect_proxies


def build_parser() -> argparse.ArgumentParser:
    d = Config(time_min=7700, time_max=9500, bin_h=50, max_error=150)
    parser = argparse.ArgumentParser(
        prog="paleopy-ccsi",
        description="Composite Climate Stress Index (CCSI) via PCA over multiple paleoclimate proxies.",
    )
    parser.add_argument("--input", required=True, help="Path to the calibrated ('pristine') p3k14c CSV")
    parser.add_argument("--outdir", default=".", help="Output directory (default: cwd)")
    parser.add_argument("--spd-from-04", default=None, help="Path to a <Site>_spd_for_06.csv from paleopy-spd to reuse, instead of rebuilding from scratch")
    add_site_arguments(parser, defaults=d)

    g = parser.add_argument_group("environmental proxies")
    g.add_argument("--env-radius-km", type=float, default=d.env_radius_km, help=f"Search radius for paleoenvironmental proxies (km) [default: {d.env_radius_km}]")
    g.add_argument("--pangaea-keyword", action="append", default=None, help="PANGAEA search keyword (repeatable) [default: stable isotope, speleothem, pollen]")
    g.add_argument("--neotoma-type", action="append", default=None, help="Neotoma dataset type (repeatable) [default: pollen, stable isotopes]")
    g.add_argument("--no-gisp2", dest="use_gisp2", action="store_false", default=d.use_gisp2, help="Exclude GISP2 from the proxy set")
    g.add_argument("--gisp2-cache", default="gisp2_cache.csv", help="Path to GISP2 cache CSV [default: gisp2_cache.csv]")
    g.add_argument("--kernel-sigma-yr", type=float, default=d.kernel_sigma_yr, help=f"Gaussian-kernel binning sigma (yr) [default: {d.kernel_sigma_yr}]")
    g.add_argument("--max-interp-gap-bins", type=int, default=d.max_interp_gap_bins, help=f"Max gap (bins) to linearly interpolate [default: {d.max_interp_gap_bins}]")

    g2 = parser.add_argument_group("PCA")
    g2.add_argument("--min-proxies-for-pca", type=int, default=d.min_proxies_for_pca, help=f"[default: {d.min_proxies_for_pca}]")
    g2.add_argument("--em-pca-max-iter", type=int, default=d.em_pca_max_iter, help=f"[default: {d.em_pca_max_iter}]")
    g2.add_argument("--em-pca-tol", type=float, default=d.em_pca_tol, help=f"[default: {d.em_pca_tol}]")
    g2.add_argument("--warn-neff-threshold", type=float, default=d.warn_neff_threshold, help=f"[default: {d.warn_neff_threshold}]")

    g3 = parser.add_argument_group("anomaly detection / resilience")
    g3.add_argument("--anomaly-threshold", type=float, default=d.anomaly_threshold, help=f"[default: {d.anomaly_threshold}]")
    g3.add_argument("--recovery-steps", type=int, default=d.recovery_steps, help=f"[default: {d.recovery_steps}]")
    g3.add_argument("--baseline-window", type=float, default=d.baseline_window, help=f"[default: {d.baseline_window}]")
    g3.add_argument("--resilience-cv-threshold", type=float, default=d.resilience_cv_threshold, help=f"[default: {d.resilience_cv_threshold}]")
    g3.add_argument("--no-confirm", dest="confirm", action="store_false", default=True, help="Skip the site-lookup confirmation prompt")

    return parser


def _review_nearby_sites(nearby: pd.DataFrame, archaeo_radius_km: float, confirm: bool) -> None:
    site_col = next((c for c in ("SiteName", "Site", "site_name") if c in nearby.columns), None)
    id_col = next((c for c in ("SiteID", "Site_ID", "site_id") if c in nearby.columns), None)
    lat_col = next((c for c in ("Lat", "lat", "latitude") if c in nearby.columns), None)
    lon_col = next((c for c in ("Long", "lon", "longitude") if c in nearby.columns), None)

    rows = []
    if site_col:
        for name, grp in nearby.groupby(site_col, sort=False):
            rows.append({
                "SiteName": name,
                "SiteID": grp[id_col].iloc[0] if id_col else "Unknown",
                "n_dates": len(grp),
                "dist_km": grp["dist_km"].min(),
                "lat": grp[lat_col].iloc[0] if lat_col else float("nan"),
                "lon": grp[lon_col].iloc[0] if lon_col else float("nan"),
            })
    df_sites = pd.DataFrame(rows).sort_values("dist_km")
    df_sites["SiteName"] = df_sites["SiteName"].fillna("Unknown")
    df_sites["SiteID"] = df_sites["SiteID"].fillna("Unknown")
    n_sites = len(df_sites)
    n_dates = df_sites["n_dates"].sum()

    print(f"  Records within {archaeo_radius_km} km: {n_dates} across {n_sites} site(s)")
    name_w = max(df_sites["SiteName"].astype(str).str.len().max(), 8)
    id_w = max(df_sites["SiteID"].astype(str).str.len().max(), 6)
    print(f"  {'SiteName':>{name_w}}  {'SiteID':>{id_w}}  {'n_dates':>7}  {'dist_km':>9}  {'lat':>8}  {'lon':>9}")
    for _, row in df_sites.iterrows():
        print(f"  {str(row['SiteName']):>{name_w}}  {str(row['SiteID']):>{id_w}}  "
              f"{int(row['n_dates']):>7}  {row['dist_km']:>9.6f}  {row['lat']:>8.4f}  {row['lon']:>9.4f}")

    if confirm:
        while True:
            ans = input("  Proceed? (yes/no) ").strip().lower()
            if ans in ("yes", "y"):
                return
            if ans in ("no", "n"):
                sys.exit("  Aborted by user.")
            print("  Please type 'yes' or 'no'.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.time_min >= args.time_max:
        sys.exit("ERROR: --time-min must be less than --time-max")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = args.site_name.replace(" ", "_")

    cfg = Config(
        site_name=args.site_name, site_lat=args.site_lat, site_lon=args.site_lon,
        archaeo_radius_km=args.radius_km, env_radius_km=args.env_radius_km,
        time_min=args.time_min, time_max=args.time_max, max_error=args.max_error, bin_h=args.bin_h,
        pangaea_keywords=args.pangaea_keyword or ["stable isotope", "speleothem", "pollen"],
        neotoma_types=args.neotoma_type or ["pollen", "stable isotopes"],
        use_gisp2=args.use_gisp2, kernel_sigma_yr=args.kernel_sigma_yr, max_interp_gap_bins=args.max_interp_gap_bins,
        min_proxies_for_pca=args.min_proxies_for_pca, em_pca_max_iter=args.em_pca_max_iter, em_pca_tol=args.em_pca_tol,
        warn_neff_threshold=args.warn_neff_threshold, anomaly_threshold=args.anomaly_threshold,
        recovery_steps=args.recovery_steps, baseline_window=args.baseline_window,
        resilience_cv_threshold=args.resilience_cv_threshold, confirm=args.confirm, outdir=outdir,
    )

    output_ccsi = outdir / f"{slug}_ccsi.csv"
    output_res = outdir / f"{slug}_ccsi_resilience.csv"
    output_plot = outdir / f"{slug}_ccsi.png"

    divider("COMPOSITE CLIMATE STRESS INDEX - PCA PIPELINE")
    print("  Databases    : PANGAEA -> Neotoma -> GISP2")
    divider("SITE LOOKUP")
    print(f"  Site         : {cfg.site_name}")
    print(f"  Coordinates  : {cfg.site_lat}N, {cfg.site_lon}E")
    print(f"  Archo radius : {cfg.archaeo_radius_km} km")
    print(f"  Env radius   : {cfg.env_radius_km} km")
    print(f"  Time window  : {cfg.time_min}-{cfg.time_max} Cal BP")
    print(f"  Resolution   : {cfg.resolution} yr  |  Kernel sigma: {cfg.kernel_sigma_yr} yr")
    print("  SPD firewall : SPD excluded from PCA matrix")

    df_input = pd.read_csv(args.input, low_memory=False, index_col=0)
    for col in ("Age", "Error", "Lat", "Long", "MedianCalBP"):
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")
    df_input = df_input.dropna(subset=["Lat", "Long"])
    df_input["dist_km"] = df_input.apply(lambda r: haversine_km(cfg.site_lat, cfg.site_lon, r["Lat"], r["Long"]), axis=1)
    df_nearby = df_input[df_input["dist_km"] <= cfg.archaeo_radius_km].copy()
    print(f"  Dates within {cfg.archaeo_radius_km} km : {len(df_nearby):,}")
    _review_nearby_sites(df_nearby, cfg.archaeo_radius_km, cfg.confirm)

    spd_from_04 = args.spd_from_04 or f"{slug}_spd_for_06.csv"
    if Path(spd_from_04).is_file():
        divider("SPD - LOADING FROM paleopy-spd")
        years_spd, spd_un, spd_lo, spd_hi = load_spd_from_04(spd_from_04)
    else:
        divider("SPD - BUILDING FROM SCRATCH")
        df_clean = apply_chronometric_hygiene(df_nearby, cfg.max_error, cfg.time_min, cfg.time_max)
        if len(df_clean) < 5:
            sys.exit("Too few dates. Increase --radius-km.")
        years_spd, spd_un, spd_lo, spd_hi = build_spd_from_scratch(df_clean, cfg.time_min, cfg.time_max, cfg.bin_h, cfg.resolution)

    divider("PROXY COLLECTION")
    try:
        proxy_df = collect_proxies(
            cfg.site_lat, cfg.site_lon, cfg.env_radius_km, cfg.time_min, cfg.time_max,
            cfg.use_gisp2, cfg.pangaea_keywords, cfg.neotoma_types, args.gisp2_cache,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    wide_df = bin_proxies(proxy_df, cfg.time_min, cfg.time_max, cfg.resolution, cfg.kernel_sigma_yr, cfg.max_interp_gap_bins)

    divider("PCA - COMPOSITE CLIMATE STRESS INDEX (climate proxies only)")
    ccsi_df, diag = compute_ccsi(wide_df, proxy_df, cfg.min_proxies_for_pca, cfg.em_pca_max_iter, cfg.em_pca_tol, cfg.warn_neff_threshold)
    ccsi_df.to_csv(output_ccsi, index=False)
    print(f"\n Saved CCSI -> {output_ccsi}")

    spd_correlations = compute_proxy_spd_correlations(wide_df, years_spd, spd_un, cfg.resolution)
    diag["spd_correlations"] = spd_correlations

    t, demo_z, ccsi_z = align_ccsi_spd(ccsi_df, years_spd, spd_un, cfg.time_min, cfg.time_max, cfg.resolution)

    divider("EPISODE DETECTION")
    records = detect_episodes(
        t, demo_z, ccsi_z, spd_lo, years_spd,
        cfg.anomaly_threshold, cfg.recovery_steps, cfg.baseline_window, cfg.resilience_cv_threshold,
    )
    df_res = pd.DataFrame(records)
    if not df_res.empty:
        df_res.to_csv(output_res, index=False)
        print(f"  Saved -> {output_res}")

    make_figure(
        t, years_spd, spd_un, spd_lo, spd_hi, demo_z, ccsi_z, ccsi_df, diag, df_res,
        cfg.site_name, cfg.time_min, cfg.time_max, cfg.archaeo_radius_km, cfg.env_radius_km,
        cfg.anomaly_threshold, cfg.baseline_window, output_plot,
    )
    print(f"[output] Cleaned figure saved -> {output_plot}")

    divider("COMPLETE")
    print(f"  Site         : {cfg.site_name}  [{cfg.site_lat}N, {cfg.site_lon}E]")
    print(f"  Proxies (PCA): {diag['n_proxies']} (climate only, SPD excluded)")
    print(f"  PC1 variance : {diag['explained_var_pc1']:.1%}")
    print(f"  N_eff        : {diag['n_eff']:.1f}")
    print(f"  Episodes     : {len(df_res)}")


if __name__ == "__main__":
    sys.exit(main())
