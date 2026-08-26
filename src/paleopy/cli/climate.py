"""paleopy-climate : compares the p3k14c radiocarbon SPD against
paleoenvironmental proxy data.

CLI wrapper around paleopy.human_climate/proxies/resilience; mirrors
Scripts/05_Human_Climate_Interaction.py's main() orchestration.

NOTE: the original script's own defaults have TIME_MIN=7700 here vs.
Scripts/04_SPD.py's TIME_MIN=7300, despite a comment in 05 claiming it
"must match 04" — this is a pre-existing inconsistency in the original
repo (not introduced by this port), preserved as-is rather than silently
normalized.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from paleopy.calibration import apply_chronometric_hygiene, load_intcal20
from paleopy.cli._console import add_site_arguments, confirm_or_abort, divider
from paleopy.config import Config
from paleopy.human_climate import build_spd, load_spd_from_04, make_plots, spd_significance_envelope
from paleopy.proxies import get_env_data
from paleopy.resilience import align, detect_anomalies
from paleopy.spd import find_nearby_sites


def build_parser() -> argparse.ArgumentParser:
    d = Config(time_min=7700, time_max=9500, bin_h=50, max_error=150)
    parser = argparse.ArgumentParser(
        prog="paleopy-climate",
        description="Compare the p3k14c radiocarbon SPD against paleoenvironmental proxy data.",
    )
    parser.add_argument("--input", required=True, help="Path to the calibrated ('pristine') p3k14c CSV")
    parser.add_argument("--outdir", default=".", help="Output directory (default: cwd)")
    parser.add_argument("--spd-from-04", default=None, help="Path to a <Site>_spd_for_06.csv from paleopy-spd to reuse, instead of rebuilding from scratch")
    add_site_arguments(parser, defaults=d)

    g = parser.add_argument_group("environmental data")
    g.add_argument("--env-radius-km", type=float, default=d.env_radius_km, help=f"Search radius for paleoenvironmental proxies (km) [default: {d.env_radius_km}]")
    g.add_argument("--env-proxy-keyword", default=d.env_proxy_keyword, help=f"PANGAEA search keyword [default: {d.env_proxy_keyword!r}]")
    g.add_argument("--env-neotoma-type", action="append", default=None, help="Neotoma dataset type (repeatable) [default: stable isotopes]")
    g.add_argument("--use-gisp2-only", action="store_true", default=d.use_gisp2_only, help="Skip live PANGAEA/Neotoma queries, use GISP2 only (default)")
    g.add_argument("--query-live-databases", dest="use_gisp2_only", action="store_false", help="Query PANGAEA/Neotoma instead of GISP2-only")
    g.add_argument("--gisp2-cache", default=None, help="Path to GISP2 cache CSV (default: <outdir>/<Site>_environment.csv)")

    g2 = parser.add_argument_group("Monte Carlo / anomaly detection")
    g2.add_argument("--mc-sims", type=int, default=d.mc_n_sim, help=f"Monte Carlo iterations [default: {d.mc_n_sim}]")
    g2.add_argument("--mc-model", default=d.mc_model, choices=["exponential", "uniform"], help=f"Null model [default: {d.mc_model}]")
    g2.add_argument("--anomaly-threshold", type=float, default=d.anomaly_threshold, help=f"Env Z-score anomaly threshold [default: {d.anomaly_threshold}]")
    g2.add_argument("--recovery-steps", type=int, default=d.recovery_steps, help=f"Consecutive above-threshold steps to close an episode [default: {d.recovery_steps}]")
    g2.add_argument("--baseline-window", type=float, default=d.baseline_window, help=f"Baseline window before onset (yr) [default: {d.baseline_window}]")
    g2.add_argument("--no-confirm", dest="confirm", action="store_false", default=True, help="Skip the site-lookup confirmation prompt")

    return parser


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
        time_min=args.time_min, time_max=args.time_max, max_error=args.max_error,
        bin_h=args.bin_h, env_proxy_keyword=args.env_proxy_keyword,
        env_neotoma_type=args.env_neotoma_type or ["stable isotopes"],
        use_gisp2_only=args.use_gisp2_only, mc_n_sim=args.mc_sims, mc_model=args.mc_model,
        anomaly_threshold=args.anomaly_threshold, recovery_steps=args.recovery_steps,
        baseline_window=args.baseline_window, confirm=args.confirm, outdir=outdir,
    )

    output_plot = outdir / f"{slug}_human_environment.png"
    output_spd = outdir / f"{slug}_spd.csv"
    output_env = outdir / (args.gisp2_cache or f"{slug}_environment.csv")
    output_csv = outdir / f"{slug}_comparison.csv"
    output_res = outdir / f"{slug}_resilience.csv"

    divider("HUMAN-ENVIRONMENT COMPARISON")
    print("  Databases : PANGAEA -> Neotoma -> GISP2")
    print(f"  Site          : {cfg.site_name}")
    print(f"  Coordinates   : {cfg.site_lat}N, {cfg.site_lon}E")
    print(f"  Archo radius  : {cfg.archaeo_radius_km} km")
    print(f"  Env radius    : {cfg.env_radius_km} km")
    print(f"  Time window   : {cfg.time_min}-{cfg.time_max} Cal BP")

    if not Path(args.input).is_file():
        sys.exit(f"ERROR: {args.input} not found.")

    df = pd.read_csv(args.input, low_memory=False, index_col=0)
    nearby = find_nearby_sites(df, cfg.site_lat, cfg.site_lon, cfg.archaeo_radius_km, site_id_fillna="Unknown")
    if nearby.empty:
        sys.exit(f"\n  No p3k14c records within {cfg.archaeo_radius_km} km.")

    sites = (
        nearby.groupby(["SiteName", "SiteID"])
        .agg(n_dates=("Age", "count"), dist_km=("dist_km", "min"), lat=("Lat", "first"), lon=("Long", "first"))
        .sort_values("dist_km").reset_index()
    )
    print(f"\n  Records within {cfg.archaeo_radius_km} km: {len(nearby):,} across {len(sites)} site(s)\n")
    print(sites[["SiteName", "SiteID", "n_dates", "dist_km", "lat", "lon"]].to_string(index=False))

    if cfg.confirm:
        confirm_or_abort("Proceed?")

    df_clean = apply_chronometric_hygiene(nearby, cfg.max_error, cfg.time_min, cfg.time_max)
    if len(df_clean) < 5:
        sys.exit("Not enough dates. Increase --radius-km.")

    spd_from_04 = args.spd_from_04 or f"{slug}_spd_for_06.csv"
    if Path(spd_from_04).is_file():
        divider("SPD - LOADING FROM paleopy-spd")
        years_spd, spd_un, spd_lo, spd_hi = load_spd_from_04(spd_from_04)
        spd_no = spd_un
        spd_pvals = (spd_un < spd_lo).astype(float)
        print("  Skipping IntCal20 rebuild and MC simulation.")
    else:
        divider("SPD - BUILDING FROM SCRATCH")
        years_spd, spd_un, spd_no = build_spd(df_clean, cfg.time_min, cfg.time_max, cfg.bin_h, cfg.resolution)
        intcal_curve = load_intcal20()
        spd_lo, spd_hi, spd_pvals = spd_significance_envelope(
            df_clean, years_spd, spd_un, cfg.bin_h, cfg.mc_n_sim, cfg.mc_model, intcal_curve,
        )

    env_ages, env_vals, env_label = get_env_data(
        cfg.site_lat, cfg.site_lon, cfg.env_radius_km, cfg.time_min, cfg.time_max, cfg.resolution,
        cfg.env_proxy_keyword, cfg.env_neotoma_type, cfg.env_bin_yr, cfg.use_gisp2_only,
        str(output_env), str(output_env),
    )
    t, demo_z, env_z = align(years_spd, spd_un, env_ages, env_vals, cfg.time_min, cfg.time_max, cfg.resolution)

    pd.DataFrame({"CalBP": t, "Demo_Z": demo_z, "Env_Z": env_z}).to_csv(output_csv, index=False)

    pd.DataFrame({
        "CalBP": years_spd, "SPD_Unnormalized": spd_un, "SPD_Normalized": spd_no,
        "MC_lo_2.5pct": spd_lo, "MC_hi_97.5pct": spd_hi, "MC_pval": spd_pvals,
    }).to_csv(output_spd, index=False)

    records = detect_anomalies(
        t, demo_z, env_z, cfg.anomaly_threshold, cfg.recovery_steps, cfg.baseline_window, cfg.resolution,
        spd_lo=spd_lo, years_spd=years_spd,
    )
    df_res = pd.DataFrame(records)
    if not df_res.empty:
        df_res.to_csv(output_res, index=False)
        print(f"\n  Saved -> {output_res}")

    make_plots(
        t, years_spd, spd_un, spd_no, demo_z, env_z, df_res,
        cfg.site_name, cfg.time_min, cfg.time_max, cfg.archaeo_radius_km, cfg.env_radius_km,
        cfg.anomaly_threshold, output_plot, spd_lo=spd_lo, spd_hi=spd_hi,
    )

    divider("COMPLETE")
    print(f"  Site      : {cfg.site_name}  [{cfg.site_lat}N, {cfg.site_lon}E]")
    print(f"  Dates     : {len(df_clean):,}  |  Anomalies: {len(df_res)}")
    print(f"  Env data  : {env_label}")
    print("  Outputs:")
    for f in (output_spd, output_env, output_csv, output_res, output_plot):
        print(f"    {f}")


if __name__ == "__main__":
    sys.exit(main())
