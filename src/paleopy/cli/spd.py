"""paleopy-spd : radiocarbon paleodemographic analysis (dates-as-data).

CLI wrapper around paleopy.spd.*; mirrors Scripts/04_SPD.py's argparse
interface and main() orchestration.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paleopy.calibration import calibrate_date_radiocarbon_pkg
from paleopy.cli._console import add_site_arguments, confirm_or_abort, divider, warn
from paleopy.config import Config
from paleopy.spd import (
    apply_hygiene_regex,
    bin_by_site_name,
    build_spd_and_caldists,
    find_nearby_sites,
    make_plots,
    monte_carlo_envelope,
    phase6_cpl,
    summarize_sites,
    taphonomic_weight,
)


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    parser = argparse.ArgumentParser(
        prog="paleopy-spd",
        description="Radiocarbon paleodemographic analysis (dates-as-data).",
    )
    parser.add_argument("--input", required=True, help="Path to the calibrated ('pristine') p3k14c CSV")
    parser.add_argument("--outdir", default=".", help="Output directory for CSV/PNG outputs (default: cwd)")
    add_site_arguments(parser, defaults=d)

    g = parser.add_argument_group("analysis options")
    g.add_argument("--sims", type=int, default=d.n_simulations, help=f"Monte Carlo iterations [default: {d.n_simulations}]")
    g.add_argument("--fast", action="store_true", help="Quick mode: 200 MC iterations, max 2 CPL hinges")
    g.add_argument("--no-cpl", action="store_true", help="Skip CPL modelling")
    g.add_argument("--confirm", action="store_true", help="Pause for y/n confirmation before long computation steps")
    g.add_argument("--no-show", action="store_true", help="Save plot without opening an interactive window")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.time_min >= args.time_max:
        sys.exit("ERROR: --time-min must be less than --time-max")

    cfg = Config(
        site_name=args.site_name, site_lat=args.site_lat, site_lon=args.site_lon,
        archaeo_radius_km=args.radius_km, time_min=args.time_min, time_max=args.time_max,
        max_error=args.max_error, bin_h=args.bin_h, n_simulations=args.sims,
        outdir=args.outdir,
    )
    effective_sims = 200 if args.fast else cfg.n_simulations
    max_hinges = 2 if args.fast else 4

    t_start = time.time()

    divider("SITE LOOKUP")
    print(f"  Site          : {cfg.site_name}")
    print(f"  Coordinates   : {cfg.site_lat}°N, {cfg.site_lon}°E")
    print(f"  Search radius : {cfg.archaeo_radius_km} km")
    print(f"  Time window   : {cfg.time_min}-{cfg.time_max} Cal BP")

    if not Path(args.input).is_file():
        sys.exit(f"\n  ERROR: {args.input!r} not found.\n")

    print("\n  Loading dataset ...", end=" ", flush=True)
    df = pd.read_csv(args.input, low_memory=False)
    print(f"{len(df):,} total records")

    nearby = find_nearby_sites(df, cfg.site_lat, cfg.site_lon, cfg.archaeo_radius_km)
    if nearby.empty:
        sys.exit(
            f"\n  ERROR: No records found within {cfg.archaeo_radius_km} km of "
            f"({cfg.site_lat}, {cfg.site_lon}).\n"
        )

    sites = summarize_sites(nearby, cfg.time_min, cfg.time_max)
    print(f"\n  Records within {cfg.archaeo_radius_km} km : {len(nearby):,} across {len(sites)} site(s)\n")
    display_df = sites[["SiteName", "SiteID", "n_total", "n_in_window", "age_range", "dist_km", "lat", "lon"]].copy()
    display_df.columns = ["SiteName", "SiteID", "N total", f"N {cfg.time_min}-{cfg.time_max}", "Cal BP range", "dist (km)", "Lat", "Lon"]
    print(display_df.to_string(index=False))

    in_window = nearby[(nearby["MedianCalBP"] >= cfg.time_min) & (nearby["MedianCalBP"] <= cfg.time_max)]
    n_usable = int(in_window["Error"].le(cfg.max_error).sum()) if not in_window.empty else 0
    print(f"\n  Dates in window with error <= {cfg.max_error} yr: {n_usable}")
    if n_usable < 10:
        warn(f"Only {n_usable} usable dates in the analysis window. Results may be unreliable.")

    if args.confirm:
        confirm_or_abort("Proceed with these settings?")
    divider()

    try:
        df_clean = apply_hygiene_regex(nearby, cfg.max_error, cfg.time_min, cfg.time_max)
    except ValueError as exc:
        sys.exit(f"\n  ERROR: {exc}\n")

    df_binned = bin_by_site_name(df_clean, cfg.bin_h)

    years = np.arange(cfg.time_min, cfg.time_max + cfg.resolution, cfg.resolution)
    spd_raw, cal_dists = build_spd_and_caldists(df_binned, years, calibrate_date_radiocarbon_pkg)

    taph_weight = taphonomic_weight(years)
    spd_taph = spd_raw / taph_weight
    spd_taph /= spd_taph.sum()

    if args.confirm:
        confirm_or_abort(
            f"About to run {effective_sims:,} MC iterations "
            f"(~{effective_sims * len(df_clean) * 0.013 / 60:.1f} min est.). Proceed?"
        )

    errors = df_binned["Error"].values
    curves = df_binned["CalCurveUsed"].values if "CalCurveUsed" in df_binned.columns else np.array(["intcal20"] * len(df_binned))

    null_exp, lower_exp, upper_exp, p_exp = monte_carlo_envelope(
        years, spd_taph, errors, curves, effective_sims, cfg.time_min, cfg.time_max,
        calibrate_date_radiocarbon_pkg, null_model="exponential",
    )
    null_log, lower_log, upper_log, p_log = monte_carlo_envelope(
        years, spd_taph, errors, curves, effective_sims, cfg.time_min, cfg.time_max,
        calibrate_date_radiocarbon_pkg, null_model="logistic",
    )

    if not args.no_cpl:
        cpl_fitted, hinge_years, hinge_probs, best_n, _results = phase6_cpl(
            years, spd_taph, cal_dists, df_binned, taph_weight,
            cfg.time_min, cfg.time_max, min_hinge_sep=200, max_hinges=max_hinges,
        )
    else:
        cpl_fitted = np.zeros_like(years)
        hinge_years = [float(cfg.time_min), float(cfg.time_max)]
        hinge_probs = [0.0, 0.0]
        best_n = 0

    output_csv = cfg.outdir / f"{cfg.slug}_population_dynamics.csv"
    output_plot = cfg.outdir / f"{cfg.slug}_population_dynamics.png"

    pd.DataFrame({
        "CalBP": years,
        "SPD_Raw": spd_raw,
        "SPD_TaphCorrected": spd_taph,
        "NullModel_Exp": null_exp,
        "Envelope_Exp_Lower": lower_exp,
        "Envelope_Exp_Upper": upper_exp,
        "NullModel_Log": null_log,
        "Envelope_Log_Lower": lower_log,
        "Envelope_Log_Upper": upper_log,
        "CPL_Fitted": cpl_fitted,
    }).to_csv(output_csv, index=False)
    print(f"\n  CSV saved -> {output_csv}")

    spd_for_06_path = cfg.outdir / f"{cfg.slug}_spd_for_06.csv"
    pd.DataFrame({
        "CalBP": years,
        "SPD_TaphCorrected": spd_taph,
        "MC_lo_2.5pct": lower_exp,
        "MC_hi_97.5pct": upper_exp,
    }).to_csv(spd_for_06_path, index=False)
    print(f"  SPD for 06  -> {spd_for_06_path}")

    fig = make_plots(
        years, spd_raw, spd_taph,
        null_exp, lower_exp, upper_exp,
        null_log, lower_log, upper_log,
        cpl_fitted, hinge_years, hinge_probs, best_n,
        cfg.site_name, cfg.time_min, cfg.time_max, effective_sims,
        output_plot,
    )
    print(f"  Plot saved -> {output_plot}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)

    divider("RUN SUMMARY")
    print(f"  Site             : {cfg.site_name}")
    print(f"  Dates analysed   : {len(df_clean):,}")
    print(f"  Site-phase bins  : {df_binned['Bin'].nunique()}")
    print(f"  MC iterations    : {effective_sims:,}")
    print(f"  Global p (exp.)  : {p_exp:.4f}")
    print(f"  Global p (log.)  : {p_log:.4f}")
    print(f"  Total runtime    : {time.time() - t_start:.0f}s")
    divider()


if __name__ == "__main__":
    sys.exit(main())
