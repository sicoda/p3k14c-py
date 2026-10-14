"""Example: SPD paleodemography pipeline using the paleopy API directly.

Replicates Scripts/04_SPD.py's Catalhoyuk case study without going
through the paleopy-spd console script. Uses a small Monte Carlo
iteration count so it runs in seconds; raise N_SIMULATIONS for a
publication-quality run (the original default was 5000). Run from the
repo root:

    python examples/04_spd.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from paleopy.calibration import calibrate_date_radiocarbon_pkg
from paleopy.config import Config
from paleopy.spd import (
    apply_hygiene_regex,
    bin_by_site_name,
    build_spd_and_caldists,
    find_nearby_sites,
    make_plots,
    monte_carlo_envelope,
    phase6_cpl,
    taphonomic_weight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)

N_SIMULATIONS = 100  # publication runs typically use 5000


def main() -> None:
    cfg = Config(outdir=OUTDIR)  # defaults match the Catalhoyuk case study
    input_path = REPO_ROOT / "Datasets" / "p3k14c_pristine_dates.csv"

    df = pd.read_csv(input_path, low_memory=False)
    nearby = find_nearby_sites(df, cfg.site_lat, cfg.site_lon, cfg.archaeo_radius_km)
    print(f"Records within {cfg.archaeo_radius_km} km: {len(nearby):,}")

    df_clean = apply_hygiene_regex(nearby, cfg.max_error, cfg.time_min, cfg.time_max)
    df_binned = bin_by_site_name(df_clean, cfg.bin_h)
    print(f"Dates after hygiene: {len(df_clean):,}  |  bins: {df_binned['Bin'].nunique()}")

    years = np.arange(cfg.time_min, cfg.time_max + cfg.resolution, cfg.resolution)
    spd_raw, cal_dists = build_spd_and_caldists(df_binned, years, calibrate_date_radiocarbon_pkg)

    taph_weight = taphonomic_weight(years)
    spd_taph = spd_raw / taph_weight
    spd_taph /= spd_taph.sum()

    errors = df_binned["Error"].values
    curves = df_binned["CalCurveUsed"].values

    null_exp, lower_exp, upper_exp, p_exp = monte_carlo_envelope(
        years, spd_taph, errors, curves, N_SIMULATIONS, cfg.time_min, cfg.time_max,
        calibrate_date_radiocarbon_pkg, null_model="exponential",
    )
    null_log, lower_log, upper_log, p_log = monte_carlo_envelope(
        years, spd_taph, errors, curves, N_SIMULATIONS, cfg.time_min, cfg.time_max,
        calibrate_date_radiocarbon_pkg, null_model="logistic",
    )

    cpl_fitted, hinge_years, hinge_probs, best_n, _ = phase6_cpl(
        years, spd_taph, cal_dists, df_binned, taph_weight,
        cfg.time_min, cfg.time_max, min_hinge_sep=200, max_hinges=2,
    )

    output_csv = OUTDIR / f"{cfg.slug}_population_dynamics.csv"
    output_plot = OUTDIR / f"{cfg.slug}_population_dynamics.png"

    pd.DataFrame({
        "CalBP": years, "SPD_Raw": spd_raw, "SPD_TaphCorrected": spd_taph,
        "NullModel_Exp": null_exp, "Envelope_Exp_Lower": lower_exp, "Envelope_Exp_Upper": upper_exp,
        "NullModel_Log": null_log, "Envelope_Log_Lower": lower_log, "Envelope_Log_Upper": upper_log,
        "CPL_Fitted": cpl_fitted,
    }).to_csv(output_csv, index=False)

    make_plots(
        years, spd_raw, spd_taph, null_exp, lower_exp, upper_exp, null_log, lower_log, upper_log,
        cpl_fitted, hinge_years, hinge_probs, best_n,
        cfg.site_name, cfg.time_min, cfg.time_max, N_SIMULATIONS, output_plot,
    )

    print(f"\nGlobal p (exponential): {p_exp:.4f}  |  Global p (logistic): {p_log:.4f}")
    print(f"CPL best model: {best_n} internal hinge(s)")
    print(f"Wrote {output_csv.name} and {output_plot.name} -> {OUTDIR}")


if __name__ == "__main__":
    main()
