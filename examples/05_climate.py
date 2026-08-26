"""Example: human-climate comparison using the paleopy API directly.

Replicates Scripts/05_Human_Climate_Interaction.py's Catalhoyuk case
study (GISP2-only, matching the original's default USE_GISP2_ONLY=True)
without going through the paleopy-climate console script. Run from the
repo root, ideally after 04_spd.py so the SPD can be reused:

    python examples/05_climate.py
"""

from pathlib import Path

import pandas as pd

from paleopy.calibration import apply_chronometric_hygiene, load_intcal20
from paleopy.config import Config
from paleopy.human_climate import build_spd, load_spd_from_04, make_plots, spd_significance_envelope
from paleopy.proxies import get_env_data
from paleopy.resilience import align, detect_anomalies
from paleopy.spd import find_nearby_sites

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)

MC_N_SIM = 100  # publication runs typically use 999


def main() -> None:
    # 05's own defaults differ slightly from 04's (TIME_MIN=7700 vs 7300,
    # BIN_H=50, MAX_ERROR=150) - see cli/climate.py's note on this
    # pre-existing discrepancy in the original scripts.
    cfg = Config(time_min=7700, time_max=9500, bin_h=50, max_error=150, outdir=OUTDIR)
    input_path = REPO_ROOT / "Datasets" / "p3k14c_pristine_dates.csv"

    df = pd.read_csv(input_path, low_memory=False, index_col=0)
    nearby = find_nearby_sites(df, cfg.site_lat, cfg.site_lon, cfg.archaeo_radius_km, site_id_fillna="Unknown")
    df_clean = apply_chronometric_hygiene(nearby, cfg.max_error, cfg.time_min, cfg.time_max)
    print(f"Dates after hygiene: {len(df_clean):,}")

    spd_from_04 = OUTDIR / f"{cfg.slug}_population_dynamics.csv"
    spd_for_06 = REPO_ROOT / "Catalhoyuk Data" / "Catalhoyuk_spd_for_06.csv"
    if spd_for_06.is_file():
        print(f"Reusing precomputed SPD from {spd_for_06}")
        years_spd, spd_un, spd_lo, spd_hi = load_spd_from_04(str(spd_for_06))
        spd_no = spd_un
    else:
        print("Building SPD from scratch (no precomputed SPD found)")
        years_spd, spd_un, spd_no = build_spd(df_clean, cfg.time_min, cfg.time_max, cfg.bin_h, cfg.resolution)
        intcal_curve = load_intcal20()
        spd_lo, spd_hi, _ = spd_significance_envelope(
            df_clean, years_spd, spd_un, cfg.bin_h, MC_N_SIM, cfg.mc_model, intcal_curve,
        )

    env_ages, env_vals, env_label = get_env_data(
        cfg.site_lat, cfg.site_lon, cfg.env_radius_km, cfg.time_min, cfg.time_max, cfg.resolution,
        cfg.env_proxy_keyword, cfg.env_neotoma_type, cfg.env_bin_yr, use_gisp2_only=True,
        gisp2_cache_path=str(OUTDIR / f"{cfg.slug}_environment.csv"),
        env_output_path=str(OUTDIR / f"{cfg.slug}_environment.csv"),
    )
    print(f"Environmental data source: {env_label}")

    t, demo_z, env_z = align(years_spd, spd_un, env_ages, env_vals, cfg.time_min, cfg.time_max, cfg.resolution)

    records = detect_anomalies(
        t, demo_z, env_z, cfg.anomaly_threshold, cfg.recovery_steps, cfg.baseline_window, cfg.resolution,
        spd_lo=spd_lo, years_spd=years_spd,
    )
    df_res = pd.DataFrame(records)
    print(f"Anomaly episodes detected: {len(df_res)}")

    output_plot = OUTDIR / f"{cfg.slug}_human_environment.png"
    make_plots(
        t, years_spd, spd_un, spd_no, demo_z, env_z, df_res,
        cfg.site_name, cfg.time_min, cfg.time_max, cfg.archaeo_radius_km, cfg.env_radius_km,
        cfg.anomaly_threshold, output_plot, spd_lo=spd_lo, spd_hi=spd_hi,
    )

    pd.DataFrame({"CalBP": t, "Demo_Z": demo_z, "Env_Z": env_z}).to_csv(OUTDIR / f"{cfg.slug}_comparison.csv", index=False)
    if not df_res.empty:
        df_res.to_csv(OUTDIR / f"{cfg.slug}_resilience.csv", index=False)

    print(f"Wrote comparison/resilience CSVs and {output_plot.name} -> {OUTDIR}")


if __name__ == "__main__":
    main()
