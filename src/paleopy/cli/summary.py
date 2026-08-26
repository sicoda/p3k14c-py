"""paleopy-summary : descriptive statistics + 4-panel dashboard for the
calibrated p3k14c dataset.

CLI wrapper around paleopy.summary_stats.compute_summary(); mirrors
Scripts/03_SummaryStats.py's interactive filtering, but also accepts
--country/--site-name flags for non-interactive use.
"""

import argparse
import os
import sys

import pandas as pd

from paleopy.summary_stats import compute_summary, default_output_name, print_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paleopy-summary",
        description="Print summary statistics and save a 4-panel dashboard for the calibrated p3k14c dataset.",
    )
    parser.add_argument("--input", required=True, help="Path to the calibrated ('pristine') p3k14c CSV")
    parser.add_argument("--outdir", default=".", help="Directory to write the dashboard PNG into (default: cwd)")
    parser.add_argument("--country", default=None, help="Filter to a single Country (non-interactive)")
    parser.add_argument("--site-name", default=None, help="Filter to a single SiteName (non-interactive)")
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip the input() prompts; use only --country/--site-name (or no filter)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print(f"Loading dataset: {args.input}...")
    if not os.path.exists(args.input):
        sys.exit(f"ERROR: '{args.input}' not found. Did you run paleopy-calibrate?")

    df = pd.read_csv(args.input, low_memory=False)

    country = args.country
    site_name = args.site_name
    if not args.no_interactive and country is None and site_name is None:
        print("\n--- Data Filtering ---")
        print("Press ENTER to skip filtering and analyze the entire dataset.")
        country = input("Enter a Country to filter by (or leave blank): ").strip() or None
        site_name = input("Enter a SiteName to filter by (or leave blank): ").strip() or None

    if country:
        print(f"\n[!] Filtered dataset down to Country: {country}")
    if site_name:
        print(f"[!] Filtered dataset down to SiteName: {site_name}")

    try:
        stats, fig = compute_summary(df, country=country, site_name=site_name)
    except ValueError as exc:
        sys.exit(f"\nERROR: {exc}")

    print_summary(stats, country, site_name)

    print("\nGenerating visual dashboard...")
    out_name = default_output_name(country, site_name)
    out_path = os.path.join(args.outdir, out_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard successfully saved → {out_path}")
    print("========================================\n")


if __name__ == "__main__":
    sys.exit(main())
