"""Example: summary statistics + dashboard using the paleopy API directly.

Replicates Scripts/03_SummaryStats.py's non-interactive (global) path
without going through the paleopy-summary console script. Run from the
repo root:

    python examples/03_summary.py
"""

from pathlib import Path

import pandas as pd

from paleopy.summary_stats import compute_summary, default_output_name, print_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    input_path = REPO_ROOT / "Datasets" / "p3k14c_pristine_dates.csv"

    df = pd.read_csv(input_path, low_memory=False)
    stats, fig = compute_summary(df)  # no country/site_name filter = global view

    print_summary(stats, country=None, site_name=None)

    out_path = OUTDIR / default_output_name(None, None)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nDashboard saved -> {out_path}")


if __name__ == "__main__":
    main()
