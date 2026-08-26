"""Descriptive statistics and 4-panel dashboard for the calibrated p3k14c
dataset.

Ported from Scripts/03_SummaryStats.py, split into a pure
`compute_summary()` (no input()/file I/O) and a thin interactive CLI
wrapper in paleopy.cli.summary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUMERIC_COLS = ["Age", "Error", "MedianCalBP"]


def default_output_name(country: str | None, site_name: str | None) -> str:
    if site_name:
        return f"summary_{site_name.replace(' ', '_')}.png"
    if country:
        return f"summary_{country.replace(' ', '_')}.png"
    return "summary_global.png"


def compute_summary(
    df: pd.DataFrame,
    country: str | None = None,
    site_name: str | None = None,
) -> tuple[dict, "plt.Figure"]:
    """Compute summary statistics and build the 4-panel dashboard figure.

    Returns (stats, fig). stats is a dict with keys: total_records,
    continental_breakdown, top_countries, missing_values, age_error_stats
    (any of which may be omitted depending on filters/available columns).

    Raises ValueError if the country/site_name filters result in an empty
    dataset.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if country:
        df = df[df["Country"].str.lower() == country.lower()]
    if site_name:
        df = df[df["SiteName"].str.lower() == site_name.lower()]

    if df.empty:
        raise ValueError("Filters resulted in an empty dataset. Try different spelling.")

    stats: dict = {"total_records": len(df)}

    if "Continent" in df.columns and not country and not site_name:
        stats["continental_breakdown"] = df["Continent"].value_counts().head()

    if "Country" in df.columns and not site_name:
        stats["top_countries"] = df["Country"].value_counts().head()

    missing = df[["Age", "Error", "Lat", "Long", "MedianCalBP"]].isna().sum()
    stats["missing_values"] = missing[missing > 0]

    stats_df = df[["Age", "Error", "MedianCalBP"]].describe().T
    stats["age_error_stats"] = stats_df[["count", "mean", "50%", "min", "max"]].round(2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Radiocarbon Dataset Geometry & Distributions", fontsize=16, fontweight="bold")

    # 1. Bar chart: top regions/countries (or materials, if heavily filtered)
    ax1 = axes[0, 0]
    if country or site_name:
        top_cats = df["Material"].value_counts().head(10)
        ax1.set_title("Top 10 Dates Materials")
    else:
        top_cats = df["Country"].value_counts().head(10)
        ax1.set_title("Top 10 Countries by Sample Count")

    top_cats.plot(kind="bar", color="coral", ax=ax1)
    ax1.set_ylabel("Number of Samples")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Histogram: error margins
    ax2 = axes[0, 1]
    error_data = df["Error"].dropna()
    if not error_data.empty:
        p95 = np.percentile(error_data, 95)
        filtered_errors = error_data[error_data <= p95]
        ax2.hist(filtered_errors, bins=50, color="purple", alpha=0.7)
        ax2.set_xlim(0, p95)
    ax2.set_title("Distribution of Error Margins")
    ax2.set_xlabel("Radiocarbon Error (± years)")
    ax2.set_ylabel("Frequency")

    # 3. Histogram: uncalibrated ages
    ax3 = axes[1, 0]
    ax3.hist(df["Age"].dropna(), bins=50, color="red", alpha=0.6)
    ax3.set_title("Geometry of Uncalibrated Ages (CRA)")
    ax3.set_xlabel("14C yr BP")
    ax3.set_ylabel("Frequency")

    # 4. Histogram: calibrated ages
    ax4 = axes[1, 1]
    if "MedianCalBP" in df.columns:
        ax4.hist(df["MedianCalBP"].dropna(), bins=50, color="blue", alpha=0.6)
        ax4.set_title("Geometry of Calibrated Ages (Median)")
        ax4.set_xlabel("Calendar yr BP")
    else:
        ax4.text(0.5, 0.5, "Calibrated Data Not Found", ha="center", va="center")

    plt.tight_layout()

    return stats, fig


def print_summary(stats: dict, country: str | None, site_name: str | None) -> None:
    print("\n========================================")
    print("        DATASET SUMMARY REPORT          ")
    print("========================================")
    print(f"Total Records: {stats['total_records']:,}")

    if "continental_breakdown" in stats:
        print("\n--- Continental Breakdown ---")
        print(stats["continental_breakdown"])

    if "top_countries" in stats:
        print("\n--- Top 5 Countries ---")
        print(stats["top_countries"])

    print("\n--- Missing Values ---")
    missing = stats["missing_values"]
    print(missing.to_string() if missing.sum() > 0 else "No missing values in key columns!")

    print("\n--- Age & Error Margins ---")
    print(stats["age_error_stats"].to_string(header=["Count", "Mean", "Median", "Min", "Max"]))
