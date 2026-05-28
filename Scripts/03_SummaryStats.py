"""
03_SummaryStats.py : Data summary and visualization.

Input  : p3k14c_pristine_dates.csv   (output of 02_Calibrating.py)
Output : summary_global.png  (or summary_<Country/Site>.png if filtered)

DEPENDENCIES : pandas numpy matplotlib
PYTHON       : 3.10+
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_summary(file_path):
    print(f"Loading dataset: {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: '{file_path}' not found. Did you run the calibration script?")
        return

    df = pd.read_csv(file_path, low_memory=False)

    # Convert essential columns to numeric to avoid math errors
    numeric_cols = ['Age', 'Error', 'MedianCalBP']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Interactive Filtering ---
    print("\n--- Data Filtering ---")
    print("Press ENTER to skip filtering and analyze the entire dataset.")
    country_filter = input("Enter a Country to filter by (or leave blank): ").strip()
    site_filter = input("Enter a SiteName to filter by (or leave blank): ").strip()

    if country_filter:
        df = df[df['Country'].str.lower() == country_filter.lower()]
        print(f"\n[!] Filtered dataset down to Country: {country_filter}")
    if site_filter:
        df = df[df['SiteName'].str.lower() == site_filter.lower()]
        print(f"[!] Filtered dataset down to SiteName: {site_filter}")

    if df.empty:
        print("\nERROR: Your filters resulted in an empty dataset. Try different spelling.")
        return

    # -- Textual Summary Statistics ------------------------------
    print("\n========================================")
    print("        DATASET SUMMARY REPORT          ")
    print("========================================")
    print(f"Total Records: {len(df):,}")
    
    if 'Continent' in df.columns and not country_filter and not site_filter:
        print("\n--- Continental Breakdown ---")
        print(df['Continent'].value_counts().head())

    if 'Country' in df.columns and not site_filter:
        print("\n--- Top 5 Countries ---")
        print(df['Country'].value_counts().head())

    print("\n--- Missing Values ---")
    missing = df[['Age', 'Error', 'Lat', 'Long', 'MedianCalBP']].isna().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values in key columns!")

    print("\n--- Age & Error Margins ---")
    stats_df = df[['Age', 'Error', 'MedianCalBP']].describe().T
    # Select just the relevant stats for a cleaner readout
    print(stats_df[['count', 'mean', '50%', 'min', 'max']].round(2).to_string(header=['Count', 'Mean', 'Median', 'Min', 'Max']))
    
    # -- Visual Dashboard ----------------------------------------
    print("\nGenerating visual dashboard...")
    
    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Radiocarbon Dataset Geometry & Distributions', fontsize=16, fontweight='bold')

    # 1. Bar Chart: Top Regions/Countries
    ax1 = axes[0, 0]
    if country_filter or site_filter:
        # If heavily filtered, show material instead
        top_cats = df['Material'].value_counts().head(10)
        ax1.set_title('Top 10 Dates Materials')
    else:
        top_cats = df['Country'].value_counts().head(10)
        ax1.set_title('Top 10 Countries by Sample Count')
    
    top_cats.plot(kind='bar', color='coral', ax=ax1)
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)

    # Histogram: Error Margins
    ax2 = axes[0, 1]
    error_data = df['Error'].dropna()
    
    if not error_data.empty:
        # Filter the data to the 95th percentile to calculate bins
        p95 = np.percentile(error_data, 95)
        filtered_errors = error_data[error_data <= p95]
        
        # Draw the histogram using only the filtered data
        ax2.hist(filtered_errors, bins=50, color='purple', alpha=0.7)
        ax2.set_xlim(0, p95)
        
    ax2.set_title('Distribution of Error Margins')
    ax2.set_xlabel('Radiocarbon Error (± years)')
    ax2.set_ylabel('Frequency')

    # Histogram: Uncalibrated Ages
    ax3 = axes[1, 0]
    ax3.hist(df['Age'].dropna(), bins=50, color='red', alpha=0.6)
    ax3.set_title('Geometry of Uncalibrated Ages (CRA)')
    ax3.set_xlabel('14C yr BP')
    ax3.set_ylabel('Frequency')

    # Histogram: Calibrated Ages
    ax4 = axes[1, 1]
    if 'MedianCalBP' in df.columns:
        ax4.hist(df['MedianCalBP'].dropna(), bins=50, color='blue', alpha=0.6)
        ax4.set_title('Geometry of Calibrated Ages (Median)')
        ax4.set_xlabel('Calendar yr BP')
    else:
        ax4.text(0.5, 0.5, 'Calibrated Data Not Found', ha='center', va='center')
        
    plt.tight_layout()
    
    # Define output filename based on filters
    if site_filter:
        out_name = f"summary_{site_filter.replace(' ', '_')}.png"
    elif country_filter:
        out_name = f"summary_{country_filter.replace(' ', '_')}.png"
    else:
        out_name = "summary_global.png"

    out_path = os.path.join(_SCRIPT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard successfully saved → {out_path}")
    print("========================================\n")

if __name__ == "__main__":
    generate_summary(os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv"))