"""Example: spatial KDE visualizer using the paleopy API directly.

Replicates Scripts/07_KDE.py for a fixed region/time window (Europe,
2000-10000 Cal BP) instead of the original's interactive prompts,
without going through the paleopy-kde console script. Run from the
repo root:

    python examples/07_kde.py
"""

from pathlib import Path

import pandas as pd

from paleopy.kde import filter_region_and_time
from paleopy.mapping import REGIONS, generate_map

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    input_path = REPO_ROOT / "Datasets" / "p3k14c_pristine_dates.csv"
    region = REGIONS["3"]  # Europe; see paleopy.mapping.REGIONS for all presets
    t_min, t_max = 2000, 10000

    df = pd.read_csv(input_path, low_memory=False)
    df_region, df_sites = filter_region_and_time(df, region["lon"], region["lat"], t_min, t_max)
    print(f"Dates in window ({t_min}-{t_max} Cal BP): {len(df_region):,}  |  unique sites: {len(df_sites):,}")

    if len(df_sites) < 3:
        raise SystemExit("Fewer than 3 unique sites - broaden the region/time window.")

    safe_name = region["name"].replace(" ", "_").replace("/", "-")
    output_path = OUTDIR / f"{safe_name}_{t_min}-{t_max}BP_KDE.png"

    # mask_res="50m" trades coastline detail for speed vs. the original's
    # "10m" default - raise it for a publication-quality map.
    generate_map(df_region, df_sites, region, t_min, t_max, output_path, grid_res=150, mask_res="50m", edge_blur=3)
    print(f"Wrote {output_path.name} -> {OUTDIR}")


if __name__ == "__main__":
    main()
