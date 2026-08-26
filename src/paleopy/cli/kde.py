"""paleopy-kde : spatial KDE visualizer over p3k14c site/date locations.

CLI wrapper around paleopy.kde/mapping; mirrors Scripts/07_KDE.py's
fully-interactive UX by default, with --region/--time-min/--time-max
(and custom-bbox flags) for non-interactive/scripted use.
"""

import argparse
import sys

import pandas as pd

from paleopy.kde import filter_region_and_time
from paleopy.mapping import REGIONS, generate_map


def prompt_region() -> dict:
    print("\n" + "=" * 50)
    print("  P3K14C - Spatial KDE Visualizer")
    print("=" * 50)
    print("Select a region:")
    for key, data in REGIONS.items():
        if data["lon"]:
            lon, lat = data["lon"], data["lat"]
            print(f"  [{key}] {data['name']:<25}  lon {lon[0]:>5}-{lon[1]:<5}  lat {lat[0]:>4}-{lat[1]}")
        else:
            print(f"  [{key}] {data['name']}")
    print("=" * 50)

    while True:
        choice = input("Enter choice (1-8): ").strip()
        if choice in REGIONS:
            region = dict(REGIONS[choice])
            if choice == "8":
                region = _prompt_custom_region(region)
            return region
        print("  Invalid - enter a number between 1 and 8.")


def _prompt_custom_region(region: dict) -> dict:
    print("\n  Define your custom bounding box.")

    def _get_float(prompt, lo, hi):
        while True:
            try:
                v = float(input(f"  {prompt} [{lo}-{hi}]: ").strip())
                if lo <= v <= hi:
                    return v
                print(f"  Must be between {lo} and {hi}.")
            except ValueError:
                print("  Please enter a number.")

    lon_min = _get_float("Longitude min", -180, 179)
    lon_max = _get_float(f"Longitude max (> {lon_min})", lon_min + 0.1, 180)
    lat_min = _get_float("Latitude  min", -90, 89)
    lat_max = _get_float(f"Latitude  max (> {lat_min})", lat_min + 0.1, 90)

    name = input("  Region label (used in title / filename): ").strip() or "Custom"
    region["name"] = name
    region["lon"] = [lon_min, lon_max]
    region["lat"] = [lat_min, lat_max]
    return region


def prompt_time_range() -> tuple:
    print("\n  Temporal filter (Calendar Years BP, e.g. 1000-12000).")
    while True:
        try:
            t_min = int(input("  Minimum Cal BP (younger limit): ").strip())
            t_max = int(input("  Maximum Cal BP (older limit): ").strip())
            if t_min < t_max:
                return t_min, t_max
            print("  Minimum must be less than maximum.")
        except ValueError:
            print("  Please enter whole numbers.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paleopy-kde",
        description="Spatial KDE visualizer over p3k14c site/date locations.",
    )
    parser.add_argument("--input", required=True, help="Path to the calibrated ('pristine') p3k14c CSV")
    parser.add_argument("--outdir", default=".", help="Output directory for the PNG (default: cwd)")
    parser.add_argument("--region", choices=list(REGIONS.keys()), default=None, help="Region preset key (1-8; 8=custom, requires --lon-min/--lon-max/--lat-min/--lat-max)")
    parser.add_argument("--region-name", default=None, help="Label for a custom region (with --region 8)")
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--time-min", type=int, default=None, help="Younger Cal BP boundary")
    parser.add_argument("--time-max", type=int, default=None, help="Older Cal BP boundary")
    parser.add_argument("--grid-res", type=int, default=300, help="KDE/land-mask grid resolution [default: 300]")
    parser.add_argument("--mask-res", default="10m", choices=["10m", "50m", "110m"], help="Natural Earth land-mask resolution [default: 10m]")
    parser.add_argument("--edge-blur", type=float, default=3, help="Gaussian sigma (grid cells) to feather the land/ocean boundary [default: 3]")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.region is not None:
        region = dict(REGIONS[args.region])
        if args.region == "8":
            if None in (args.lon_min, args.lon_max, args.lat_min, args.lat_max):
                sys.exit("ERROR: --region 8 (custom) requires --lon-min/--lon-max/--lat-min/--lat-max")
            region["name"] = args.region_name or "Custom"
            region["lon"] = [args.lon_min, args.lon_max]
            region["lat"] = [args.lat_min, args.lat_max]
    else:
        region = prompt_region()

    if args.time_min is not None and args.time_max is not None:
        if args.time_min >= args.time_max:
            sys.exit("ERROR: --time-min must be less than --time-max")
        t_min, t_max = args.time_min, args.time_max
    else:
        t_min, t_max = prompt_time_range()

    print(f"\n  Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    df_region, df_sites = filter_region_and_time(df, region["lon"], region["lat"], t_min, t_max)
    print(f"  Dates in temporal window ({t_min}-{t_max} Cal BP): {len(df_region):,}")
    print(f"  Unique site locations: {len(df_sites):,}")

    if len(df_region) < 3:
        sys.exit("  ERROR: fewer than 3 data points - KDE cannot be computed.")
    if len(df_sites) < 3:
        sys.exit("  ERROR: fewer than 3 unique sites. Broaden the region/time window.")

    safe_name = region["name"].replace(" ", "_").replace("/", "-")
    output_path = f"{args.outdir}/{safe_name}_{t_min}-{t_max}BP_KDE.png"

    generate_map(df_region, df_sites, region, t_min, t_max, output_path, args.grid_res, args.mask_res, args.edge_blur)
    print(f"\n  Saved -> {output_path}")


if __name__ == "__main__":
    sys.exit(main())
