"""Cartopy figure assembly for the spatial KDE visualizer.

Ported from Scripts/07_KDE.py. The original disabled HTTPS certificate
verification GLOBALLY at import time
(`ssl._create_default_https_context = ssl._create_unverified_context`)
and raised the process-wide recursion limit to 5000 - both as permanent
side effects of merely importing the module. Here both are scoped to just
the one call that needs them (cartopy's Natural Earth geometry download in
build_land_mask), via paleopy.net's context managers, and restored
afterward.
"""

import numpy as np
import pandas as pd

from paleopy.net import temporary_recursion_limit, temporary_unverified_ssl

REGIONS = {
    "1": {"name": "North America", "lon": [-170, -50], "lat": [15, 80]},
    "2": {"name": "South America", "lon": [-85, -30], "lat": [-60, 15]},
    "3": {"name": "Europe", "lon": [-15, 45], "lat": [35, 72]},
    "4": {"name": "Asia", "lon": [40, 180], "lat": [0, 80]},
    "5": {"name": "Africa", "lon": [-20, 55], "lat": [-35, 40]},
    "6": {"name": "Oceania", "lon": [110, 180], "lat": [-50, 10]},
    "7": {"name": "Arctic / Circumpolar", "lon": [-180, 180], "lat": [60, 90]},
    "8": {"name": "Custom region", "lon": None, "lat": None},
}


def build_land_mask(lon_range: list, lat_range: list, res: int = 300, mask_res: str = "10m", edge_blur: float = 3) -> np.ndarray:
    """Returns a float alpha mask [0..1] shaped (res, res) in (lat, lon)
    order, rasterizing Natural Earth land polygons via
    matplotlib.path.Path.contains_points().

    Fetching the Natural Earth geometry (via cartopy) is wrapped in
    temporary_unverified_ssl()/temporary_recursion_limit(5000), scoped to
    just this call, rather than the original's permanent global side effects.
    """
    import cartopy.feature as cfeature
    from matplotlib.path import Path
    from scipy.ndimage import gaussian_filter

    print(f"  Building {mask_res} land mask at {res}x{res} (edge blur sigma={edge_blur}) ...")

    lons = np.linspace(lon_range[0], lon_range[1], res)
    lats = np.linspace(lat_range[0], lat_range[1], res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    binary = np.zeros(res * res, dtype=bool)

    with temporary_unverified_ssl(), temporary_recursion_limit(5000):
        geometries = list(cfeature.NaturalEarthFeature("physical", "land", mask_res).geometries())

    for geom in geometries:
        polys = geom.geoms if hasattr(geom, "geoms") else [geom]
        for poly in polys:
            ext = np.array(poly.exterior.coords)
            if (ext[:, 0].max() < lon_range[0] or ext[:, 0].min() > lon_range[1]
                    or ext[:, 1].max() < lat_range[0] or ext[:, 1].min() > lat_range[1]):
                continue
            binary |= Path(ext).contains_points(points)

    binary = binary.reshape(lon_grid.shape).astype(float)

    if edge_blur > 0:
        soft = gaussian_filter(binary, sigma=edge_blur)
        alpha = np.clip(soft / soft.max(), 0, 1)
    else:
        alpha = binary

    print("  Land mask ready.")
    return alpha


def generate_map(
    df_region: pd.DataFrame, df_sites: pd.DataFrame, region: dict, t_min: int, t_max: int,
    output_path, grid_res: int = 300, mask_res: str = "10m", edge_blur: float = 3,
) -> "plt.Figure":
    """Builds the 2-panel (unweighted site density / date-weighted
    density) KDE map figure and saves it to output_path.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    from paleopy.kde import build_grid, kde_unweighted_sites, kde_weighted_dates

    # Applied here (scoped to actually generating a map) rather than at
    # module import time, unlike the original script's unconditional
    # global rcParams.update() at import.
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
    })

    name = region["name"]
    lon_range = region["lon"]
    lat_range = region["lat"]

    print("  Computing KDE surfaces...")
    X, Y, positions = build_grid(lon_range, lat_range, grid_res)
    Z_sites = kde_unweighted_sites(df_sites, positions, X)
    Z_dates = kde_weighted_dates(df_region, positions, X)
    alpha_mask = build_land_mask(lon_range, lat_range, grid_res, mask_res, edge_blur)

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 1, figsize=(10, 13), subplot_kw={"projection": projection}, facecolor="white")

    panels = [
        {"ax": axes[0], "Z": Z_sites, "title": f"(A) Unweighted Site Density  -  {len(df_sites):,} unique sites", "label": "Relative site density", "cmap": "magma"},
        {"ax": axes[1], "Z": Z_dates, "title": f"(B) Date-Weighted Density  -  {len(df_region):,} radiocarbon dates", "label": "Relative date density", "cmap": "inferno"},
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_facecolor("white")
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=projection)

        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#3d3d3d", zorder=1)
        ax.add_feature(cfeature.LAKES, facecolor="white", zorder=2)

        Z_raw = p["Z"].T
        vmax = np.nanpercentile(Z_raw, 99.5)
        norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax)

        cmap_obj = plt.get_cmap(p["cmap"]).copy()
        rgba = cmap_obj(norm(np.clip(Z_raw, 0, vmax)))
        rgba[..., 3] = alpha_mask * 0.92

        ax.imshow(rgba, origin="lower", extent=[*lon_range, *lat_range], transform=projection, interpolation="bilinear", zorder=3)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#1a1a1a", zorder=4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#555555", linestyle=":", alpha=0.8, zorder=4)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.03, shrink=0.80, aspect=25)
        cb.set_label(p["label"], fontsize=10)
        cb.ax.tick_params(labelsize=10)
        cb.set_ticks([])

        gl = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.3, color="#aaaaaa", alpha=0.6, linestyle="--", zorder=5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
        gl.xlabel_style = {"size": 10, "color": "#333333"}
        gl.ylabel_style = {"size": 10, "color": "#333333"}

        ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=6, color="#1a1a1a", loc="left")

    fig.suptitle(
        f"P3K14C Radiocarbon Database  -  {name}\n{t_max:,}-{t_min:,} Cal BP",
        fontsize=14, fontweight="bold", color="#1a1a1a", y=0.995,
    )
    fig.subplots_adjust(hspace=0.08, top=0.955, bottom=0.03, left=0.05, right=0.96)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
