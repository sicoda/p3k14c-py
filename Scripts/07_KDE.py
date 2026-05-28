"""
05_KDE.py

Spatial KDE visualizer

Input  : p3k14c_pristine_dates.csv   (output of 02_Calibrating.py)
Output : <Region>_<tmin>-<tmax>BP_KDE.png

IN TERMINAL :
  python 05_KDE.py

DEPENDENCIES : pandas numpy scipy matplotlib cartopy
PYTHON       : 3.10+
NOTE         : cartopy may require additional system libraries.
               See https://scitools.org.uk/cartopy/docs/latest/installing.html
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import os
import sys
import ssl
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
 
 
 
ssl._create_default_https_context = ssl._create_unverified_context
sys.setrecursionlimit(5000)
 
# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = _SCRIPT_DIR
DATA_FILE   = os.path.join(_SCRIPT_DIR, "p3k14c_pristine_dates.csv")
GRID_RES    = 300   # higher res → smoother edges (110m mask: ~20 s; 10m: ~2 min)
MASK_RES    = "10m" # "10m" gives clean coast edges; "50m" is faster, "110m" is blocky
EDGE_BLUR   = 3     # gaussian sigma (grid cells) to feather the land/ocean boundary

# ---------------------------------------------------------------------------
# END OF USER CONFIGURATION
# ---------------------------------------------------------------------------
 
rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize":   10,
    "axes.titlesize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
})
 
REGIONS = {
    "1": {"name": "North America",        "lon": [-170, -50], "lat": [15,  80]},
    "2": {"name": "South America",        "lon": [-85,  -30], "lat": [-60, 15]},
    "3": {"name": "Europe",               "lon": [-15,   45], "lat": [35,  72]},
    "4": {"name": "Asia",                 "lon": [40,   180], "lat": [0,   80]},
    "5": {"name": "Africa",               "lon": [-20,   55], "lat": [-35, 40]},
    "6": {"name": "Oceania",              "lon": [110,  180], "lat": [-50, 10]},
    "7": {"name": "Arctic / Circumpolar", "lon": [-180, 180], "lat": [60,  90]},
    "8": {"name": "Custom region",        "lon": None,        "lat": None},
}
 
 
# ---------------------------------------------------------------------------
# Terminal prompts
# ---------------------------------------------------------------------------
 
def prompt_region() -> dict:
    print("\n" + "=" * 50)
    print("  P3K14C  —  Spatial KDE Visualizer")
    print("=" * 50)
    print("Select a region:")
    for key, data in REGIONS.items():
        if data["lon"]:
            lon, lat = data["lon"], data["lat"]
            print(f"  [{key}] {data['name']:<25}  "
                  f"lon {lon[0]:>5}–{lon[1]:<5}  lat {lat[0]:>4}–{lat[1]}")
        else:
            print(f"  [{key}] {data['name']}")
    print("=" * 50)
 
    while True:
        choice = input("Enter choice (1–8): ").strip()
        if choice in REGIONS:
            region = dict(REGIONS[choice])
            if choice == "8":
                region = _prompt_custom_region(region)
            return region
        print("  Invalid — enter a number between 1 and 8.")
 
 
def _prompt_custom_region(region: dict) -> dict:
    print("\n  Define your custom bounding box.")
 
    def _get_float(prompt, lo, hi):
        while True:
            try:
                v = float(input(f"  {prompt} [{lo}–{hi}]: ").strip())
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
    region["lon"]  = [lon_min, lon_max]
    region["lat"]  = [lat_min, lat_max]
    return region
 
 
def prompt_time_range() -> tuple[int, int]:
    print("\n  Temporal filter (Calendar Years BP, e.g. 1000–12000).")
    while True:
        try:
            t_min = int(input("  Minimum Cal BP (younger limit): ").strip())
            t_max = int(input("  Maximum Cal BP (older limit): ").strip())
            if t_min < t_max:
                return t_min, t_max
            print("  Minimum must be less than maximum.")
        except ValueError:
            print("  Please enter whole numbers.")
 
 
# ---------------------------------------------------------------------------
# Land masking
# ---------------------------------------------------------------------------
 
def build_land_mask(lon_range, lat_range, res=GRID_RES,
                    mask_res=MASK_RES, edge_blur=EDGE_BLUR):
    """
    Returns a float alpha mask [0..1] shaped (res, res) in (lat, lon) order.
 
    Uses matplotlib.path.Path.contains_points() to rasterize each land polygon
    directly — completes in seconds at 300x300 with 10m geometry because it
    never unions polygons and never loops point-by-point in Python.
    """
    from matplotlib.path import Path
 
    print(f"  Building {mask_res} land mask at {res}x{res} "
          f"(edge blur sigma={edge_blur}) ...")
 
    lons = np.linspace(lon_range[0], lon_range[1], res)
    lats = np.linspace(lat_range[0], lat_range[1], res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
 
    binary = np.zeros(res * res, dtype=bool)
 
    geometries = list(
        cfeature.NaturalEarthFeature("physical", "land", mask_res).geometries()
    )
 
    for geom in geometries:
        polys = geom.geoms if hasattr(geom, "geoms") else [geom]
        for poly in polys:
            ext = np.array(poly.exterior.coords)
            # Quick bbox pre-filter: skip polygons entirely outside view
            if (ext[:, 0].max() < lon_range[0] or ext[:, 0].min() > lon_range[1] or
                    ext[:, 1].max() < lat_range[0] or ext[:, 1].min() > lat_range[1]):
                continue
            binary |= Path(ext).contains_points(points)
 
    binary = binary.reshape(lon_grid.shape).astype(float)
 
    if edge_blur > 0:
        soft  = gaussian_filter(binary, sigma=edge_blur)
        alpha = np.clip(soft / soft.max(), 0, 1)
    else:
        alpha = binary
 
    print("  Land mask ready.")
    return alpha
 
 
# ---------------------------------------------------------------------------
# KDE 
# ---------------------------------------------------------------------------
 
def build_grid(lon_range, lat_range, res=GRID_RES):
    X, Y = np.mgrid[
        lon_range[0]:lon_range[1]:complex(res),
        lat_range[0]:lat_range[1]:complex(res),
    ]
    return X, Y, np.vstack([X.ravel(), Y.ravel()])
 
 
def kde_unweighted_sites(df_sites, positions, X):
    """Plot (A) — each unique site location counts equally."""
    values = np.vstack([df_sites["Long"], df_sites["Lat"]])
    kernel = gaussian_kde(values)
    Z = kernel(positions).reshape(X.shape)
    return Z * len(df_sites)
 
 
def kde_weighted_dates(df_dates, positions, X):
    """Plot (B) — sites with more 14C dates contribute proportionally more."""
    weight_map = (df_dates
                  .groupby(["Long", "Lat"])
                  .size()
                  .reset_index(name="n_dates"))
    values  = np.vstack([weight_map["Long"].values, weight_map["Lat"].values])
    weights = weight_map["n_dates"].values.astype(float)
    kernel  = gaussian_kde(values, weights=weights)
    Z = kernel(positions).reshape(X.shape)
    return Z * df_dates.shape[0]
 
 
# ---------------------------------------------------------------------------
# Map generator
# ---------------------------------------------------------------------------
 
def generate_map(region: dict, t_min: int, t_max: int):
    os.chdir(WORKING_DIR)
 
    name      = region["name"]
    lon_range = region["lon"]
    lat_range = region["lat"]
    safe_name = name.replace(" ", "_").replace("/", "-")
    filename  = f"{safe_name}_{t_min}-{t_max}BP_KDE.png"
 
    # -- Load & filter ------------------------------------------------------
    print(f"\n  Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, low_memory=False).dropna(subset=["Lat", "Long"])
    df["MedianCalBP"] = pd.to_numeric(df["MedianCalBP"], errors="coerce")
    df = df.dropna(subset=["MedianCalBP"])
    df = df[(df["MedianCalBP"] >= t_min) & (df["MedianCalBP"] <= t_max)]
    print(f"  Dates in temporal window ({t_min}–{t_max} Cal BP): {len(df):,}")
 
    mask = (
        (df["Long"] >= lon_range[0]) & (df["Long"] <= lon_range[1]) &
        (df["Lat"]  >= lat_range[0]) & (df["Lat"]  <= lat_range[1])
    )
    df_region = df[mask].copy()
    print(f"  Dates in region '{name}': {len(df_region):,}")
 
    if len(df_region) < 3:
        print("  ERROR: fewer than 3 data points — KDE cannot be computed.")
        return
 
    df_sites = df_region.drop_duplicates(subset=["Lat", "Long"])
    print(f"  Unique site locations: {len(df_sites):,}")
 
    if len(df_sites) < 3:
        print("  ERROR: fewer than 3 unique sites. Broaden the region/time window.")
        return
 
    # -- KDE surfaces -------------------------------------------------------
    print("  Computing KDE surfaces...")
    X, Y, positions = build_grid(lon_range, lat_range)
    Z_sites = kde_unweighted_sites(df_sites, positions, X)
    Z_dates = kde_weighted_dates(df_region, positions, X)
    alpha_mask = build_land_mask(lon_range, lat_range)  
 
    # -- Figure -------------------------------------------------------------
    projection = ccrs.PlateCarree()
    fig, axes  = plt.subplots(
        2, 1, figsize=(10, 13),
        subplot_kw={"projection": projection},
        facecolor="white",
    )
 
    panels = [
        {
            "ax":    axes[0],
            "Z":     Z_sites,
            "title": f"(A) Unweighted Site Density  -  {len(df_sites):,} unique sites",
            "label": "Relative site density",
            "cmap":  "magma",
        },
        {
            "ax":    axes[1],
            "Z":     Z_dates,
            "title": f"(B) Date-Weighted Density  -  {len(df_region):,} radiocarbon dates",
            "label": "Relative date density",
            "cmap":  "inferno",
        },
    ]
 
    for p in panels:
        ax = p["ax"]
        ax.set_facecolor("white")
        ax.set_extent([lon_range[0], lon_range[1],
                       lat_range[0], lat_range[1]], crs=projection)
 
        # -- Base geography ----------------------------------------------------
        ax.add_feature(cfeature.OCEAN,     facecolor="white",   zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor="#3d3d3d", zorder=1)
        ax.add_feature(cfeature.LAKES,     facecolor="white",   zorder=2)
 
        # -- KDE: apply soft land alpha so gradient fades at coastlines --------
        Z_raw  = p["Z"].T                               # (lat, lon)
        vmax   = np.nanpercentile(Z_raw, 99.5)
        norm   = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
 
        # Convert raw density to RGBA using the colormap + norm
        cmap_obj  = plt.get_cmap(p["cmap"]).copy()
        rgba      = cmap_obj(norm(np.clip(Z_raw, 0, vmax)))  # (lat, lon, 4)
 
        # Multiply alpha channel by the soft land mask → ocean cells fade out
        rgba[..., 3] = alpha_mask * 0.92   # max opacity 0.92 keeps land base visible
 
        ax.imshow(
            rgba,
            origin="lower",
            extent=[*lon_range, *lat_range],
            transform=projection,
            interpolation="bilinear",       # bilinear smooths pixel edges further
            zorder=3,
        )
 
        # -- Overlay clean coastlines on top of the KDE ------------------------
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5,  edgecolor="#1a1a1a", zorder=4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor="#555555",
                       linestyle=":", alpha=0.8, zorder=4)
 
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.03, shrink=0.80,
                          aspect=25)
        cb.set_label(p["label"], fontsize=10)
        cb.ax.tick_params(labelsize=10)
        cb.set_ticks([])   
 
        # -- Gridlines ---------------------------------------------------------
        gl = ax.gridlines(crs=projection, draw_labels=True, linewidth=0.3,
                          color="#aaaaaa", alpha=0.6, linestyle="--", zorder=5)
        gl.top_labels   = False
        gl.right_labels = False
        gl.xformatter   = cticker.LongitudeFormatter()
        gl.yformatter   = cticker.LatitudeFormatter()
        gl.xlabel_style = {"size": 10, "color": "#333333"}
        gl.ylabel_style = {"size": 10, "color": "#333333"}
 
        ax.set_title(p["title"], fontsize=12, fontweight="bold",
                     pad=6, color="#1a1a1a", loc="left")
 
    # -- Figure-level labels ---------------------------------------------------
    fig.suptitle(
        f"P3K14C Radiocarbon Database  —  {name}\n"
        f"{t_max:,}–{t_min:,} Cal BP",
        fontsize=14, fontweight="bold", color="#1a1a1a", y=0.995,
    )
 
    plt.subplots_adjust(hspace=0.08, top=0.955, bottom=0.03,
                        left=0.05, right=0.96)
 
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved → {filename}")
    plt.close()
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    region       = prompt_region()
    t_min, t_max = prompt_time_range()
    generate_map(region, t_min, t_max)