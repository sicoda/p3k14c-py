"""Spatial KDE (kernel density estimation) over p3k14c site/date locations.

Ported from Scripts/07_KDE.py. Pure numerics only; land-masking and the
cartopy figure live in paleopy.mapping since they require the optional
`cartopy` extra.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def filter_region_and_time(
    df: pd.DataFrame, lon_range: list, lat_range: list, t_min: int, t_max: int
) -> tuple:
    """Filter to the given bounding box and Cal BP time window.

    Returns (df_region, df_sites) where df_sites is the unique
    (Lat, Long) locations within df_region.
    """
    df = df.dropna(subset=["Lat", "Long"]).copy()
    df["MedianCalBP"] = pd.to_numeric(df["MedianCalBP"], errors="coerce")
    df = df.dropna(subset=["MedianCalBP"])
    df = df[(df["MedianCalBP"] >= t_min) & (df["MedianCalBP"] <= t_max)]

    mask = (
        (df["Long"] >= lon_range[0]) & (df["Long"] <= lon_range[1])
        & (df["Lat"] >= lat_range[0]) & (df["Lat"] <= lat_range[1])
    )
    df_region = df[mask].copy()
    df_sites = df_region.drop_duplicates(subset=["Lat", "Long"])
    return df_region, df_sites


def build_grid(lon_range: list, lat_range: list, res: int) -> tuple:
    X, Y = np.mgrid[
        lon_range[0]:lon_range[1]:complex(res),
        lat_range[0]:lat_range[1]:complex(res),
    ]
    return X, Y, np.vstack([X.ravel(), Y.ravel()])


def kde_unweighted_sites(df_sites: pd.DataFrame, positions: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Plot (A): each unique site location counts equally."""
    values = np.vstack([df_sites["Long"], df_sites["Lat"]])
    kernel = gaussian_kde(values)
    Z = kernel(positions).reshape(X.shape)
    return Z * len(df_sites)


def kde_weighted_dates(df_dates: pd.DataFrame, positions: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Plot (B): sites with more 14C dates contribute proportionally more."""
    weight_map = df_dates.groupby(["Long", "Lat"]).size().reset_index(name="n_dates")
    values = np.vstack([weight_map["Long"].values, weight_map["Lat"].values])
    weights = weight_map["n_dates"].values.astype(float)
    kernel = gaussian_kde(values, weights=weights)
    Z = kernel(positions).reshape(X.shape)
    return Z * df_dates.shape[0]
