"""Paleoenvironmental proxy data fetchers: PANGAEA, Neotoma, NOAA GISP2.

Ported from Scripts/05_Human_Climate_Interaction.py. fetch_neotoma()
includes the `nearby_ids` -> `nearby_siteids` bugfix applied to the
original script before this migration began (it previously raised
NameError whenever this Neotoma-querying path was actually exercised).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d

from paleopy.geo import haversine_km
from paleopy.resilience import sg_smooth

GISP2_URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
    "summit/gisp2/isotopes/gisp2_temp_accum_alley2000.txt"
)
NEOTOMA_API = "https://api.neotomadb.org/v2.0"


def fetch_pangaea(
    keyword: str, site_lat: float, site_lon: float, env_radius_km: float,
    time_min: float, time_max: float,
) -> pd.DataFrame | None:
    """Search PANGAEA by keyword via its Elasticsearch API. Returns a
    standardized DataFrame [age_bp, value, variable, site_name,
    dataset_type, dist_km, doi], or None if nothing usable was found.
    """
    print(f"\n  [PANGAEA] Searching: '{keyword}' within {env_radius_km} km...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            "https://ws.pangaea.de/es/pangaea/panmd/_search",
            params={"q": keyword, "size": 50, "_source": "true"},
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        print(f"  PANGAEA status: {response.status_code}")
        response.raise_for_status()
        results_raw = response.json()
        raw_hits = results_raw.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"  PANGAEA search failed: {e}")
        return None

    print(f"  PANGAEA hits: {len(raw_hits)}")
    if not raw_hits:
        return None

    rows = []
    for hit in raw_hits:
        src = hit.get("_source", {})
        doi = src.get("URI", "")
        if not doi:
            continue

        lat = src.get("meanPosition", {}).get("lat")
        lon = src.get("meanPosition", {}).get("lon")
        if lat is None:
            n = src.get("northBoundLatitude")
            s = src.get("southBoundLatitude")
            lat = (n + s) / 2 if n is not None and s is not None else None
        if lon is None:
            e = src.get("eastBoundLongitude")
            w = src.get("westBoundLongitude")
            lon = (e + w) / 2 if e is not None and w is not None else None
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue

        dist = haversine_km(site_lat, site_lon, lat, lon)
        if dist > env_radius_km:
            continue

        site_name = ""
        xml_thumb = src.get("xml-thumb", "")
        if "<md:title>" in xml_thumb:
            try:
                site_name = xml_thumb.split("<md:title>")[1].split("</md:title>")[0][:60]
            except IndexError:
                pass
        if not site_name:
            site_name = doi

        print(f"    + {site_name[:50]:<50}  dist={dist:.0f} km  doi={doi}")

        xml = src.get("xml", "")
        child_ids = []
        if "collectionChilds" in xml:
            try:
                child_str = xml.split('key="collectionChilds" value="')[1].split('"')[0]
                child_ids = [c.replace("D", "") for c in child_str.split(",")]
            except IndexError:
                pass

        dois_to_fetch = (
            [f"https://doi.org/10.1594/PANGAEA.{cid}" for cid in child_ids]
            if child_ids else [doi]
        )

        for fetch_doi in dois_to_fetch:
            data_url = fetch_doi.replace("https://doi.org/", "https://doi.pangaea.de/") + "?format=textfile"
            try:
                time.sleep(0.3)
                dr = requests.get(data_url, headers=headers, timeout=30)
                dr.raise_for_status()
                text = dr.text

                lines = text.splitlines()
                header_end = 0
                for i, line in enumerate(lines):
                    if line.startswith("*/"):
                        header_end = i + 1
                        break
                if header_end == 0:
                    continue

                data_lines = lines[header_end:]
                if not data_lines:
                    continue

                col_line = data_lines[0].split("\t")
                age_col = None
                for j, col in enumerate(col_line):
                    if any(term in col.lower() for term in
                           ["age", "cal bp", "ka bp", "cal yr", "yr bp"]):
                        age_col = j
                        break
                if age_col is None:
                    continue

                skip_terms = {"latitude", "longitude", "depth", "event",
                              "elevation", "sample", "label", "comment",
                              "reference", "age", "date"}
                val_cols = [(j, col) for j, col in enumerate(col_line)
                            if j != age_col and not any(s in col.lower() for s in skip_terms)]

                for line in data_lines[1:]:
                    parts = line.split("\t")
                    if len(parts) <= age_col:
                        continue
                    try:
                        raw_age = float(parts[age_col])
                        if raw_age < 200:
                            raw_age *= 1000
                        if not (time_min <= raw_age <= time_max):
                            continue
                        for j, col_name in val_cols:
                            if j < len(parts) and parts[j].strip():
                                try:
                                    val = float(parts[j])
                                    rows.append({
                                        "age_bp": raw_age,
                                        "value": val,
                                        "variable": col_name,
                                        "site_name": site_name[:60],
                                        "dataset_type": keyword,
                                        "dist_km": dist,
                                        "doi": fetch_doi,
                                    })
                                except ValueError:
                                    pass
                    except ValueError:
                        continue
            except Exception as e:
                print(f"    WARNING downloading {fetch_doi}: {e}")
                continue

    if not rows:
        print("  PANGAEA: no usable data rows in time window.")
        return None

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(time_min, time_max)].copy()
    print(f"  PANGAEA: {len(df):,} observations in time window")
    return df if len(df) >= 5 else None


def fetch_neotoma(
    dtypes: list, site_lat: float, site_lon: float, env_radius_km: float,
    time_min: float, time_max: float,
) -> pd.DataFrame | None:
    """Query the Neotoma API for sites/datasets/samples within
    env_radius_km, for each dataset type in dtypes.
    """
    deg = env_radius_km / 111.0
    rows = []

    print(f"\n  [Neotoma] Searching sites within {env_radius_km} km...")
    sites_url = (
        f"{NEOTOMA_API}/data/sites"
        f"?bbox={site_lon-deg:.4f},{site_lat-deg:.4f},"
        f"{site_lon+deg:.4f},{site_lat+deg:.4f}&limit=100"
    )
    try:
        r = requests.get(sites_url, timeout=30)
        r.raise_for_status()
        all_sites = r.json().get("data", [])
    except Exception as e:
        print(f"  Neotoma sites query failed: {e}")
        return None

    nearby_siteids = []
    for s in all_sites:
        if not isinstance(s, dict):
            continue
        geo_raw = s.get("geography", "{}")
        try:
            geo = json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
            coords = geo.get("coordinates", [None, None])
            slon, slat = coords[0], coords[1]
            if slat and slon:
                dist = haversine_km(site_lat, site_lon, slat, slon)
                if dist <= env_radius_km:
                    nearby_siteids.append(s.get("siteid"))
        except Exception:
            continue

    print(f"  Neotoma sites within radius: {len(nearby_siteids)}")
    if not nearby_siteids:
        return None

    for dtype in dtypes:
        ids_str = ",".join(str(i) for i in nearby_siteids)
        ds_url = (
            f"{NEOTOMA_API}/data/datasets"
            f"?siteid={ids_str}"
            f"&datasettype={dtype.replace(' ', '%20')}&limit=200"
        )
        try:
            r = requests.get(ds_url, timeout=30)
            r.raise_for_status()
            datasets = r.json().get("data", [])
        except Exception as e:
            print(f"  WARNING: {e}")
            continue

        print(f"  Neotoma '{dtype}' datasets: {len(datasets)}")
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            try:
                si = ds.get("site", {})
                if not isinstance(si, dict):
                    continue
                name = si.get("sitename", "?")
                geo_raw = si.get("geography", "{}")
                try:
                    geo = json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
                    coords = geo.get("coordinates", [None, None])
                    slon, slat = coords[0], coords[1]
                    dist = haversine_km(site_lat, site_lon, slat, slon) if slat and slon else None
                except Exception:
                    dist = None

                site_ds = si.get("datasets", [])
                if not site_ds:
                    continue
                dsid = site_ds[0].get("datasetid")
                if dsid is None:
                    continue

                print(f"    + {name:<35}  dist={dist:.0f} km  id={dsid}" if dist else f"    + {name}  id={dsid}")
                time.sleep(0.3)
                sr = requests.get(f"{NEOTOMA_API}/data/downloads/{dsid}", timeout=30)
                sr.raise_for_status()
                for item in sr.json().get("data", []):
                    for sample in item.get("samples", []):
                        age = sample.get("age")
                        if age is None:
                            continue
                        for datum in sample.get("data", []):
                            rows.append({
                                "age_bp": float(age),
                                "value": datum.get("value"),
                                "variable": datum.get("variablename", ""),
                                "site_name": name,
                                "dataset_type": dtype,
                                "dist_km": dist,
                                "doi": "",
                            })
            except Exception as e:
                print(f"    WARNING: {e}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(time_min, time_max)].copy()
    print(f"  Neotoma: {len(df):,} observations in time window")
    return df if len(df) >= 5 else None


def fetch_gisp2(
    cache_path: str, time_min: float, time_max: float, resolution: float
) -> tuple:
    """Download (or load cached) NOAA GISP2 ice-core temperature series,
    interpolated onto [time_min, time_max] at step=resolution and smoothed.
    """
    cache = Path(cache_path)
    df = None
    if cache.is_file():
        c = pd.read_csv(cache)
        if "Age_BP" in c.columns:
            print(f"  [GISP2] Loading cached data from {cache_path}")
            df = c
    if df is None:
        print("  [GISP2] Downloading from NOAA...")
        resp = requests.get(GISP2_URL, timeout=30)
        resp.raise_for_status()
        rows, in_data = [], False
        for line in resp.text.splitlines():
            if "Age" in line and "Temperature" in line:
                in_data = True
                continue
            if not in_data:
                continue
            if "Accumulation" in line:
                break
            p = line.strip().split()
            if len(p) >= 2:
                try:
                    rows.append((float(p[0]) * 1000, float(p[1])))
                except ValueError:
                    pass
        df = pd.DataFrame(rows, columns=["Age_BP", "Temp_C"]).sort_values("Age_BP")
        df.to_csv(cache_path, index=False)
        print(f"  [GISP2] {len(df):,} points -> {cache_path}")

    ages = np.arange(time_min, time_max + resolution, resolution)
    f = interp1d(df["Age_BP"], df["Temp_C"], bounds_error=False, fill_value=np.nan)
    return ages, sg_smooth(f(ages))


def env_series_from_df(
    env_df: pd.DataFrame, time_min: float, time_max: float, resolution: float, env_bin_yr: float
) -> tuple:
    """Z-score each proxy variable individually before averaging (so
    proxies with wildly different units/scales are commensurable), then
    linearly detrend the composite to remove orbital-scale long-term
    trends.
    """
    ages = np.arange(time_min, time_max + resolution, resolution)
    composite = np.full(len(ages), np.nan)
    n_proxies = 0

    env_df = env_df.copy()

    for var_name, grp in env_df.groupby("variable"):
        if len(grp) < 5:
            continue

        grp_sorted = grp.sort_values("age_bp")
        var_series = np.full(len(ages), np.nan)

        for i, yr in enumerate(ages):
            window = grp_sorted[
                (grp_sorted["age_bp"] >= yr - env_bin_yr / 2)
                & (grp_sorted["age_bp"] < yr + env_bin_yr / 2)
            ]
            if len(window) > 0:
                var_series[i] = window["value"].mean()

        mask = np.isfinite(var_series)
        if mask.sum() < 5:
            continue
        idx = np.where(mask)[0]
        var_interp = np.interp(np.arange(len(ages)), idx, var_series[mask])
        var_interp[:idx[0]] = np.nan
        var_interp[idx[-1]:] = np.nan
        var_series = var_interp

        mu, sd = np.nanmean(var_series), np.nanstd(var_series)
        if sd < 1e-10:
            continue
        z_series = (var_series - mu) / sd

        print(f"    Proxy '{var_name[:40]}': {mask.sum()} pts  mean={mu:.3f}  sd={sd:.3f}")

        if np.all(np.isnan(composite)):
            composite = z_series.copy()
        else:
            composite = np.nanmean(np.stack([composite, z_series]), axis=0)
        n_proxies += 1

    print(f"  Composite built from {n_proxies} proxy variable(s)")

    if n_proxies == 0:
        return ages, np.full(len(ages), np.nan)

    from scipy.signal import detrend as scipy_detrend

    finite_mask = np.isfinite(composite)
    if finite_mask.sum() > 10:
        composite[finite_mask] = scipy_detrend(composite[finite_mask])

    return ages, sg_smooth(composite)


# ---------------------------------------------------------------------------
# 06-specific proxy fetchers (Scripts/06_Composite_Human_Environment.py)
# ---------------------------------------------------------------------------
#
# These are NOT parameterized variants of fetch_gisp2/fetch_pangaea/
# fetch_neotoma above - they return a different column schema
# ([age_bp, value, variable, source], with the search keyword/dtype baked
# into `variable`) suited to wide-format PCA input, whereas the 05
# versions return per-variable-name breakdowns and (for PANGAEA) expand
# parent collections into child datasets, which 06 does not do. Verified
# genuinely different during porting - kept as separate functions rather
# than forcing a shared interface that would risk subtle behavioral drift.


def fetch_gisp2_proxy(cache_path: str = "gisp2_cache.csv") -> pd.DataFrame:
    """06's GISP2 fetcher: returns a [age_bp, value, variable, source]
    DataFrame for the wide-format PCA proxy matrix (vs. fetch_gisp2's
    (ages, smoothed_values) tuple for direct plotting).
    """
    print(" [GISP2] Fetching temperature ...")
    cache = Path(cache_path)
    if cache.is_file():
        df = pd.read_csv(cache)
        print(f" [GISP2] Loaded from cache ({len(df):,} pts)")
    else:
        resp = requests.get(GISP2_URL, timeout=30)
        resp.raise_for_status()
        rows, in_data = [], False
        for line in resp.text.splitlines():
            if "Age" in line and "Temperature" in line:
                in_data = True
                continue
            if not in_data:
                continue
            if "Accumulation" in line:
                break
            p = line.strip().split()
            if len(p) >= 2:
                try:
                    rows.append((float(p[0]) * 1000, float(p[1])))
                except ValueError:
                    pass
        df = pd.DataFrame(rows, columns=["age_bp", "value"]).sort_values("age_bp")
        df["variable"] = "GISP2_Temp_C"
        df["source"] = "GISP2"
        df.to_csv(cache, index=False)
        print(f" [GISP2] {len(df):,} points cached")
    df["variable"] = "GISP2_Temp_C"
    df["source"] = "GISP2"
    return df[["age_bp", "value", "variable", "source"]]


def fetch_pangaea_proxy(
    keyword: str, site_lat: float, site_lon: float, env_radius_km: float,
    time_min: float, time_max: float,
) -> pd.DataFrame | None:
    print(f"\n [PANGAEA] '{keyword}' within {env_radius_km} km ...")
    try:
        resp = requests.get(
            "https://ws.pangaea.de/es/pangaea/panmd/_search",
            params={"q": keyword, "size": 30, "_source": "true"},
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
    except Exception as e:
        print(f" PANGAEA failed: {e}")
        return None

    rows = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    for hit in hits:
        src = hit.get("_source", {})
        doi = src.get("URI", "")
        if not doi:
            continue
        lat = src.get("meanPosition", {}).get("lat")
        lon = src.get("meanPosition", {}).get("lon")
        if lat is None:
            n = src.get("northBoundLatitude")
            s = src.get("southBoundLatitude")
            lat = (n + s) / 2 if n is not None and s is not None else None
        if lon is None:
            e = src.get("eastBoundLongitude")
            w = src.get("westBoundLongitude")
            lon = (e + w) / 2 if e is not None and w is not None else None
        if lat is None or lon is None:
            continue
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if haversine_km(site_lat, site_lon, lat, lon) > env_radius_km:
            continue

        data_url = doi.replace("https://doi.org/", "https://doi.pangaea.de/") + "?format=textfile"
        try:
            time.sleep(0.3)
            dr = requests.get(data_url, headers=headers, timeout=30)
            dr.raise_for_status()
            text = dr.text
            lines = text.splitlines()
            he = next((i + 1 for i, l in enumerate(lines) if l.startswith("*/")), 0)
            if he == 0:
                continue
            data_lines = lines[he:]
            if not data_lines:
                continue
            col_line = data_lines[0].split("\t")
            age_col = next(
                (j for j, c in enumerate(col_line)
                 if any(t in c.lower() for t in ["age", "cal bp", "ka bp", "cal yr", "yr bp"])),
                None,
            )
            if age_col is None:
                continue
            skip = {"latitude", "longitude", "depth", "event", "elevation",
                    "sample", "label", "comment", "reference", "age", "date"}
            val_cols = [(j, c) for j, c in enumerate(col_line) if j != age_col and not any(s in c.lower() for s in skip)]
            for line in data_lines[1:]:
                parts = line.split("\t")
                if len(parts) <= age_col:
                    continue
                try:
                    raw_age = float(parts[age_col])
                    if raw_age < 200:
                        raw_age *= 1000
                    if not (time_min <= raw_age <= time_max):
                        continue
                    for j, col_name in val_cols:
                        if j < len(parts) and parts[j].strip():
                            try:
                                rows.append({
                                    "age_bp": raw_age,
                                    "value": float(parts[j]),
                                    "variable": f"PANGAEA_{keyword[:15]}_{col_name[:20]}",
                                    "source": "PANGAEA",
                                })
                            except ValueError:
                                pass
                except ValueError:
                    continue
        except Exception:
            continue

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(time_min, time_max)]
    print(f" PANGAEA '{keyword}': {len(df):,} observations")
    return df if len(df) >= 5 else None


def fetch_neotoma_proxy(
    dtypes: list, site_lat: float, site_lon: float, env_radius_km: float,
    time_min: float, time_max: float,
) -> pd.DataFrame | None:
    deg = env_radius_km / 111.0
    rows = []
    print(f"\n [Neotoma] Searching within {env_radius_km} km ...")
    try:
        r = requests.get(
            f"{NEOTOMA_API}/data/sites"
            f"?bbox={site_lon-deg:.4f},{site_lat-deg:.4f},"
            f"{site_lon+deg:.4f},{site_lat+deg:.4f}&limit=100",
            timeout=30,
        )
        r.raise_for_status()
        all_sites = r.json().get("data", [])
    except Exception as e:
        print(f" Neotoma failed: {e}")
        return None

    nearby_ids = []
    for s in all_sites:
        if not isinstance(s, dict):
            continue
        geo_raw = s.get("geography", "{}")
        try:
            geo = json.loads(geo_raw) if isinstance(geo_raw, str) else geo_raw
            coords = geo.get("coordinates", [None, None])
            if coords[0] and coords[1]:
                if haversine_km(site_lat, site_lon, coords[1], coords[0]) <= env_radius_km:
                    nearby_ids.append(s.get("siteid"))
        except Exception:
            continue

    print(f" Neotoma sites: {len(nearby_ids)}")
    if not nearby_ids:
        return None

    for dtype in dtypes:
        ids_str = ",".join(str(i) for i in nearby_ids)
        try:
            r = requests.get(
                f"{NEOTOMA_API}/data/datasets?siteid={ids_str}"
                f"&datasettype={dtype.replace(' ', '%20')}&limit=200",
                timeout=30,
            )
            r.raise_for_status()
            datasets = r.json().get("data", [])
        except Exception:
            continue
        print(f" Neotoma '{dtype}': {len(datasets)} datasets")
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            try:
                si = ds.get("site", {})
                if not isinstance(si, dict):
                    continue
                site_ds = si.get("datasets", [])
                if not site_ds:
                    continue
                dsid = site_ds[0].get("datasetid")
                if dsid is None:
                    continue
                time.sleep(0.3)
                sr = requests.get(f"{NEOTOMA_API}/data/downloads/{dsid}", timeout=30)
                sr.raise_for_status()
                for item in sr.json().get("data", []):
                    for sample in item.get("samples", []):
                        age = sample.get("age")
                        if age is None:
                            continue
                        for datum in sample.get("data", []):
                            rows.append({
                                "age_bp": float(age),
                                "value": datum.get("value"),
                                "variable": f"Neotoma_{dtype[:10]}_{datum.get('variablename', '')[:20]}",
                                "source": "Neotoma",
                            })
            except Exception:
                continue

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["age_bp"].between(time_min, time_max)]
    print(f" Neotoma total: {len(df):,} observations")
    return df if len(df) >= 5 else None


def collect_proxies(
    site_lat: float, site_lon: float, env_radius_km: float, time_min: float, time_max: float,
    use_gisp2: bool, pangaea_keywords: list, neotoma_types: list, gisp2_cache_path: str,
) -> pd.DataFrame:
    """Orchestrate 06's multi-keyword/multi-dtype proxy collection: GISP2
    (optional) + all PANGAEA keywords + all Neotoma dataset types,
    combined into one long-format DataFrame for wide-format PCA binning.
    """
    print(f"  PANGAEA keywords : {pangaea_keywords}")
    print(f"  Neotoma types    : {neotoma_types}")
    print(f"  Search radius    : {env_radius_km} km")
    print("  Database order   : PANGAEA -> Neotoma -> GISP2")

    frames = []
    if use_gisp2:
        frames.append(fetch_gisp2_proxy(gisp2_cache_path))
    for kw in pangaea_keywords:
        df = fetch_pangaea_proxy(kw, site_lat, site_lon, env_radius_km, time_min, time_max)
        if df is not None:
            frames.append(df)
    df_neo = fetch_neotoma_proxy(neotoma_types, site_lat, site_lon, env_radius_km, time_min, time_max)
    if df_neo is not None:
        frames.append(df_neo)

    if not frames:
        raise ValueError("No proxy data retrieved.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["age_bp"].between(time_min, time_max)]
    n_vars = combined["variable"].nunique()
    print(f"\n  Total proxy observations : {len(combined):,}")
    print(f"  Distinct proxy variables : {n_vars}")
    for v in combined["variable"].unique():
        n = (combined["variable"] == v).sum()
        print(f"    {v:<55} {n:>5} obs")
    return combined


def get_env_data(
    site_lat: float, site_lon: float, env_radius_km: float,
    time_min: float, time_max: float, resolution: float,
    env_proxy_keyword: str, env_neotoma_type: list, env_bin_yr: float,
    use_gisp2_only: bool, gisp2_cache_path: str, env_output_path: str,
) -> tuple:
    """Orchestrate PANGAEA -> Neotoma -> GISP2 fallback chain. Returns
    (ages, values, label).
    """
    print(f"  PANGAEA keyword : '{env_proxy_keyword}'")
    print(f"  Neotoma type    : {env_neotoma_type}")
    print(f"  Search radius   : {env_radius_km} km")
    print("  Database order  : PANGAEA -> Neotoma -> GISP2")

    if use_gisp2_only:
        print("\n  USE_GISP2_ONLY=True, skipping live database queries.")
        ages, vals = fetch_gisp2(gisp2_cache_path, time_min, time_max, resolution)
        return ages, vals, "GISP2 Ice Core Temperature (Alley 2000, NOAA)"

    all_rows = []

    pang_df = fetch_pangaea(env_proxy_keyword, site_lat, site_lon, env_radius_km, time_min, time_max)
    if pang_df is not None:
        all_rows.append(pang_df)
        print(f"  PANGAEA: SUCCESS ({len(pang_df):,} rows)")

    neot_df = fetch_neotoma(env_neotoma_type, site_lat, site_lon, env_radius_km, time_min, time_max)
    if neot_df is not None:
        all_rows.append(neot_df)
        print(f"  Neotoma: SUCCESS ({len(neot_df):,} rows)")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(env_output_path, index=False)
        ages, vals = env_series_from_df(combined, time_min, time_max, resolution, env_bin_yr)

        sources = []
        if pang_df is not None:
            sources.append(f"PANGAEA '{env_proxy_keyword}'")
        if neot_df is not None:
            sources.append(f"Neotoma {env_neotoma_type}")
        label = " + ".join(sources) + f" ({env_radius_km} km)"

        if np.sum(~np.isnan(vals)) >= 10:
            return ages, vals, label

    print(f"\n  No local proxy data found within {env_radius_km} km.")
    print("  Falling back to GISP2 ice core (global signal).")
    ages, vals = fetch_gisp2(gisp2_cache_path, time_min, time_max, resolution)
    return ages, vals, f"GISP2 Ice Core (fallback - no local data within {env_radius_km} km)"
